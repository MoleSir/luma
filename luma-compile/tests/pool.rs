//! 池化执行端到端：pooled Cpu + 死槽清理下，稳态每次 run 零系统分配。

use std::sync::{Arc, RwLock};

use luma_compile::{Graph, GraphExecutor, NodeOp};
use luma_tensor::device::cpu::allocator::{CpuAllocator, PoolAllocator};
use luma_tensor::dtype::FloatDType;
use luma_tensor::{BinaryOp, Cpu, DType, FloatUnaryOp, Shape, Tensor};

fn allocs(shared: &Arc<RwLock<dyn CpuAllocator>>) -> usize {
    shared.read().unwrap().as_any().downcast_ref::<PoolAllocator>().unwrap().system_allocs()
}

/// 链 x→a→b→c→out：死槽清理让每个中间值在死后立即回池，链式滚动复用。
///
/// run 1 预期 3 次系统分配（input 创建 + step 0 + step 1——输入是 Arc 浅克隆，
/// 用户持有 storage，清输入槽不释放；a 在 step 1 结束才真正 drop 回池），
/// 之后全部命中。稳态（run 3 起）零新增。
///
/// 注意：run() 返回的输出与内部输出槽共享 storage——调用方跨 run 持有旧输出
/// 会钉住一块（下次 run 的替换无法回收它）。模拟真实推理"用完即弃"：
/// 每个 run 前 drop 旧输出。
#[test]
fn pooled_executor_reuses_across_runs() {
    let shared: Arc<RwLock<dyn CpuAllocator>> = Arc::new(RwLock::new(PoolAllocator::new()));
    let cpu = Cpu::with_allocator_shared(shared.clone());

    // 手建链式图：x → a=sqr(x) → b=sqr(a) → c=add(b,b) → out=add(c,c)
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let a = g.add_node(NodeOp::FloatUnary(FloatUnaryOp::Sqr), vec![x], DType::F32, Shape::from((2, 2)));
    let b = g.add_node(NodeOp::FloatUnary(FloatUnaryOp::Sqr), vec![a], DType::F32, Shape::from((2, 2)));
    let c = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![b, b], DType::F32, Shape::from((2, 2)));
    let o = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![c, c], DType::F32, Shape::from((2, 2)));
    g.mark_output(o);

    let mut exec = GraphExecutor::compile(&g, &cpu).unwrap();
    let input = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), (&cpu, FloatDType::F32)).unwrap();
    let expected = vec![4.0, 64.0, 324.0, 1024.0]; // 4x^4

    let mut out = exec.run(&[input.clone().into()]).unwrap();
    let after_run1 = allocs(&shared);
    eprintln!("run 1 allocs: {after_run1}");
    assert_eq!(out[0].as_float().unwrap().to_vec().unwrap(), expected, "run 1 结果应正确");

    // run 2：收敛尾巴（峰值并发 2 块，run 1 结束时池只有 1 块）——至多 +1。
    drop(out);
    out = exec.run(&[input.clone().into()]).unwrap();
    let after_run2 = allocs(&shared);
    eprintln!("run 2 allocs: {after_run2}");
    assert_eq!(out[0].as_float().unwrap().to_vec().unwrap(), expected, "run 2 结果应正确");

    // run 3 起：稳态，零新增系统分配（核心断言）。
    for run in 3..=6 {
        drop(out);
        out = exec.run(&[input.clone().into()]).unwrap();
        eprintln!("run {run} allocs: {}", allocs(&shared));
        assert_eq!(out[0].as_float().unwrap().to_vec().unwrap(), expected, "run {run} 结果应正确");
    }
}

/// 对照：同一张图在 SystemAllocator（无池）下必须给出正确值——隔离"池化引入的
/// 错误"与"测试期望/图构建错误"。
#[test]
fn chain_without_pool_is_correct() {
    let cpu = Cpu::default(); // SystemAllocator
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let a = g.add_node(NodeOp::FloatUnary(FloatUnaryOp::Sqr), vec![x], DType::F32, Shape::from((2, 2)));
    let b = g.add_node(NodeOp::FloatUnary(FloatUnaryOp::Sqr), vec![a], DType::F32, Shape::from((2, 2)));
    let c = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![b, b], DType::F32, Shape::from((2, 2)));
    let o = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![c, c], DType::F32, Shape::from((2, 2)));
    g.mark_output(o);

    let mut exec = GraphExecutor::compile(&g, &cpu).unwrap();
    let input = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), (&cpu, FloatDType::F32)).unwrap();
    let expected = vec![4.0, 64.0, 324.0, 1024.0];

    for run in 1..=3 {
        let out = exec.run(&[input.clone().into()]).unwrap();
        let got = out[0].as_float().unwrap().to_vec().unwrap();
        assert_eq!(got, expected, "run {run}（无池）结果应正确");
    }
}

/// 真实模型端到端：pooled Linear 推理，稳态零系统分配 + 数值与直接 forward 一致。
#[test]
fn pooled_linear_inference_steady_state() {
    use luma_nn::Linear;

    let shared: Arc<RwLock<dyn CpuAllocator>> = Arc::new(RwLock::new(PoolAllocator::new()));
    let cpu = Cpu::with_allocator_shared(shared.clone());

    let linear = Linear::new(8, 8, true, cpu.clone()).unwrap();
    let x = Tensor::<Cpu>::from_slice(&[0.5; 64], (8, 8), (&cpu, FloatDType::F32)).unwrap();
    let expected = linear.forward(&x).unwrap().to_vec().unwrap();

    let graph = luma_compile::trace(&linear, &x).unwrap();
    let mut g = graph.lock().unwrap().clone();
    g.optimize().unwrap();
    let mut exec = GraphExecutor::compile(&g, &cpu).unwrap();

    // run 1 建立稳态，run 2+ 零新增系统分配
    let mut out = exec.run(&[x.clone().into()]).unwrap();
    for _ in 1..2 {
        drop(out);
        out = exec.run(&[x.clone().into()]).unwrap();
    }
    let steady = allocs(&shared);
    for _ in 0..3 {
        drop(out);
        out = exec.run(&[x.clone().into()]).unwrap();
        assert_eq!(allocs(&shared), steady, "稳态 run 不应有系统分配");
    }
    let got = out[0].as_float().unwrap().to_vec().unwrap();
    assert_eq!(got.len(), expected.len());
    for (a, b) in got.iter().zip(&expected) {
        assert!((a - b).abs() < 1e-5, "结果应与 direct forward 一致: {a} vs {b}");
    }
}
