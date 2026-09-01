use crate::frontend::verify::verify;
use crate::{Graph, NodeOp, Scalar};
use luma_nn::Linear;
use luma_tensor::dtype::FloatDType;
use luma_tensor::{BinaryOp, Cpu, DType, FloatUnaryOp, Shape, Tensor};

/// 端到端：trace Linear → optimize → compile/run，结果与直接 forward 一致。
#[test]
fn optimize_linear_matches_forward() {
    let linear = Linear::new(3, 4, true, Cpu::default()).unwrap();
    let x = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0], (1, 3), FloatDType::F32).unwrap();

    let graph = crate::trace(&linear, &x).unwrap();
    let mut g = graph.lock().unwrap().clone();
    let n_nodes_before = g.nodes.len();
    g.optimize().unwrap();
    verify(&g).unwrap();
    assert!(g.nodes.len() <= n_nodes_before, "优化后节点数不应增加");

    let expected = linear.forward(&x).unwrap().to_vec().unwrap();
    let mut exec = g.compile(&Cpu::default()).unwrap();
    let out = exec.run(&[x.into()]).unwrap();
    let got = out[0].as_float().unwrap().to_vec().unwrap();
    // 容差断言：fold 折叠 transpose(weight) 后 matmul 输入布局变化，
    // CPU 内核可能走不同累加路径 → 1 ulp 级差异（既有测试同样用 1e-5）
    assert_eq!(got.len(), expected.len());
    for (a, b) in got.iter().zip(&expected) {
        assert!((a - b).abs() < 1e-5, "graph execution must match the module forward: {a} vs {b}");
    }
}

/// 混合场景：恒等元 + 常量子图 + 重复子图一起被优化掉。
#[test]
fn optimize_mixed_graph() {
    let filled = |v: f32| vec![v; 4].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);

    // 常量子图：2 + 3（可折叠）
    let a = g.add_constant(DType::F32, Shape::from((2, 2)), filled(2.0));
    let b = g.add_constant(DType::F32, Shape::from((2, 2)), filled(3.0));
    let s = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![a, b], DType::F32, Shape::from((2, 2)));

    // 恒等元：x + 0（可消）
    let z = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(0.0), BinaryOp::Add), vec![x], DType::F32, Shape::from((2, 2)));

    // 重复子图：sqr(z) 算两次（CSE 合并）
    let r1 = g.add_node(NodeOp::FloatUnary(FloatUnaryOp::Sqr), vec![z], DType::F32, Shape::from((2, 2)));
    let r2 = g.add_node(NodeOp::FloatUnary(FloatUnaryOp::Sqr), vec![z], DType::F32, Shape::from((2, 2)));

    // 输出：s + r1 与 s + r2
    let o1 = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![s, r1], DType::F32, Shape::from((2, 2)));
    let o2 = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![s, r2], DType::F32, Shape::from((2, 2)));
    g.mark_output(o1);
    g.mark_output(o2);

    let n_before = g.nodes.len();
    g.optimize().unwrap();
    verify(&g).unwrap();

    // x+0 消掉 1 个，常量子图折叠 1 个，sqr 合并 1 个，s 相关 add 合并 1 个 → 至少减 4
    assert!(g.nodes.len() <= n_before - 4, "混合场景应至少减少 4 个节点，实际 {} -> {}", n_before, g.nodes.len());

    // 执行：x² + 5（两个输出相同）
    let input = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), FloatDType::F32).unwrap();
    let expected = input.clone().sqr().unwrap().add_scalar(5.0).unwrap().to_vec().unwrap();
    let mut exec = g.compile(&Cpu::default()).unwrap();
    let out = exec.run(&[input.into()]).unwrap();
    assert_close(&out[0].as_float().unwrap().to_vec().unwrap(), &expected, 1e-5);
    assert_close(&out[1].as_float().unwrap().to_vec().unwrap(), &expected, 1e-5);
}

fn assert_close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol, "mismatch at {i}: {x} vs {y}");
    }
}

/// optimize 是幂等的：优化后再优化不改变图。
#[test]
fn optimize_idempotent() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let a = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(0.0), BinaryOp::Add), vec![x], DType::F32, Shape::from((2, 2)));
    let b = g.add_node(NodeOp::FloatUnary(FloatUnaryOp::Sqr), vec![a], DType::F32, Shape::from((2, 2)));
    g.mark_output(b);

    g.optimize().unwrap();
    let once = g.clone();
    g.optimize().unwrap();
    verify(&g).unwrap();
    assert_eq!(g.nodes.len(), once.nodes.len(), "二次优化不应再改变图");
    assert_eq!(g.values.len(), once.values.len());
}
