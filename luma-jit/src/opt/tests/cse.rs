use crate::opt::cse::cse;
use crate::opt::verify::verify;
use crate::{Graph, NodeOp, Scalar};
use luma_tensor::dtype::FloatDType;
use luma_tensor::{BinaryOp, Cpu, DType, FloatUnaryOp, Shape, Tensor};

/// 相同子图合并：两个 add(x, y) → 一个。
#[test]
fn duplicate_removed() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    let y = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    g.mark_input(y);
    let a = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![x, y], DType::F32, Shape::from((2, 2)));
    let b = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![x, y], DType::F32, Shape::from((2, 2)));
    g.mark_output(a);
    g.mark_output(b);

    let g2 = cse(g).unwrap();
    verify(&g2).unwrap();
    assert_eq!(g2.nodes.len(), 1, "重复的 add 应合并成一个");
    // dce 后 values = [x, y, add.out]，两个输出都指向 add.out (id 2)
    assert_eq!(g2.outputs, vec![2, 2], "两个输出都指向同一个节点");
}

/// 不同 op 不合并；相同 op 不同输入不合并。
#[test]
fn different_kept() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    let y = g.add_value(DType::F32, Shape::from((2, 2)));
    let z = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    g.mark_input(y);
    g.mark_input(z);
    let a = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![x, y], DType::F32, Shape::from((2, 2)));
    let b = g.add_node(NodeOp::Binary(BinaryOp::Mul), vec![x, y], DType::F32, Shape::from((2, 2)));
    let c = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![x, z], DType::F32, Shape::from((2, 2)));
    g.mark_output(a);
    g.mark_output(b);
    g.mark_output(c);

    let g2 = cse(g).unwrap();
    verify(&g2).unwrap();
    assert_eq!(g2.nodes.len(), 3, "add/mul、add(x,y)/add(x,z) 都不相同，全部保留");
}

/// 端到端：合并后执行结果与合并前一致（两个重复子图的输出值相同）。
#[test]
fn cse_executes_same() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let a = g.add_node(NodeOp::FloatUnary(FloatUnaryOp::Sqr), vec![x], DType::F32, Shape::from((2, 2)));
    let b = g.add_node(NodeOp::FloatUnary(FloatUnaryOp::Sqr), vec![x], DType::F32, Shape::from((2, 2)));
    g.mark_output(a);
    g.mark_output(b);

    let g2 = cse(g).unwrap();
    verify(&g2).unwrap();
    assert_eq!(g2.nodes.len(), 1);

    let input = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), FloatDType::F32).unwrap();
    let expected = input.sqr().unwrap().to_vec().unwrap();
    let mut exec = g2.compile(&Cpu).unwrap();
    let out = exec.run(&[input.into()]).unwrap();
    assert_eq!(out[0].as_float().unwrap().to_vec().unwrap(), expected);
    assert_eq!(out[1].as_float().unwrap().to_vec().unwrap(), expected, "两个输出都应指向合并后的值");
}

/// 标量参数 op 也能合并（key 包含标量）。
#[test]
fn scalar_op_duplicate_removed() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let a = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(2.0), BinaryOp::Mul), vec![x], DType::F32, Shape::from((2, 2)));
    let b = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(2.0), BinaryOp::Mul), vec![x], DType::F32, Shape::from((2, 2)));
    let c = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(3.0), BinaryOp::Mul), vec![x], DType::F32, Shape::from((2, 2)));
    g.mark_output(a);
    g.mark_output(b);
    g.mark_output(c);

    let g2 = cse(g).unwrap();
    verify(&g2).unwrap();
    assert_eq!(g2.nodes.len(), 2, "x*2 合并，x*3 保留");
}

/// 常量叶子去重不在 CSE 范围（节点级）；输入是同一常量时节点仍合并。
#[test]
fn shared_const_input_merged() {
    let c = vec![2.0f32; 4].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let mut g = Graph::default();
    let k = g.add_constant(DType::F32, Shape::from((2, 2)), c);
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let a = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![x, k], DType::F32, Shape::from((2, 2)));
    let b = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![x, k], DType::F32, Shape::from((2, 2)));
    g.mark_output(a);
    g.mark_output(b);

    let g2 = cse(g).unwrap();
    verify(&g2).unwrap();
    assert_eq!(g2.nodes.len(), 1);
}
