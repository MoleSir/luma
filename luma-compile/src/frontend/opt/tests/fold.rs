use crate::frontend::opt::fold::fold;
use crate::frontend::opt::verify::verify;
use crate::{Graph, NodeOp, Scalar};
use luma_tensor::dtype::FloatDType;
use luma_tensor::{BinaryOp, Cpu, DType, Shape, Tensor, UnaryOp};

/// arange（无输入常量节点）折叠成常量叶子，compile 只物化一次。
#[test]
fn arange_folded() {
    let mut g = Graph::default();
    let out = g.add_node(NodeOp::Arange(0, 5, 1), vec![], DType::I32, Shape::from((5,)));
    g.mark_output(out);

    let g2 = fold(g).unwrap();
    verify(&g2).unwrap();
    assert!(g2.nodes.is_empty(), "arange 应折叠成常量叶子");
    assert_eq!(g2.values.len(), 1);
    assert!(g2.values[0].data.is_some(), "应留下带数据的常量");

    // 端到端：执行结果等于 arange
    let mut exec = g2.compile(&Cpu).unwrap();
    let out = exec.run(&[]).unwrap();
    let got = out[0].as_int().unwrap().to_vec().unwrap();
    assert_eq!(got, vec![0, 1, 2, 3, 4]);
}

/// 常量子图折叠：2.0 + 3.0 → 常量 5.0。
#[test]
fn const_binary_folded() {
    let mut g = Graph::default();
    let a = g.add_constant(DType::F32, Shape::from(()), 2.0f32.to_le_bytes().to_vec());
    let b = g.add_constant(DType::F32, Shape::from(()), 3.0f32.to_le_bytes().to_vec());
    let out = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![a, b], DType::F32, Shape::from(()));
    g.mark_output(out);

    let g2 = fold(g).unwrap();
    verify(&g2).unwrap();
    assert!(g2.nodes.is_empty());
    assert_eq!(g2.values.len(), 1);
    let t = Tensor::<Cpu>::from_bytes(g2.values[0].data.as_ref().unwrap().0.as_slice(), Shape::from(()), (&Cpu, FloatDType::F32)).unwrap();
    assert_eq!(t.to_vec().unwrap(), vec![5.0]);
}

/// 常量节点作为输出：折叠后 outputs 指向常量（-2.0）。
#[test]
fn const_output_folded() {
    let mut g = Graph::default();
    let a = g.add_constant(DType::F32, Shape::from(()), 2.0f32.to_le_bytes().to_vec());
    let out = g.add_node(NodeOp::Unary(UnaryOp::Neg), vec![a], DType::F32, Shape::from(()));
    g.mark_output(out);

    let g2 = fold(g).unwrap();
    verify(&g2).unwrap();
    assert!(g2.nodes.is_empty());
    let t = Tensor::<Cpu>::from_bytes(g2.values[0].data.as_ref().unwrap().0.as_slice(), Shape::from(()), (&Cpu, FloatDType::F32)).unwrap();
    assert_eq!(t.to_vec().unwrap(), vec![-2.0]);
}

/// 输入依赖的节点不折叠。
#[test]
fn input_dependent_kept() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let c = g.add_constant(DType::F32, Shape::from(()), 3.0f32.to_le_bytes().to_vec());
    let out = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![x, c], DType::F32, Shape::from((2, 2)));
    g.mark_output(out);

    let g2 = fold(g).unwrap();
    verify(&g2).unwrap();
    assert_eq!(g2.nodes.len(), 1, "输入依赖节点不折叠");
}

/// 端到端：混合图 fold 后执行一致（x + (2 + 3) → x + 5）。
#[test]
fn folded_graph_executes_same() {
    let filled = |v: f32| vec![v; 4].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let a = g.add_constant(DType::F32, Shape::from((2, 2)), filled(2.0));
    let b = g.add_constant(DType::F32, Shape::from((2, 2)), filled(3.0));
    let s = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![a, b], DType::F32, Shape::from((2, 2)));
    let out = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![x, s], DType::F32, Shape::from((2, 2)));
    g.mark_output(out);

    let g2 = fold(g).unwrap();
    verify(&g2).unwrap();
    assert_eq!(g2.nodes.len(), 1, "常量子图折叠后只剩一个 add");

    let input = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), FloatDType::F32).unwrap();
    let expected = input.clone().add_scalar(5.0).unwrap().to_vec().unwrap();
    let mut exec = g2.compile(&Cpu).unwrap();
    let out = exec.run(&[input.into()]).unwrap();
    assert_eq!(out[0].as_float().unwrap().to_vec().unwrap(), expected);
}

/// 常量 scalar 参与运算（BinaryScalarRhs 常量标量本身在 op 里，不在图里）。
#[test]
fn scalar_const_kept() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let out = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(3.0), BinaryOp::Mul), vec![x], DType::F32, Shape::from((2, 2)));
    g.mark_output(out);

    let g2 = fold(g).unwrap();
    verify(&g2).unwrap();
    assert_eq!(g2.nodes.len(), 1, "标量在 op 属性里，没有常量节点可折叠");
}
