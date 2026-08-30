use crate::opt::dce::dce;
use crate::opt::verify::verify;
use crate::{Graph, NodeOp, trace};
use luma_nn::Linear;
use luma_tensor::dtype::FloatDType;
use luma_tensor::{BinaryOp, Cpu, DType, FloatUnaryOp, Shape, Tensor, UnaryOp};

/// 无死代码的图必须原样保留（节点数、值数不变），且通过 verify。
#[test]
fn traced_linear_survives_dce() {
    let linear = Linear::new(3, 4, true, Cpu).unwrap();
    let x = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0], (1, 3), FloatDType::F32).unwrap();
    let g = trace(&linear, &x).unwrap();
    let g = g.lock().unwrap().clone();
    let (n_values, n_nodes) = (g.values.len(), g.nodes.len());

    let g2 = dce(g).unwrap();
    verify(&g2).unwrap();
    assert_eq!(g2.values.len(), n_values, "无死代码时 values 不能变");
    assert_eq!(g2.nodes.len(), n_nodes, "无死代码时 nodes 不能变");
}

/// 一条死链（relu→neg→sqr 无人消费）必须整体消除，同时保住活路径。
#[test]
fn dead_chain_removed() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let r = g.add_node(NodeOp::FloatUnary(FloatUnaryOp::Relu), vec![x], DType::F32, Shape::from((2, 2)));
    let n = g.add_node(NodeOp::Unary(UnaryOp::Neg), vec![r], DType::F32, Shape::from((2, 2)));
    let _s = g.add_node(NodeOp::FloatUnary(FloatUnaryOp::Sqr), vec![n], DType::F32, Shape::from((2, 2)));
    let out = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![x, x], DType::F32, Shape::from((2, 2)));
    g.mark_output(out);

    let g2 = dce(g).unwrap();
    verify(&g2).unwrap();
    // 只剩输入叶子 + 输出值，只剩 add 一个节点
    assert_eq!(g2.values.len(), 2);
    assert_eq!(g2.nodes.len(), 1);
}

/// 无人消费的常量叶子必须被剔除（死常量消除）。
#[test]
fn dead_constant_removed() {
    let mut g = Graph::default();
    let _c = g.add_constant(DType::F32, Shape::from(()), vec![0u8; 8]);
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let out = g.add_node(NodeOp::FloatUnary(FloatUnaryOp::Relu), vec![x], DType::F32, Shape::from((2, 2)));
    g.mark_output(out);

    let g2 = dce(g).unwrap();
    verify(&g2).unwrap();
    assert_eq!(g2.values.len(), 2, "死常量应被剔除，只剩输入 + 输出");
}

/// 输出直接是叶子（恒等模块）：不 panic，图原样保留。
#[test]
fn identity_output_leaf() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    g.mark_output(x);

    let g2 = dce(g).unwrap();
    verify(&g2).unwrap();
    assert_eq!(g2.values.len(), 1);
    assert!(g2.nodes.is_empty());
    assert_eq!(g2.outputs, vec![0]);
}
