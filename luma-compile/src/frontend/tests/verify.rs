use crate::frontend::verify::{VerifyError, verify};
use crate::{ConstData, Graph, Node, NodeOp, trace};
use luma_nn::Linear;
use luma_tensor::dtype::FloatDType;
use luma_tensor::{Cpu, DType, FloatUnaryOp, Shape, Tensor};

/// A minimal valid graph: one input, one Relu node, one output.
fn valid_graph() -> Graph {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let y = g.add_node(NodeOp::FloatUnary(FloatUnaryOp::Relu), vec![x], DType::F32, Shape::from((2, 2)));
    g.mark_output(y);
    g
}

#[test]
fn valid_hand_built_graph_passes() {
    verify(&valid_graph()).unwrap();
}

#[test]
fn valid_traced_module_passes() {
    let linear = Linear::new(3, 4, true, Cpu).unwrap();
    let x = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0], (1, 3), FloatDType::F32).unwrap();
    let graph = trace(&linear, &x).unwrap();
    verify(&graph.lock().unwrap()).unwrap();
}

#[test]
fn value_id_mismatch() {
    let mut g = valid_graph();
    g.values[1].id = 42;
    assert!(matches!(verify(&g), Err(VerifyError::ValueIdMismatch(1, 42))));
}

#[test]
fn node_input_out_of_range() {
    let mut g = valid_graph();
    g.nodes.push(Node { op: NodeOp::Not, inputs: vec![99], outputs: vec![2] });
    assert!(matches!(verify(&g), Err(VerifyError::NodeInputOutOfRange(..))));
}

#[test]
fn node_output_out_of_range() {
    let mut g = valid_graph();
    g.nodes.push(Node { op: NodeOp::Not, inputs: vec![0], outputs: vec![7] });
    assert!(matches!(verify(&g), Err(VerifyError::NodeOutputOutOfRange(..))));
}

#[test]
fn node_with_two_outputs() {
    let mut g = valid_graph();
    g.nodes.push(Node { op: NodeOp::Not, inputs: vec![0], outputs: vec![1, 2] });
    assert!(matches!(verify(&g), Err(VerifyError::NodeOutputCount(..))));
}

#[test]
fn consumer_precedes_producer() {
    // 位置 0 的节点消费 %1，但 %1 由位置 1 的节点生产 → 违反节点位置拓扑
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    g.add_value(DType::F32, Shape::from((2, 2)));
    g.add_value(DType::F32, Shape::from((2, 2)));
    g.nodes.push(Node { op: NodeOp::Not, inputs: vec![1], outputs: vec![2] });
    g.nodes.push(Node { op: NodeOp::FloatUnary(FloatUnaryOp::Relu), inputs: vec![0], outputs: vec![1] });
    assert!(matches!(verify(&g), Err(VerifyError::ProduceEarly(..))));
}

#[test]
fn multi_producer() {
    let mut g = valid_graph();
    g.nodes.push(Node { op: NodeOp::Not, inputs: vec![0], outputs: vec![1] });
    assert!(matches!(verify(&g), Err(VerifyError::MultiProducer(1))));
}

#[test]
fn node_output_has_data() {
    let mut g = valid_graph();
    g.values[1].data = Some(ConstData(vec![]));
    assert!(matches!(verify(&g), Err(VerifyError::NodeOutputHasData(..))));
}

#[test]
fn phantom_value() {
    let mut g = Graph::default();
    g.add_value(DType::F32, Shape::from((2, 2)));
    assert!(matches!(verify(&g), Err(VerifyError::PhantomValue(0))));
}

#[test]
fn constant_marked_input() {
    let mut g = Graph::default();
    let c = g.add_constant(DType::F32, Shape::from((2, 2)), vec![0u8; 16]);
    g.mark_input(c);
    assert!(matches!(verify(&g), Err(VerifyError::ConstMarkedInput(0))));
}

#[test]
fn input_has_producer() {
    let mut g = Graph::default();
    let a = g.add_value(DType::F32, Shape::from((2, 2)));
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(a);
    g.mark_input(x);
    g.nodes.push(Node { op: NodeOp::Not, inputs: vec![a], outputs: vec![x] });
    assert!(matches!(verify(&g), Err(VerifyError::InputHasProducer(1))));
}

#[test]
fn input_out_of_range() {
    let mut g = valid_graph();
    g.inputs.push(99);
    assert!(matches!(verify(&g), Err(VerifyError::InputOutOfRange(..))));
}

#[test]
fn output_out_of_range() {
    let mut g = valid_graph();
    g.outputs.push(99);
    assert!(matches!(verify(&g), Err(VerifyError::OutputOutOfRange(..))));
}

#[test]
fn constant_node() {
    let mut g = Graph::default();
    g.add_value(DType::F32, Shape::from((2, 2)));
    g.nodes.push(Node { op: NodeOp::Constant, inputs: vec![], outputs: vec![1] });
    assert!(matches!(verify(&g), Err(VerifyError::ConstantNode(0))));
}
