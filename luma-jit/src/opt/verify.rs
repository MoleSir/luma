//! Structural validation for a [`Graph`](crate::Graph): every invariant the
//! executor (and the optimization passes) rely on, checked in one linear scan.
//!
//! `GraphExecutor::compile` runs `verify` first, so a broken graph fails here
//! with a precise message instead of panicking deep inside lowering.

use crate::graph::NodeOp;
use crate::{Graph, ValueId};

#[derive(thiserror::Error, Debug)]
pub enum VerifyError {
    /// `values[i].id` must equal its position `i` — ids index the `values` vec.
    #[error("value id mismatch: values[{0}].id = {1}")]
    ValueIdMismatch(usize, usize),

    #[error("node {0} input %{1} out of range (values.len() = {2})")]
    NodeInputOutOfRange(usize, ValueId, usize),

    #[error("node {0} output %{1} out of range (values.len() = {2})")]
    NodeOutputOutOfRange(usize, ValueId, usize),

    /// The IR is single-output: every node produces exactly one value.
    #[error("node {0} has {1} outputs (the IR is single-output)")]
    NodeOutputCount(usize, usize),

    /// SSA: every input must be produced before the node that consumes it.
    #[error("node {0} produces %{1} but consumes %{2} (a consumer must precede its producer)")]
    ProduceEarly(usize, ValueId, ValueId),

    #[error("value %{0} is produced by more than one node")]
    MultiProducer(ValueId),

    /// Op outputs are data-less; only constant leaves carry data.
    #[error("node {0} output %{1} carries data (op outputs must be data-less)")]
    NodeOutputHasData(usize, ValueId),

    /// Data-less, unproduced, unmarked value: the classic trace-time mistake
    /// (`Tensor::zeros`/`full`/`rand` inside a traced forward).
    #[error(
        "value %{0} has no producer, no data and is not a graph input \
         (traced from Tensor::zeros/full/rand inside forward?)"
    )]
    PhantomValue(ValueId),

    #[error("constant %{0} is marked as a graph input (inputs must be data-less)")]
    ConstMarkedInput(ValueId),

    #[error("graph input %{0} is produced by a node (inputs must be leaves)")]
    InputHasProducer(ValueId),

    #[error("graph input %{0} out of range (values.len() = {1})")]
    InputOutOfRange(ValueId, usize),

    #[error("graph output %{0} out of range (values.len() = {1})")]
    OutputOutOfRange(ValueId, usize),

    /// `NodeOp::Constant` exists in the enum but is never emitted — constants
    /// are data-carrying leaves.
    #[error("node {0} is a Constant node (constants are data-carrying leaves, not nodes)")]
    ConstantNode(usize),
}

/// Check every structural invariant of `graph` in one pass.
pub fn verify(graph: &Graph) -> Result<(), VerifyError> {
    // 1. ids must equal their position in `values` (ids index the vec).
    for (i, v) in graph.values.iter().enumerate() {
        if v.id != i {
            return Err(VerifyError::ValueIdMismatch(i, v.id));
        }
    }

    let n_values = graph.values.len();
    // produced[id] = true once some node produces value `id`.
    let mut produced = vec![false; n_values];

    // 2-6. per-node checks.
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        if matches!(&node.op, NodeOp::Constant) {
            return Err(VerifyError::ConstantNode(node_idx));
        }
        if node.outputs.len() != 1 {
            return Err(VerifyError::NodeOutputCount(node_idx, node.outputs.len()));
        }
        let out = node.outputs[0];

        for &input in &node.inputs {
            if input >= n_values {
                return Err(VerifyError::NodeInputOutOfRange(node_idx, input, n_values));
            }
            if input >= out {
                return Err(VerifyError::ProduceEarly(node_idx, out, input));
            }
        }
        if out >= n_values {
            return Err(VerifyError::NodeOutputOutOfRange(node_idx, out, n_values));
        }
        if graph.values[out].data.is_some() {
            return Err(VerifyError::NodeOutputHasData(node_idx, out));
        }
        if produced[out] {
            return Err(VerifyError::MultiProducer(out));
        }
        produced[out] = true;
    }

    // 7. input/output bookkeeping.
    let mut is_input = vec![false; n_values];
    for &id in &graph.inputs {
        if id >= n_values {
            return Err(VerifyError::InputOutOfRange(id, n_values));
        }
        is_input[id] = true;
    }
    for &id in &graph.outputs {
        if id >= n_values {
            return Err(VerifyError::OutputOutOfRange(id, n_values));
        }
    }

    // 8. leaf trichotomy: every value is exactly one of
    //    constant (data) | input | computed (produced by a node).
    //    A node-produced value carrying data was already rejected above, so
    //    only the three reachable violations remain.
    for (id, v) in graph.values.iter().enumerate() {
        match (v.data.is_some(), is_input[id], produced[id]) {
            (true, true, _) => return Err(VerifyError::ConstMarkedInput(id)),
            (false, true, true) => return Err(VerifyError::InputHasProducer(id)),
            (false, false, false) => return Err(VerifyError::PhantomValue(id)),
            _ => {}
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use luma_nn::Linear;
    use luma_tensor::dtype::FloatDType;
    use luma_tensor::{Cpu, DType, FloatUnaryOp, Shape, Tensor};

    use super::{VerifyError, verify};
    use crate::{ConstData, Graph, Node, NodeOp, trace};

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
        let mut g = valid_graph();
        g.nodes.push(Node { op: NodeOp::Not, inputs: vec![1], outputs: vec![0] });
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
}
