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

    /// SSA: every input's producing node must appear *before* the consuming
    /// node in the node array (the executor runs nodes in array order). Value
    /// ids are just indices — a rewriting rule may insert a node whose output
    /// id is large but whose position is early.
    #[error("node {0} consumes %{1} produced by node {2} (producers must precede consumers)")]
    ProduceEarly(usize, ValueId, usize),

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
    // 容错版 producer：越界的输出跳过（随后由 NodeOutputOutOfRange 报错）。
    // verify 是最前端防御，不能对非法图 panic。
    let mut producer = vec![None; n_values];
    for (i, node) in graph.nodes.iter().enumerate() {
        for &out in &node.outputs {
            if out < n_values {
                producer[out] = Some(i);
            }
        }
    }

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
            // SSA 拓扑按节点位置：输入的生产者必须在该消费者之前执行。
            if let Some(p) = producer[input] {
                if p >= node_idx {
                    return Err(VerifyError::ProduceEarly(node_idx, input, p));
                }
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
