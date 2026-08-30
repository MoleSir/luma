//! Kind inference: validate the graph and assign a `KindTag` to every value.

use std::collections::HashSet;

use luma_tensor::{Error, KindTag, Result};

use crate::graph::{Graph, Node, NodeOp, Scalar, ValueId};

// ============================================================================
//    Kind inference — validate the graph and assign a kind to every value
// ============================================================================

fn expect_kind(got: KindTag, expected: KindTag, what: &str) -> Result<()> {
    if got != expected {
        return Err(Error::Msg(format!("executor: {what}: expected {expected:?}, got {got:?}")));
    }
    Ok(())
}

fn scalar_matches(s: &Scalar, kind: KindTag) -> Result<()> {
    let ok = match (s, kind) {
        (Scalar::F64(_), KindTag::Float) | (Scalar::I64(_), KindTag::Int) | (Scalar::Bool(_), KindTag::Bool) => true,
        _ => false,
    };
    if ok { Ok(()) } else { Err(Error::Msg(format!("executor: scalar {s} does not match operand kind {kind:?}"))) }
}

fn infer_node_kind(node: &Node, kinds: &[KindTag]) -> Result<KindTag> {
    let input = |i: usize| -> Result<KindTag> {
        let id = *node
            .inputs
            .get(i)
            .ok_or_else(|| Error::Msg(format!("executor: {:?} expects more than {} inputs", node.op, node.inputs.len())))?;
        kinds.get(id).copied().ok_or_else(|| Error::Msg(format!("executor: unknown value {id}")))
    };
    let same_pair = || -> Result<KindTag> {
        let a = input(0)?;
        let b = input(1)?;
        if a != b || a == KindTag::Bool {
            return Err(Error::Msg(format!("executor: {:?} needs two operands of the same numeric kind, got {a:?}/{b:?}", node.op)));
        }
        Ok(a)
    };

    Ok(match &node.op {
        NodeOp::Constant => return Err(Error::Msg("executor: unexpected Constant node (constants are data-carrying leaves)".to_string())),
        NodeOp::Binary(_) => same_pair()?,
        NodeOp::BinaryScalarRhs(s, _) => {
            let a = input(0)?;
            scalar_matches(s, a)?;
            a
        }
        NodeOp::BinaryScalarLhs(s, _) => {
            let a = input(0)?;
            scalar_matches(s, a)?;
            a
        }
        NodeOp::Unary(_) => {
            expect_kind(input(0)?, KindTag::Float, "Unary")?;
            KindTag::Float
        }
        NodeOp::UnaryI(_) => {
            expect_kind(input(0)?, KindTag::Int, "UnaryI")?;
            KindTag::Int
        }
        NodeOp::FloatUnary(_) => {
            expect_kind(input(0)?, KindTag::Float, "FloatUnary")?;
            KindTag::Float
        }
        NodeOp::Cmp(_) => {
            same_pair()?;
            KindTag::Bool
        }
        NodeOp::CmpScalar(s, _) => {
            let a = input(0)?;
            scalar_matches(s, a)?;
            KindTag::Bool
        }
        NodeOp::And | NodeOp::Or | NodeOp::Xor => {
            expect_kind(input(0)?, KindTag::Bool, "bool logic")?;
            expect_kind(input(1)?, KindTag::Bool, "bool logic")?;
            KindTag::Bool
        }
        NodeOp::Not => {
            expect_kind(input(0)?, KindTag::Bool, "Not")?;
            KindTag::Bool
        }
        NodeOp::Cast(dt) => dt.kind(),
        NodeOp::Reduce(_, _) => {
            let a = input(0)?;
            if a == KindTag::Bool {
                return Err(Error::Msg("executor: Reduce on Bool (use ReduceAll/ReduceAny)".to_string()));
            }
            a
        }
        NodeOp::ReduceAll(_) | NodeOp::ReduceAny(_) => {
            expect_kind(input(0)?, KindTag::Bool, "bool reduce")?;
            KindTag::Bool
        }
        NodeOp::ArgReduce(_, _) => {
            let a = input(0)?;
            if a == KindTag::Bool {
                return Err(Error::Msg("executor: ArgReduce on Bool".to_string()));
            }
            KindTag::Int
        }
        NodeOp::Matmul => same_pair()?,
        NodeOp::IndexSelect(_) | NodeOp::Gather(_) => {
            let a = input(0)?;
            expect_kind(input(1)?, KindTag::Int, "index tensor")?;
            if a == KindTag::Bool {
                return Err(Error::Msg("executor: index op on Bool".to_string()));
            }
            a
        }
        NodeOp::IndexAdd(_) | NodeOp::ScatterAdd(_) => {
            let a = input(0)?;
            expect_kind(input(1)?, KindTag::Int, "index tensor")?;
            let b = input(2)?;
            if a != b || a == KindTag::Bool {
                return Err(Error::Msg(format!("executor: {:?} needs matching numeric operands, got {a:?}/{b:?}", node.op)));
            }
            a
        }
        NodeOp::Cat(_) => {
            let first = input(0)?;
            for i in 1..node.inputs.len() {
                if input(i)? != first {
                    return Err(Error::Msg("executor: Cat inputs must share one kind".to_string()));
                }
            }
            first
        }
        NodeOp::Softmax(_) | NodeOp::RmsNorm(_) => {
            expect_kind(input(0)?, KindTag::Float, "nn op")?;
            KindTag::Float
        }
        NodeOp::Pick => {
            expect_kind(input(0)?, KindTag::Bool, "Pick mask")?;
            if node.inputs.len() != 3 {
                return Err(Error::Msg(
                    "executor: scalar Pick is not yet supported — the IR does not record the scalar operand".to_string(),
                ));
            }
            let a = input(1)?;
            let b = input(2)?;
            if a != b {
                return Err(Error::Msg(format!("executor: Pick branches must share one kind, got {a:?}/{b:?}")));
            }
            a
        }
        NodeOp::PickTrue(s) | NodeOp::PickFalse(s) => {
            expect_kind(input(0)?, KindTag::Bool, "Pick mask")?;
            let a = input(1)?;
            scalar_matches(s, a)?;
            a
        }
        NodeOp::Arange(_, _, _) => KindTag::Int,
        NodeOp::Reshape
        | NodeOp::Transpose(..)
        | NodeOp::Permute(_)
        | NodeOp::Narrow(..)
        | NodeOp::Slice(..)
        | NodeOp::Broadcast
        | NodeOp::Squeeze(_)
        | NodeOp::Unsqueeze(_) => input(0)?,
    })
}

pub(crate) fn infer_kinds(graph: &Graph) -> Result<Vec<KindTag>> {
    let mut kinds = vec![KindTag::Float; graph.values.len()];
    let mut producer: Vec<Option<&Node>> = vec![None; graph.values.len()];
    for node in &graph.nodes {
        for &out in &node.outputs {
            if let Some(slot) = producer.get_mut(out) {
                *slot = Some(node);
            }
        }
    }
    let inputs: HashSet<ValueId> = graph.inputs.iter().copied().collect();

    // Leaves (constants + inputs) have small ids; op outputs are appended in
    // SSA order, so a forward pass always sees an input's kind first.
    for v in &graph.values {
        if v.data.is_some() || inputs.contains(&v.id) {
            kinds[v.id] = v.dtype.kind();
            continue;
        }
        let node = producer[v.id]
            .ok_or_else(|| Error::Msg(format!("executor: dangling value {} — no constant, input, or producing node", v.id)))?;
        kinds[v.id] = infer_node_kind(node, &kinds)?;
    }
    Ok(kinds)
}
