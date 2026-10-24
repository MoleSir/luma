//! Kind inference: validate the graph and assign a `DTypeKindTag` to every value.

use std::collections::HashSet;

use luma_tensor::DTypeKindTag;

use crate::graph::{Graph, Node, NodeOp, Scalar, ValueId};
use crate::{CompileResult, ExecuteError};

// ============================================================================
//    Kind inference — validate the graph and assign a kind to every value
// ============================================================================

fn expect_kind(got: DTypeKindTag, expected: DTypeKindTag, what: &'static str) -> CompileResult<()> {
    if got != expected {
        return Err(ExecuteError::KindMismatch { what, expected, got }.into());
    }
    Ok(())
}

fn scalar_matches(s: &Scalar, kind: DTypeKindTag) -> CompileResult<()> {
    let ok = match (s, kind) {
        (Scalar::F64(_), DTypeKindTag::Float) | (Scalar::I64(_), DTypeKindTag::Int) | (Scalar::Bool(_), DTypeKindTag::Bool) => true,
        _ => false,
    };
    if ok { Ok(()) } else { Err(ExecuteError::ScalarKindMismatch { scalar: *s, kind }.into()) }
}

fn infer_node_kind(node: &Node, kinds: &[DTypeKindTag]) -> CompileResult<DTypeKindTag> {
    let input = |i: usize| -> CompileResult<DTypeKindTag> {
        let id = *node.inputs.get(i).ok_or_else(|| ExecuteError::ExpectMoreInputs { op: node.op.to_string(), got: node.inputs.len() })?;
        Ok(kinds.get(id).copied().ok_or_else(|| ExecuteError::UnknownValue(id))?)
    };
    let same_pair = || -> CompileResult<DTypeKindTag> {
        let a = input(0)?;
        let b = input(1)?;
        if a != b || a == DTypeKindTag::Bool {
            return Err(ExecuteError::PairKindMismatch { op: node.op.to_string(), a, b }.into());
        }
        Ok(a)
    };

    Ok(match &node.op {
        NodeOp::Constant => return Err(ExecuteError::UnexpectedConstantNode.into()),
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
            expect_kind(input(0)?, DTypeKindTag::Float, "Unary")?;
            DTypeKindTag::Float
        }
        NodeOp::UnaryI(_) => {
            expect_kind(input(0)?, DTypeKindTag::Int, "UnaryI")?;
            DTypeKindTag::Int
        }
        NodeOp::FloatUnary(_) => {
            expect_kind(input(0)?, DTypeKindTag::Float, "FloatUnary")?;
            DTypeKindTag::Float
        }
        NodeOp::Cmp(_) => {
            same_pair()?;
            DTypeKindTag::Bool
        }
        NodeOp::CmpScalar(s, _) => {
            let a = input(0)?;
            scalar_matches(s, a)?;
            DTypeKindTag::Bool
        }
        NodeOp::And | NodeOp::Or | NodeOp::Xor => {
            expect_kind(input(0)?, DTypeKindTag::Bool, "bool logic")?;
            expect_kind(input(1)?, DTypeKindTag::Bool, "bool logic")?;
            DTypeKindTag::Bool
        }
        NodeOp::Not => {
            expect_kind(input(0)?, DTypeKindTag::Bool, "Not")?;
            DTypeKindTag::Bool
        }
        NodeOp::Cast(dt) => dt.kind(),
        NodeOp::Reduce(_, _) => {
            let a = input(0)?;
            if a == DTypeKindTag::Bool {
                return Err(ExecuteError::UnsupportedOp("Reduce on Bool (use ReduceAll/ReduceAny)".to_string()).into());
            }
            a
        }
        NodeOp::ReduceAll(_) | NodeOp::ReduceAny(_) => {
            expect_kind(input(0)?, DTypeKindTag::Bool, "bool reduce")?;
            DTypeKindTag::Bool
        }
        NodeOp::ArgReduce(_, _) => {
            let a = input(0)?;
            if a == DTypeKindTag::Bool {
                return Err(ExecuteError::UnsupportedOp("ArgReduce on Bool".to_string()).into());
            }
            DTypeKindTag::Int
        }
        NodeOp::Matmul => same_pair()?,
        NodeOp::IndexSelect(_) | NodeOp::Gather(_) => {
            let a = input(0)?;
            expect_kind(input(1)?, DTypeKindTag::Int, "index tensor")?;
            if a == DTypeKindTag::Bool {
                return Err(ExecuteError::UnsupportedOp("index op on Bool".to_string()).into());
            }
            a
        }
        NodeOp::IndexAdd(_) | NodeOp::ScatterAdd(_) => {
            let a = input(0)?;
            expect_kind(input(1)?, DTypeKindTag::Int, "index tensor")?;
            let b = input(2)?;
            if a != b || a == DTypeKindTag::Bool {
                return Err(ExecuteError::PairKindMismatch { op: node.op.to_string(), a, b }.into());
            }
            a
        }
        NodeOp::Cat(_) => {
            let first = input(0)?;
            for i in 1..node.inputs.len() {
                let other = input(i)?;
                if other != first {
                    return Err(ExecuteError::CatKindMismatch { expected: first, got: other }.into());
                }
            }
            first
        }
        NodeOp::Softmax(_) | NodeOp::RmsNorm(_) => {
            expect_kind(input(0)?, DTypeKindTag::Float, "nn op")?;
            DTypeKindTag::Float
        }
        NodeOp::Pick => {
            expect_kind(input(0)?, DTypeKindTag::Bool, "Pick mask")?;
            if node.inputs.len() != 3 {
                return Err(ExecuteError::PickScalarUnsupported.into());
            }
            let a = input(1)?;
            let b = input(2)?;
            if a != b {
                return Err(ExecuteError::BranchKindMismatch { a, b }.into());
            }
            a
        }
        NodeOp::PickTrue(s) | NodeOp::PickFalse(s) => {
            expect_kind(input(0)?, DTypeKindTag::Bool, "Pick mask")?;
            let a = input(1)?;
            scalar_matches(s, a)?;
            a
        }
        NodeOp::Arange(_, _, _) => DTypeKindTag::Int,
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

pub(crate) fn infer_kinds(graph: &Graph) -> CompileResult<Vec<DTypeKindTag>> {
    let mut kinds = vec![DTypeKindTag::Float; graph.values.len()];
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
        let node = producer[v.id].ok_or_else(|| ExecuteError::DanglingValue(v.id))?;
        kinds[v.id] = infer_node_kind(node, &kinds)?;
    }
    Ok(kinds)
}
