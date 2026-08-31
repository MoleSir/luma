//! Lowering: nodes become type-specialised steps.

use luma_tensor::{KindTag, ReduceOp, Shape};

use crate::graph::{Graph, Node, NodeOp, Scalar, ValueId};
use crate::{ExecuteError, JitResult};

use super::step::{Slot, Step, ViewStep};

// ============================================================================
//    Lowering — nodes become type-specialised steps
// ============================================================================

fn lower_node(node: &Node, kinds: &[KindTag], slots: &[Slot], graph: &Graph) -> JitResult<Step> {
    let slot = |i: usize| -> JitResult<Slot> {
        let id = *node.inputs.get(i).ok_or_else(|| ExecuteError::ExpectMoreInputs { op: node.op.to_string(), got: node.inputs.len() })?;
        Ok(slots.get(id).copied().ok_or_else(|| ExecuteError::UnknownValue(id))?)
    };
    let out = slots.get(node.outputs[0]).copied().ok_or_else(|| ExecuteError::NodeWithoutOutput(node.op.to_string()))?;

    let num = |id: ValueId| kinds[id];
    let in_num = |i: usize| -> JitResult<KindTag> {
        let id = *node.inputs.get(i).expect("infer validated input count");
        Ok(num(id))
    };
    let in_shape = |i: usize| -> &Shape { &graph.values[node.inputs[i]].shape };
    let out_shape = || -> &Shape { &graph.values[node.outputs[0]].shape };

    Ok(match &node.op {
        NodeOp::Constant => return Err(ExecuteError::UnexpectedConstantNode.into()),
        NodeOp::Binary(op) => match in_num(0)? {
            KindTag::Float => Step::BinaryF(*op, slot(0)?, slot(1)?, out),
            KindTag::Int => Step::BinaryI(*op, slot(0)?, slot(1)?, out),
            KindTag::Bool => unreachable!("kind inference validated"),
        },
        NodeOp::BinaryScalarRhs(s, op) => match s {
            Scalar::F64(v) => Step::BinaryScalarRhsF(*op, *v, slot(0)?, out),
            Scalar::I64(v) => Step::BinaryScalarRhsI(*op, *v, slot(0)?, out),
            Scalar::Bool(_) => unreachable!("kind inference validated"),
        },
        NodeOp::BinaryScalarLhs(s, op) => match s {
            Scalar::F64(v) => Step::BinaryScalarLhsF(*op, *v, slot(0)?, out),
            Scalar::I64(v) => Step::BinaryScalarLhsI(*op, *v, slot(0)?, out),
            Scalar::Bool(_) => unreachable!("kind inference validated"),
        },
        NodeOp::Unary(op) => Step::UnaryF(op.clone(), slot(0)?, out),
        NodeOp::UnaryI(op) => Step::UnaryI(op.clone(), slot(0)?, out),
        NodeOp::FloatUnary(op) => Step::FloatUnaryF(*op, slot(0)?, out),
        NodeOp::Cmp(op) => match in_num(0)? {
            KindTag::Float => Step::CmpF(*op, slot(0)?, slot(1)?, out),
            KindTag::Int => Step::CmpI(*op, slot(0)?, slot(1)?, out),
            KindTag::Bool => unreachable!("kind inference validated"),
        },
        NodeOp::CmpScalar(s, op) => match s {
            Scalar::F64(v) => Step::CmpScalarF(*op, *v, slot(0)?, out),
            Scalar::I64(v) => Step::CmpScalarI(*op, *v, slot(0)?, out),
            Scalar::Bool(_) => unreachable!("kind inference validated"),
        },
        NodeOp::And => Step::And(slot(0)?, slot(1)?, out),
        NodeOp::Or => Step::Or(slot(0)?, slot(1)?, out),
        NodeOp::Xor => Step::Xor(slot(0)?, slot(1)?, out),
        NodeOp::Not => Step::Not(slot(0)?, out),
        NodeOp::Cast(dt) => match in_num(0)? {
            KindTag::Float => Step::CastFromF(*dt, slot(0)?, out),
            KindTag::Int => Step::CastFromI(*dt, slot(0)?, out),
            KindTag::Bool => Step::CastFromB(*dt, slot(0)?, out),
        },
        NodeOp::Reduce(op, dims) => {
            let keepdim = out_shape().rank() == in_shape(0).rank();
            match in_num(0)? {
                KindTag::Float => Step::ReduceF(*op, dims.clone(), keepdim, slot(0)?, out),
                KindTag::Int => {
                    if *op == ReduceOp::Mean {
                        return Err(
                            ExecuteError::UnsupportedOp("int mean is not yet supported (public API is float-only)".to_string()).into()
                        );
                    }
                    Step::ReduceI(*op, dims.clone(), keepdim, slot(0)?, out)
                }
                KindTag::Bool => unreachable!("kind inference validated"),
            }
        }
        NodeOp::ReduceAll(_) | NodeOp::ReduceAny(_) => {
            return Err(ExecuteError::UnsupportedOp(
                "ReduceAll/ReduceAny are not yet supported (no public dim-wise bool reduction)".to_string(),
            )
            .into());
        }
        NodeOp::ArgReduce(dim, take_max) => {
            let keepdim = out_shape().rank() == in_shape(0).rank();
            match in_num(0)? {
                KindTag::Float => Step::ArgReduceF(*dim, *take_max, keepdim, slot(0)?, out),
                KindTag::Int => Step::ArgReduceI(*dim, *take_max, keepdim, slot(0)?, out),
                KindTag::Bool => unreachable!("kind inference validated"),
            }
        }
        NodeOp::Matmul => match in_num(0)? {
            KindTag::Float => Step::MatmulF(slot(0)?, slot(1)?, out),
            KindTag::Int => Step::MatmulI(slot(0)?, slot(1)?, out),
            KindTag::Bool => unreachable!("kind inference validated"),
        },
        NodeOp::IndexSelect(dim) => match in_num(0)? {
            KindTag::Float => Step::IndexSelectF(*dim, slot(0)?, slot(1)?, out),
            KindTag::Int => Step::IndexSelectI(*dim, slot(0)?, slot(1)?, out),
            KindTag::Bool => unreachable!("kind inference validated"),
        },
        NodeOp::Gather(dim) => match in_num(0)? {
            KindTag::Float => Step::GatherF(*dim, slot(0)?, slot(1)?, out),
            KindTag::Int => Step::GatherI(*dim, slot(0)?, slot(1)?, out),
            KindTag::Bool => unreachable!("kind inference validated"),
        },
        NodeOp::IndexAdd(dim) => match in_num(0)? {
            KindTag::Float => Step::IndexAddF(*dim, slot(0)?, slot(1)?, slot(2)?, out),
            KindTag::Int => Step::IndexAddI(*dim, slot(0)?, slot(1)?, slot(2)?, out),
            KindTag::Bool => unreachable!("kind inference validated"),
        },
        NodeOp::ScatterAdd(dim) => match in_num(0)? {
            KindTag::Float => Step::ScatterAddF(*dim, slot(0)?, slot(1)?, slot(2)?, out),
            KindTag::Int => Step::ScatterAddI(*dim, slot(0)?, slot(1)?, slot(2)?, out),
            KindTag::Bool => unreachable!("kind inference validated"),
        },
        NodeOp::Cat(dim) => {
            let inputs: Vec<Slot> = (0..node.inputs.len()).map(slot).collect::<JitResult<_>>()?;
            match in_num(0)? {
                KindTag::Float => Step::CatF(*dim, inputs, out),
                KindTag::Int => Step::CatI(*dim, inputs, out),
                KindTag::Bool => Step::CatB(*dim, inputs, out),
            }
        }
        NodeOp::Softmax(dim) => Step::Softmax(*dim, slot(0)?, out),
        NodeOp::RmsNorm(eps) => Step::RmsNorm(*eps, slot(0)?, slot(1)?, out),
        NodeOp::Pick => match in_num(1)? {
            KindTag::Float => Step::PickF(slot(0)?, slot(1)?, slot(2)?, out),
            KindTag::Int => Step::PickI(slot(0)?, slot(1)?, slot(2)?, out),
            KindTag::Bool => Step::PickB(slot(0)?, slot(1)?, slot(2)?, out),
        },
        NodeOp::PickTrue(s) => match (s, in_num(1)?) {
            (Scalar::F64(v), KindTag::Float) => Step::PickTrueF(*v, slot(0)?, slot(1)?, out),
            (Scalar::I64(v), KindTag::Int) => Step::PickTrueI(*v, slot(0)?, slot(1)?, out),
            (Scalar::Bool(v), KindTag::Bool) => Step::PickTrueB(*v, slot(0)?, slot(1)?, out),
            _ => unreachable!("kind inference validated"),
        },
        NodeOp::PickFalse(s) => match (s, in_num(1)?) {
            (Scalar::F64(v), KindTag::Float) => Step::PickFalseF(*v, slot(0)?, slot(1)?, out),
            (Scalar::I64(v), KindTag::Int) => Step::PickFalseI(*v, slot(0)?, slot(1)?, out),
            (Scalar::Bool(v), KindTag::Bool) => Step::PickFalseB(*v, slot(0)?, slot(1)?, out),
            _ => unreachable!("kind inference validated"),
        },
        NodeOp::Arange(start, end, step) => Step::Arange(*start, *end, *step, out),
        NodeOp::Reshape => Step::View(ViewStep::Reshape(out_shape().clone()), slot(0)?, out),
        NodeOp::Transpose(a, b) => Step::View(ViewStep::Transpose(*a, *b), slot(0)?, out),
        NodeOp::Permute(dims) => Step::View(ViewStep::Permute(dims.clone()), slot(0)?, out),
        NodeOp::Narrow(d, s, l) => Step::View(ViewStep::Narrow(*d, *s, *l), slot(0)?, out),
        NodeOp::Slice(d, s, e, st) => Step::View(ViewStep::Slice(*d, *s, *e, *st), slot(0)?, out),
        NodeOp::Broadcast => Step::View(ViewStep::Broadcast(out_shape().clone()), slot(0)?, out),
        NodeOp::Squeeze(dim) => Step::View(ViewStep::Squeeze(*dim), slot(0)?, out),
        NodeOp::Unsqueeze(dim) => Step::View(ViewStep::Unsqueeze(*dim), slot(0)?, out),
    })
}

pub(crate) fn lower_steps(graph: &Graph, kinds: &[KindTag], slots: &[Slot]) -> JitResult<Vec<Step>> {
    graph.nodes.iter().map(|node| lower_node(node, kinds, slots, graph)).collect()
}
