//! Lowering: nodes become type-specialised steps.

use luma_tensor::{Error, KindTag, ReduceOp, Result, Shape};

use crate::graph::{Graph, Node, NodeOp, Scalar, ValueId};

use super::step::{Slot, Step, ViewStep};

// ============================================================================
//    Lowering — nodes become type-specialised steps
// ============================================================================

/// The single removed size-1 dim of a squeeze, derived from the recorded shapes.
fn infer_squeeze_dim(in_shape: &Shape, out_shape: &Shape) -> Result<usize> {
    if in_shape.rank() != out_shape.rank() + 1 {
        return Err(Error::Msg(format!("executor: squeeze shape mismatch {in_shape:?} -> {out_shape:?}")));
    }
    let cands: Vec<usize> = (0..in_shape.rank())
        .filter(|&d| {
            in_shape.dims()[d] == 1 && {
                let mut ds = in_shape.dims().to_vec();
                ds.remove(d);
                ds == out_shape.dims()
            }
        })
        .collect();
    if cands.len() != 1 {
        return Err(Error::Msg(format!("executor: ambiguous squeeze dim (candidates {cands:?}) for {in_shape:?} -> {out_shape:?}")));
    }
    Ok(cands[0])
}

/// The single added size-1 dim of an unsqueeze, derived from the recorded shapes.
fn infer_unsqueeze_dim(in_shape: &Shape, out_shape: &Shape) -> Result<usize> {
    if out_shape.rank() != in_shape.rank() + 1 {
        return Err(Error::Msg(format!("executor: unsqueeze shape mismatch {in_shape:?} -> {out_shape:?}")));
    }
    let cands: Vec<usize> = (0..out_shape.rank())
        .filter(|&d| {
            out_shape.dims()[d] == 1 && {
                let mut ds = out_shape.dims().to_vec();
                ds.remove(d);
                ds == in_shape.dims()
            }
        })
        .collect();
    if cands.len() != 1 {
        return Err(Error::Msg(format!("executor: ambiguous unsqueeze dim (candidates {cands:?}) for {in_shape:?} -> {out_shape:?}")));
    }
    Ok(cands[0])
}

fn lower_node(node: &Node, kinds: &[KindTag], slots: &[Slot], graph: &Graph) -> Result<Step> {
    let slot = |i: usize| -> Result<Slot> {
        let id = *node
            .inputs
            .get(i)
            .ok_or_else(|| Error::Msg(format!("executor: {:?} expects more than {} inputs", node.op, node.inputs.len())))?;
        slots.get(id).copied().ok_or_else(|| Error::Msg(format!("executor: unknown value {id}")))
    };
    let out = slots.get(node.outputs[0]).copied().ok_or_else(|| Error::Msg(format!("executor: node with no output: {:?}", node.op)))?;

    let num = |id: ValueId| kinds[id];
    let in_num = |i: usize| -> Result<KindTag> {
        let id = *node.inputs.get(i).expect("infer validated input count");
        Ok(num(id))
    };
    let in_shape = |i: usize| -> &Shape { &graph.values[node.inputs[i]].shape };
    let out_shape = || -> &Shape { &graph.values[node.outputs[0]].shape };

    Ok(match &node.op {
        NodeOp::Constant => return Err(Error::Msg("executor: unexpected Constant node (constants are data-carrying leaves)".to_string())),
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
                        return Err(Error::Msg("executor: int mean is not yet supported (public API is float-only)".to_string()));
                    }
                    Step::ReduceI(*op, dims.clone(), keepdim, slot(0)?, out)
                }
                KindTag::Bool => unreachable!("kind inference validated"),
            }
        }
        NodeOp::ReduceAll(_) | NodeOp::ReduceAny(_) => {
            return Err(Error::Msg("executor: ReduceAll/ReduceAny are not yet supported (no public dim-wise bool reduction)".to_string()));
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
            let inputs: Vec<Slot> = (0..node.inputs.len()).map(slot).collect::<Result<_>>()?;
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
        NodeOp::Squeeze => {
            let dim = infer_squeeze_dim(in_shape(0), out_shape())?;
            Step::View(ViewStep::Squeeze(dim), slot(0)?, out)
        }
        NodeOp::Unsqueeze => {
            let dim = infer_unsqueeze_dim(in_shape(0), out_shape())?;
            Step::View(ViewStep::Unsqueeze(dim), slot(0)?, out)
        }
    })
}

pub(crate) fn lower_steps(graph: &Graph, kinds: &[KindTag], slots: &[Slot]) -> Result<Vec<Step>> {
    graph.nodes.iter().map(|node| lower_node(node, kinds, slots, graph)).collect()
}
