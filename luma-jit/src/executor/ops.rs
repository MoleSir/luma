//! Typed op helpers: forward a `Step` against concrete tensors.

use luma_tensor::ops::{BaseOpsDTypeKind, NumOpsDTypeKind};
use luma_tensor::{BinaryOp, Bool, CmpOp, Device, Float, FloatUnaryOp, Int, ReduceOp, Result, Tensor, UnaryOp};

use super::step::ViewStep;

// ============================================================================
//    Typed op helpers
// ============================================================================

pub(crate) fn binary_result<D: Device, K>(x: &Tensor<D, K>, y: &Tensor<D, K>, op: BinaryOp) -> Result<Tensor<D, K>>
where
    K: NumOpsDTypeKind<D>
{
    match op {
        BinaryOp::Add => x.add(y),
        BinaryOp::Sub => x.sub(y),
        BinaryOp::Mul => x.mul(y),
        BinaryOp::Div => x.div(y),
        BinaryOp::Maximum => x.maximum(y),
        BinaryOp::Minimum => x.minimum(y),
    }
}

pub(crate) fn cmp_result<D: Device, K>(x: &Tensor<D, K>, y: &Tensor<D, K>, op: CmpOp) -> Result<Tensor<D, Bool>>
where
    K: NumOpsDTypeKind<D>
{
    match op {
        CmpOp::Eq => x.eq(y),
        CmpOp::Ne => x.ne(y),
        CmpOp::Le => x.le(y),
        CmpOp::Ge => x.ge(y),
        CmpOp::Lt => x.lt(y),
        CmpOp::Gt => x.gt(y),
    }
}

pub(crate) fn unary_result<D: Device, K>(x: &Tensor<D, K>, op: UnaryOp<K::Scalar>) -> Result<Tensor<D, K>>
where
    K: NumOpsDTypeKind<D>
{
    match op {
        UnaryOp::Neg => x.neg(),
        UnaryOp::Abs => x.abs(),
        UnaryOp::Sign => x.sign(),
        UnaryOp::Affine(m, a) => x.affine(m, a),
        UnaryOp::Pow(p) => x.pow(p),
        UnaryOp::Clamp(mn, mx) => x.clamp(mn, mx),
    }
}

pub(crate) fn float_unary_result<D: Device>(x: &Tensor<D, Float>, op: FloatUnaryOp) -> Result<Tensor<D, Float>> {
    match op {
        FloatUnaryOp::Exp => x.exp(),
        FloatUnaryOp::Ln => x.ln(),
        FloatUnaryOp::Sin => x.sin(),
        FloatUnaryOp::Cos => x.cos(),
        FloatUnaryOp::Tanh => x.tanh(),
        FloatUnaryOp::Sqr => x.sqr(),
        FloatUnaryOp::Sqrt => x.sqrt(),
        FloatUnaryOp::Recip => x.recip(),
        FloatUnaryOp::Gelu => x.gelu(),
        FloatUnaryOp::GeluErf => x.gelu_erf(),
        FloatUnaryOp::Erf => x.erf(),
        FloatUnaryOp::Relu => x.relu(),
        FloatUnaryOp::LeakyRelu(s) => x.leaky_relu(s),
        FloatUnaryOp::Silu => x.silu(),
        FloatUnaryOp::Sigmoid => x.sigmoid(),
        FloatUnaryOp::Floor => x.floor(),
        FloatUnaryOp::Ceil => x.ceil(),
        FloatUnaryOp::Round => x.round(),
    }
}

/// Reduce over `dims` by applying single-dim reductions iteratively
/// (descending, keepdim on) and squeezing the reduced dims at the end when
/// `keepdim` is false. Correct for Sum/Max/Min/Prod/Mean because each dim
/// reduction is associative and equally weighted.
pub(crate) fn f_reduce<D: Device>(x: &Tensor<D, Float>, op: ReduceOp, dims: &[usize], keepdim: bool) -> Result<Tensor<D, Float>> {
    let mut ds: Vec<usize> = dims.to_vec();
    ds.sort_unstable_by(|a, b| b.cmp(a));
    let mut t = x.clone();
    for &d in &ds {
        t = match op {
            ReduceOp::Sum => t.sum_keepdim(d)?,
            ReduceOp::Max => t.max_keepdim(d)?,
            ReduceOp::Min => t.min_keepdim(d)?,
            ReduceOp::Prod => t.prod_keepdim(d)?,
            ReduceOp::Mean => t.mean_keepdim(d)?,
        };
    }
    if !keepdim {
        for &d in &ds {
            t = t.squeeze(d)?;
        }
    }
    Ok(t)
}

pub(crate) fn i_reduce<D: Device>(x: &Tensor<D, Int>, op: ReduceOp, dims: &[usize], keepdim: bool) -> Result<Tensor<D, Int>> {
    let mut ds: Vec<usize> = dims.to_vec();
    ds.sort_unstable_by(|a, b| b.cmp(a));
    let mut t = x.clone();
    for &d in &ds {
        t = match op {
            ReduceOp::Sum => t.sum_keepdim(d)?,
            ReduceOp::Max => t.max_keepdim(d)?,
            ReduceOp::Min => t.min_keepdim(d)?,
            ReduceOp::Prod => t.prod_keepdim(d)?,
            ReduceOp::Mean => unreachable!("int mean rejected at lowering"),
        };
    }
    if !keepdim {
        for &d in &ds {
            t = t.squeeze(d)?;
        }
    }
    Ok(t)
}

pub(crate) fn apply_view<D: Device, K: BaseOpsDTypeKind<D>>(t: &Tensor<D, K>, v: &ViewStep) -> Result<Tensor<D, K>> {
    match v {
        ViewStep::Reshape(s) => t.reshape(s.clone()),
        ViewStep::Transpose(a, b) => t.transpose(*a, *b),
        ViewStep::Permute(dims) => t.permute(dims.clone()),
        ViewStep::Narrow(d, s, l) => t.narrow(*d, *s, *l),
        ViewStep::Slice(d, s, e, st) => t.slice(*d, *s, *e, *st),
        ViewStep::Broadcast(s) => t.broadcast_as(s.clone()),
        ViewStep::Squeeze(d) => t.squeeze(*d),
        ViewStep::Unsqueeze(d) => t.unsqueeze(*d),
    }
}
