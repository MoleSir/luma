use std::sync::RwLock;
use crate::{BinaryOp, Bool, Device, Float, Int, Op, ReduceOp, Tensor, UnaryOp, DTypeKind, is_grad_enabled};

/// Trait for metadata that knows how to construct itself when a tensor operation is performed.
pub trait TensorMeta<D: Device, K: DTypeKind<D> + Sized>: Default + Send + Sync {
    // ---- Binary operations ----
    fn on_binary(lhs: &Tensor<D, K>, rhs: &Tensor<D, K>, op: BinaryOp) -> Self;
    fn on_binary_scalar_rhs(lhs: &Tensor<D, K>, rhs: K::Scalar, op: BinaryOp) -> Self;
    fn on_binary_scalar_lhs(lhs: K::Scalar, rhs: &Tensor<D, K>, op: BinaryOp) -> Self;

    // ---- Unary operations ----
    fn on_unary(t: &Tensor<D, K>, op: UnaryOp) -> Self;

    // ---- Elementwise ops shared by Float and Int ----
    fn on_neg(t: &Tensor<D, K>) -> Self;
    fn on_abs(t: &Tensor<D, K>) -> Self;
    fn on_sign(t: &Tensor<D, K>) -> Self;
    fn on_pow(t: &Tensor<D, K>, exp: K::Scalar) -> Self;
    fn on_affine(t: &Tensor<D, K>, mul: K::Scalar, add: K::Scalar) -> Self;
    fn on_clamp(t: &Tensor<D, K>, min: Option<K::Scalar>, max: Option<K::Scalar>) -> Self;

    // ---- Reductions ----
    fn on_reduce(t: &Tensor<D, K>, dims: &[usize], op: ReduceOp) -> Self;

    // ---- Matrix operations ----
    fn on_matmul(lhs: &Tensor<D, K>, rhs: &Tensor<D, K>) -> Self;

    // ---- Shape operations ----
    fn on_broadcast(t: &Tensor<D, K>) -> Self;
    fn on_narrow(t: &Tensor<D, K>, dim: usize, start: usize, len: usize) -> Self;
    fn on_slice(t: &Tensor<D, K>, dim: usize, start: usize, end: usize, step: usize) -> Self;
    fn on_reshape(t: &Tensor<D, K>) -> Self;
    fn on_transpose(t: &Tensor<D, K>, dim1: usize, dim2: usize) -> Self;
    fn on_permute(t: &Tensor<D, K>, dims: Vec<usize>) -> Self;
    fn on_cat<A: AsRef<Tensor<D, K>>>(args: &[A], dim: usize) -> Self;
    fn on_copy(t: &Tensor<D, K>) -> Self;

    // ---- Type conversions ----
    fn on_cast(t: &Tensor<D, K>) -> Self;

    // ---- Indexing operations ----
    fn on_index_select(t: &Tensor<D, K>, idx: &Tensor<D, Int>, dim: usize) -> Self;
    fn on_gather(src: &Tensor<D, K>, idx: &Tensor<D, Int>, dim: usize) -> Self;
    fn on_index_add(init: &Tensor<D, K>, idx: &Tensor<D, Int>, src: &Tensor<D, K>, dim: usize) -> Self;
    fn on_scatter_add(init: &Tensor<D, K>, idx: &Tensor<D, Int>, src: &Tensor<D, K>, dim: usize) -> Self;

    // ---- Conditional operations ----
    fn on_pick(mask: &Tensor<D, Bool>, tv: Option<&Tensor<D, K>>, fv: Option<&Tensor<D, K>>) -> Self;

    // ---- NN operations (Float-specific, but included for completeness) ----
    fn on_rms_norm(input: &Tensor<D, K>, weight: &Tensor<D, K>, eps: f64) -> Self;
    fn on_softmax(input: &Tensor<D, K>, dim: usize) -> Self;
}

pub struct FloatMeta<D: Device> {
    pub op: Option<Op<D>>,
    pub requires_grad: RwLock<bool>,
}

impl<D: Device> FloatMeta<D> {
    /// A leaf variable that accumulates gradients.
    pub fn var() -> Self {
        Self { op: None, requires_grad: RwLock::new(true) }
    }

    /// A constant value that does not track gradients.
    pub fn val() -> Self {
        Self { op: None, requires_grad: RwLock::new(false) }
    }

    /// A non-leaf node produced by `op`.
    pub fn from_op(op: Op<D>) -> Self {
        Self { op: Some(op), requires_grad: RwLock::new(true) }
    }

    pub fn op(&self) -> Option<&Op<D>> {
        self.op.as_ref()
    }

    pub fn requires_grad(&self) -> bool {
        *self.requires_grad.read().unwrap()
    }

    pub fn set_requires_grad(&self, mode: bool) {
        *self.requires_grad.write().unwrap() = mode;
    }

    /// A tensor is a leaf when it requires grad but was not produced by an op.
    pub fn is_leaf(&self) -> bool {
        self.requires_grad() && self.op.is_none()
    }
}

impl<D: Device> Default for FloatMeta<D> {
    fn default() -> Self {
        Self::val()
    }
}

/// Records an op into a `FloatMeta` iff grad is globally enabled and `record` is
/// true (i.e. some input requires grad). Otherwise produces a plain value meta.
impl<D: Device> FloatMeta<D> {
    fn record(record: bool, op: impl FnOnce() -> Op<D>) -> Self {
        if is_grad_enabled() && record { Self::from_op(op()) } else { Self::val() }
    }

    pub fn on_binary(lhs: &Tensor<D, Float>, rhs: &Tensor<D, Float>, op: BinaryOp) -> Self {
        Self::record(lhs.requires_grad() || rhs.requires_grad(), || Op::Binary(lhs.clone(), rhs.clone(), op))
    }

    pub fn on_binary_scalar_rhs(lhs: &Tensor<D, Float>, rhs: f64, op: BinaryOp) -> Self {
        Self::record(lhs.requires_grad(), || Op::BinaryScalarRhs(lhs.clone(), rhs, op))
    }

    pub fn on_binary_scalar_lhs(lhs: f64, rhs: &Tensor<D, Float>, op: BinaryOp) -> Self {
        Self::record(rhs.requires_grad(), || Op::BinaryScalarLhs(lhs, rhs.clone(), op))
    }

    pub fn on_unary(t: &Tensor<D, Float>, op: UnaryOp) -> Self {
        Self::record(t.requires_grad(), || Op::Unary(t.clone(), op))
    }

    pub fn on_broadcast(t: &Tensor<D, Float>) -> Self {
        Self::record(t.requires_grad(), || Op::Broadcast(t.clone()))
    }

    pub fn on_reduce(t: &Tensor<D, Float>, dims: &[usize], op: ReduceOp) -> Self {
        Self::record(t.requires_grad(), || Op::Reduce(t.clone(), op, dims.to_vec()))
    }

    pub fn on_matmul(lhs: &Tensor<D, Float>, rhs: &Tensor<D, Float>) -> Self {
        Self::record(lhs.requires_grad() || rhs.requires_grad(), || Op::Matmul(lhs.clone(), rhs.clone()))
    }

    pub fn on_narrow(t: &Tensor<D, Float>, dim: usize, start: usize, len: usize) -> Self {
        Self::record(t.requires_grad(), || Op::Narrow(t.clone(), dim, start, len))
    }

    pub fn on_slice(t: &Tensor<D, Float>, dim: usize, start: usize, end: usize, step: usize) -> Self {
        Self::record(t.requires_grad(), || Op::Slice(t.clone(), dim, start, end, step))
    }

    pub fn on_reshape(t: &Tensor<D, Float>) -> Self {
        Self::record(t.requires_grad(), || Op::Reshape(t.clone()))
    }

    pub fn on_transpose(t: &Tensor<D, Float>, dim1: usize, dim2: usize) -> Self {
        Self::record(t.requires_grad(), || Op::Transpose(t.clone(), dim1, dim2))
    }

    pub fn on_permute(t: &Tensor<D, Float>, dims: Vec<usize>) -> Self {
        Self::record(t.requires_grad(), || Op::Permute(t.clone(), dims))
    }

    pub fn on_cat<A: AsRef<Tensor<D, Float>>>(args: &[A], dim: usize) -> Self {
        let record = args.iter().any(|t| t.as_ref().requires_grad());
        Self::record(record, || {
            let vec = args.iter().map(|a| a.as_ref().clone()).collect();
            Op::Cat(vec, dim)
        })
    }

    pub fn on_copy(t: &Tensor<D, Float>) -> Self {
        Self::record(t.requires_grad(), || Op::Copy(t.clone()))
    }

    pub fn on_cast(t: &Tensor<D, Float>) -> Self {
        Self::record(t.requires_grad(), || Op::Cast(t.clone()))
    }

    pub fn on_neg(t: &Tensor<D, Float>) -> Self {
        Self::record(t.requires_grad(), || Op::Neg(t.clone()))
    }

    pub fn on_abs(t: &Tensor<D, Float>) -> Self {
        Self::record(t.requires_grad(), || Op::Abs(t.clone()))
    }

    pub fn on_sign(t: &Tensor<D, Float>) -> Self {
        Self::record(t.requires_grad(), || Op::Sign(t.clone()))
    }

    pub fn on_pow(t: &Tensor<D, Float>, exp: f64) -> Self {
        Self::record(t.requires_grad(), || Op::Pow(t.clone(), exp))
    }

    pub fn on_affine(t: &Tensor<D, Float>, mul: f64, add: f64) -> Self {
        Self::record(t.requires_grad(), || Op::Affine(t.clone(), mul, add))
    }

    pub fn on_clamp(t: &Tensor<D, Float>, min: Option<f64>, max: Option<f64>) -> Self {
        Self::record(t.requires_grad(), || Op::Clamp(t.clone(), min, max))
    }

    pub fn on_pick(mask: &Tensor<D, Bool>, tv: Option<&Tensor<D, Float>>, fv: Option<&Tensor<D, Float>>) -> Self {
        let record = tv.map(|t| t.requires_grad()).unwrap_or(false) || fv.map(|f| f.requires_grad()).unwrap_or(false);
        Self::record(record, || Op::Pick(mask.clone(), tv.cloned(), fv.cloned()))
    }

    pub fn on_index_select(t: &Tensor<D, Float>, idx: &Tensor<D, Int>, dim: usize) -> Self {
        Self::record(t.requires_grad(), || Op::IndexSelect(t.clone(), idx.clone(), dim))
    }

    pub fn on_index_add(init: &Tensor<D, Float>, idx: &Tensor<D, Int>, src: &Tensor<D, Float>, dim: usize) -> Self {
        Self::record(init.requires_grad() || src.requires_grad(), || Op::IndexAdd(init.clone(), idx.clone(), src.clone(), dim))
    }

    pub fn on_scatter_add(init: &Tensor<D, Float>, idx: &Tensor<D, Int>, src: &Tensor<D, Float>, dim: usize) -> Self {
        Self::record(init.requires_grad() || src.requires_grad(), || Op::ScatterAdd(init.clone(), idx.clone(), src.clone(), dim))
    }

    pub fn on_gather(src: &Tensor<D, Float>, idx: &Tensor<D, Int>, dim: usize) -> Self {
        Self::record(src.requires_grad(), || Op::Gather(src.clone(), idx.clone(), dim))
    }

    pub fn on_rms_norm(input: &Tensor<D, Float>, weight: &Tensor<D, Float>, eps: f64) -> Self {
        Self::record(input.requires_grad() || weight.requires_grad(), || Op::RmsNorm(input.clone(), weight.clone(), eps))
    }

    pub fn on_softmax(input: &Tensor<D, Float>, dim: usize) -> Self {
        Self::record(input.requires_grad(), || Op::Softmax(input.clone(), dim))
    }
}

/// Convenience accessors on a `Float` tensor for its autograd state.
impl<D: Device> Tensor<D, Float> {
    pub fn requires_grad(&self) -> bool {
        self.0.meta.requires_grad()
    }

    pub fn set_requires_grad(&self, mode: bool) {
        self.0.meta.set_requires_grad(mode)
    }

    pub fn is_leaf(&self) -> bool {
        self.0.meta.is_leaf()
    }

    pub(crate) fn op(&self) -> Option<&Op<D>> {
        self.0.meta.op()
    }
}

// ============================================================================
// TensorMeta trait implementation for FloatMeta
// ============================================================================

impl<D: Device> TensorMeta<D, Float> for FloatMeta<D> {
    fn on_binary(lhs: &Tensor<D, Float>, rhs: &Tensor<D, Float>, op: BinaryOp) -> Self {
        FloatMeta::on_binary(lhs, rhs, op)
    }

    fn on_binary_scalar_rhs(lhs: &Tensor<D, Float>, rhs: f64, op: BinaryOp) -> Self {
        FloatMeta::on_binary_scalar_rhs(lhs, rhs, op)
    }

    fn on_binary_scalar_lhs(lhs: f64, rhs: &Tensor<D, Float>, op: BinaryOp) -> Self {
        FloatMeta::on_binary_scalar_lhs(lhs, rhs, op)
    }

    fn on_unary(t: &Tensor<D, Float>, op: UnaryOp) -> Self {
        FloatMeta::on_unary(t, op)
    }

    fn on_reduce(t: &Tensor<D, Float>, dims: &[usize], op: ReduceOp) -> Self {
        FloatMeta::on_reduce(t, dims, op)
    }

    fn on_matmul(lhs: &Tensor<D, Float>, rhs: &Tensor<D, Float>) -> Self {
        FloatMeta::on_matmul(lhs, rhs)
    }

    fn on_broadcast(t: &Tensor<D, Float>) -> Self {
        FloatMeta::on_broadcast(t)
    }

    fn on_narrow(t: &Tensor<D, Float>, dim: usize, start: usize, len: usize) -> Self {
        FloatMeta::on_narrow(t, dim, start, len)
    }

    fn on_slice(t: &Tensor<D, Float>, dim: usize, start: usize, end: usize, step: usize) -> Self {
        FloatMeta::on_slice(t, dim, start, end, step)
    }

    fn on_reshape(t: &Tensor<D, Float>) -> Self {
        FloatMeta::on_reshape(t)
    }

    fn on_transpose(t: &Tensor<D, Float>, dim1: usize, dim2: usize) -> Self {
        FloatMeta::on_transpose(t, dim1, dim2)
    }

    fn on_permute(t: &Tensor<D, Float>, dims: Vec<usize>) -> Self {
        FloatMeta::on_permute(t, dims)
    }

    fn on_cat<A: AsRef<Tensor<D, Float>>>(args: &[A], dim: usize) -> Self {
        FloatMeta::on_cat(args, dim)
    }

    fn on_copy(t: &Tensor<D, Float>) -> Self {
        FloatMeta::on_copy(t)
    }

    fn on_cast(t: &Tensor<D, Float>) -> Self {
        FloatMeta::on_cast(t)
    }

    fn on_index_select(t: &Tensor<D, Float>, idx: &Tensor<D, Int>, dim: usize) -> Self {
        FloatMeta::on_index_select(t, idx, dim)
    }

    fn on_gather(src: &Tensor<D, Float>, idx: &Tensor<D, Int>, dim: usize) -> Self {
        FloatMeta::on_gather(src, idx, dim)
    }

    fn on_index_add(init: &Tensor<D, Float>, idx: &Tensor<D, Int>, src: &Tensor<D, Float>, dim: usize) -> Self {
        FloatMeta::on_index_add(init, idx, src, dim)
    }

    fn on_scatter_add(init: &Tensor<D, Float>, idx: &Tensor<D, Int>, src: &Tensor<D, Float>, dim: usize) -> Self {
        FloatMeta::on_scatter_add(init, idx, src, dim)
    }

    fn on_pick(mask: &Tensor<D, Bool>, tv: Option<&Tensor<D, Float>>, fv: Option<&Tensor<D, Float>>) -> Self {
        FloatMeta::on_pick(mask, tv, fv)
    }

    fn on_rms_norm(input: &Tensor<D, Float>, weight: &Tensor<D, Float>, eps: f64) -> Self {
        FloatMeta::on_rms_norm(input, weight, eps)
    }

    fn on_softmax(input: &Tensor<D, Float>, dim: usize) -> Self {
        FloatMeta::on_softmax(input, dim)
    }

    fn on_neg(t: &Tensor<D, Float>) -> Self {
        FloatMeta::on_neg(t)
    }

    fn on_abs(t: &Tensor<D, Float>) -> Self {
        FloatMeta::on_abs(t)
    }

    fn on_sign(t: &Tensor<D, Float>) -> Self {
        FloatMeta::on_sign(t)
    }

    fn on_pow(t: &Tensor<D, Float>, exp: f64) -> Self {
        FloatMeta::on_pow(t, exp)
    }

    fn on_affine(t: &Tensor<D, Float>, mul: f64, add: f64) -> Self {
        FloatMeta::on_affine(t, mul, add)
    }

    fn on_clamp(t: &Tensor<D, Float>, min: Option<f64>, max: Option<f64>) -> Self {
        FloatMeta::on_clamp(t, min, max)
    }
}

// ============================================================================
// TensorMeta trait implementation for () (Int and Bool)
// ============================================================================

impl<D: Device, K: crate::DTypeKind<D>> TensorMeta<D, K> for () {
    fn on_binary(_: &Tensor<D, K>, _: &Tensor<D, K>, _: BinaryOp) -> Self {}
    fn on_binary_scalar_rhs(_: &Tensor<D, K>, _: K::Scalar, _: BinaryOp) -> Self {}
    fn on_binary_scalar_lhs(_: K::Scalar, _: &Tensor<D, K>, _: BinaryOp) -> Self {}
    fn on_unary(_: &Tensor<D, K>, _: UnaryOp) -> Self {}
    fn on_reduce(_: &Tensor<D, K>, _: &[usize], _: ReduceOp) -> Self {}
    fn on_matmul(_: &Tensor<D, K>, _: &Tensor<D, K>) -> Self {}
    fn on_broadcast(_: &Tensor<D, K>) -> Self {}
    fn on_narrow(_: &Tensor<D, K>, _: usize, _: usize, _: usize) -> Self {}
    fn on_slice(_: &Tensor<D, K>, _: usize, _: usize, _: usize, _: usize) -> Self {}
    fn on_reshape(_: &Tensor<D, K>) -> Self {}
    fn on_transpose(_: &Tensor<D, K>, _: usize, _: usize) -> Self {}
    fn on_permute(_: &Tensor<D, K>, _: Vec<usize>) -> Self {}
    fn on_cat<A: AsRef<Tensor<D, K>>>(_: &[A], _: usize) -> Self {}
    fn on_copy(_: &Tensor<D, K>) -> Self {}
    fn on_cast(_: &Tensor<D, K>) -> Self {}
    fn on_index_select(_: &Tensor<D, K>, _: &Tensor<D, Int>, _: usize) -> Self {}
    fn on_gather(_: &Tensor<D, K>, _: &Tensor<D, Int>, _: usize) -> Self {}
    fn on_index_add(_: &Tensor<D, K>, _: &Tensor<D, Int>, _: &Tensor<D, K>, _: usize) -> Self {}
    fn on_scatter_add(_: &Tensor<D, K>, _: &Tensor<D, Int>, _: &Tensor<D, K>, _: usize) -> Self {}
    fn on_pick(_: &Tensor<D, Bool>, _: Option<&Tensor<D, K>>, _: Option<&Tensor<D, K>>) -> Self {}
    fn on_rms_norm(_: &Tensor<D, K>, _: &Tensor<D, K>, _: f64) -> Self {}
    fn on_softmax(_: &Tensor<D, K>, _: usize) -> Self {}
    fn on_neg(_: &Tensor<D, K>) -> Self {}
    fn on_abs(_: &Tensor<D, K>) -> Self {}
    fn on_sign(_: &Tensor<D, K>) -> Self {}
    fn on_pow(_: &Tensor<D, K>, _: K::Scalar) -> Self {}
    fn on_affine(_: &Tensor<D, K>, _: K::Scalar, _: K::Scalar) -> Self {}
    fn on_clamp(_: &Tensor<D, K>, _: Option<K::Scalar>, _: Option<K::Scalar>) -> Self {}
}
