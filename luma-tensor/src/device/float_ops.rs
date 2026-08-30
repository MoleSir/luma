use std::borrow::Cow;

use crate::Result;
use crate::dtype::{BoolDType, FloatDType, IntDType};
use crate::tensor::{Layout, Shape};

/// Operations for floating-point tensors.
pub trait FloatOps<D: super::Device> {
    // construction
    fn f_zeros(shape: &Shape, device: &D, dtype: FloatDType) -> Result<D::FloatStorage>;
    fn f_ones(shape: &Shape, device: &D, dtype: FloatDType) -> Result<D::FloatStorage>;
    fn f_full(shape: &Shape, value: f64, device: &D, dtype: FloatDType) -> Result<D::FloatStorage>;
    fn f_from_f64<'a>(data: impl Into<Cow<'a, [f64]>>, device: &D) -> Result<D::FloatStorage>;

    fn f_from_f32<'a>(data: impl Into<Cow<'a, [f32]>>, device: &D) -> Result<D::FloatStorage>;

    fn f_from_bytes<'a>(bytes: impl Into<Cow<'a, [u8]>>, shape: &Shape, device: &D, dtype: FloatDType) -> Result<D::FloatStorage>;
    fn f_rand_uniform(shape: &Shape, lo: f64, hi: f64, device: &D, dtype: FloatDType) -> Result<D::FloatStorage>;
    fn f_rand_normal(shape: &Shape, mean: f64, std: f64, device: &D, dtype: FloatDType) -> Result<D::FloatStorage>;

    // materialization / read-back
    fn f_contiguous(x: &D::FloatStorage, layout: &Layout) -> Result<D::FloatStorage>;
    fn f_cast_float(x: &D::FloatStorage, layout: &Layout, to: FloatDType) -> Result<D::FloatStorage>;
    fn f_cast_int(x: &D::FloatStorage, layout: &Layout, to: IntDType) -> Result<D::IntStorage>;
    fn f_cast_bool(x: &D::FloatStorage, layout: &Layout, to: BoolDType) -> Result<D::BoolStorage>;

    /// Read all elements into a `Vec<f64>` in logical (layout) order.
    fn f_to_vec(x: &D::FloatStorage, layout: &Layout) -> Result<Vec<f64>>;

    /// Read raw little-endian bytes in logical (layout) order.
    /// Returns `Cow::Borrowed` when the underlying storage is already contiguous
    /// (zero-copy); `Cow::Owned` otherwise.
    fn f_to_bytes<'a>(x: &'a D::FloatStorage, layout: &Layout) -> Result<Cow<'a, [u8]>>;

    // binary (elementwise, same shape; broadcasting handled above this layer)
    fn f_binary(
        lhs: &D::FloatStorage,
        lhs_l: &Layout,
        rhs: &D::FloatStorage,
        rhs_l: &Layout,
        op: crate::BinaryOp,
    ) -> Result<D::FloatStorage>;
    fn f_binary_(dst: &mut D::FloatStorage, dst_l: &Layout, src: &D::FloatStorage, src_l: &Layout, op: crate::BinaryOp) -> Result<()>;
    fn f_binary_scalar(lhs: &D::FloatStorage, lhs_l: &Layout, rhs: f64, op: crate::BinaryOp) -> Result<D::FloatStorage>;
    fn f_binary_scalar_(dst: &mut D::FloatStorage, dst_l: &Layout, rhs: f64, op: crate::BinaryOp) -> Result<()>;
    fn f_binary_scalar_lhs(scalar: f64, rhs: &D::FloatStorage, rhs_l: &Layout, op: crate::BinaryOp) -> Result<D::FloatStorage>;

    // comparison -> bool storage
    fn f_cmp(lhs: &D::FloatStorage, lhs_l: &Layout, rhs: &D::FloatStorage, rhs_l: &Layout, op: crate::CmpOp) -> Result<D::BoolStorage>;
    fn f_cmp_scalar(lhs: &D::FloatStorage, lhs_l: &Layout, rhs: f64, op: crate::CmpOp) -> Result<D::BoolStorage>;

    // unary
    fn f_unary(x: &D::FloatStorage, layout: &Layout, op: crate::UnaryOp<f64>) -> Result<D::FloatStorage>;
    fn f_unary_(dst: &mut D::FloatStorage, dst_l: &Layout, op: crate::UnaryOp<f64>) -> Result<()>;
    fn f_float_unary(x: &D::FloatStorage, layout: &Layout, op: crate::FloatUnaryOp) -> Result<D::FloatStorage>;
    fn f_float_unary_(dst: &mut D::FloatStorage, dst_l: &Layout, op: crate::FloatUnaryOp) -> Result<()>;

    // reduction: `out_shape` is computed by the tensor layer (see
    // `ops::shape_infer::reduce_out_shape`) and passed in; the kernel must
    // produce a buffer of exactly `out_shape.element_count()` elements.
    fn f_reduce(
        x: &D::FloatStorage,
        layout: &Layout,
        dims: &[usize],
        keepdim: bool,
        op: crate::ReduceOp,
        out_shape: &Shape,
    ) -> Result<D::FloatStorage>;
    fn f_arg_reduce(
        x: &D::FloatStorage,
        layout: &Layout,
        dim: usize,
        keepdim: bool,
        take_max: bool,
        out_shape: &Shape,
    ) -> Result<D::IntStorage>;

    // matmul (batched); `out_shape` computed by the tensor layer
    // (`ops::shape_infer::matmul_out_shape`).
    fn f_matmul(lhs: &D::FloatStorage, lhs_l: &Layout, rhs: &D::FloatStorage, rhs_l: &Layout, out_shape: &Shape)
    -> Result<D::FloatStorage>;
    fn f_add_matmul_(
        dst: &mut D::FloatStorage,
        dst_l: &Layout,
        lhs: &D::FloatStorage,
        lhs_l: &Layout,
        rhs: &D::FloatStorage,
        rhs_l: &Layout,
    ) -> Result<()>;

    // indexing (`out_shape` computed by the tensor layer)
    fn f_index_select(
        x: &D::FloatStorage,
        x_l: &Layout,
        idx: &D::IntStorage,
        idx_l: &Layout,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<D::FloatStorage>;

    fn f_gather(
        x: &D::FloatStorage,
        x_l: &Layout,
        idx: &D::IntStorage,
        idx_l: &Layout,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<D::FloatStorage>;

    fn f_index_add(
        init: &D::FloatStorage,
        init_l: &Layout,
        idx: &D::IntStorage,
        idx_l: &Layout,
        src: &D::FloatStorage,
        src_l: &Layout,
        dim: usize,
    ) -> Result<D::FloatStorage>;

    fn f_scatter_add(
        init: &D::FloatStorage,
        init_l: &Layout,
        idx: &D::IntStorage,
        idx_l: &Layout,
        src: &D::FloatStorage,
        src_l: &Layout,
        dim: usize,
    ) -> Result<D::FloatStorage>;

    // shape ops that need data movement (`out_shape` computed by the tensor layer)
    fn f_cat(srcs: &[(&D::FloatStorage, &Layout)], dim: usize, out_shape: &Shape) -> Result<D::FloatStorage>;

    // views (alias on compute devices; a fresh SSA value on tracing devices)
    fn f_view(_src: &D::FloatStorage, _src_l: &Layout, _dst_l: &Layout, _view: crate::ViewOp) -> Result<Option<D::FloatStorage>> {
        Ok(None)
    }

    // nn fused kernels
    fn f_softmax(x: &D::FloatStorage, layout: &Layout, dim: usize) -> Result<D::FloatStorage>;

    fn f_rms_norm(x: &D::FloatStorage, x_l: &Layout, weight: &D::FloatStorage, weight_l: &Layout, eps: f64) -> Result<D::FloatStorage>;

    // pick via a bool mask
    fn f_pick(
        mask: &D::BoolStorage,
        mask_l: &Layout,
        on_true: &D::FloatStorage,
        true_l: &Layout,
        on_false: &D::FloatStorage,
        false_l: &Layout,
    ) -> Result<D::FloatStorage>;

    fn f_pick_true(
        mask: &D::BoolStorage,
        mask_l: &Layout,
        value: f64,
        on_false: &D::FloatStorage,
        false_l: &Layout,
    ) -> Result<D::FloatStorage>;

    fn f_pick_false(
        mask: &D::BoolStorage,
        mask_l: &Layout,
        on_true: &D::FloatStorage,
        true_l: &Layout,
        value: f64,
    ) -> Result<D::FloatStorage>;

    // allclose
    fn f_allclose(a: &D::FloatStorage, a_l: &Layout, b: &D::FloatStorage, b_l: &Layout, rtol: f64, atol: f64) -> Result<bool>;
}
