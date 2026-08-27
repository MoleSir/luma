use std::borrow::Cow;

use crate::Result;
use crate::dtype::{BoolDType, FloatDType, IntDType};
use crate::tensor::{Layout, Shape};

/// Operations on `Int`-kind tensors. No autograd. Scalars are `i64` (wide enough
/// for all int precisions).
pub trait IntOps<D: super::Device> {
    // construction
    fn i_zeros(shape: &Shape, device: &D, dtype: IntDType) -> Result<D::IntStorage>;
    fn i_ones(shape: &Shape, device: &D, dtype: IntDType) -> Result<D::IntStorage>;
    fn i_full(shape: &Shape, value: i64, device: &D, dtype: IntDType) -> Result<D::IntStorage>;
    fn i_from_i64<'a>(data: impl Into<Cow<'a, [i64]>>, device: &D) -> Result<D::IntStorage>;

    fn i_from_i32<'a>(data: impl Into<Cow<'a, [i32]>>, device: &D) -> Result<D::IntStorage>;

    fn i_from_u32<'a>(data: impl Into<Cow<'a, [u32]>>, device: &D) -> Result<D::IntStorage>;

    fn i_from_u8<'a>(data: impl Into<Cow<'a, [u8]>>, device: &D) -> Result<D::IntStorage>;

    fn i_from_bytes<'a>(bytes: impl Into<Cow<'a, [u8]>>, shape: &Shape, device: &D, dtype: IntDType) -> Result<D::IntStorage>;

    fn i_arange(start: i64, end: i64, step: i64, device: &D, dtype: IntDType) -> Result<(D::IntStorage, usize)>;

    // materialization / read-back
    fn i_contiguous(x: &D::IntStorage, layout: &Layout) -> Result<D::IntStorage>;
    fn i_cast_float(x: &D::IntStorage, layout: &Layout, to: FloatDType) -> Result<D::FloatStorage>;
    fn i_cast_int(x: &D::IntStorage, layout: &Layout, to: IntDType) -> Result<D::IntStorage>;
    fn i_cast_bool(x: &D::IntStorage, layout: &Layout, to: BoolDType) -> Result<D::BoolStorage>;

    /// Read all elements into a `Vec<i64>` in logical (layout) order.
    fn i_to_vec(x: &D::IntStorage, layout: &Layout) -> Result<Vec<i64>>;

    /// Read raw little-endian bytes in logical (layout) order.
    /// Returns `Cow::Borrowed` when the underlying storage is already contiguous
    /// (zero-copy); `Cow::Owned` otherwise.
    fn i_to_bytes<'a>(x: &'a D::IntStorage, layout: &Layout) -> Result<Cow<'a, [u8]>>;

    // arithmetic
    fn i_binary(lhs: &D::IntStorage, lhs_l: &Layout, rhs: &D::IntStorage, rhs_l: &Layout, op: crate::BinaryOp) -> Result<D::IntStorage>;

    fn i_binary_(dst: &mut D::IntStorage, dst_l: &Layout, src: &D::IntStorage, src_l: &Layout, op: crate::BinaryOp) -> Result<()>;

    fn i_binary_scalar(lhs: &D::IntStorage, lhs_l: &Layout, rhs: i64, op: crate::BinaryOp) -> Result<D::IntStorage>;

    fn i_binary_scalar_(dst: &mut D::IntStorage, dst_l: &Layout, rhs: i64, op: crate::BinaryOp) -> Result<()>;

    fn i_binary_scalar_lhs(scalar: i64, rhs: &D::IntStorage, rhs_l: &Layout, op: crate::BinaryOp) -> Result<D::IntStorage>;

    fn i_unary(x: &D::IntStorage, layout: &Layout, op: crate::UnaryOp<i64>) -> Result<D::IntStorage>;

    fn i_unary_(dst: &mut D::IntStorage, dst_l: &Layout, op: crate::UnaryOp<i64>) -> Result<()>;

    // matmul (batched); out shape computed by the caller / this fn
    fn i_matmul(lhs: &D::IntStorage, lhs_l: &Layout, rhs: &D::IntStorage, rhs_l: &Layout) -> Result<(D::IntStorage, Shape)>;

    // comparison -> bool
    fn i_cmp(lhs: &D::IntStorage, lhs_l: &Layout, rhs: &D::IntStorage, rhs_l: &Layout, op: crate::CmpOp) -> Result<D::BoolStorage>;

    fn i_cmp_scalar(lhs: &D::IntStorage, lhs_l: &Layout, rhs: i64, op: crate::CmpOp) -> Result<D::BoolStorage>;

    // reduction
    fn i_reduce(x: &D::IntStorage, layout: &Layout, dims: &[usize], keepdim: bool, op: crate::ReduceOp) -> Result<(D::IntStorage, Shape)>;

    /// argmin/argmax return int-kind indices.
    fn i_arg_reduce(x: &D::IntStorage, layout: &Layout, dim: usize, keepdim: bool, take_max: bool) -> Result<(D::IntStorage, Shape)>;

    // indexing
    fn i_index_select(x: &D::IntStorage, x_l: &Layout, idx: &D::IntStorage, idx_l: &Layout, dim: usize) -> Result<(D::IntStorage, Shape)>;

    fn i_gather(x: &D::IntStorage, x_l: &Layout, idx: &D::IntStorage, idx_l: &Layout, dim: usize) -> Result<(D::IntStorage, Shape)>;

    fn i_index_add(
        init: &D::IntStorage,
        init_l: &Layout,
        idx: &D::IntStorage,
        idx_l: &Layout,
        src: &D::IntStorage,
        src_l: &Layout,
        dim: usize,
    ) -> Result<D::IntStorage>;

    fn i_scatter_add(
        init: &D::IntStorage,
        init_l: &Layout,
        idx: &D::IntStorage,
        idx_l: &Layout,
        src: &D::IntStorage,
        src_l: &Layout,
        dim: usize,
    ) -> Result<D::IntStorage>;

    // shape
    fn i_cat(srcs: &[(&D::IntStorage, &Layout)], dim: usize) -> Result<(D::IntStorage, Shape)>;

    /// Produce the storage for a view of `src` under `dst_l`. See [`FloatOps::f_view`].
    fn i_view(_src: &D::IntStorage, _src_l: &Layout, _dst_l: &Layout, _view: crate::ViewOp) -> Result<Option<D::IntStorage>> {
        Ok(None)
    }

    // pick via a bool mask
    fn i_pick(
        mask: &D::BoolStorage,
        mask_l: &Layout,
        on_true: &D::IntStorage,
        true_l: &Layout,
        on_false: &D::IntStorage,
        false_l: &Layout,
    ) -> Result<D::IntStorage>;

    fn i_pick_true(mask: &D::BoolStorage, mask_l: &Layout, value: i64, on_false: &D::IntStorage, false_l: &Layout)
    -> Result<D::IntStorage>;

    fn i_pick_false(mask: &D::BoolStorage, mask_l: &Layout, on_true: &D::IntStorage, true_l: &Layout, value: i64) -> Result<D::IntStorage>;

    // allclose
    fn i_allclose(a: &D::IntStorage, a_l: &Layout, b: &D::IntStorage, b_l: &Layout) -> Result<bool>;
}
