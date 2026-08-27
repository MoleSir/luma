use std::borrow::Cow;

use crate::Result;
use crate::dtype::{BoolDType, FloatDType, IntDType};
use crate::tensor::{Layout, Shape};

/// Operations on `Bool`-kind tensors: logical ops, casts, reductions, masking.
pub trait BoolOps<D: super::Device> {
    // construction
    fn b_falses(shape: &Shape, device: &D, dtype: BoolDType) -> Result<D::BoolStorage>;
    fn b_trues(shape: &Shape, device: &D, dtype: BoolDType) -> Result<D::BoolStorage>;
    fn b_from_bool<'a>(data: impl Into<Cow<'a, [bool]>>, device: &D) -> Result<D::BoolStorage>;

    fn b_from_bytes<'a>(bytes: impl Into<Cow<'a, [u8]>>, shape: &Shape, device: &D, dtype: BoolDType) -> Result<D::BoolStorage>;

    // materialization / read-back
    fn b_contiguous(x: &D::BoolStorage, layout: &Layout) -> Result<D::BoolStorage>;
    fn b_cast_float(x: &D::BoolStorage, layout: &Layout, to: FloatDType) -> Result<D::FloatStorage>;
    fn b_cast_int(x: &D::BoolStorage, layout: &Layout, to: IntDType) -> Result<D::IntStorage>;
    fn b_cast_bool(x: &D::BoolStorage, layout: &Layout, to: BoolDType) -> Result<D::BoolStorage>;

    /// Read all elements into a `Vec<bool>` in logical (layout) order.
    fn b_to_vec(x: &D::BoolStorage, layout: &Layout) -> Result<Vec<bool>>;

    /// Read raw bytes (0/1 per element) in logical (layout) order.
    /// Returns `Cow::Borrowed` when the underlying storage is already contiguous
    /// (zero-copy); `Cow::Owned` otherwise.
    fn b_to_bytes<'a>(x: &'a D::BoolStorage, layout: &Layout) -> Result<Cow<'a, [u8]>>;

    // logical ops
    fn b_and(lhs: &D::BoolStorage, lhs_l: &Layout, rhs: &D::BoolStorage, rhs_l: &Layout) -> Result<D::BoolStorage>;
    fn b_or(lhs: &D::BoolStorage, lhs_l: &Layout, rhs: &D::BoolStorage, rhs_l: &Layout) -> Result<D::BoolStorage>;
    fn b_xor(lhs: &D::BoolStorage, lhs_l: &Layout, rhs: &D::BoolStorage, rhs_l: &Layout) -> Result<D::BoolStorage>;
    fn b_not(x: &D::BoolStorage, layout: &Layout) -> Result<D::BoolStorage>;

    // reductions (all/any over dims)
    fn b_reduce_all(x: &D::BoolStorage, layout: &Layout, dims: &[usize], keepdim: bool) -> Result<(D::BoolStorage, Shape)>;
    fn b_reduce_any(x: &D::BoolStorage, layout: &Layout, dims: &[usize], keepdim: bool) -> Result<(D::BoolStorage, Shape)>;

    fn b_true_count(x: &D::BoolStorage, layout: &Layout) -> Result<usize>;

    // shape
    fn b_cat(srcs: &[(&D::BoolStorage, &Layout)], dim: usize) -> Result<(D::BoolStorage, Shape)>;

    /// Produce the storage for a view of `src` under `dst_l`. See [`FloatOps::f_view`].
    fn b_view(_src: &D::BoolStorage, _src_l: &Layout, _dst_l: &Layout, _view: crate::ViewOp) -> Result<Option<D::BoolStorage>> {
        Ok(None)
    }

    // pick via a bool mask
    fn b_pick(
        mask: &D::BoolStorage,
        mask_l: &Layout,
        on_true: &D::BoolStorage,
        true_l: &Layout,
        on_false: &D::BoolStorage,
        false_l: &Layout,
    ) -> Result<D::BoolStorage>;

    fn b_pick_true(
        mask: &D::BoolStorage,
        mask_l: &Layout,
        value: bool,
        on_false: &D::BoolStorage,
        false_l: &Layout,
    ) -> Result<D::BoolStorage>;

    fn b_pick_false(
        mask: &D::BoolStorage,
        mask_l: &Layout,
        on_true: &D::BoolStorage,
        true_l: &Layout,
        value: bool,
    ) -> Result<D::BoolStorage>;

    // allclose
    fn b_allclose(a: &D::BoolStorage, a_l: &Layout, b: &D::BoolStorage, b_l: &Layout) -> Result<bool>;
}
