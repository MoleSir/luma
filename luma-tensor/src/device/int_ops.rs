use crate::Result;
use crate::dtype::{BoolDType, FloatDType, IntDType};
use crate::tensor::{Layout, Shape};

/// Operations on `Int`-kind tensors. No autograd. Scalars are `i64` (wide enough
/// for all int precisions).
pub trait IntOps<D: super::Device> {
    // ---- construction ----
    fn i_zeros(shape: &Shape, device: &D, dtype: IntDType) -> Result<D::IntStorage>;
    fn i_ones(shape: &Shape, device: &D, dtype: IntDType) -> Result<D::IntStorage>;
    fn i_full(shape: &Shape, value: i64, device: &D, dtype: IntDType) -> Result<D::IntStorage>;
    fn i_from_i64(data: &[i64], device: &D, dtype: IntDType) -> Result<D::IntStorage>;
    fn i_arange(start: i64, end: i64, step: i64, device: &D, dtype: IntDType) -> Result<(D::IntStorage, usize)>;

    // ---- materialization / read-back ----
    fn i_contiguous(x: &D::IntStorage, layout: &Layout) -> Result<D::IntStorage>;
    fn i_cast_float(x: &D::IntStorage, layout: &Layout, to: FloatDType) -> Result<D::FloatStorage>;
    fn i_cast_int(x: &D::IntStorage, layout: &Layout, to: IntDType) -> Result<D::IntStorage>;
    fn i_cast_bool(x: &D::IntStorage, layout: &Layout, to: BoolDType) -> Result<D::BoolStorage>;

    /// Read all elements into a `Vec<i64>` in logical (layout) order.
    fn i_to_vec(x: &D::IntStorage, layout: &Layout) -> Result<Vec<i64>>;

    // ---- arithmetic ----
    fn i_binary(lhs: &D::IntStorage, lhs_l: &Layout, rhs: &D::IntStorage, rhs_l: &Layout, op: crate::BinaryOp) -> Result<D::IntStorage>;

    fn i_binary_scalar(lhs: &D::IntStorage, lhs_l: &Layout, rhs: i64, op: crate::BinaryOp) -> Result<D::IntStorage>;

    fn i_neg(x: &D::IntStorage, layout: &Layout) -> Result<D::IntStorage>;
    fn i_abs(x: &D::IntStorage, layout: &Layout) -> Result<D::IntStorage>;
    fn i_sign(x: &D::IntStorage, layout: &Layout) -> Result<D::IntStorage>;
    fn i_affine(x: &D::IntStorage, layout: &Layout, mul: i64, add: i64) -> Result<D::IntStorage>;
    fn i_pow(x: &D::IntStorage, layout: &Layout, exp: i64) -> Result<D::IntStorage>;
    fn i_clamp(x: &D::IntStorage, layout: &Layout, min: Option<i64>, max: Option<i64>) -> Result<D::IntStorage>;

    // ---- matmul (batched); out shape computed by the caller / this fn ----
    fn i_matmul(lhs: &D::IntStorage, lhs_l: &Layout, rhs: &D::IntStorage, rhs_l: &Layout) -> Result<(D::IntStorage, Shape)>;

    // ---- comparison -> bool ----
    fn i_cmp(lhs: &D::IntStorage, lhs_l: &Layout, rhs: &D::IntStorage, rhs_l: &Layout, op: crate::CmpOp) -> Result<D::BoolStorage>;

    // ---- reduction ----
    fn i_reduce(x: &D::IntStorage, layout: &Layout, dims: &[usize], keepdim: bool, op: crate::ReduceOp) -> Result<(D::IntStorage, Shape)>;

    /// argmin/argmax return int-kind indices.
    fn i_arg_reduce(x: &D::IntStorage, layout: &Layout, dim: usize, keepdim: bool, take_max: bool) -> Result<(D::IntStorage, Shape)>;

    // ---- indexing ----
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

    // ---- shape ----
    fn i_cat(srcs: &[(&D::IntStorage, &Layout)], dim: usize) -> Result<(D::IntStorage, Shape)>;

    // ---- pick via a bool mask ----
    fn i_pick(
        mask: &D::BoolStorage,
        mask_l: &Layout,
        on_true: &D::IntStorage,
        true_l: &Layout,
        on_false: &D::IntStorage,
        false_l: &Layout,
    ) -> Result<D::IntStorage>;

    fn i_pick_true(
        mask: &D::BoolStorage,
        mask_l: &Layout,
        value: i64,
        on_false: &D::IntStorage,
        false_l: &Layout,
    ) -> Result<D::IntStorage>;

    fn i_pick_false(
        mask: &D::BoolStorage,
        mask_l: &Layout,
        on_true: &D::IntStorage,
        true_l: &Layout,
        value: i64,
    ) -> Result<D::IntStorage>;

    // ---- allclose ----
    fn i_allclose(a: &D::IntStorage, a_l: &Layout, b: &D::IntStorage, b_l: &Layout) -> Result<bool>;
}
