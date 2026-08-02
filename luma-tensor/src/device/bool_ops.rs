use crate::dtype::{BoolDType, FloatDType, IntDType};
use crate::Result;
use crate::tensor::{Layout, Shape};

/// Operations on `Bool`-kind tensors: logical ops, casts, reductions, masking.
pub trait BoolOps<D: super::Device> {
    // ---- construction ----
    fn b_falses(shape: &Shape, device: &D, dtype: BoolDType) -> Result<D::BoolStorage>;
    fn b_trues(shape: &Shape, device: &D, dtype: BoolDType) -> Result<D::BoolStorage>;
    fn b_from_bool(data: &[bool], device: &D, dtype: BoolDType) -> Result<D::BoolStorage>;

    // ---- materialization / read-back ----
    fn b_contiguous(x: &D::BoolStorage, layout: &Layout) -> Result<D::BoolStorage>;
    fn b_cast_float(x: &D::BoolStorage, layout: &Layout, to: FloatDType) -> Result<D::FloatStorage>;
    fn b_cast_int(x: &D::BoolStorage, layout: &Layout, to: IntDType) -> Result<D::IntStorage>;
    fn b_cast_bool(x: &D::BoolStorage, layout: &Layout, to: BoolDType) -> Result<D::BoolStorage>;

    /// Read all elements into a `Vec<bool>` in logical (layout) order.
    fn b_to_vec(x: &D::BoolStorage, layout: &Layout) -> Result<Vec<bool>>;

    // ---- logical ops ----
    fn b_and(lhs: &D::BoolStorage, lhs_l: &Layout, rhs: &D::BoolStorage, rhs_l: &Layout) -> Result<D::BoolStorage>;
    fn b_or(lhs: &D::BoolStorage, lhs_l: &Layout, rhs: &D::BoolStorage, rhs_l: &Layout) -> Result<D::BoolStorage>;
    fn b_xor(lhs: &D::BoolStorage, lhs_l: &Layout, rhs: &D::BoolStorage, rhs_l: &Layout) -> Result<D::BoolStorage>;
    fn b_not(x: &D::BoolStorage, layout: &Layout) -> Result<D::BoolStorage>;

    // ---- reductions (all/any over dims) ----
    fn b_reduce_all(x: &D::BoolStorage, layout: &Layout, dims: &[usize], keepdim: bool) -> Result<(D::BoolStorage, Shape)>;
    fn b_reduce_any(x: &D::BoolStorage, layout: &Layout, dims: &[usize], keepdim: bool) -> Result<(D::BoolStorage, Shape)>;

    fn b_true_count(x: &D::BoolStorage, layout: &Layout) -> Result<usize>;

    // ---- shape ----
    fn b_cat(srcs: &[(&D::BoolStorage, &Layout)], dim: usize) -> Result<(D::BoolStorage, Shape)>;
}
