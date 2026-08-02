use crate::Result;
use crate::dtype::{BoolDType, FloatDType, IntDType};
use crate::tensor::{Layout, Shape};

/// Operations for floating-point tensors.
pub trait FloatOps<D: super::Device> {
    // ---- construction ----
    fn f_zeros(shape: &Shape, device: &D, dtype: FloatDType) -> Result<D::FloatStorage>;
    fn f_ones(shape: &Shape, device: &D, dtype: FloatDType) -> Result<D::FloatStorage>;
    fn f_full(shape: &Shape, value: f64, device: &D, dtype: FloatDType) -> Result<D::FloatStorage>;
    fn f_from_f64(data: &[f64], dtype: FloatDType) -> Result<D::FloatStorage>;
    fn f_rand_uniform(shape: &Shape, lo: f64, hi: f64, device: &D, dtype: FloatDType) -> Result<D::FloatStorage>;
    fn f_rand_normal(shape: &Shape, mean: f64, std: f64, device: &D, dtype: FloatDType) -> Result<D::FloatStorage>;

    // ---- materialization / read-back ----
    fn f_contiguous(x: &D::FloatStorage, layout: &Layout) -> Result<D::FloatStorage>;
    fn f_cast_float(x: &D::FloatStorage, layout: &Layout, to: FloatDType) -> Result<D::FloatStorage>;
    fn f_cast_int(x: &D::FloatStorage, layout: &Layout, to: IntDType) -> Result<D::IntStorage>;
    fn f_cast_bool(x: &D::FloatStorage, layout: &Layout, to: BoolDType) -> Result<D::BoolStorage>;

    /// Read all elements into a `Vec<f64>` in logical (layout) order.
    fn f_to_vec(x: &D::FloatStorage, layout: &Layout) -> Result<Vec<f64>>;

    // ---- binary (elementwise, same shape; broadcasting handled above this layer) ----
    fn f_binary(
        lhs: &D::FloatStorage,
        lhs_l: &Layout,
        rhs: &D::FloatStorage,
        rhs_l: &Layout,
        op: crate::BinaryOp,
    ) -> Result<D::FloatStorage>;

    fn f_binary_scalar(lhs: &D::FloatStorage, lhs_l: &Layout, rhs: f64, op: crate::BinaryOp) -> Result<D::FloatStorage>;

    // ---- unary ----
    fn f_unary(x: &D::FloatStorage, layout: &Layout, op: crate::UnaryOp) -> Result<D::FloatStorage>;

    // ---- comparison -> bool storage ----
    fn f_cmp(lhs: &D::FloatStorage, lhs_l: &Layout, rhs: &D::FloatStorage, rhs_l: &Layout, op: crate::CmpOp) -> Result<D::BoolStorage>;

    // ---- reduction: returns (storage, resulting shape) ----
    fn f_reduce(
        x: &D::FloatStorage,
        layout: &Layout,
        dims: &[usize],
        keepdim: bool,
        op: crate::ReduceOp,
    ) -> Result<(D::FloatStorage, Shape)>;

    /// argmin/argmax return int-kind indices.
    fn f_arg_reduce(x: &D::FloatStorage, layout: &Layout, dim: usize, keepdim: bool, take_max: bool) -> Result<(D::IntStorage, Shape)>;

    // ---- matmul (batched); out shape computed by the caller / this fn ----
    fn f_matmul(lhs: &D::FloatStorage, lhs_l: &Layout, rhs: &D::FloatStorage, rhs_l: &Layout) -> Result<(D::FloatStorage, Shape)>;

    /// In-place accumulate: `dst += lhs @ rhs`. Used by the backward pass.
    fn f_add_matmul_(
        dst: &mut D::FloatStorage,
        dst_l: &Layout,
        lhs: &D::FloatStorage,
        lhs_l: &Layout,
        rhs: &D::FloatStorage,
        rhs_l: &Layout,
    ) -> Result<()>;

    // ---- in-place elementwise accumulation used during backprop ----
    /// `dst = f(dst, src)` for a binary op, respecting both layouts.
    fn f_binary_(dst: &mut D::FloatStorage, dst_l: &Layout, src: &D::FloatStorage, src_l: &Layout, op: crate::BinaryOp) -> Result<()>;

    // ---- indexing ----
    fn f_index_select(
        x: &D::FloatStorage,
        x_l: &Layout,
        idx: &D::IntStorage,
        idx_l: &Layout,
        dim: usize,
    ) -> Result<(D::FloatStorage, Shape)>;

    fn f_gather(x: &D::FloatStorage, x_l: &Layout, idx: &D::IntStorage, idx_l: &Layout, dim: usize) -> Result<(D::FloatStorage, Shape)>;

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

    // ---- shape ops that need data movement ----
    fn f_cat(srcs: &[(&D::FloatStorage, &Layout)], dim: usize) -> Result<(D::FloatStorage, Shape)>;

    // ---- nn fused kernels ----
    fn f_softmax(x: &D::FloatStorage, layout: &Layout, dim: usize) -> Result<D::FloatStorage>;

    fn f_rms_norm(x: &D::FloatStorage, x_l: &Layout, weight: &D::FloatStorage, weight_l: &Layout, eps: f64) -> Result<D::FloatStorage>;

    // ---- masked select via a bool mask ----
    fn f_if_else(
        mask: &D::BoolStorage,
        mask_l: &Layout,
        on_true: &D::FloatStorage,
        true_l: &Layout,
        on_false: &D::FloatStorage,
        false_l: &Layout,
    ) -> Result<D::FloatStorage>;
}
