use crate::Result;
use crate::dtype::{BoolDType, FloatDType, IntDType, Storage};
use crate::tensor::{Layout, Shape};
use crate::Float;

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

    fn f_neg(x: &D::FloatStorage, layout: &Layout) -> Result<D::FloatStorage>;
    fn f_abs(x: &D::FloatStorage, layout: &Layout) -> Result<D::FloatStorage>;
    fn f_sign(x: &D::FloatStorage, layout: &Layout) -> Result<D::FloatStorage>;
    fn f_affine(x: &D::FloatStorage, layout: &Layout, mul: f64, add: f64) -> Result<D::FloatStorage>;
    fn f_pow(x: &D::FloatStorage, layout: &Layout, exp: f64) -> Result<D::FloatStorage>;
    fn f_clamp(x: &D::FloatStorage, layout: &Layout, min: Option<f64>, max: Option<f64>) -> Result<D::FloatStorage>;

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
    fn f_softmax(x: &D::FloatStorage, layout: &Layout, dim: usize) -> Result<D::FloatStorage> {
        let data = D::f_to_vec(x, layout)?;
        let dims = layout.dims();
        let reduce_size = dims[dim];
        let outer: usize = dims[..dim].iter().product();
        let inner: usize = dims[dim + 1..].iter().product();
        let mut out = vec![0f64; data.len()];

        for o in 0..outer {
            for i in 0..inner {
                let row: Vec<f64> = (0..reduce_size).map(|r| data[o * reduce_size * inner + r * inner + i]).collect();
                let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_sum: f64 = row.iter().map(|&v| (v - max_val).exp()).sum();
                for r in 0..reduce_size {
                    out[o * reduce_size * inner + r * inner + i] = ((row[r] - max_val).exp()) / exp_sum;
                }
            }
        }

        let dtype = <D::FloatStorage as Storage<D, Float>>::dtype(x);
        D::f_from_f64(&out, dtype)
    }

    fn f_rms_norm(x: &D::FloatStorage, x_l: &Layout, weight: &D::FloatStorage, weight_l: &Layout, eps: f64) -> Result<D::FloatStorage> {
        let x_data = D::f_to_vec(x, x_l)?;
        let w_data = D::f_to_vec(weight, weight_l)?;
        let last_dim = x_l.shape().rank() - 1;
        let last_dim_size = x_l.dims()[last_dim];
        let batch = x_data.len() / last_dim_size;

        let mut out = vec![0f64; x_data.len()];
        for b in 0..batch {
            let (b_start, b_end) = (b * last_dim_size, (b + 1) * last_dim_size);
            let mean_sq = x_data[b_start..b_end].iter().map(|&v| v * v).sum::<f64>() / last_dim_size as f64;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            for i in 0..last_dim_size {
                out[b_start + i] = x_data[b_start + i] * inv_rms * w_data[i];
            }
        }

        let dtype = <D::FloatStorage as Storage<D, Float>>::dtype(x);
        D::f_from_f64(&out, dtype)
    }

    // ---- pick via a bool mask ----
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

    // ---- allclose ----
    fn f_allclose(a: &D::FloatStorage, a_l: &Layout, b: &D::FloatStorage, b_l: &Layout, rtol: f64, atol: f64) -> Result<bool>;
}
