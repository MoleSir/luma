//! CUDA tests — grouped by op category. Each #[test] function covers a group.
//! Run with: cargo test -p luma-cuda

use std::sync::LazyLock;

use luma_tensor::tests::*;
use luma_tensor::dtype::{FloatDType, IntDType};
use luma_tensor::{Int, Shape, Tensor};

use luma_cpu::Cpu;
use luma_cuda::Cuda;

static CUDA: LazyLock<Cuda> = LazyLock::new(|| Cuda::new(0).expect("cuda device 0"));

// CPU convenience constructors used to seed cross-device roundtrips.
fn tensor_f32<S: Into<Shape>>(data: &[f64], shape: S) -> Tensor<Cpu> {
    Tensor::<Cpu>::from_slice(data, shape, FloatDType::F32).unwrap()
}

fn tensor_i32<S: Into<Shape>>(data: &[i64], shape: S) -> Tensor<Cpu, Int> {
    Tensor::<Cpu, Int>::from_slice(data, shape, IntDType::I32).unwrap()
}

#[test]
fn cuda_binary() {
    let dev = &*CUDA;
    numeric::test_add_f32(dev);
    numeric::test_sub_f32(dev);
    numeric::test_mul_f32(dev);
    numeric::test_div_f32(dev);
    numeric::test_maximum_f32(dev);
    numeric::test_minimum_f32(dev);
}

#[test]
fn cuda_unary() {
    let dev = &*CUDA;
    numeric::test_neg_f32(dev);
    numeric::test_abs_f32(dev);
    numeric::test_relu_f32(dev);
    numeric::test_exp_f32(dev);
    numeric::test_sigmoid_f32(dev);
    numeric::test_tanh_f32(dev);
    numeric::test_ln_f32(dev);
    numeric::test_sin_f32(dev);
    numeric::test_cos_f32(dev);
    numeric::test_sqr_f32(dev);
    numeric::test_sqrt_f32(dev);
    numeric::test_recip_f32(dev);
    numeric::test_gelu_f32(dev);
    numeric::test_silu_f32(dev);
    numeric::test_floor_f32(dev);
    numeric::test_ceil_f32(dev);
    numeric::test_sign_f32(dev);
    numeric::test_leaky_relu_f32(dev);
    numeric::test_pow_f32(dev);
    numeric::test_affine_f32(dev);
    numeric::test_erf_f32(dev);
    numeric::test_gelu_erf_f32(dev);
    numeric::test_round_f32(dev);
}

#[test]
fn cuda_dtype() {
    let dev = &*CUDA;
    dtype::test_u8_construct(dev);
    dtype::test_u8_add(dev);
    dtype::test_u8_sub(dev);
    dtype::test_u8_clamp(dev);
    dtype::test_u8_cast_to_i32(dev);
    dtype::test_u8_cast_to_f32(dev);
    dtype::test_u32_construct(dev);
    dtype::test_u32_add(dev);
    dtype::test_u32_mul(dev);
}

#[test]
fn cuda_scalar() {
    let dev = &*CUDA;
    numeric::test_add_scalar_f32(dev);
    numeric::test_sub_scalar_f32(dev);
    numeric::test_sub_scalar_lhs_f32(dev);
    numeric::test_mul_scalar_f32(dev);
    numeric::test_div_scalar_f32(dev);
    numeric::test_div_scalar_lhs_f32(dev);
}

#[test]
fn cuda_cmp() {
    let dev = &*CUDA;
    numeric::test_eq_f32(dev);
    numeric::test_lt_f32(dev);
    numeric::test_ne_f32(dev);
    numeric::test_ge_f32(dev);
    numeric::test_gt_f32(dev);
    numeric::test_le_f32(dev);
}

#[test]
fn cuda_reduce() {
    let dev = &*CUDA;
    reduce::test_sum_dim_f32(dev);
    reduce::test_sum_keepdim_f32(dev);
    reduce::test_sum_all_f32(dev);
    reduce::test_sum_dims_f32(dev);
    reduce::test_max_dim_f32(dev);
    reduce::test_max_keepdim_f32(dev);
    reduce::test_max_all_f32(dev);
    reduce::test_min_dim_f32(dev);
    reduce::test_min_all_f32(dev);
    reduce::test_mean_dim_f32(dev);
    reduce::test_mean_all_f32(dev);
    reduce::test_prod_dim_f32(dev);
    reduce::test_prod_all_f32(dev);
    reduce::test_argmax_f32(dev);
    reduce::test_argmin_f32(dev);
    reduce::test_argmax_keepdim(dev);
    reduce::test_var_f32(dev);
    reduce::test_var_unbiased_f32(dev);
    reduce::test_std_f32(dev);
    reduce::test_std_all_f32(dev);
    reduce::test_logsumexp_f32(dev);
    reduce::test_logsumexp_keepdim(dev);
    reduce::test_sum_i32(dev);
    reduce::test_sum_all_i32(dev);
    reduce::test_max_dim_i32(dev);
    reduce::test_min_dim_i32(dev);
    reduce::test_sum_u8(dev);
    reduce::test_max_u8(dev);
    reduce::test_sum_u32(dev);
    reduce::test_min_u32(dev);
}

#[test]
fn cuda_bool() {
    let dev = &*CUDA;
    boolean::test_bool_and(dev);
    boolean::test_bool_or(dev);
    boolean::test_bool_xor(dev);
    boolean::test_bool_not(dev);
    boolean::test_pick_f32(dev);
    boolean::test_pick_scalar_true(dev);
    boolean::test_pick_scalar_false(dev);
    boolean::test_pick_bool(dev);
    boolean::test_pick_int(dev);
    boolean::test_pick_int_scalar_true(dev);
    boolean::test_pick_int_scalar_false(dev);
    boolean::test_pick_bool_scalar_true(dev);
    boolean::test_pick_bool_scalar_false(dev);
    boolean::test_bool_all_all(dev);
    boolean::test_bool_any_all(dev);
    boolean::test_bool_true_count(dev);
    boolean::test_bool_false_count(dev);
    boolean::test_allclose_exact(dev);
    boolean::test_allclose_false(dev);
    boolean::test_allclose_int(dev);
    boolean::test_allclose_bool(dev);
}

#[test]
fn cuda_clamp() {
    let dev = &*CUDA;
    numeric::test_clamp_both(dev);
    numeric::test_clamp_min_only(dev);
    numeric::test_clamp_max_only(dev);
    numeric::test_clamp_none(dev);
    numeric::test_pow_exp_zero(dev);
    numeric::test_pow_exp_one(dev);
}

#[test]
fn cuda_broadcast() {
    let dev = &*CUDA;
    numeric::test_broadcast_add_f32(dev);
    numeric::test_broadcast_mul_f32(dev);
    numeric::test_broadcast_eq_f32(dev);
}

#[test]
fn cuda_cast() {
    let dev = &*CUDA;
    cast::test_cast_f32_to_f64(dev);
    cast::test_cast_f32_to_i32(dev);
    cast::test_cast_f32_to_bool(dev);
    cast::test_cast_i32_to_f32(dev);
    cast::test_cast_bool_to_f32(dev);
    cast::test_cast_bool_to_i32(dev);
    cast::test_cast_i32_to_bool(dev);
    cast::test_cast_f64_to_f32(dev);
    cast::test_cast_i32_to_u32(dev);
    cast::test_cast_bool_to_bool(dev);
}

#[test]
fn cuda_f64() {
    let dev = &*CUDA;
    cast::test_f64_zeros(dev);
    cast::test_f64_add(dev);
    f64::test_f64_neg(dev);
    f64::test_f64_abs(dev);
    f64::test_f64_relu(dev);
    f64::test_f64_exp(dev);
    f64::test_f64_ln(dev);
    f64::test_f64_sqrt(dev);
    f64::test_f64_sigmoid(dev);
    f64::test_f64_tanh(dev);
    f64::test_f64_sin(dev);
    f64::test_f64_cos(dev);
    f64::test_f64_sqr(dev);
    f64::test_f64_recip(dev);
    f64::test_f64_floor(dev);
    f64::test_f64_ceil(dev);
    f64::test_f64_sign(dev);
    f64::test_f64_pow(dev);
    f64::test_f64_affine(dev);
    f64::test_f64_eq(dev);
    f64::test_f64_lt(dev);
    f64::test_f64_gt(dev);
    f64::test_f64_le(dev);
    f64::test_f64_ge(dev);
    f64::test_f64_ne(dev);
    f64::test_f64_add_scalar(dev);
    f64::test_f64_sub_scalar(dev);
    f64::test_f64_sub_scalar_lhs(dev);
    f64::test_f64_mul_scalar(dev);
    f64::test_f64_div_scalar(dev);
    f64::test_f64_div_scalar_lhs(dev);
    f64::test_f64_sum_dim(dev);
    f64::test_f64_max_all(dev);
    f64::test_f64_grad_add(dev);
    f64::test_f64_grad_mul(dev);
}

#[test]
fn cuda_display() {
    let dev = &*CUDA;
    display::test_display_scalar(dev);
    display::test_display_1d(dev);
}

#[test]
fn cuda_shape() {
    let dev = &*CUDA;
    shape::test_cat_dim0_f32(dev);
    shape::test_contiguous_after_transpose(dev);
    shape::test_reshape_f32(dev);
    shape::test_transpose_f32(dev);
    shape::test_broadcast_as_f32(dev);
    shape::test_narrow_dim0(dev);
    shape::test_squeeze_dim1(dev);
    shape::test_unsqueeze(dev);
    shape::test_flatten_all(dev);
    shape::test_permute_f32(dev);
    shape::test_split_f32(dev);
    shape::test_repeat_dim_f32(dev);
    shape::test_transpose_last(dev);
    shape::test_already_contiguous_is_noop(dev);
    shape::test_flatten_range(dev);
    shape::test_stack_f32(dev);
    shape::test_chunk_f32(dev);
}

#[test]
fn cuda_matmul() {
    let dev = &*CUDA;
    matmul::test_matmul_2x2(dev);
    matmul::test_matmul_2x3_3x2(dev);
    matmul::test_matmul_f64(dev);
}

#[test]
fn cuda_int() {
    let dev = &*CUDA;
    numeric::test_add_i32(dev);
    numeric::test_neg_i32(dev);
    numeric::test_abs_i32(dev);
    numeric::test_sign_i32(dev);
    numeric::test_pow_i32(dev);
    numeric::test_affine_i32(dev);
    numeric::test_clamp_i32(dev);
    numeric::test_add_scalar_i32(dev);
    numeric::test_sub_scalar_i32(dev);
    numeric::test_sub_scalar_lhs_i32(dev);
    numeric::test_mul_scalar_i32(dev);
    numeric::test_div_scalar_i32(dev);
    numeric::test_div_scalar_lhs_i32(dev);
}

#[test]
fn cuda_construct() {
    let dev = &*CUDA;
    construct::test_zeros_like_f32(dev);
    construct::test_ones_like_f32(dev);
    construct::test_from_slice_f32(dev);
    construct::test_full_scalar(dev);
    construct::test_rand_like_shape(dev);
    construct::test_randn_like_shape(dev);
}

#[test]
fn cuda_grad() {
    let dev = &*CUDA;
    grad::test_grad_add(dev);
    grad::test_grad_sub(dev);
    grad::test_grad_mul(dev);
    grad::test_grad_div(dev);
    grad::test_grad_relu(dev);
    grad::test_grad_sum(dev);
    grad::test_grad_mean(dev);
    grad::test_grad_exp(dev);
    grad::test_grad_sigmoid(dev);
    grad::test_grad_clamp(dev);
    grad::test_grad_clamp_min(dev);
    grad::test_grad_prod(dev);
    grad::test_grad_reshape(dev);
    grad::test_grad_transpose(dev);
    grad::test_grad_matmul(dev);
    grad::test_grad_accumulate(dev);
    grad::test_no_grad_disabled(dev);
}

#[test]
fn cuda_indexing() {
    let dev = &*CUDA;
    indexing::test_index_select_dim0(dev);
    indexing::test_index_select_dim1(dev);
    indexing::test_gather_dim1(dev);
    indexing::test_index_add_f32(dev);
    indexing::test_scatter_add_f32(dev);
    indexing::test_index_add_2d(dev);
    indexing::test_i_select_row(dev);
    indexing::test_i_select_negative(dev);
    indexing::test_i_slice_range(dev);
    indexing::test_i_slice_full(dev);
    indexing::test_i_slice_with_step(dev);
    indexing::test_i_tuple_select_slice(dev);
    indexing::test_i_tuple_slice_slice(dev);
    indexing::test_i_boolean_mask(dev);
    indexing::test_i_boolean_mask_2d(dev);
    indexing::test_get_element(dev);
    indexing::test_get_row_2d(dev);
}

#[test]
fn cuda_nn() {
    let dev = &*CUDA;
    nn::test_softmax_dim0(dev);
    nn::test_softmax_dim1(dev);
    nn::test_softmax_numerical_stability(dev);
    nn::test_rms_norm_f32(dev);
    nn::test_rms_norm_weighted(dev);
    nn::test_cross_entropy_chain_f32(dev);
    nn::test_cross_entropy_basic_f32(dev);
    nn::test_cross_entropy_mnist_shape_f32(dev);
    nn::test_matmul_transposed_weight_small_f32(dev);
    nn::test_matmul_transposed_weight_f32(dev);
    nn::test_matmul_add_bias_f32(dev);
    nn::test_broadcast_add_precision(dev);
    nn::test_cross_entropy_precision(dev);
    nn::test_broadcast_add_grad_f32(dev);
    nn::test_broadcast_reduce_backward_f32(dev);
    nn::test_sum_keepdim_nonuniform_f32(dev);
    nn::test_argmax_eval_pipeline_f32(dev);
    nn::test_mini_training_step_f32(dev);
    nn::test_argmax_eval_large_f32(dev);
}

#[test]
fn cuda_edge() {
    let dev = &*CUDA;
    numeric::test_sqrt_negative(dev);
    numeric::test_ln_zero(dev);
    numeric::test_exp_large(dev);
    numeric::test_div_zero_f32(dev);
    numeric::test_add_nan_f32(dev);
}

#[test]
fn cuda_large() {
    let dev = &*CUDA;
    reduce::test_large_sum_f32(dev);
    matmul::test_large_matmul_f32(dev);
    nn::test_large_softmax(dev);
}

#[test]
fn cuda_cross() {
    let dev = &*CUDA;
    cross::test_transpose_add(dev);
    cross::test_transpose_sub(dev);
    cross::test_transpose_sum(dev);
    cross::test_transpose_max(dev);
    cross::test_permute_add(dev);
    cross::test_slice_sum(dev);
    cross::test_narrow_add(dev);
    cross::test_permute_contiguous_add(dev);
    cross::test_broadcast_sum(dev);
}

#[test]
fn cuda_error() {
    let dev = &*CUDA;
    error::test_binary_shape_mismatch(dev);
    error::test_matmul_shape_mismatch(dev);
    error::test_narrow_out_of_range(dev);
    error::test_dim_out_of_range(dev);
    error::test_allclose_shape_mismatch(dev);
    error::test_f64_to_f32_add(dev);
    error::test_reshape_wrong_elements(dev);
}

#[test]
fn cuda_to_device() {
    let dev = &*CUDA;

    // Cpu -> Cuda -> Cpu roundtrip (f32), through the public to_device API.
    let src = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
    let gpu = src.to_device(dev).unwrap();
    assert_eq!(gpu.dtype(), src.dtype());
    assert_eq!(gpu.dims(), &[2, 3]);
    let back = gpu.to_device(&Cpu::default()).unwrap();
    assert_close(&back.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-5, 1e-5);

    // f64 roundtrip — dtype must be preserved on both sides.
    let src64 = tensor_f64_dev(&[1.5, 2.5, 3.5], (3,), &Cpu::default());
    let gpu64 = src64.to_device(dev).unwrap();
    assert_eq!(gpu64.dtype(), FloatDType::F64);
    let back64 = gpu64.to_device(&Cpu::default()).unwrap();
    assert_close(&back64.to_vec().unwrap(), &[1.5, 2.5, 3.5], 1e-5, 1e-5);

    // Int roundtrip.
    let srci = tensor_i32(&[1, 2, 3, 4], (4,));
    let gpui = srci.to_device(dev).unwrap();
    let backi = gpui.to_device(&Cpu::default()).unwrap();
    assert_eq!(backi.to_vec().unwrap(), vec![1, 2, 3, 4]);

    // Bool roundtrip (stored as u8 on device — the bytes path bridges this).
    let srcb = tensor_bool_dev(&[true, false, true, true], (4,), &Cpu::default());
    let gpub = srcb.to_device(dev).unwrap();
    let backb = gpub.to_device(&Cpu::default()).unwrap();
    assert_eq!(backb.to_vec().unwrap(), vec![true, false, true, true]);

    // Non-contiguous cpu tensor -> cuda: result is contiguous, values in
    // logical order.
    let nc = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3)).transpose(0usize, 1usize).unwrap();
    assert!(!nc.is_contiguous());
    let gpun = nc.to_device(dev).unwrap();
    assert!(gpun.is_contiguous());
    assert_close(&gpun.to_vec().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-5, 1e-5);

    // requires_grad preserved across the transfer, graph severed.
    let gr = tensor_f32_dev(&[1.0, 2.0], (2,), &Cpu::default());
    gr.set_requires_grad(true);
    let grg = gr.to_device(dev).unwrap();
    assert!(grg.requires_grad());
    assert!(grg.op().is_none());

    // Same-device fast path: same handle, and a fresh handle to the same
    // ordinal must also hit the no-op path (Cuda::same_device override).
    let same = gpun.to_device(dev).unwrap();
    assert_eq!(same.id(), gpun.id());
    let dev2 = Cuda::new(0).expect("cuda device 0 (second handle)");
    let same2 = gpun.to_device(&dev2).unwrap();
    assert_eq!(same2.id(), gpun.id(), "same ordinal must be a no-op");

    // Explicit `to_device(&Cuda)` sugar (the removed `.cuda()` / `.cpu()`
    // convenience methods are gone from `luma-tensor`).
    let g = Cuda::new(0).expect("cuda device 0 (sugar)");
    let sug = src.to_device(&g).unwrap();
    let sugback = sug.to_device(&Cpu::default()).unwrap();
    assert_close(&sugback.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-5, 1e-5);

    // Identity tests (common delegation) run on Cuda too.
    transfer::test_to_device_identity_f32(dev);
    transfer::test_to_device_identity_int(dev);
    transfer::test_to_device_identity_bool(dev);
    transfer::test_to_device_identity_requires_grad(dev);
}

#[test]
fn cuda_to_device_severs_graph_and_grad_flows() {
    let dev = &*CUDA;

    // CPU non-leaf → GPU: the op is severed (result is a fresh leaf), but the
    // trainability flag is preserved and gradients still flow on the GPU.
    let x = tensor_f32(&[2.0, 3.0], (2,));
    x.set_requires_grad(true);
    let y = x.mul(&x).unwrap();
    assert!(y.op().is_some());

    let yg = y.to_device(dev).unwrap();
    assert!(yg.requires_grad());
    assert!(yg.op().is_none(), "cross-device transfer severs the graph");
    assert!(yg.is_leaf());

    let z = yg.mul(&yg).unwrap();
    let grads = z.backward().unwrap();
    let gy = grads.get_by_id(yg.id()).unwrap();
    assert_close(&gy.to_vec().unwrap(), &[8.0, 18.0], 1e-5, 1e-5);

    // GPU non-leaf → CPU: the same severing in the other direction.
    let y2 = yg.mul(&yg).unwrap();
    assert!(y2.op().is_some());
    let back = y2.to_device(&Cpu::default()).unwrap();
    assert!(back.op().is_none());
    assert!(back.is_leaf());
    assert!(back.requires_grad());
}

#[test]
fn cuda_cross_ordinal_transfer() {
    let dev0 = &*CUDA; // ordinal 0
    let Ok(dev1) = Cuda::new(1) else {
        eprintln!("skipping: a second CUDA device (ordinal 1) is not available");
        return;
    };

    // Cuda(0) → Cuda(1): different ordinals share the `Cuda` type, so this
    // bypasses the no-op fast path and goes through the host copy.
    let src = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (2, 2)).to_device(dev0).unwrap();
    let dst = src.to_device(&dev1).unwrap();
    assert_ne!(dst.id(), src.id(), "cross-ordinal transfer must copy");
    assert_close(&dst.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0], 1e-5, 1e-5);
}
