#![cfg(feature = "cuda")]
//! CUDA tests — grouped by op category. Each #[test] function covers a group.
//! Run with: cargo test --test cuda

mod common;
use luma_tensor::device::cuda::Cuda;
use std::sync::LazyLock;

static CUDA: LazyLock<Cuda> = LazyLock::new(|| Cuda::new(0).expect("cuda device 0"));

#[test]
fn cuda_binary() {
    let dev = &*CUDA;
    common::numeric::test_add_f32(dev);
    common::numeric::test_sub_f32(dev);
    common::numeric::test_mul_f32(dev);
    common::numeric::test_div_f32(dev);
    common::numeric::test_maximum_f32(dev);
    common::numeric::test_minimum_f32(dev);
}

#[test]
fn cuda_unary() {
    let dev = &*CUDA;
    common::numeric::test_neg_f32(dev);
    common::numeric::test_abs_f32(dev);
    common::numeric::test_relu_f32(dev);
    common::numeric::test_exp_f32(dev);
    common::numeric::test_sigmoid_f32(dev);
    common::numeric::test_tanh_f32(dev);
    common::numeric::test_ln_f32(dev);
    common::numeric::test_sin_f32(dev);
    common::numeric::test_cos_f32(dev);
    common::numeric::test_sqr_f32(dev);
    common::numeric::test_sqrt_f32(dev);
    common::numeric::test_recip_f32(dev);
    common::numeric::test_gelu_f32(dev);
    common::numeric::test_silu_f32(dev);
    common::numeric::test_floor_f32(dev);
    common::numeric::test_ceil_f32(dev);
    common::numeric::test_sign_f32(dev);
    common::numeric::test_leaky_relu_f32(dev);
    common::numeric::test_pow_f32(dev);
    common::numeric::test_affine_f32(dev);
    common::numeric::test_erf_f32(dev);
    common::numeric::test_gelu_erf_f32(dev);
    common::numeric::test_round_f32(dev);
}

#[test]
fn cuda_dtype() {
    let dev = &*CUDA;
    common::dtype::test_u8_construct(dev);
    common::dtype::test_u8_add(dev);
    common::dtype::test_u8_sub(dev);
    common::dtype::test_u8_clamp(dev);
    common::dtype::test_u8_cast_to_i32(dev);
    common::dtype::test_u8_cast_to_f32(dev);
    common::dtype::test_u32_construct(dev);
    common::dtype::test_u32_add(dev);
    common::dtype::test_u32_mul(dev);
}

#[test]
fn cuda_scalar() {
    let dev = &*CUDA;
    common::numeric::test_add_scalar_f32(dev);
    common::numeric::test_sub_scalar_f32(dev);
    common::numeric::test_sub_scalar_lhs_f32(dev);
    common::numeric::test_mul_scalar_f32(dev);
    common::numeric::test_div_scalar_f32(dev);
    common::numeric::test_div_scalar_lhs_f32(dev);
}

#[test]
fn cuda_cmp() {
    let dev = &*CUDA;
    common::numeric::test_eq_f32(dev);
    common::numeric::test_lt_f32(dev);
    common::numeric::test_ne_f32(dev);
    common::numeric::test_ge_f32(dev);
    common::numeric::test_gt_f32(dev);
    common::numeric::test_le_f32(dev);
}

#[test]
fn cuda_reduce() {
    let dev = &*CUDA;
    common::reduce::test_sum_dim_f32(dev);
    common::reduce::test_sum_keepdim_f32(dev);
    common::reduce::test_sum_all_f32(dev);
    common::reduce::test_sum_dims_f32(dev);
    common::reduce::test_max_dim_f32(dev);
    common::reduce::test_max_keepdim_f32(dev);
    common::reduce::test_max_all_f32(dev);
    common::reduce::test_min_dim_f32(dev);
    common::reduce::test_min_all_f32(dev);
    common::reduce::test_mean_dim_f32(dev);
    common::reduce::test_mean_all_f32(dev);
    common::reduce::test_prod_dim_f32(dev);
    common::reduce::test_prod_all_f32(dev);
    common::reduce::test_argmax_f32(dev);
    common::reduce::test_argmin_f32(dev);
    common::reduce::test_argmax_keepdim(dev);
    common::reduce::test_var_f32(dev);
    common::reduce::test_var_unbiased_f32(dev);
    common::reduce::test_std_f32(dev);
    common::reduce::test_std_all_f32(dev);
    common::reduce::test_logsumexp_f32(dev);
    common::reduce::test_logsumexp_keepdim(dev);
    common::reduce::test_sum_i32(dev);
    common::reduce::test_sum_all_i32(dev);
    common::reduce::test_max_dim_i32(dev);
    common::reduce::test_min_dim_i32(dev);
    common::reduce::test_sum_u8(dev);
    common::reduce::test_max_u8(dev);
    common::reduce::test_sum_u32(dev);
    common::reduce::test_min_u32(dev);
}

#[test]
fn cuda_bool() {
    let dev = &*CUDA;
    common::boolean::test_bool_and(dev);
    common::boolean::test_bool_or(dev);
    common::boolean::test_bool_xor(dev);
    common::boolean::test_bool_not(dev);
    common::boolean::test_pick_f32(dev);
    common::boolean::test_pick_scalar_true(dev);
    common::boolean::test_pick_scalar_false(dev);
    common::boolean::test_pick_bool(dev);
    common::boolean::test_pick_int(dev);
    common::boolean::test_pick_int_scalar_true(dev);
    common::boolean::test_pick_int_scalar_false(dev);
    common::boolean::test_pick_bool_scalar_true(dev);
    common::boolean::test_pick_bool_scalar_false(dev);
    common::boolean::test_bool_all_all(dev);
    common::boolean::test_bool_any_all(dev);
    common::boolean::test_bool_true_count(dev);
    common::boolean::test_bool_false_count(dev);
    common::boolean::test_allclose_exact(dev);
    common::boolean::test_allclose_false(dev);
    common::boolean::test_allclose_int(dev);
    common::boolean::test_allclose_bool(dev);
}

#[test]
fn cuda_clamp() {
    let dev = &*CUDA;
    common::numeric::test_clamp_both(dev);
    common::numeric::test_clamp_min_only(dev);
    common::numeric::test_clamp_max_only(dev);
    common::numeric::test_clamp_none(dev);
    common::numeric::test_pow_exp_zero(dev);
    common::numeric::test_pow_exp_one(dev);
}

#[test]
fn cuda_broadcast() {
    let dev = &*CUDA;
    common::numeric::test_broadcast_add_f32(dev);
    common::numeric::test_broadcast_mul_f32(dev);
    common::numeric::test_broadcast_eq_f32(dev);
}

#[test]
fn cuda_cast() {
    let dev = &*CUDA;
    common::cast::test_cast_f32_to_f64(dev);
    common::cast::test_cast_f32_to_i32(dev);
    common::cast::test_cast_f32_to_bool(dev);
    common::cast::test_cast_i32_to_f32(dev);
    common::cast::test_cast_bool_to_f32(dev);
    common::cast::test_cast_bool_to_i32(dev);
    common::cast::test_cast_i32_to_bool(dev);
    common::cast::test_cast_f64_to_f32(dev);
    common::cast::test_cast_i32_to_u32(dev);
    common::cast::test_cast_bool_to_bool(dev);
}

#[test]
fn cuda_f64() {
    let dev = &*CUDA;
    common::cast::test_f64_zeros(dev);
    common::cast::test_f64_add(dev);
    common::f64::test_f64_neg(dev);
    common::f64::test_f64_abs(dev);
    common::f64::test_f64_relu(dev);
    common::f64::test_f64_exp(dev);
    common::f64::test_f64_ln(dev);
    common::f64::test_f64_sqrt(dev);
    common::f64::test_f64_sigmoid(dev);
    common::f64::test_f64_tanh(dev);
    common::f64::test_f64_sin(dev);
    common::f64::test_f64_cos(dev);
    common::f64::test_f64_sqr(dev);
    common::f64::test_f64_recip(dev);
    common::f64::test_f64_floor(dev);
    common::f64::test_f64_ceil(dev);
    common::f64::test_f64_sign(dev);
    common::f64::test_f64_pow(dev);
    common::f64::test_f64_affine(dev);
    common::f64::test_f64_eq(dev);
    common::f64::test_f64_lt(dev);
    common::f64::test_f64_gt(dev);
    common::f64::test_f64_le(dev);
    common::f64::test_f64_ge(dev);
    common::f64::test_f64_ne(dev);
    common::f64::test_f64_add_scalar(dev);
    common::f64::test_f64_sub_scalar(dev);
    common::f64::test_f64_sub_scalar_lhs(dev);
    common::f64::test_f64_mul_scalar(dev);
    common::f64::test_f64_div_scalar(dev);
    common::f64::test_f64_div_scalar_lhs(dev);
    common::f64::test_f64_sum_dim(dev);
    common::f64::test_f64_max_all(dev);
    common::f64::test_f64_grad_add(dev);
    common::f64::test_f64_grad_mul(dev);
}

#[test]
fn cuda_display() {
    let dev = &*CUDA;
    common::display::test_display_scalar(dev);
    common::display::test_display_1d(dev);
}

#[test]
fn cuda_shape() {
    let dev = &*CUDA;
    common::shape::test_cat_dim0_f32(dev);
    common::shape::test_contiguous_after_transpose(dev);
    common::shape::test_reshape_f32(dev);
    common::shape::test_transpose_f32(dev);
    common::shape::test_broadcast_as_f32(dev);
    common::shape::test_narrow_dim0(dev);
    common::shape::test_squeeze_dim1(dev);
    common::shape::test_unsqueeze(dev);
    common::shape::test_flatten_all(dev);
    common::shape::test_permute_f32(dev);
    common::shape::test_split_f32(dev);
    common::shape::test_repeat_dim_f32(dev);
    common::shape::test_transpose_last(dev);
    common::shape::test_already_contiguous_is_noop(dev);
    common::shape::test_flatten_range(dev);
    common::shape::test_stack_f32(dev);
    common::shape::test_chunk_f32(dev);
}

#[test]
fn cuda_matmul() {
    let dev = &*CUDA;
    common::matmul::test_matmul_2x2(dev);
    common::matmul::test_matmul_2x3_3x2(dev);
    common::matmul::test_matmul_f64(dev);
}

#[test]
fn cuda_int() {
    let dev = &*CUDA;
    common::numeric::test_add_i32(dev);
    common::numeric::test_neg_i32(dev);
    common::numeric::test_abs_i32(dev);
    common::numeric::test_sign_i32(dev);
    common::numeric::test_pow_i32(dev);
    common::numeric::test_affine_i32(dev);
    common::numeric::test_clamp_i32(dev);
    common::numeric::test_add_scalar_i32(dev);
    common::numeric::test_sub_scalar_i32(dev);
    common::numeric::test_sub_scalar_lhs_i32(dev);
    common::numeric::test_mul_scalar_i32(dev);
    common::numeric::test_div_scalar_i32(dev);
    common::numeric::test_div_scalar_lhs_i32(dev);
}

#[test]
fn cuda_construct() {
    let dev = &*CUDA;
    common::construct::test_zeros_like_f32(dev);
    common::construct::test_ones_like_f32(dev);
    common::construct::test_from_slice_f32(dev);
    common::construct::test_full_scalar(dev);
    common::construct::test_rand_like_shape(dev);
    common::construct::test_randn_like_shape(dev);
}

#[test]
fn cuda_grad() {
    let dev = &*CUDA;
    common::grad::test_grad_add(dev);
    common::grad::test_grad_sub(dev);
    common::grad::test_grad_mul(dev);
    common::grad::test_grad_div(dev);
    common::grad::test_grad_relu(dev);
    common::grad::test_grad_sum(dev);
    common::grad::test_grad_mean(dev);
    common::grad::test_grad_exp(dev);
    common::grad::test_grad_sigmoid(dev);
    common::grad::test_grad_clamp(dev);
    common::grad::test_grad_clamp_min(dev);
    common::grad::test_grad_prod(dev);
    common::grad::test_grad_reshape(dev);
    common::grad::test_grad_transpose(dev);
    common::grad::test_grad_matmul(dev);
    common::grad::test_grad_accumulate(dev);
    common::grad::test_no_grad_disabled(dev);
}

#[test]
fn cuda_indexing() {
    let dev = &*CUDA;
    common::indexing::test_index_select_dim0(dev);
    common::indexing::test_index_select_dim1(dev);
    common::indexing::test_gather_dim1(dev);
    common::indexing::test_index_add_f32(dev);
    common::indexing::test_scatter_add_f32(dev);
    common::indexing::test_index_add_2d(dev);
    common::indexing::test_i_select_row(dev);
    common::indexing::test_i_select_negative(dev);
    common::indexing::test_i_slice_range(dev);
    common::indexing::test_i_slice_full(dev);
    common::indexing::test_i_slice_with_step(dev);
    common::indexing::test_i_tuple_select_slice(dev);
    common::indexing::test_i_tuple_slice_slice(dev);
    common::indexing::test_i_boolean_mask(dev);
    common::indexing::test_i_boolean_mask_2d(dev);
    common::indexing::test_get_element(dev);
    common::indexing::test_get_row_2d(dev);
}

#[test]
fn cuda_nn() {
    let dev = &*CUDA;
    common::nn::test_softmax_dim0(dev);
    common::nn::test_softmax_dim1(dev);
    common::nn::test_softmax_numerical_stability(dev);
    common::nn::test_rms_norm_f32(dev);
    common::nn::test_rms_norm_weighted(dev);
    common::nn::test_cross_entropy_chain_f32(dev);
    common::nn::test_cross_entropy_basic_f32(dev);
    common::nn::test_cross_entropy_mnist_shape_f32(dev);
    common::nn::test_matmul_transposed_weight_small_f32(dev);
    common::nn::test_matmul_transposed_weight_f32(dev);
    common::nn::test_matmul_add_bias_f32(dev);
    common::nn::test_broadcast_add_precision(dev);
    common::nn::test_cross_entropy_precision(dev);
    common::nn::test_broadcast_add_grad_f32(dev);
    common::nn::test_broadcast_reduce_backward_f32(dev);
    common::nn::test_sum_keepdim_nonuniform_f32(dev);
    common::nn::test_argmax_eval_pipeline_f32(dev);
    common::nn::test_mini_training_step_f32(dev);
    common::nn::test_argmax_eval_large_f32(dev);
}

#[test]
fn cuda_edge() {
    let dev = &*CUDA;
    common::numeric::test_sqrt_negative(dev);
    common::numeric::test_ln_zero(dev);
    common::numeric::test_exp_large(dev);
    common::numeric::test_div_zero_f32(dev);
    common::numeric::test_add_nan_f32(dev);
}

#[test]
fn cuda_large() {
    let dev = &*CUDA;
    common::reduce::test_large_sum_f32(dev);
    common::matmul::test_large_matmul_f32(dev);
    common::nn::test_large_softmax(dev);
}

#[test]
fn cuda_cross() {
    let dev = &*CUDA;
    common::cross::test_transpose_add(dev);
    common::cross::test_transpose_sub(dev);
    common::cross::test_transpose_sum(dev);
    common::cross::test_transpose_max(dev);
    common::cross::test_permute_add(dev);
    common::cross::test_slice_sum(dev);
    common::cross::test_narrow_add(dev);
    common::cross::test_permute_contiguous_add(dev);
    common::cross::test_broadcast_sum(dev);
}

#[test]
fn cuda_error() {
    let dev = &*CUDA;
    common::error::test_binary_shape_mismatch(dev);
    common::error::test_matmul_shape_mismatch(dev);
    common::error::test_narrow_out_of_range(dev);
    common::error::test_dim_out_of_range(dev);
    common::error::test_allclose_shape_mismatch(dev);
    common::error::test_f64_to_f32_add(dev);
    common::error::test_reshape_wrong_elements(dev);
}

#[test]
fn cuda_to_device() {
    use common::{assert_close, tensor_bool_dev, tensor_f32, tensor_f32_dev, tensor_f64_dev, tensor_i32};
    use luma_tensor::Cpu;
    use luma_tensor::dtype::FloatDType;

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

    // .cuda() / .cpu() sugar.
    let sug = src.cuda(0).unwrap();
    let sugback = sug.cpu().unwrap();
    assert_close(&sugback.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-5, 1e-5);

    // Identity tests (common delegation) run on Cuda too.
    common::transfer::test_to_device_identity_f32(dev);
    common::transfer::test_to_device_identity_int(dev);
    common::transfer::test_to_device_identity_bool(dev);
    common::transfer::test_to_device_identity_requires_grad(dev);
}

#[test]
fn cuda_to_device_severs_graph_and_grad_flows() {
    use common::{assert_close, tensor_f32};
    use luma_tensor::Cpu;

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
    use common::{assert_close, tensor_f32};

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
