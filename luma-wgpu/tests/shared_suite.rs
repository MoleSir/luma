//! Broad coverage of WGPU using the device-generic suite in `luma_tensor::tests`.
//!
//! Tests that require F64/BF16/U8 are intentionally omitted. The former
//! F64-only NN tests are parameterized by dtype and run here with F32.

use luma_tensor::dtype::{BoolDType, FloatDType, IntDType};
use luma_tensor::tests::*;
use luma_tensor::{Bool, Int, Tensor};
use luma_wgpu::Wgpu;

use std::sync::LazyLock;

static DEVICE: LazyLock<Option<Wgpu>> = LazyLock::new(|| match Wgpu::new(0) {
    Ok(device) => Some(device),
    Err(e) => {
        eprintln!("skip wgpu test: {e}");
        None
    }
});

fn device() -> Option<Wgpu> {
    DEVICE.clone()
}

#[test]
fn shared_binary() {
    let Some(dev) = device() else { return };
    numeric::test_add_f32(&dev);
    numeric::test_sub_f32(&dev);
    numeric::test_mul_f32(&dev);
    numeric::test_div_f32(&dev);
    numeric::test_maximum_f32(&dev);
    numeric::test_minimum_f32(&dev);
}

#[test]
fn shared_unary() {
    let Some(dev) = device() else { return };
    numeric::test_neg_f32(&dev);
    numeric::test_abs_f32(&dev);
    numeric::test_relu_f32(&dev);
    numeric::test_exp_f32(&dev);
    numeric::test_sigmoid_f32(&dev);
    numeric::test_tanh_f32(&dev);
    numeric::test_ln_f32(&dev);
    numeric::test_sin_f32(&dev);
    numeric::test_cos_f32(&dev);
    numeric::test_sqr_f32(&dev);
    numeric::test_sqrt_f32(&dev);
    numeric::test_recip_f32(&dev);
    numeric::test_gelu_f32(&dev);
    numeric::test_silu_f32(&dev);
    numeric::test_floor_f32(&dev);
    numeric::test_ceil_f32(&dev);
    numeric::test_sign_f32(&dev);
    numeric::test_leaky_relu_f32(&dev);
    numeric::test_pow_f32(&dev);
    numeric::test_affine_f32(&dev);
    numeric::test_erf_f32(&dev);
    numeric::test_gelu_erf_f32(&dev);
    numeric::test_round_f32(&dev);
}

#[test]
fn shared_dtype_u32() {
    let Some(dev) = device() else { return };
    dtype::test_u32_construct(&dev);
    dtype::test_u32_add(&dev);
    dtype::test_u32_mul(&dev);
}

#[test]
fn shared_scalar() {
    let Some(dev) = device() else { return };
    numeric::test_add_scalar_f32(&dev);
    numeric::test_sub_scalar_f32(&dev);
    numeric::test_sub_scalar_lhs_f32(&dev);
    numeric::test_mul_scalar_f32(&dev);
    numeric::test_div_scalar_f32(&dev);
    numeric::test_div_scalar_lhs_f32(&dev);
}

#[test]
fn shared_cmp() {
    let Some(dev) = device() else { return };
    numeric::test_eq_f32(&dev);
    numeric::test_lt_f32(&dev);
    numeric::test_ne_f32(&dev);
    numeric::test_ge_f32(&dev);
    numeric::test_gt_f32(&dev);
    numeric::test_le_f32(&dev);
    numeric::test_cmp_scalar_f32(&dev);
}

#[test]
fn shared_clamp() {
    let Some(dev) = device() else { return };
    numeric::test_clamp_both(&dev);
    numeric::test_clamp_min_only(&dev);
    numeric::test_clamp_max_only(&dev);
    numeric::test_clamp_none(&dev);
    numeric::test_pow_exp_zero(&dev);
    numeric::test_pow_exp_one(&dev);
}

#[test]
fn shared_broadcast() {
    let Some(dev) = device() else { return };
    numeric::test_broadcast_add_f32(&dev);
    numeric::test_broadcast_mul_f32(&dev);
    numeric::test_broadcast_eq_f32(&dev);
}

#[test]
fn shared_reduce() {
    let Some(dev) = device() else { return };
    reduce::test_sum_dim_f32(&dev);
    reduce::test_sum_keepdim_f32(&dev);
    reduce::test_sum_all_f32(&dev);
    reduce::test_sum_dims_f32(&dev);
    reduce::test_max_dim_f32(&dev);
    reduce::test_max_keepdim_f32(&dev);
    reduce::test_max_all_f32(&dev);
    reduce::test_min_dim_f32(&dev);
    reduce::test_min_all_f32(&dev);
    reduce::test_mean_dim_f32(&dev);
    reduce::test_mean_all_f32(&dev);
    reduce::test_prod_dim_f32(&dev);
    reduce::test_prod_all_f32(&dev);
    reduce::test_argmax_f32(&dev);
    reduce::test_argmin_f32(&dev);
    reduce::test_argmax_keepdim(&dev);
    reduce::test_var_f32(&dev);
    reduce::test_var_unbiased_f32(&dev);
    reduce::test_std_f32(&dev);
    reduce::test_std_all_f32(&dev);
    reduce::test_logsumexp_f32(&dev);
    reduce::test_logsumexp_keepdim(&dev);
    reduce::test_sum_i32(&dev);
    reduce::test_sum_all_i32(&dev);
    reduce::test_max_dim_i32(&dev);
    reduce::test_min_dim_i32(&dev);
    reduce::test_sum_u32(&dev);
    reduce::test_min_u32(&dev);
}

#[test]
fn shared_bool() {
    let Some(dev) = device() else { return };
    boolean::test_bool_and(&dev);
    boolean::test_bool_or(&dev);
    boolean::test_bool_xor(&dev);
    boolean::test_bool_not(&dev);
    boolean::test_pick_f32(&dev);
    boolean::test_bool_all_all(&dev);
    boolean::test_bool_any_all(&dev);
    boolean::test_bool_true_count(&dev);
    boolean::test_bool_false_count(&dev);
    boolean::test_pick_scalar_true(&dev);
    boolean::test_pick_scalar_false(&dev);
    boolean::test_pick_bool(&dev);
    boolean::test_pick_int(&dev);
    boolean::test_pick_int_scalar_true(&dev);
    boolean::test_pick_int_scalar_false(&dev);
    boolean::test_pick_bool_scalar_true(&dev);
    boolean::test_pick_bool_scalar_false(&dev);
    boolean::test_allclose_exact(&dev);
    boolean::test_allclose_false(&dev);
    boolean::test_allclose_int(&dev);
    boolean::test_allclose_bool(&dev);
}

#[test]
fn shared_cast() {
    let Some(dev) = device() else { return };
    cast::test_cast_f32_to_i32(&dev);
    cast::test_cast_f32_to_bool(&dev);
    cast::test_cast_i32_to_f32(&dev);
    cast::test_cast_bool_to_f32(&dev);
    cast::test_cast_bool_to_i32(&dev);
    cast::test_cast_i32_to_bool(&dev);
    cast::test_cast_i32_to_u32(&dev);
    cast::test_cast_bool_to_bool(&dev);
}

#[test]
fn shared_f16() {
    let Some(dev) = device() else { return };
    if !dev.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }
    let dt = FloatDType::F16;
    let tol = 2e-2;
    half_precision::test_half_binary(&dev, dt, tol);
    half_precision::test_half_unary(&dev, dt, tol);
    half_precision::test_half_cmp(&dev, dt, tol);
    half_precision::test_half_scalar(&dev, dt, tol);
    half_precision::test_half_reduce(&dev, dt, tol);
    half_precision::test_half_softmax(&dev, dt, tol);
    half_precision::test_half_rms_norm(&dev, dt, tol);
    half_precision::test_half_matmul(&dev, dt, tol);
    half_precision::test_half_cast(&dev, dt, tol);
    half_precision::test_half_grad(&dev, dt, tol);
}

#[test]
fn shared_display() {
    let Some(dev) = device() else { return };
    display::test_display_scalar(&dev);
    display::test_display_1d(&dev);
}

#[test]
fn shared_shape() {
    let Some(dev) = device() else { return };
    shape::test_cat_dim0_f32(&dev);
    shape::test_contiguous_after_transpose(&dev);
    shape::test_reshape_f32(&dev);
    shape::test_transpose_f32(&dev);
    shape::test_broadcast_as_f32(&dev);
    shape::test_narrow_dim0(&dev);
    shape::test_squeeze_dim1(&dev);
    shape::test_unsqueeze(&dev);
    shape::test_flatten_all(&dev);
    shape::test_permute_f32(&dev);
    shape::test_split_f32(&dev);
    shape::test_repeat_dim_f32(&dev);
    shape::test_transpose_last(&dev);
    shape::test_already_contiguous_is_noop(&dev);
    shape::test_flatten_range(&dev);
    shape::test_stack_f32(&dev);
    shape::test_chunk_f32(&dev);
    shape::test_copy_float(&dev);
    shape::test_copy_shape_mismatch(&dev);
    shape::test_phantom(&dev);
    shape::test_phantom_then_copy(&dev);
    shape::test_copy_preserves_requires_grad(&dev);
}

#[test]
fn shared_matmul() {
    let Some(dev) = device() else { return };
    matmul::test_matmul_2x2(&dev);
    matmul::test_matmul_2x3_3x2(&dev);
    matmul::test_large_matmul_f32(&dev);
}

#[test]
fn shared_int() {
    let Some(dev) = device() else { return };
    numeric::test_add_i32(&dev);
    numeric::test_neg_i32(&dev);
    numeric::test_abs_i32(&dev);
    numeric::test_sign_i32(&dev);
    numeric::test_pow_i32(&dev);
    numeric::test_affine_i32(&dev);
    numeric::test_clamp_i32(&dev);
    numeric::test_add_scalar_i32(&dev);
    numeric::test_sub_scalar_i32(&dev);
    numeric::test_sub_scalar_lhs_i32(&dev);
    numeric::test_mul_scalar_i32(&dev);
    numeric::test_div_scalar_i32(&dev);
    numeric::test_div_scalar_lhs_i32(&dev);
}

#[test]
fn shared_construct() {
    let Some(dev) = device() else { return };
    construct::test_zeros_like_f32(&dev);
    construct::test_ones_like_f32(&dev);
    construct::test_from_slice_f32(&dev);
    construct::test_full_scalar(&dev);
    construct::test_rand_like_shape(&dev);
    construct::test_randn_like_shape(&dev);
    construct::test_bytes_roundtrip_f32(&dev);
    construct::test_bytes_roundtrip_i32(&dev);
    construct::test_bytes_roundtrip_bool(&dev);
    construct::test_bytes_non_contiguous(&dev);
    construct::test_dyn_tensor_roundtrip_float(&dev);
    construct::test_dyn_tensor_roundtrip_int(&dev);
    construct::test_dyn_tensor_roundtrip_bool(&dev);
    construct::test_dyn_tensor_accessors(&dev);
}

#[test]
fn shared_grad() {
    let Some(dev) = device() else { return };
    grad::test_grad_add(&dev);
    grad::test_grad_sub(&dev);
    grad::test_grad_mul(&dev);
    grad::test_grad_div(&dev);
    grad::test_grad_relu(&dev);
    grad::test_grad_sum(&dev);
    grad::test_grad_mean(&dev);
    grad::test_grad_exp(&dev);
    grad::test_grad_sigmoid(&dev);
    grad::test_grad_clamp(&dev);
    grad::test_grad_clamp_min(&dev);
    grad::test_grad_prod(&dev);
    grad::test_grad_matmul(&dev);
    grad::test_grad_reshape(&dev);
    grad::test_grad_transpose(&dev);
    grad::test_grad_accumulate(&dev);
    grad::test_no_grad_disabled(&dev);
    grad::test_custom_op3_forward_backward(&dev);
}

#[test]
fn shared_indexing() {
    let Some(dev) = device() else { return };
    indexing::test_index_select_dim0(&dev);
    indexing::test_index_select_dim1(&dev);
    indexing::test_gather_dim1(&dev);
    indexing::test_index_add_f32(&dev);
    indexing::test_scatter_add_f32(&dev);
    indexing::test_index_add_2d(&dev);
    indexing::test_i_select_row(&dev);
    indexing::test_i_select_negative(&dev);
    indexing::test_i_slice_range(&dev);
    indexing::test_i_slice_full(&dev);
    indexing::test_i_slice_with_step(&dev);
    indexing::test_i_tuple_select_slice(&dev);
    indexing::test_i_tuple_slice_slice(&dev);
    indexing::test_i_boolean_mask(&dev);
    indexing::test_i_boolean_mask_2d(&dev);
    indexing::test_get_element(&dev);
    indexing::test_get_row_2d(&dev);
}

#[test]
fn shared_nn_f32() {
    let Some(dev) = device() else { return };
    nn::test_softmax_dim0(&dev);
    nn::test_softmax_dim1(&dev);
    nn::test_softmax_numerical_stability(&dev);
    nn::test_softmax_grad(&dev);
    nn::test_rms_norm_f32(&dev);
    nn::test_rms_norm_weighted(&dev);
    nn::test_dropout(&dev);
    nn::test_cross_entropy_chain_f32(&dev);
    nn::test_cross_entropy_basic_f32(&dev);
    nn::test_cross_entropy_mnist_shape_f32(&dev);
    nn::test_matmul_transposed_weight_small_f32(&dev);
    nn::test_matmul_transposed_weight_f32(&dev);
    nn::test_matmul_add_bias_f32(&dev);
    nn::test_broadcast_add_precision(&dev);
    nn::test_cross_entropy_precision(&dev);
    nn::test_broadcast_add_grad_f32(&dev);
    nn::test_broadcast_reduce_backward_f32(&dev);
    nn::test_sum_keepdim_nonuniform_f32(&dev);
    nn::test_argmax_eval_pipeline_f32(&dev);
    nn::test_mini_training_step_f32(&dev);
    nn::test_argmax_eval_large_f32(&dev);
}

#[test]
fn shared_nn_dtype_f32() {
    let Some(dev) = device() else { return };
    let dt = FloatDType::F32;
    let tol = 1e-4;
    nn::test_rms_norm_bias(&dev, dt, tol);
    nn::test_rms_norm_grad(&dev, dt, tol);
    nn::test_default_softmax(&dev, dt, tol);
    nn::test_log_softmax(&dev, dt, tol);
    nn::test_default_rms_norm(&dev, dt, tol);
    nn::test_layer_norm(&dev, dt, tol);
    nn::test_group_norm(&dev, dt, tol);
    nn::test_conv2d(&dev, dt, tol);
    nn::test_attention(&dev, dt, tol);
    nn::test_max_pool2d(&dev, dt, tol);
    nn::test_avg_pool2d(&dev, dt, tol);
    nn::test_adaptive_avg_pool2d(&dev, dt, tol);
    nn::test_max_pool2d_grad(&dev, dt, tol);
    nn::test_avg_pool2d_grad(&dev, dt, tol);
    nn::test_adaptive_avg_pool2d_grad(&dev, dt, tol);
    nn::test_linear(&dev, dt, tol);
    nn::test_embedding(&dev, dt, tol);
    nn::test_losses(&dev, dt, tol);
    nn::test_loss_grads(&dev, dt, tol);
}

#[test]
fn shared_edge() {
    let Some(dev) = device() else { return };
    numeric::test_sqrt_negative(&dev);
    numeric::test_ln_zero(&dev);
    numeric::test_exp_large(&dev);
    numeric::test_div_zero_f32(&dev);
    numeric::test_add_nan_f32(&dev);
    numeric::test_empty_zeros(&dev);
    numeric::test_empty_add(&dev);
}

#[test]
fn shared_large() {
    let Some(dev) = device() else { return };
    reduce::test_large_sum_f32(&dev);
    matmul::test_large_matmul_f32(&dev);
    nn::test_large_softmax(&dev);
}

#[test]
fn shared_cross() {
    let Some(dev) = device() else { return };
    cross::test_transpose_add(&dev);
    cross::test_transpose_sub(&dev);
    cross::test_transpose_sum(&dev);
    cross::test_transpose_max(&dev);
    cross::test_permute_add(&dev);
    cross::test_slice_sum(&dev);
    cross::test_narrow_add(&dev);
    cross::test_permute_contiguous_add(&dev);
    cross::test_broadcast_sum(&dev);
}

#[test]
fn shared_error() {
    let Some(dev) = device() else { return };
    error::test_binary_shape_mismatch(&dev);
    error::test_matmul_shape_mismatch(&dev);
    error::test_narrow_out_of_range(&dev);
    error::test_dim_out_of_range(&dev);
    error::test_allclose_shape_mismatch(&dev);
    error::test_reshape_wrong_elements(&dev);
}

#[test]
fn shared_strided_readback() {
    let Some(dev) = device() else { return };

    let x = Tensor::<Wgpu>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), (&dev, FloatDType::F32))
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    assert_eq!(x.to_vec().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    let xi = Tensor::<Wgpu, Int>::from_slice(&[1i64, 2, 3, 4, 5, 6], (2, 3), (&dev, IntDType::I32)).unwrap().transpose(0, 1).unwrap();
    assert_eq!(xi.to_vec().unwrap(), vec![1, 4, 2, 5, 3, 6]);
    let ibytes = xi.to_bytes().unwrap();
    let ivals: Vec<i32> = ibytes.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    assert_eq!(ivals, vec![1, 4, 2, 5, 3, 6]);

    let xb = Tensor::<Wgpu, Bool>::from_slice(&[true, false, false, true, true, false], (2, 3), (&dev, BoolDType::Bool))
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    assert_eq!(xb.to_vec().unwrap(), vec![true, true, false, true, false, false]);
    assert_eq!(xb.to_bytes().unwrap().to_vec(), vec![1, 1, 0, 1, 0, 0]);
}

#[test]
fn shared_transfer() {
    let Some(dev) = device() else { return };
    transfer::test_to_device_identity_f32(&dev);
    transfer::test_to_device_identity_int(&dev);
    transfer::test_to_device_identity_bool(&dev);
    transfer::test_to_device_identity_requires_grad(&dev);
}
