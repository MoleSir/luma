#![cfg(feature = "cuda")]
//! CUDA tests — trace → compile → run the same closed loop on the GPU.
//! Run with: cargo test -p luma-jit --features cuda --test cuda

mod common;

use luma_tensor::Cuda;
use std::sync::LazyLock;

static CUDA: LazyLock<Cuda> = LazyLock::new(|| Cuda::new(0).expect("cuda device 0"));

// ---- module-level (trace a real module) --------------------------------------

#[test]
fn cuda_traced_linear_matches_forward() {
    common::module::test_traced_linear_matches_forward(&*CUDA);
}

#[test]
fn cuda_repeated_runs_different_inputs() {
    common::module::test_repeated_runs_different_inputs(&*CUDA);
}

#[test]
fn cuda_bool_ops() {
    common::module::test_bool_ops(&*CUDA);
}

#[test]
fn cuda_traced_cross_entropy_matches_forward() {
    common::module::test_traced_cross_entropy_matches_forward(&*CUDA);
}

// ---- op-family coverage ------------------------------------------------------

#[test]
fn cuda_binary() {
    common::numeric::test_add(&*CUDA);
    common::numeric::test_sub(&*CUDA);
    common::numeric::test_mul(&*CUDA);
    common::numeric::test_div(&*CUDA);
    common::numeric::test_maximum(&*CUDA);
    common::numeric::test_minimum(&*CUDA);
}

#[test]
fn cuda_scalar() {
    common::numeric::test_add_scalar(&*CUDA);
    common::numeric::test_sub_scalar(&*CUDA);
    common::numeric::test_mul_scalar(&*CUDA);
    common::numeric::test_div_scalar(&*CUDA);
    common::numeric::test_maximum_scalar(&*CUDA);
    common::numeric::test_minimum_scalar(&*CUDA);
    common::numeric::test_sub_scalar_lhs(&*CUDA);
    common::numeric::test_div_scalar_lhs(&*CUDA);
}

#[test]
fn cuda_unary() {
    common::numeric::test_neg(&*CUDA);
    common::numeric::test_abs(&*CUDA);
    common::numeric::test_sign(&*CUDA);
    common::numeric::test_exp(&*CUDA);
    common::numeric::test_ln(&*CUDA);
    common::numeric::test_sin(&*CUDA);
    common::numeric::test_cos(&*CUDA);
    common::numeric::test_tanh(&*CUDA);
    common::numeric::test_sqr(&*CUDA);
    common::numeric::test_sqrt(&*CUDA);
    common::numeric::test_recip(&*CUDA);
    common::numeric::test_relu(&*CUDA);
    common::numeric::test_sigmoid(&*CUDA);
    common::numeric::test_silu(&*CUDA);
    common::numeric::test_gelu(&*CUDA);
    common::numeric::test_gelu_erf(&*CUDA);
    common::numeric::test_erf(&*CUDA);
    common::numeric::test_floor(&*CUDA);
    common::numeric::test_ceil(&*CUDA);
    common::numeric::test_round(&*CUDA);
    common::numeric::test_affine(&*CUDA);
    common::numeric::test_pow(&*CUDA);
    common::numeric::test_clamp(&*CUDA);
    common::numeric::test_leaky_relu(&*CUDA);
}

#[test]
fn cuda_int_unary() {
    common::numeric::test_neg_i32(&*CUDA);
    common::numeric::test_abs_i32(&*CUDA);
    common::numeric::test_sign_i32(&*CUDA);
}

#[test]
fn cuda_cmp() {
    common::numeric::test_eq(&*CUDA);
    common::numeric::test_ne(&*CUDA);
    common::numeric::test_lt(&*CUDA);
    common::numeric::test_gt(&*CUDA);
    common::numeric::test_le(&*CUDA);
    common::numeric::test_ge(&*CUDA);
    common::numeric::test_gt_scalar(&*CUDA);
}

#[test]
fn cuda_reduce() {
    common::reduce::test_sum_dim(&*CUDA);
    common::reduce::test_sum_keepdim(&*CUDA);
    common::reduce::test_sum_all(&*CUDA);
    common::reduce::test_max_dim(&*CUDA);
    common::reduce::test_max_keepdim(&*CUDA);
    common::reduce::test_max_all(&*CUDA);
    common::reduce::test_min_dim(&*CUDA);
    common::reduce::test_min_keepdim(&*CUDA);
    common::reduce::test_min_all(&*CUDA);
    common::reduce::test_prod_dim(&*CUDA);
    common::reduce::test_prod_keepdim(&*CUDA);
    common::reduce::test_prod_all(&*CUDA);
    common::reduce::test_mean_dim(&*CUDA);
    common::reduce::test_mean_keepdim(&*CUDA);
    common::reduce::test_mean_all(&*CUDA);
}

#[test]
fn cuda_argreduce() {
    common::reduce::test_argmax(&*CUDA);
    common::reduce::test_argmin(&*CUDA);
    common::reduce::test_argmax_keepdim(&*CUDA);
    common::reduce::test_argmin_keepdim(&*CUDA);
}

#[test]
fn cuda_shape() {
    common::shape::test_reshape(&*CUDA);
    common::shape::test_transpose(&*CUDA);
    common::shape::test_permute(&*CUDA);
    common::shape::test_narrow(&*CUDA);
    common::shape::test_slice(&*CUDA);
    common::shape::test_squeeze(&*CUDA);
    common::shape::test_unsqueeze(&*CUDA);
    common::shape::test_broadcast_as(&*CUDA);
}

#[test]
fn cuda_indexing() {
    common::indexing::test_index_select(&*CUDA);
    common::indexing::test_gather(&*CUDA);
    common::indexing::test_index_add(&*CUDA);
    common::indexing::test_scatter_add(&*CUDA);
    common::indexing::test_cat(&*CUDA);
}

#[test]
fn cuda_nn() {
    common::nn::test_softmax(&*CUDA);
    common::nn::test_rms_norm(&*CUDA);
    common::nn::test_arange(&*CUDA);
}

#[test]
fn cuda_cast() {
    common::cast::test_cast_f32_f64(&*CUDA);
    common::cast::test_cast_f32_i32(&*CUDA);
    common::cast::test_cast_f32_bool(&*CUDA);
    common::cast::test_cast_i32_f32(&*CUDA);
    common::cast::test_cast_i32_bool(&*CUDA);
    common::cast::test_cast_bool_f32(&*CUDA);
    common::cast::test_cast_bool_i32(&*CUDA);
}

#[test]
fn cuda_boolean() {
    common::boolean::test_and(&*CUDA);
    common::boolean::test_or(&*CUDA);
    common::boolean::test_xor(&*CUDA);
    common::boolean::test_not(&*CUDA);
    common::boolean::test_pick_f32(&*CUDA);
    common::boolean::test_pick_i32(&*CUDA);
    common::boolean::test_pick_bool(&*CUDA);
    common::boolean::test_pick_true_f32(&*CUDA);
    common::boolean::test_pick_true_i32(&*CUDA);
    common::boolean::test_pick_true_bool(&*CUDA);
    common::boolean::test_pick_false_f32(&*CUDA);
    common::boolean::test_pick_false_i32(&*CUDA);
    common::boolean::test_pick_false_bool(&*CUDA);
}
