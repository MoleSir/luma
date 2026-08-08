//! CPU tests — delegates ALL tests to common/, plus truly CPU-only tests.
//! Run with: cargo test --test cpu

mod common;

use common::*;
use luma_tensor::{Bool, Cpu, Int, Tensor};
use luma_tensor::dtype::{FloatDType, IntDType};

// ---- common delegation groups (all device-generic tests) ----

#[test]
fn test_binary() {
    common::numeric::test_add_f32(&Cpu); common::numeric::test_sub_f32(&Cpu);
    common::numeric::test_mul_f32(&Cpu); common::numeric::test_div_f32(&Cpu);
    common::numeric::test_maximum_f32(&Cpu); common::numeric::test_minimum_f32(&Cpu);
}

#[test]
fn test_unary() {
    common::numeric::test_neg_f32(&Cpu); common::numeric::test_abs_f32(&Cpu);
    common::numeric::test_relu_f32(&Cpu); common::numeric::test_exp_f32(&Cpu);
    common::numeric::test_sigmoid_f32(&Cpu); common::numeric::test_tanh_f32(&Cpu);
    common::numeric::test_ln_f32(&Cpu); common::numeric::test_sin_f32(&Cpu);
    common::numeric::test_cos_f32(&Cpu); common::numeric::test_sqr_f32(&Cpu);
    common::numeric::test_sqrt_f32(&Cpu); common::numeric::test_recip_f32(&Cpu);
    common::numeric::test_gelu_f32(&Cpu); common::numeric::test_silu_f32(&Cpu);
    common::numeric::test_floor_f32(&Cpu); common::numeric::test_ceil_f32(&Cpu);
    common::numeric::test_sign_f32(&Cpu); common::numeric::test_leaky_relu_f32(&Cpu);
    common::numeric::test_pow_f32(&Cpu); common::numeric::test_affine_f32(&Cpu);
    common::numeric::test_erf_f32(&Cpu); common::numeric::test_gelu_erf_f32(&Cpu);
    common::numeric::test_round_f32(&Cpu);
}

#[test]
fn test_dtype() {
    common::dtype::test_u8_construct(&Cpu); common::dtype::test_u8_add(&Cpu);
    common::dtype::test_u8_sub(&Cpu); common::dtype::test_u8_clamp(&Cpu);
    common::dtype::test_u8_cast_to_i32(&Cpu); common::dtype::test_u8_cast_to_f32(&Cpu);
    common::dtype::test_u32_construct(&Cpu); common::dtype::test_u32_add(&Cpu);
    common::dtype::test_u32_mul(&Cpu);
}

#[test]
fn test_scalar() {
    common::numeric::test_add_scalar_f32(&Cpu); common::numeric::test_mul_scalar_f32(&Cpu);
    common::numeric::test_add_scalar_lhs_f32(&Cpu); common::numeric::test_div_scalar_f32(&Cpu);
}

#[test]
fn test_cmp() {
    common::numeric::test_eq_f32(&Cpu); common::numeric::test_lt_f32(&Cpu);
    common::numeric::test_ne_f32(&Cpu); common::numeric::test_ge_f32(&Cpu);
    common::numeric::test_gt_f32(&Cpu); common::numeric::test_le_f32(&Cpu);
}

#[test]
fn test_clamp() {
    common::numeric::test_clamp_both(&Cpu); common::numeric::test_clamp_min_only(&Cpu);
    common::numeric::test_clamp_max_only(&Cpu);
    common::numeric::test_clamp_none(&Cpu);
    common::numeric::test_pow_exp_zero(&Cpu);
    common::numeric::test_pow_exp_one(&Cpu);
}

#[test]
fn test_broadcast() {
    common::numeric::test_broadcast_add_f32(&Cpu);
    common::numeric::test_broadcast_mul_f32(&Cpu);
    common::numeric::test_broadcast_eq_f32(&Cpu);
}

#[test]
fn test_reduce() {
    common::reduce::test_sum_dim_f32(&Cpu); common::reduce::test_sum_keepdim_f32(&Cpu);
    common::reduce::test_sum_all_f32(&Cpu); common::reduce::test_sum_dims_f32(&Cpu);
    common::reduce::test_max_dim_f32(&Cpu); common::reduce::test_max_keepdim_f32(&Cpu);
    common::reduce::test_max_all_f32(&Cpu); common::reduce::test_min_dim_f32(&Cpu);
    common::reduce::test_min_all_f32(&Cpu); common::reduce::test_mean_dim_f32(&Cpu);
    common::reduce::test_mean_all_f32(&Cpu); common::reduce::test_prod_dim_f32(&Cpu);
    common::reduce::test_prod_all_f32(&Cpu); common::reduce::test_argmax_f32(&Cpu);
    common::reduce::test_argmin_f32(&Cpu); common::reduce::test_argmax_keepdim(&Cpu);
    common::reduce::test_var_f32(&Cpu); common::reduce::test_var_unbiased_f32(&Cpu);
    common::reduce::test_std_f32(&Cpu); common::reduce::test_std_all_f32(&Cpu);
    common::reduce::test_logsumexp_f32(&Cpu); common::reduce::test_logsumexp_keepdim(&Cpu);
    common::reduce::test_sum_i32(&Cpu); common::reduce::test_sum_all_i32(&Cpu);
    common::reduce::test_max_dim_i32(&Cpu); common::reduce::test_min_dim_i32(&Cpu);
    common::reduce::test_sum_f64(&Cpu); common::reduce::test_mean_f64(&Cpu);
    common::reduce::test_sum_u8(&Cpu); common::reduce::test_max_u8(&Cpu);
    common::reduce::test_sum_u32(&Cpu); common::reduce::test_min_u32(&Cpu);
}

#[test]
fn test_bool() {
    common::boolean::test_bool_and(&Cpu); common::boolean::test_bool_or(&Cpu);
    common::boolean::test_bool_xor(&Cpu); common::boolean::test_bool_not(&Cpu);
    common::boolean::test_pick_f32(&Cpu); common::boolean::test_bool_all_all(&Cpu);
    common::boolean::test_bool_any_all(&Cpu); common::boolean::test_bool_true_count(&Cpu);
    common::boolean::test_bool_false_count(&Cpu);
    common::boolean::test_pick_scalar_true(&Cpu);
    common::boolean::test_pick_scalar_false(&Cpu);
    common::boolean::test_pick_bool(&Cpu);
    common::boolean::test_pick_int(&Cpu);
    common::boolean::test_pick_int_scalar_true(&Cpu);
    common::boolean::test_pick_int_scalar_false(&Cpu);
    common::boolean::test_pick_bool_scalar_true(&Cpu);
    common::boolean::test_pick_bool_scalar_false(&Cpu);
    common::boolean::test_allclose_exact(&Cpu); common::boolean::test_allclose_false(&Cpu);
    common::boolean::test_allclose_int(&Cpu); common::boolean::test_allclose_bool(&Cpu);
}

#[test]
fn test_cast() {
    common::cast::test_cast_f32_to_f64(&Cpu); common::cast::test_cast_f32_to_i32(&Cpu);
    common::cast::test_cast_f32_to_bool(&Cpu); common::cast::test_cast_i32_to_f32(&Cpu);
    common::cast::test_cast_bool_to_f32(&Cpu); common::cast::test_cast_bool_to_i32(&Cpu);
    common::cast::test_cast_i32_to_bool(&Cpu);
    common::cast::test_cast_f64_to_f32(&Cpu);
    common::cast::test_cast_i32_to_u32(&Cpu); common::cast::test_cast_bool_to_bool(&Cpu);
}

#[test]
fn test_shape() {
    common::shape::test_cat_dim0_f32(&Cpu);
    common::shape::test_contiguous_after_transpose(&Cpu);
    common::shape::test_reshape_f32(&Cpu); common::shape::test_transpose_f32(&Cpu);
    common::shape::test_broadcast_as_f32(&Cpu);
    common::shape::test_narrow_dim0(&Cpu); common::shape::test_squeeze_dim1(&Cpu);
    common::shape::test_unsqueeze(&Cpu); common::shape::test_flatten_all(&Cpu);
    common::shape::test_permute_f32(&Cpu); common::shape::test_split_f32(&Cpu);
    common::shape::test_repeat_dim_f32(&Cpu);
    common::shape::test_transpose_last(&Cpu);
    common::shape::test_already_contiguous_is_noop(&Cpu);
    common::shape::test_flatten_range(&Cpu);
    common::shape::test_stack_f32(&Cpu); common::shape::test_chunk_f32(&Cpu);
}

#[test]
fn test_matmul() {
    common::matmul::test_matmul_2x2(&Cpu); common::matmul::test_matmul_2x3_3x2(&Cpu);
    common::matmul::test_matmul_f64(&Cpu);
}

#[test]
fn test_indexing() {
    common::indexing::test_index_select_dim0(&Cpu);
    common::indexing::test_index_select_dim1(&Cpu);
    common::indexing::test_gather_dim1(&Cpu);
    common::indexing::test_index_add_f32(&Cpu);
    common::indexing::test_scatter_add_f32(&Cpu);
    common::indexing::test_index_add_2d(&Cpu);
    common::indexing::test_i_select_row(&Cpu); common::indexing::test_i_select_negative(&Cpu);
    common::indexing::test_i_slice_range(&Cpu); common::indexing::test_i_slice_full(&Cpu);
    common::indexing::test_i_slice_with_step(&Cpu);
    common::indexing::test_i_tuple_select_slice(&Cpu);
    common::indexing::test_i_tuple_slice_slice(&Cpu);
    common::indexing::test_i_boolean_mask(&Cpu); common::indexing::test_i_boolean_mask_2d(&Cpu);
    common::indexing::test_get_element(&Cpu); common::indexing::test_get_row_2d(&Cpu);
}

#[test]
fn test_int() {
    common::numeric::test_add_i32(&Cpu); common::numeric::test_neg_i32(&Cpu);
    common::numeric::test_abs_i32(&Cpu);
    common::numeric::test_sign_i32(&Cpu);
    common::numeric::test_pow_i32(&Cpu);
    common::numeric::test_affine_i32(&Cpu);
    common::numeric::test_clamp_i32(&Cpu);
}

#[test]
fn test_construct() {
    common::construct::test_zeros_like_f32(&Cpu);
    common::construct::test_ones_like_f32(&Cpu);
    common::construct::test_from_slice_f32(&Cpu);
    common::construct::test_full_scalar(&Cpu);
    common::construct::test_rand_like_shape(&Cpu); common::construct::test_randn_like_shape(&Cpu);
}

#[test]
fn test_grad() {
    common::grad::test_grad_add(&Cpu); common::grad::test_grad_sub(&Cpu);
    common::grad::test_grad_mul(&Cpu); common::grad::test_grad_div(&Cpu);
    common::grad::test_grad_relu(&Cpu); common::grad::test_grad_sum(&Cpu);
    common::grad::test_grad_mean(&Cpu); common::grad::test_grad_exp(&Cpu);
    common::grad::test_grad_sigmoid(&Cpu); common::grad::test_grad_clamp(&Cpu);
    common::grad::test_grad_clamp_min(&Cpu); common::grad::test_grad_prod(&Cpu);
    common::grad::test_grad_matmul(&Cpu); common::grad::test_grad_reshape(&Cpu);
    common::grad::test_grad_transpose(&Cpu);
    common::grad::test_no_grad_disabled(&Cpu);
}

#[test]
fn test_display() {
    common::display::test_display_scalar(&Cpu);
    common::display::test_display_1d(&Cpu);
}

#[test]
fn test_f64() {
    common::cast::test_f64_zeros(&Cpu);
    common::cast::test_f64_add(&Cpu);
    common::reduce::test_sum_f64(&Cpu); common::reduce::test_mean_f64(&Cpu);
    common::f64::test_f64_neg(&Cpu); common::f64::test_f64_abs(&Cpu);
    common::f64::test_f64_relu(&Cpu); common::f64::test_f64_exp(&Cpu);
    common::f64::test_f64_ln(&Cpu); common::f64::test_f64_sqrt(&Cpu);
    common::f64::test_f64_sigmoid(&Cpu); common::f64::test_f64_tanh(&Cpu);
    common::f64::test_f64_sin(&Cpu); common::f64::test_f64_cos(&Cpu);
    common::f64::test_f64_sqr(&Cpu); common::f64::test_f64_recip(&Cpu);
    common::f64::test_f64_floor(&Cpu); common::f64::test_f64_ceil(&Cpu);
    common::f64::test_f64_sign(&Cpu); common::f64::test_f64_pow(&Cpu);
    common::f64::test_f64_affine(&Cpu);
    common::f64::test_f64_eq(&Cpu); common::f64::test_f64_lt(&Cpu);
    common::f64::test_f64_gt(&Cpu); common::f64::test_f64_le(&Cpu);
    common::f64::test_f64_ge(&Cpu); common::f64::test_f64_ne(&Cpu);
    common::f64::test_f64_add_scalar(&Cpu); common::f64::test_f64_mul_scalar(&Cpu);
    common::f64::test_f64_sum_dim(&Cpu); common::f64::test_f64_max_all(&Cpu);
    common::f64::test_f64_grad_add(&Cpu); common::f64::test_f64_grad_mul(&Cpu);
}

// ---- truly CPU-only (no device-generic API available) ----

#[test]
fn test_cpu_only() {
    // Error cases
    assert!(Tensor::<Cpu>::from_slice(&[1.0, 2.0], (2, 2), FloatDType::F32).is_err());
    assert!(tensor_f32(&[1.0, 2.0, 3.0, 4.0], (2, 2)).reshape((3, 2)).is_err());
    assert!(tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3)).broadcast_as((3, 2)).is_err());

    // Cat empty (needs concrete Cpu type)
    let arrs: &[&Tensor<Cpu>] = &[];
    assert!(Tensor::<Cpu>::cat(arrs, 0usize).is_err());

    // Eye (CPU-specific API)
    assert_close(&Tensor::<Cpu>::eye(3).unwrap().to_vec().unwrap(),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 1e-7, 1e-7);
    assert_eq!(Tensor::<Cpu, Int>::eye(2).unwrap().to_vec().unwrap(), vec![1i64, 0, 0, 1]);

    // Tril/Triu (CPU-specific API)
    assert_close(&Tensor::<Cpu>::tril(3, false).unwrap().to_vec().unwrap(),
        &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], 1e-7, 1e-7);
    assert_close(&Tensor::<Cpu>::tril(3, true).unwrap().to_vec().unwrap(),
        &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0], 1e-7, 1e-7);
    assert_close(&Tensor::<Cpu>::triu(3, false).unwrap().to_vec().unwrap(),
        &[0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 1e-7, 1e-7);
    assert_close(&Tensor::<Cpu>::triu(3, true).unwrap().to_vec().unwrap(),
        &[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], 1e-7, 1e-7);
    assert_eq!(Tensor::<Cpu, Int>::tril(3, true).unwrap().to_vec().unwrap(),
        vec![1i64, 0, 0, 1, 1, 0, 1, 1, 1]);

    // Linspace (CPU-specific API)
    let t = Tensor::<Cpu>::linspace(0.0, 1.0, 5).unwrap();
    assert_eq!(t.dims(), &[5]);
    let v = t.to_vec().unwrap();
    assert!((v[0] - 0.0).abs() < 1e-5);
    assert!((v[4] - 1.0).abs() < 1e-5);
    assert!((v[0] - 0.0).abs() < 1e-5);
    assert!((v[4] - 1.0).abs() < 1e-5);
    let t = Tensor::<Cpu>::linspace(3.0, 3.0, 1).unwrap();
    assert!((t.to_scalar().unwrap() - 3.0).abs() < 1e-5);

    // Arange
    assert_eq!(Tensor::<Cpu, Int>::arange(0, 5, 1, IntDType::I32).unwrap().to_vec().unwrap(),
        vec![0i64, 1, 2, 3, 4]);
    assert_eq!(Tensor::<Cpu, Int>::arange(0, 10, 2, IntDType::I32).unwrap().to_vec().unwrap(),
        vec![0i64, 2, 4, 6, 8]);
    assert_eq!(Tensor::<Cpu, Int>::arange(5, 0, -1, IntDType::I32).unwrap().to_vec().unwrap(),
        vec![5i64, 4, 3, 2, 1]);

    // Int slice construct
    assert_eq!(Tensor::<Cpu, Int>::from_slice(&[10i64, 20, 30, 40], (2, 2), IntDType::I32).unwrap().to_vec().unwrap(),
        vec![10i64, 20, 30, 40]);
    assert_eq!(tensor_i32(&[1, 2, 3, 4], (4,)).sum_all().unwrap().to_vec().unwrap(), vec![10]);
    assert_eq!(tensor_i32(&[1, 2, 3, 4], (4,)).prod_all().unwrap().to_vec().unwrap(), vec![24]);

    // Bool construct
    assert!(!Tensor::<Cpu, Bool>::falses((), ()).unwrap().to_vec().unwrap()[0]);
    assert!(Tensor::<Cpu, Bool>::trues((), ()).unwrap().to_vec().unwrap()[0]);
    assert_eq!(Tensor::<Cpu, Bool>::from_slice(&[true, false, true], (3,), ()).unwrap().to_vec().unwrap(),
        vec![true, false, true]);

    // Diag
    assert_close(&Tensor::<Cpu>::diag(&[1.0, 2.0, 3.0]).unwrap().to_vec().unwrap(),
        &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], 1e-7, 1e-7);
}

#[test]
fn test_new_construct() {
    // IntoTensor constructors (CPU-specific trait)
    assert!((3.14 - Tensor::<Cpu>::new(3.14).unwrap().to_scalar().unwrap()).abs() < 1e-5);
    assert_close(&Tensor::<Cpu>::new(&[1.0, 2.0, 3.0][..]).unwrap().to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-7, 1e-7);
    assert_close(&Tensor::<Cpu>::new([1.0, 2.0, 3.0]).unwrap().to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-7, 1e-7);
    assert_close(&Tensor::<Cpu>::new(&[1.0, 2.0, 3.0]).unwrap().to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-7, 1e-7);
    assert_close(&Tensor::<Cpu>::new(vec![1.0, 2.0, 3.0]).unwrap().to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-7, 1e-7);
    assert_close(&Tensor::<Cpu>::new(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap().to_vec().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-7, 1e-7);
    assert_close(&Tensor::<Cpu>::new(&[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).unwrap().to_vec().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 1e-7, 1e-7);
    assert_eq!(Tensor::<Cpu, Int>::new(42i64).unwrap().to_vec().unwrap(), vec![42i64]);
    assert_eq!(Tensor::<Cpu, Int>::new(&[1i64, 2, 3][..]).unwrap().to_vec().unwrap(), vec![1i64, 2, 3]);
    assert_eq!(Tensor::<Cpu, Int>::new(&[[1i64, 2], [3, 4]]).unwrap().to_vec().unwrap(), vec![1i64, 2, 3, 4]);
    assert_eq!(Tensor::<Cpu, Bool>::new(true).unwrap().to_vec().unwrap(), vec![true]);
    assert_eq!(Tensor::<Cpu, Bool>::new(&[true, false, true]).unwrap().to_vec().unwrap(), vec![true, false, true]);
    assert_eq!(Tensor::<Cpu, Bool>::new(&[[true, false], [false, true]]).unwrap().to_vec().unwrap(),
        vec![true, false, false, true]);
}

#[test]
fn test_cpu_slice() {
    use luma_tensor::Slice;
    let s = Slice::new(0, None, 1);
    assert_eq!(s.resolve(10), (0, 10, 1));
    assert_eq!(format!("{}", Slice::new(1, Some(5), 1)), "1:5");
    assert_eq!(format!("{}", Slice::new(0, None, 1)), "0:");
}

#[test]
fn test_nn() {
    common::nn::test_softmax_dim0(&Cpu); common::nn::test_softmax_dim1(&Cpu);
    common::nn::test_softmax_numerical_stability(&Cpu);
    common::nn::test_softmax_grad(&Cpu);
    common::nn::test_rms_norm_f32(&Cpu); common::nn::test_rms_norm_weighted(&Cpu);
}

#[test]
fn test_edge() {
    common::numeric::test_sqrt_negative(&Cpu);
    common::numeric::test_ln_zero(&Cpu);
    common::numeric::test_exp_large(&Cpu);
    common::numeric::test_div_zero_f32(&Cpu);
    common::numeric::test_add_nan_f32(&Cpu);
    common::numeric::test_empty_zeros(&Cpu);
    common::numeric::test_empty_add(&Cpu);
}

#[test]
fn test_large() {
    common::reduce::test_large_sum_f32(&Cpu);
    common::matmul::test_large_matmul_f32(&Cpu);
    common::nn::test_large_softmax(&Cpu);
}

#[test]
fn test_cross() {
    common::cross::test_transpose_add(&Cpu);
    common::cross::test_transpose_sub(&Cpu);
    common::cross::test_transpose_sum(&Cpu);
    common::cross::test_transpose_max(&Cpu);
    common::cross::test_permute_add(&Cpu);
    common::cross::test_slice_sum(&Cpu);
    common::cross::test_narrow_add(&Cpu);
    common::cross::test_permute_contiguous_add(&Cpu);
    common::cross::test_broadcast_sum(&Cpu);
}

#[test]
fn test_error() {
    common::error::test_binary_shape_mismatch(&Cpu);
    common::error::test_matmul_shape_mismatch(&Cpu);
    common::error::test_narrow_out_of_range(&Cpu);
    common::error::test_dim_out_of_range(&Cpu);
    common::error::test_allclose_shape_mismatch(&Cpu);
    common::error::test_f64_to_f32_add(&Cpu);
    common::error::test_reshape_wrong_elements(&Cpu);
}
