//! CPU tests — delegates ALL tests to common/, plus truly CPU-only tests.
//! Run with: cargo test --test cpu

mod common;

use common::*;
use luma_tensor::dtype::{FloatDType, IntDType};
use luma_tensor::{Bool, Cpu, Int, Tensor};

// ---- common delegation groups (all device-generic tests) ----

#[test]
fn test_binary() {
    common::numeric::test_add_f32(&Cpu::default());
    common::numeric::test_sub_f32(&Cpu::default());
    common::numeric::test_mul_f32(&Cpu::default());
    common::numeric::test_div_f32(&Cpu::default());
    common::numeric::test_maximum_f32(&Cpu::default());
    common::numeric::test_minimum_f32(&Cpu::default());
}

#[test]
fn test_unary() {
    common::numeric::test_neg_f32(&Cpu::default());
    common::numeric::test_abs_f32(&Cpu::default());
    common::numeric::test_relu_f32(&Cpu::default());
    common::numeric::test_exp_f32(&Cpu::default());
    common::numeric::test_sigmoid_f32(&Cpu::default());
    common::numeric::test_tanh_f32(&Cpu::default());
    common::numeric::test_ln_f32(&Cpu::default());
    common::numeric::test_sin_f32(&Cpu::default());
    common::numeric::test_cos_f32(&Cpu::default());
    common::numeric::test_sqr_f32(&Cpu::default());
    common::numeric::test_sqrt_f32(&Cpu::default());
    common::numeric::test_recip_f32(&Cpu::default());
    common::numeric::test_gelu_f32(&Cpu::default());
    common::numeric::test_silu_f32(&Cpu::default());
    common::numeric::test_floor_f32(&Cpu::default());
    common::numeric::test_ceil_f32(&Cpu::default());
    common::numeric::test_sign_f32(&Cpu::default());
    common::numeric::test_leaky_relu_f32(&Cpu::default());
    common::numeric::test_pow_f32(&Cpu::default());
    common::numeric::test_affine_f32(&Cpu::default());
    common::numeric::test_erf_f32(&Cpu::default());
    common::numeric::test_gelu_erf_f32(&Cpu::default());
    common::numeric::test_round_f32(&Cpu::default());
}

#[test]
fn test_dtype() {
    common::dtype::test_u8_construct(&Cpu::default());
    common::dtype::test_u8_add(&Cpu::default());
    common::dtype::test_u8_sub(&Cpu::default());
    common::dtype::test_u8_clamp(&Cpu::default());
    common::dtype::test_u8_cast_to_i32(&Cpu::default());
    common::dtype::test_u8_cast_to_f32(&Cpu::default());
    common::dtype::test_u32_construct(&Cpu::default());
    common::dtype::test_u32_add(&Cpu::default());
    common::dtype::test_u32_mul(&Cpu::default());
}

#[test]
fn test_scalar() {
    common::numeric::test_add_scalar_f32(&Cpu::default());
    common::numeric::test_sub_scalar_f32(&Cpu::default());
    common::numeric::test_sub_scalar_lhs_f32(&Cpu::default());
    common::numeric::test_mul_scalar_f32(&Cpu::default());
    common::numeric::test_div_scalar_f32(&Cpu::default());
    common::numeric::test_div_scalar_lhs_f32(&Cpu::default());
}

#[test]
fn test_cmp() {
    common::numeric::test_eq_f32(&Cpu::default());
    common::numeric::test_lt_f32(&Cpu::default());
    common::numeric::test_ne_f32(&Cpu::default());
    common::numeric::test_ge_f32(&Cpu::default());
    common::numeric::test_gt_f32(&Cpu::default());
    common::numeric::test_le_f32(&Cpu::default());
}

#[test]
fn test_clamp() {
    common::numeric::test_clamp_both(&Cpu::default());
    common::numeric::test_clamp_min_only(&Cpu::default());
    common::numeric::test_clamp_max_only(&Cpu::default());
    common::numeric::test_clamp_none(&Cpu::default());
    common::numeric::test_pow_exp_zero(&Cpu::default());
    common::numeric::test_pow_exp_one(&Cpu::default());
}

#[test]
fn test_broadcast() {
    common::numeric::test_broadcast_add_f32(&Cpu::default());
    common::numeric::test_broadcast_mul_f32(&Cpu::default());
    common::numeric::test_broadcast_eq_f32(&Cpu::default());
}

#[test]
fn test_reduce() {
    common::reduce::test_sum_dim_f32(&Cpu::default());
    common::reduce::test_sum_keepdim_f32(&Cpu::default());
    common::reduce::test_sum_all_f32(&Cpu::default());
    common::reduce::test_sum_dims_f32(&Cpu::default());
    common::reduce::test_max_dim_f32(&Cpu::default());
    common::reduce::test_max_keepdim_f32(&Cpu::default());
    common::reduce::test_max_all_f32(&Cpu::default());
    common::reduce::test_min_dim_f32(&Cpu::default());
    common::reduce::test_min_all_f32(&Cpu::default());
    common::reduce::test_mean_dim_f32(&Cpu::default());
    common::reduce::test_mean_all_f32(&Cpu::default());
    common::reduce::test_prod_dim_f32(&Cpu::default());
    common::reduce::test_prod_all_f32(&Cpu::default());
    common::reduce::test_argmax_f32(&Cpu::default());
    common::reduce::test_argmin_f32(&Cpu::default());
    common::reduce::test_argmax_keepdim(&Cpu::default());
    common::reduce::test_var_f32(&Cpu::default());
    common::reduce::test_var_unbiased_f32(&Cpu::default());
    common::reduce::test_std_f32(&Cpu::default());
    common::reduce::test_std_all_f32(&Cpu::default());
    common::reduce::test_logsumexp_f32(&Cpu::default());
    common::reduce::test_logsumexp_keepdim(&Cpu::default());
    common::reduce::test_sum_i32(&Cpu::default());
    common::reduce::test_sum_all_i32(&Cpu::default());
    common::reduce::test_max_dim_i32(&Cpu::default());
    common::reduce::test_min_dim_i32(&Cpu::default());
    common::reduce::test_sum_f64(&Cpu::default());
    common::reduce::test_mean_f64(&Cpu::default());
    common::reduce::test_sum_u8(&Cpu::default());
    common::reduce::test_max_u8(&Cpu::default());
    common::reduce::test_sum_u32(&Cpu::default());
    common::reduce::test_min_u32(&Cpu::default());
}

#[test]
fn test_bool() {
    common::boolean::test_bool_and(&Cpu::default());
    common::boolean::test_bool_or(&Cpu::default());
    common::boolean::test_bool_xor(&Cpu::default());
    common::boolean::test_bool_not(&Cpu::default());
    common::boolean::test_pick_f32(&Cpu::default());
    common::boolean::test_bool_all_all(&Cpu::default());
    common::boolean::test_bool_any_all(&Cpu::default());
    common::boolean::test_bool_true_count(&Cpu::default());
    common::boolean::test_bool_false_count(&Cpu::default());
    common::boolean::test_pick_scalar_true(&Cpu::default());
    common::boolean::test_pick_scalar_false(&Cpu::default());
    common::boolean::test_pick_bool(&Cpu::default());
    common::boolean::test_pick_int(&Cpu::default());
    common::boolean::test_pick_int_scalar_true(&Cpu::default());
    common::boolean::test_pick_int_scalar_false(&Cpu::default());
    common::boolean::test_pick_bool_scalar_true(&Cpu::default());
    common::boolean::test_pick_bool_scalar_false(&Cpu::default());
    common::boolean::test_allclose_exact(&Cpu::default());
    common::boolean::test_allclose_false(&Cpu::default());
    common::boolean::test_allclose_int(&Cpu::default());
    common::boolean::test_allclose_bool(&Cpu::default());
}

#[test]
fn test_cast() {
    common::cast::test_cast_f32_to_f64(&Cpu::default());
    common::cast::test_cast_f32_to_i32(&Cpu::default());
    common::cast::test_cast_f32_to_bool(&Cpu::default());
    common::cast::test_cast_i32_to_f32(&Cpu::default());
    common::cast::test_cast_bool_to_f32(&Cpu::default());
    common::cast::test_cast_bool_to_i32(&Cpu::default());
    common::cast::test_cast_i32_to_bool(&Cpu::default());
    common::cast::test_cast_f64_to_f32(&Cpu::default());
    common::cast::test_cast_i32_to_u32(&Cpu::default());
    common::cast::test_cast_bool_to_bool(&Cpu::default());
}

#[test]
fn test_shape() {
    common::shape::test_cat_dim0_f32(&Cpu::default());
    common::shape::test_contiguous_after_transpose(&Cpu::default());
    common::shape::test_reshape_f32(&Cpu::default());
    common::shape::test_transpose_f32(&Cpu::default());
    common::shape::test_broadcast_as_f32(&Cpu::default());
    common::shape::test_narrow_dim0(&Cpu::default());
    common::shape::test_squeeze_dim1(&Cpu::default());
    common::shape::test_unsqueeze(&Cpu::default());
    common::shape::test_flatten_all(&Cpu::default());
    common::shape::test_permute_f32(&Cpu::default());
    common::shape::test_split_f32(&Cpu::default());
    common::shape::test_repeat_dim_f32(&Cpu::default());
    common::shape::test_transpose_last(&Cpu::default());
    common::shape::test_already_contiguous_is_noop(&Cpu::default());
    common::shape::test_flatten_range(&Cpu::default());
    common::shape::test_stack_f32(&Cpu::default());
    common::shape::test_chunk_f32(&Cpu::default());
    common::shape::test_copy_float(&Cpu::default());
    common::shape::test_copy_shape_mismatch(&Cpu::default());
    common::shape::test_phantom(&Cpu::default());
    common::shape::test_phantom_then_copy(&Cpu::default());
    common::shape::test_copy_preserves_requires_grad(&Cpu::default());
}

#[test]
fn test_matmul() {
    common::matmul::test_matmul_2x2(&Cpu::default());
    common::matmul::test_matmul_2x3_3x2(&Cpu::default());
    common::matmul::test_matmul_f64(&Cpu::default());
}

#[test]
fn test_indexing() {
    common::indexing::test_index_select_dim0(&Cpu::default());
    common::indexing::test_index_select_dim1(&Cpu::default());
    common::indexing::test_gather_dim1(&Cpu::default());
    common::indexing::test_index_add_f32(&Cpu::default());
    common::indexing::test_scatter_add_f32(&Cpu::default());
    common::indexing::test_index_add_2d(&Cpu::default());
    common::indexing::test_i_select_row(&Cpu::default());
    common::indexing::test_i_select_negative(&Cpu::default());
    common::indexing::test_i_slice_range(&Cpu::default());
    common::indexing::test_i_slice_full(&Cpu::default());
    common::indexing::test_i_slice_with_step(&Cpu::default());
    common::indexing::test_i_tuple_select_slice(&Cpu::default());
    common::indexing::test_i_tuple_slice_slice(&Cpu::default());
    common::indexing::test_i_boolean_mask(&Cpu::default());
    common::indexing::test_i_boolean_mask_2d(&Cpu::default());
    common::indexing::test_get_element(&Cpu::default());
    common::indexing::test_get_row_2d(&Cpu::default());
}

#[test]
fn test_int() {
    common::numeric::test_add_i32(&Cpu::default());
    common::numeric::test_neg_i32(&Cpu::default());
    common::numeric::test_abs_i32(&Cpu::default());
    common::numeric::test_sign_i32(&Cpu::default());
    common::numeric::test_pow_i32(&Cpu::default());
    common::numeric::test_affine_i32(&Cpu::default());
    common::numeric::test_clamp_i32(&Cpu::default());
    common::numeric::test_add_scalar_i32(&Cpu::default());
    common::numeric::test_sub_scalar_i32(&Cpu::default());
    common::numeric::test_sub_scalar_lhs_i32(&Cpu::default());
    common::numeric::test_mul_scalar_i32(&Cpu::default());
    common::numeric::test_div_scalar_i32(&Cpu::default());
    common::numeric::test_div_scalar_lhs_i32(&Cpu::default());
}

#[test]
fn test_construct() {
    common::construct::test_zeros_like_f32(&Cpu::default());
    common::construct::test_ones_like_f32(&Cpu::default());
    common::construct::test_from_slice_f32(&Cpu::default());
    common::construct::test_full_scalar(&Cpu::default());
    common::construct::test_rand_like_shape(&Cpu::default());
    common::construct::test_randn_like_shape(&Cpu::default());
    common::construct::test_bytes_roundtrip_f32(&Cpu::default());
    common::construct::test_bytes_roundtrip_f64(&Cpu::default());
    common::construct::test_bytes_roundtrip_i32(&Cpu::default());
    common::construct::test_bytes_roundtrip_u8(&Cpu::default());
    common::construct::test_bytes_roundtrip_bool(&Cpu::default());
    common::construct::test_bytes_non_contiguous(&Cpu::default());
    common::construct::test_dyn_tensor_roundtrip_float(&Cpu::default());
    common::construct::test_dyn_tensor_roundtrip_int(&Cpu::default());
    common::construct::test_dyn_tensor_roundtrip_bool(&Cpu::default());
    common::construct::test_dyn_tensor_accessors(&Cpu::default());
}

#[test]
fn test_grad() {
    common::grad::test_grad_add(&Cpu::default());
    common::grad::test_grad_sub(&Cpu::default());
    common::grad::test_grad_mul(&Cpu::default());
    common::grad::test_grad_div(&Cpu::default());
    common::grad::test_grad_relu(&Cpu::default());
    common::grad::test_grad_sum(&Cpu::default());
    common::grad::test_grad_mean(&Cpu::default());
    common::grad::test_grad_exp(&Cpu::default());
    common::grad::test_grad_sigmoid(&Cpu::default());
    common::grad::test_grad_clamp(&Cpu::default());
    common::grad::test_grad_clamp_min(&Cpu::default());
    common::grad::test_grad_prod(&Cpu::default());
    common::grad::test_grad_matmul(&Cpu::default());
    common::grad::test_grad_reshape(&Cpu::default());
    common::grad::test_grad_transpose(&Cpu::default());
    common::grad::test_grad_accumulate(&Cpu::default());
    common::grad::test_no_grad_disabled(&Cpu::default());
}

#[test]
fn test_display() {
    common::display::test_display_scalar(&Cpu::default());
    common::display::test_display_1d(&Cpu::default());
}

#[test]
fn test_f64() {
    common::cast::test_f64_zeros(&Cpu::default());
    common::cast::test_f64_add(&Cpu::default());
    common::reduce::test_sum_f64(&Cpu::default());
    common::reduce::test_mean_f64(&Cpu::default());
    common::f64::test_f64_neg(&Cpu::default());
    common::f64::test_f64_abs(&Cpu::default());
    common::f64::test_f64_relu(&Cpu::default());
    common::f64::test_f64_exp(&Cpu::default());
    common::f64::test_f64_ln(&Cpu::default());
    common::f64::test_f64_sqrt(&Cpu::default());
    common::f64::test_f64_sigmoid(&Cpu::default());
    common::f64::test_f64_tanh(&Cpu::default());
    common::f64::test_f64_sin(&Cpu::default());
    common::f64::test_f64_cos(&Cpu::default());
    common::f64::test_f64_sqr(&Cpu::default());
    common::f64::test_f64_recip(&Cpu::default());
    common::f64::test_f64_floor(&Cpu::default());
    common::f64::test_f64_ceil(&Cpu::default());
    common::f64::test_f64_sign(&Cpu::default());
    common::f64::test_f64_pow(&Cpu::default());
    common::f64::test_f64_affine(&Cpu::default());
    common::f64::test_f64_eq(&Cpu::default());
    common::f64::test_f64_lt(&Cpu::default());
    common::f64::test_f64_gt(&Cpu::default());
    common::f64::test_f64_le(&Cpu::default());
    common::f64::test_f64_ge(&Cpu::default());
    common::f64::test_f64_ne(&Cpu::default());
    common::f64::test_f64_add_scalar(&Cpu::default());
    common::f64::test_f64_sub_scalar(&Cpu::default());
    common::f64::test_f64_sub_scalar_lhs(&Cpu::default());
    common::f64::test_f64_mul_scalar(&Cpu::default());
    common::f64::test_f64_div_scalar(&Cpu::default());
    common::f64::test_f64_div_scalar_lhs(&Cpu::default());
    common::f64::test_f64_sum_dim(&Cpu::default());
    common::f64::test_f64_max_all(&Cpu::default());
    common::f64::test_f64_grad_add(&Cpu::default());
    common::f64::test_f64_grad_mul(&Cpu::default());
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
    assert_close(&Tensor::<Cpu>::eye(3, ()).unwrap().to_vec().unwrap(), &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 1e-7, 1e-7);
    assert_eq!(Tensor::<Cpu, Int>::eye(2, ()).unwrap().to_vec().unwrap(), vec![1i64, 0, 0, 1]);

    // Tril/Triu (CPU-specific API)
    assert_close(
        &Tensor::<Cpu>::tril(3, false, ()).unwrap().to_vec().unwrap(),
        &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        1e-7,
        1e-7,
    );
    assert_close(&Tensor::<Cpu>::tril(3, true, ()).unwrap().to_vec().unwrap(), &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0], 1e-7, 1e-7);
    assert_close(
        &Tensor::<Cpu>::triu(3, false, ()).unwrap().to_vec().unwrap(),
        &[0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        1e-7,
        1e-7,
    );
    assert_close(&Tensor::<Cpu>::triu(3, true, ()).unwrap().to_vec().unwrap(), &[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], 1e-7, 1e-7);
    assert_eq!(Tensor::<Cpu, Int>::tril(3, true, ()).unwrap().to_vec().unwrap(), vec![1i64, 0, 0, 1, 1, 0, 1, 1, 1]);

    // Linspace (CPU-specific API)
    let t = Tensor::<Cpu>::linspace(0.0, 1.0, 5, ()).unwrap();
    assert_eq!(t.dims(), &[5]);
    let v = t.to_vec().unwrap();
    assert!((v[0] - 0.0).abs() < 1e-5);
    assert!((v[4] - 1.0).abs() < 1e-5);
    assert!((v[0] - 0.0).abs() < 1e-5);
    assert!((v[4] - 1.0).abs() < 1e-5);
    let t = Tensor::<Cpu>::linspace(3.0, 3.0, 1, ()).unwrap();
    assert!((t.to_scalar().unwrap() - 3.0).abs() < 1e-5);

    // Arange
    assert_eq!(Tensor::<Cpu, Int>::arange(0, 5, 1, IntDType::I32).unwrap().to_vec().unwrap(), vec![0i64, 1, 2, 3, 4]);
    assert_eq!(Tensor::<Cpu, Int>::arange(0, 10, 2, IntDType::I32).unwrap().to_vec().unwrap(), vec![0i64, 2, 4, 6, 8]);
    assert_eq!(Tensor::<Cpu, Int>::arange(5, 0, -1, IntDType::I32).unwrap().to_vec().unwrap(), vec![5i64, 4, 3, 2, 1]);

    // Int slice construct
    assert_eq!(
        Tensor::<Cpu, Int>::from_slice(&[10i64, 20, 30, 40], (2, 2), IntDType::I32).unwrap().to_vec().unwrap(),
        vec![10i64, 20, 30, 40]
    );
    assert_eq!(tensor_i32(&[1, 2, 3, 4], (4,)).sum_all().unwrap().to_vec().unwrap(), vec![10]);
    assert_eq!(tensor_i32(&[1, 2, 3, 4], (4,)).prod_all().unwrap().to_vec().unwrap(), vec![24]);

    // Bool construct
    assert!(!Tensor::<Cpu, Bool>::falses((), ()).unwrap().to_vec().unwrap()[0]);
    assert!(Tensor::<Cpu, Bool>::trues((), ()).unwrap().to_vec().unwrap()[0]);
    assert_eq!(Tensor::<Cpu, Bool>::from_slice(&[true, false, true], (3,), ()).unwrap().to_vec().unwrap(), vec![true, false, true]);

    // Diag
    assert_close(
        &Tensor::<Cpu>::diag(&[1.0, 2.0, 3.0], ()).unwrap().to_vec().unwrap(),
        &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
        1e-7,
        1e-7,
    );
}

#[test]
fn test_new_construct() {
    // IntoTensor constructors (CPU-specific trait)
    assert!((3.14 - Tensor::<Cpu>::new(3.14, &Cpu::default()).unwrap().to_scalar().unwrap()).abs() < 1e-5);
    assert_close(&Tensor::<Cpu>::new(&[1.0, 2.0, 3.0][..], &Cpu::default()).unwrap().to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-7, 1e-7);
    assert_close(&Tensor::<Cpu>::new([1.0, 2.0, 3.0], &Cpu::default()).unwrap().to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-7, 1e-7);
    assert_close(&Tensor::<Cpu>::new(&[1.0, 2.0, 3.0], &Cpu::default()).unwrap().to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-7, 1e-7);
    assert_close(&Tensor::<Cpu>::new(vec![1.0, 2.0, 3.0], &Cpu::default()).unwrap().to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-7, 1e-7);
    assert_close(
        &Tensor::<Cpu>::new(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &Cpu::default()).unwrap().to_vec().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        1e-7,
        1e-7,
    );
    assert_close(
        &Tensor::<Cpu>::new(&[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], &Cpu::default()).unwrap().to_vec().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        1e-7,
        1e-7,
    );
    assert_eq!(Tensor::<Cpu, Int>::new(42i64, &Cpu::default()).unwrap().to_vec().unwrap(), vec![42i64]);
    assert_eq!(Tensor::<Cpu, Int>::new(&[1i64, 2, 3][..], &Cpu::default()).unwrap().to_vec().unwrap(), vec![1i64, 2, 3]);
    assert_eq!(Tensor::<Cpu, Int>::new(&[[1i64, 2], [3, 4]], &Cpu::default()).unwrap().to_vec().unwrap(), vec![1i64, 2, 3, 4]);
    assert_eq!(Tensor::<Cpu, Bool>::new(true, &Cpu::default()).unwrap().to_vec().unwrap(), vec![true]);
    assert_eq!(Tensor::<Cpu, Bool>::new(&[true, false, true], &Cpu::default()).unwrap().to_vec().unwrap(), vec![true, false, true]);
    assert_eq!(
        Tensor::<Cpu, Bool>::new(&[[true, false], [false, true]], &Cpu::default()).unwrap().to_vec().unwrap(),
        vec![true, false, false, true]
    );
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
    common::nn::test_softmax_dim0(&Cpu::default());
    common::nn::test_softmax_dim1(&Cpu::default());
    common::nn::test_softmax_numerical_stability(&Cpu::default());
    common::nn::test_softmax_grad(&Cpu::default());
    common::nn::test_rms_norm_f32(&Cpu::default());
    common::nn::test_rms_norm_weighted(&Cpu::default());
    common::nn::test_cross_entropy_chain_f32(&Cpu::default());
    common::nn::test_cross_entropy_basic_f32(&Cpu::default());
    common::nn::test_cross_entropy_mnist_shape_f32(&Cpu::default());
    common::nn::test_matmul_transposed_weight_small_f32(&Cpu::default());
    common::nn::test_matmul_transposed_weight_f32(&Cpu::default());
    common::nn::test_matmul_add_bias_f32(&Cpu::default());
    common::nn::test_broadcast_add_precision(&Cpu::default());
    common::nn::test_cross_entropy_precision(&Cpu::default());
    common::nn::test_broadcast_add_grad_f32(&Cpu::default());
    common::nn::test_broadcast_reduce_backward_f32(&Cpu::default());
    common::nn::test_sum_keepdim_nonuniform_f32(&Cpu::default());
    common::nn::test_argmax_eval_pipeline_f32(&Cpu::default());
    common::nn::test_mini_training_step_f32(&Cpu::default());
    common::nn::test_argmax_eval_large_f32(&Cpu::default());
}

#[test]
fn test_edge() {
    common::numeric::test_sqrt_negative(&Cpu::default());
    common::numeric::test_ln_zero(&Cpu::default());
    common::numeric::test_exp_large(&Cpu::default());
    common::numeric::test_div_zero_f32(&Cpu::default());
    common::numeric::test_add_nan_f32(&Cpu::default());
    common::numeric::test_empty_zeros(&Cpu::default());
    common::numeric::test_empty_add(&Cpu::default());
}

#[test]
fn test_large() {
    common::reduce::test_large_sum_f32(&Cpu::default());
    common::matmul::test_large_matmul_f32(&Cpu::default());
    common::nn::test_large_softmax(&Cpu::default());
}

#[test]
fn test_cross() {
    common::cross::test_transpose_add(&Cpu::default());
    common::cross::test_transpose_sub(&Cpu::default());
    common::cross::test_transpose_sum(&Cpu::default());
    common::cross::test_transpose_max(&Cpu::default());
    common::cross::test_permute_add(&Cpu::default());
    common::cross::test_slice_sum(&Cpu::default());
    common::cross::test_narrow_add(&Cpu::default());
    common::cross::test_permute_contiguous_add(&Cpu::default());
    common::cross::test_broadcast_sum(&Cpu::default());
}

#[test]
fn test_error() {
    common::error::test_binary_shape_mismatch(&Cpu::default());
    common::error::test_matmul_shape_mismatch(&Cpu::default());
    common::error::test_narrow_out_of_range(&Cpu::default());
    common::error::test_dim_out_of_range(&Cpu::default());
    common::error::test_allclose_shape_mismatch(&Cpu::default());
    common::error::test_f64_to_f32_add(&Cpu::default());
    common::error::test_reshape_wrong_elements(&Cpu::default());
}

#[test]
fn test_operators() {
    let a = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (2, 2));
    let b = tensor_f32(&[5.0, 6.0, 7.0, 8.0], (2, 2));

    let c = &a + &b;
    assert_close(&c.to_vec().unwrap(), &[6.0, 8.0, 10.0, 12.0], 1e-5, 1e-5);

    let d = &a - &b;
    assert_close(&d.to_vec().unwrap(), &[-4.0, -4.0, -4.0, -4.0], 1e-5, 1e-5);

    let e = &a * &b;
    assert_close(&e.to_vec().unwrap(), &[5.0, 12.0, 21.0, 32.0], 1e-5, 1e-5);

    let f = &a / &b;
    assert_close(&f.to_vec().unwrap(), &[0.2, 1.0 / 3.0, 3.0 / 7.0, 0.5], 1e-5, 1e-5);

    let g = -&a;
    assert_close(&g.to_vec().unwrap(), &[-1.0, -2.0, -3.0, -4.0], 1e-5, 1e-5);

    // assign
    let mut h = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (2, 2));
    h += &b;
    assert_close(&h.to_vec().unwrap(), &[6.0, 8.0, 10.0, 12.0], 1e-5, 1e-5);

    let mut i = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (2, 2));
    i *= &tensor_f32(&[2.0, 2.0, 2.0, 2.0], (2, 2));
    assert_close(&i.to_vec().unwrap(), &[2.0, 4.0, 6.0, 8.0], 1e-5, 1e-5);

    // scalar ops
    let j = &a + 10.0;
    assert_close(&j.to_vec().unwrap(), &[11.0, 12.0, 13.0, 14.0], 1e-5, 1e-5);

    let k = &a * 2.0;
    assert_close(&k.to_vec().unwrap(), &[2.0, 4.0, 6.0, 8.0], 1e-5, 1e-5);

    let l = 3.0 + &a;
    assert_close(&l.to_vec().unwrap(), &[4.0, 5.0, 6.0, 7.0], 1e-5, 1e-5);

    let m = 10.0 - &a;
    assert_close(&m.to_vec().unwrap(), &[9.0, 8.0, 7.0, 6.0], 1e-5, 1e-5);

    let n = 2.0 * &a;
    assert_close(&n.to_vec().unwrap(), &[2.0, 4.0, 6.0, 8.0], 1e-5, 1e-5);

    // Int scalar
    use luma_tensor::Int;
    let ia = Tensor::<Cpu, Int>::from_slice(&[1i64, 2, 3, 4], (2, 2), IntDType::I32).unwrap();
    let ic = &ia + 10;
    assert_eq!(ic.to_vec().unwrap(), vec![11i64, 12, 13, 14]);
    let id = 5 + &ia;
    assert_eq!(id.to_vec().unwrap(), vec![6i64, 7, 8, 9]);
}
