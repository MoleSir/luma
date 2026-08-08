#![allow(dead_code)]

use luma_tensor::Device;
use super::*;

// ---- Binary (elementwise, same-shape) ----

#[allow(dead_code)]
pub fn test_add_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let b = tensor_f32_dev(&[5.0, 6.0, 7.0, 8.0], (2, 2), device);
    let c = a.add(&b).unwrap();
    assert_close(&c.to_vec().unwrap(), &[6.0, 8.0, 10.0, 12.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_sub_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[5.0, 6.0, 7.0, 8.0], (2, 2), device);
    let b = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let c = a.sub(&b).unwrap();
    assert_close(&c.to_vec().unwrap(), &[4.0, 4.0, 4.0, 4.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_mul_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let b = tensor_f32_dev(&[2.0, 3.0, 4.0, 5.0], (2, 2), device);
    let c = a.mul(&b).unwrap();
    assert_close(&c.to_vec().unwrap(), &[2.0, 6.0, 12.0, 20.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_div_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[6.0, 8.0, 10.0, 12.0], (2, 2), device);
    let b = tensor_f32_dev(&[2.0, 2.0, 2.0, 3.0], (2, 2), device);
    let c = a.div(&b).unwrap();
    assert_close(&c.to_vec().unwrap(), &[3.0, 4.0, 5.0, 4.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_maximum_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 5.0, 3.0], (3,), device);
    let b = tensor_f32_dev(&[4.0, 2.0, 6.0], (3,), device);
    let c = a.maximum(&b).unwrap();
    assert_close(&c.to_vec().unwrap(), &[4.0, 5.0, 6.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_minimum_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 5.0, 3.0], (3,), device);
    let b = tensor_f32_dev(&[4.0, 2.0, 6.0], (3,), device);
    let c = a.minimum(&b).unwrap();
    assert_close(&c.to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-5, 1e-5);
}

// ---- Unary float ops ----

#[allow(dead_code)]
pub fn test_neg_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, -2.0, 3.0, -4.0], (2, 2), device);
    let c = a.neg().unwrap();
    assert_close(&c.to_vec().unwrap(), &[-1.0, 2.0, -3.0, 4.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_abs_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[-1.0, 2.0, -3.0, 4.0], (2, 2), device);
    let c = a.abs().unwrap();
    assert_close(&c.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_relu_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[-1.0, 0.0, 2.0, -3.0], (2, 2), device);
    let c = a.relu().unwrap();
    assert_close(&c.to_vec().unwrap(), &[0.0, 0.0, 2.0, 0.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_exp_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[0.0, 1.0, 2.0], (3,), device);
    let c = a.exp().unwrap();
    let v = c.to_vec().unwrap();
    let expected: Vec<f64> = [0.0f64, 1.0, 2.0].iter().map(|&x| x.exp()).collect();
    assert_close(&v, &expected, 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_sigmoid_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[0.0], (1,), device);
    let c = a.sigmoid().unwrap();
    assert!((c.to_vec().unwrap()[0] - 0.5).abs() < 1e-5);
}

#[allow(dead_code)]
pub fn test_tanh_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[0.0], (1,), device);
    let c = a.tanh().unwrap();
    assert!(c.to_vec().unwrap()[0].abs() < 1e-5);
}

#[allow(dead_code)]
pub fn test_ln_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, std::f64::consts::E], (2,), device);
    let c = a.ln().unwrap();
    let v = c.to_vec().unwrap();
    assert!((v[0] - 0.0).abs() < 1e-5);
    assert!((v[1] - 1.0).abs() < 1e-5);
}

#[allow(dead_code)]
pub fn test_sin_f32(device: &impl Device) {
    use std::f64::consts::PI;
    let a = tensor_f32_dev(&[0.0, PI / 2.0], (2,), device);
    let c = a.sin().unwrap();
    let v = c.to_vec().unwrap();
    assert!(v[0].abs() < 1e-5);
    assert!((v[1] - 1.0).abs() < 1e-5);
}

#[allow(dead_code)]
pub fn test_cos_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[0.0], (1,), device);
    let c = a.cos().unwrap();
    assert!((c.to_vec().unwrap()[0] - 1.0).abs() < 1e-5);
}

#[allow(dead_code)]
pub fn test_sqr_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[2.0, -3.0], (2,), device);
    let c = a.sqr().unwrap();
    assert_close(&c.to_vec().unwrap(), &[4.0, 9.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_sqrt_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[4.0, 9.0], (2,), device);
    let c = a.sqrt().unwrap();
    assert_close(&c.to_vec().unwrap(), &[2.0, 3.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_recip_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[2.0, 4.0], (2,), device);
    let c = a.recip().unwrap();
    assert_close(&c.to_vec().unwrap(), &[0.5, 0.25], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_gelu_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[0.0, 1.0, -1.0], (3,), device);
    let c = a.gelu().unwrap();
    let v = c.to_vec().unwrap();
    assert!(v[0].abs() < 1e-5);
    assert!(v[1] > 0.0);
    assert!(v[2] <= 0.0);
}

#[allow(dead_code)]
pub fn test_silu_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[0.0, 2.0], (2,), device);
    let c = a.silu().unwrap();
    let v = c.to_vec().unwrap();
    assert!(v[0].abs() < 1e-5);
    assert!(v[1] > 1.0);
}

#[allow(dead_code)]
pub fn test_floor_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.7, -2.3], (2,), device);
    let c = a.floor().unwrap();
    assert_close(&c.to_vec().unwrap(), &[1.0, -3.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_ceil_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.2, -2.7], (2,), device);
    let c = a.ceil().unwrap();
    assert_close(&c.to_vec().unwrap(), &[2.0, -2.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_sign_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[3.0, -1.0, 0.0], (3,), device);
    let c = a.sign().unwrap();
    assert_close(&c.to_vec().unwrap(), &[1.0, -1.0, 1.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_leaky_relu_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[-1.0, 2.0], (2,), device);
    let c = a.leaky_relu(0.1).unwrap();
    let v = c.to_vec().unwrap();
    assert!((v[0] - (-0.1)).abs() < 1e-5);
    assert!((v[1] - 2.0).abs() < 1e-5);
}

#[allow(dead_code)]
pub fn test_pow_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[2.0, 3.0], (2,), device);
    let c = a.pow(2.0).unwrap();
    assert_close(&c.to_vec().unwrap(), &[4.0, 9.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_affine_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0], (2,), device);
    let c = a.affine(2.0, 1.0).unwrap();
    assert_close(&c.to_vec().unwrap(), &[3.0, 5.0], 1e-5, 1e-5);
}

// ---- Scalar binary ops ----

#[allow(dead_code)]
pub fn test_add_scalar_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0], (3,), device);
    let c = a.add_scalar(10.0).unwrap();
    assert_close(&c.to_vec().unwrap(), &[11.0, 12.0, 13.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_mul_scalar_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0], (3,), device);
    let c = a.mul_scalar(3.0).unwrap();
    assert_close(&c.to_vec().unwrap(), &[3.0, 6.0, 9.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_add_scalar_lhs_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0], (2,), device);
    let c = a.add_scalar_lhs(10.0).unwrap();
    assert_close(&c.to_vec().unwrap(), &[11.0, 12.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_div_scalar_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[6.0, 8.0, 10.0], (3,), device);
    let c = a.div_scalar(2.0).unwrap();
    assert_close(&c.to_vec().unwrap(), &[3.0, 4.0, 5.0], 1e-5, 1e-5);
}

// ---- Cmp ops ----

#[allow(dead_code)]
pub fn test_eq_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0], (3,), device);
    let b = tensor_f32_dev(&[1.0, 0.0, 3.0], (3,), device);
    let c = a.eq(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![true, false, true]);
}

#[allow(dead_code)]
pub fn test_lt_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 5.0, 3.0], (3,), device);
    let b = tensor_f32_dev(&[2.0, 1.0, 3.0], (3,), device);
    let c = a.lt(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![true, false, false]);
}

#[allow(dead_code)]
pub fn test_ne_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0], (3,), device);
    let b = tensor_f32_dev(&[1.0, 0.0, 3.0], (3,), device);
    let c = a.ne(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![false, true, false]);
}

#[allow(dead_code)]
pub fn test_ge_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 5.0, 3.0], (3,), device);
    let b = tensor_f32_dev(&[1.0, 1.0, 3.0], (3,), device);
    let c = a.ge(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![true, true, true]);
}

#[allow(dead_code)]
pub fn test_gt_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 5.0, 3.0], (3,), device);
    let b = tensor_f32_dev(&[2.0, 1.0, 3.0], (3,), device);
    let c = a.gt(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![false, true, false]);
}

// ---- Clamp ----

#[allow(dead_code)]
pub fn test_clamp_both(device: &impl Device) {
    let a = tensor_f32_dev(&[-1.0, 0.0, 2.0, 5.0, 10.0], (5,), device);
    let c = a.clamp(Some(0.0), Some(6.0)).unwrap();
    assert_close(&c.to_vec().unwrap(), &[0.0, 0.0, 2.0, 5.0, 6.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_clamp_min_only(device: &impl Device) {
    let a = tensor_f32_dev(&[-1.0, 0.0, 2.0], (3,), device);
    let c = a.clamp(Some(0.0), None).unwrap();
    assert_close(&c.to_vec().unwrap(), &[0.0, 0.0, 2.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_clamp_max_only(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 5.0, 10.0], (3,), device);
    let c = a.clamp(None, Some(6.0)).unwrap();
    assert_close(&c.to_vec().unwrap(), &[1.0, 5.0, 6.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_clamp_none(device: &impl Device) {
    let a = tensor_f32_dev(&[-1.0, 0.0, 5.0], (3,), device);
    let c = a.clamp(None, None).unwrap();
    assert_close(&c.to_vec().unwrap(), &[-1.0, 0.0, 5.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_pow_exp_zero(device: &impl Device) {
    let a = tensor_f32_dev(&[2.0, -3.0, 0.0], (3,), device);
    let c = a.pow(0.0).unwrap();
    assert_close(&c.to_vec().unwrap(), &[1.0, 1.0, 1.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_pow_exp_one(device: &impl Device) {
    let a = tensor_f32_dev(&[2.0, -3.0, 0.5], (3,), device);
    let c = a.pow(1.0).unwrap();
    assert_close(&c.to_vec().unwrap(), &[2.0, -3.0, 0.5], 1e-5, 1e-5);
}

// ---- Int arithmetic ----

#[allow(dead_code)]
pub fn test_add_i32(device: &impl Device) {
    let a = tensor_i32_dev(&[10, 20, 30], (3,), device);
    let b = tensor_i32_dev(&[1, 2, 3], (3,), device);
    let c = a.add(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![11, 22, 33]);
}

#[allow(dead_code)]
pub fn test_neg_i32(device: &impl Device) {
    let a = tensor_i32_dev(&[1, -2, 3], (3,), device);
    let c = a.neg().unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![-1, 2, -3]);
}

#[allow(dead_code)]
pub fn test_abs_i32(device: &impl Device) {
    let a = tensor_i32_dev(&[-1, 2, -3], (3,), device);
    let c = a.abs().unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![1, 2, 3]);
}

#[allow(dead_code)]
pub fn test_sign_i32(device: &impl Device) {
    let a = tensor_i32_dev(&[-5, 0, 3], (3,), device);
    let c = a.sign().unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![-1, 0, 1]);
}

#[allow(dead_code)]
pub fn test_pow_i32(device: &impl Device) {
    let a = tensor_i32_dev(&[2, 3, -2], (3,), device);
    let c = a.pow(3).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![8, 27, -8]);
}

#[allow(dead_code)]
pub fn test_affine_i32(device: &impl Device) {
    let a = tensor_i32_dev(&[1, 2, 3], (3,), device);
    let c = a.affine(10, 5).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![15, 25, 35]);
}

#[allow(dead_code)]
pub fn test_clamp_i32(device: &impl Device) {
    let a = tensor_i32_dev(&[-5, 0, 10], (3,), device);
    let c = a.clamp(Some(0), Some(5)).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![0, 0, 5]);
}

// ---- Broadcast ops ----

#[allow(dead_code)]
pub fn test_broadcast_add_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0], (1, 3), device);
    let b = tensor_f32_dev(&[10.0, 20.0], (2, 1), device);
    let c = a.broadcast_add(&b).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_close(&c.to_vec().unwrap(), &[11.0, 12.0, 13.0, 21.0, 22.0, 23.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_broadcast_mul_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[2.0, 3.0], (1, 2), device);
    let b = tensor_f32_dev(&[10.0, 100.0], (2, 1), device);
    let c = a.broadcast_mul(&b).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    assert_close(&c.to_vec().unwrap(), &[20.0, 30.0, 200.0, 300.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_broadcast_eq_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0], (1,), device);
    let b = tensor_f32_dev(&[1.0, 2.0, 0.0], (3,), device);
    let c = a.broadcast_eq(&b).unwrap();
    assert_eq!(c.dims(), &[3]);
    assert_eq!(c.to_vec().unwrap(), vec![true, false, false]);
}

// ---- Missing unary ops ----

#[allow(dead_code)]
pub fn test_erf_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[0.0, 1.0, -1.0], (3,), device);
    let c = a.erf().unwrap();
    let v = c.to_vec().unwrap();
    assert!((v[0] - 0.0).abs() < 1e-5);
    assert!((v[1] - 0.8427).abs() < 1e-3);
    assert!((v[2] + 0.8427).abs() < 1e-3);
}

#[allow(dead_code)]
pub fn test_gelu_erf_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[0.0, 1.0, -1.0], (3,), device);
    let c = a.gelu_erf().unwrap();
    let v = c.to_vec().unwrap();
    assert!((v[0]).abs() < 1e-5);
    assert!(v[1] > 0.5);
    assert!(v[2] < 0.0);
}

#[allow(dead_code)]
pub fn test_round_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.3, 2.7, -1.7, 0.6], (4,), device);
    let c = a.round().unwrap();
    assert_close(&c.to_vec().unwrap(), &[1.0, 3.0, -2.0, 1.0], 1e-5, 1e-5);
}

// ---- Missing cmp op ----

#[allow(dead_code)]
pub fn test_le_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0], (3,), device);
    let b = tensor_f32_dev(&[1.0, 1.0, 5.0], (3,), device);
    let c = a.le(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![true, false, true]);
}

// ---- Edge cases: NaN / Inf ----

#[allow(dead_code)]
pub fn test_sqrt_negative(device: &impl Device) {
    let a = tensor_f32_dev(&[4.0, -1.0], (2,), device);
    let c = a.sqrt().unwrap();
    let v = c.to_vec().unwrap();
    assert!((v[0] - 2.0).abs() < 1e-5);
    assert!(v[1].is_nan(), "sqrt(-1) should be NaN, got {}", v[1]);
}

#[allow(dead_code)]
pub fn test_ln_zero(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 0.0], (2,), device);
    let c = a.ln().unwrap();
    let v = c.to_vec().unwrap();
    assert!((v[0] - 0.0).abs() < 1e-5);
    assert!(v[1].is_infinite() && v[1].is_sign_negative(), "ln(0) = {}", v[1]);
}

#[allow(dead_code)]
pub fn test_exp_large(device: &impl Device) {
    let a = tensor_f32_dev(&[0.0, 80.0], (2,), device);
    let c = a.exp().unwrap();
    let v = c.to_vec().unwrap();
    assert!((v[0] - 1.0).abs() < 1e-5);
    assert!(v[1] > 1e30, "exp(80) = {}", v[1]);
    assert!(v[1].is_finite(), "exp(80) should be finite");
}

#[allow(dead_code)]
pub fn test_div_zero_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, -1.0], (2,), device);
    let b = tensor_f32_dev(&[0.0, 0.0], (2,), device);
    let c = a.div(&b).unwrap();
    let v = c.to_vec().unwrap();
    assert!(v[0].is_infinite() && v[0].is_sign_positive(), "1/0 = {}", v[0]);
    assert!(v[1].is_infinite() && v[1].is_sign_negative(), "-1/0 = {}", v[1]);
}

#[allow(dead_code)]
pub fn test_add_nan_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, f64::NAN], (2,), device);
    let b = tensor_f32_dev(&[2.0, 3.0], (2,), device);
    let c = a.add(&b).unwrap();
    let v = c.to_vec().unwrap();
    assert!((v[0] - 3.0).abs() < 1e-5);
    assert!(v[1].is_nan());
}

// ---- Empty tensor ops ----

#[allow(dead_code)]
pub fn test_empty_zeros(device: &impl Device) {
    let t = tensor_f32_dev(&[], (0,), device);
    assert_eq!(t.element_count(), 0);
    assert_eq!(t.to_vec().unwrap(), vec![]);
}

#[allow(dead_code)]
pub fn test_empty_add(device: &impl Device) {
    let a = tensor_f32_dev(&[], (0, 3), device);
    let b = tensor_f32_dev(&[], (0, 3), device);
    let c = a.add(&b).unwrap();
    assert_eq!(c.element_count(), 0);
    assert_eq!(c.to_vec().unwrap(), vec![]);
}
