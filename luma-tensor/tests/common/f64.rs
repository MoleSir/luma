#![allow(dead_code)]

use super::*;
use luma_tensor::Device;

// ---- unary ----

#[allow(dead_code)]
pub fn test_f64_neg(device: &impl Device) {
    let a = tensor_f64_dev(&[1.0, -2.0, 3.0], (3,), device);
    assert_close(&a.neg().unwrap().to_vec().unwrap(), &[-1.0, 2.0, -3.0], 1e-10, 1e-10);
}

#[allow(dead_code)]
pub fn test_f64_abs(device: &impl Device) {
    let a = tensor_f64_dev(&[-1.0, 2.0, -3.0], (3,), device);
    assert_close(&a.abs().unwrap().to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-10, 1e-10);
}

#[allow(dead_code)]
pub fn test_f64_relu(device: &impl Device) {
    let a = tensor_f64_dev(&[-1.0, 0.0, 2.0], (3,), device);
    assert_close(&a.relu().unwrap().to_vec().unwrap(), &[0.0, 0.0, 2.0], 1e-10, 1e-10);
}

#[allow(dead_code)]
pub fn test_f64_exp(device: &impl Device) {
    let a = tensor_f64_dev(&[0.0, 1.0, 2.0], (3,), device);
    let v = a.exp().unwrap().to_vec().unwrap();
    assert!((v[0] - 1.0).abs() < 1e-10);
    assert!((v[1] - 2.718281).abs() < 1e-5);
}

#[allow(dead_code)]
pub fn test_f64_ln(device: &impl Device) {
    let a = tensor_f64_dev(&[1.0, 2.718281, 7.389056], (3,), device);
    let v = a.ln().unwrap().to_vec().unwrap();
    assert!((v[0] - 0.0).abs() < 1e-10);
    assert!((v[1] - 1.0).abs() < 1e-5);
    assert!((v[2] - 2.0).abs() < 1e-5);
}

#[allow(dead_code)]
pub fn test_f64_sqrt(device: &impl Device) {
    let a = tensor_f64_dev(&[0.0, 1.0, 4.0, 9.0], (4,), device);
    assert_close(&a.sqrt().unwrap().to_vec().unwrap(), &[0.0, 1.0, 2.0, 3.0], 1e-10, 1e-10);
}

#[allow(dead_code)]
pub fn test_f64_sigmoid(device: &impl Device) {
    let a = tensor_f64_dev(&[0.0], (1,), device);
    let v = a.sigmoid().unwrap().to_vec().unwrap();
    assert!((v[0] - 0.5).abs() < 1e-10);
}

#[allow(dead_code)]
pub fn test_f64_tanh(device: &impl Device) {
    let a = tensor_f64_dev(&[0.0], (1,), device);
    assert!((a.tanh().unwrap().to_vec().unwrap()[0] - 0.0).abs() < 1e-10);
}

#[allow(dead_code)]
pub fn test_f64_sin(device: &impl Device) {
    let a = tensor_f64_dev(&[0.0], (1,), device);
    assert!((a.sin().unwrap().to_vec().unwrap()[0] - 0.0).abs() < 1e-10);
}

#[allow(dead_code)]
pub fn test_f64_cos(device: &impl Device) {
    let a = tensor_f64_dev(&[0.0], (1,), device);
    assert!((a.cos().unwrap().to_vec().unwrap()[0] - 1.0).abs() < 1e-10);
}

#[allow(dead_code)]
pub fn test_f64_sqr(device: &impl Device) {
    let a = tensor_f64_dev(&[2.0, -3.0], (2,), device);
    assert_close(&a.sqr().unwrap().to_vec().unwrap(), &[4.0, 9.0], 1e-10, 1e-10);
}

#[allow(dead_code)]
pub fn test_f64_recip(device: &impl Device) {
    let a = tensor_f64_dev(&[2.0, 4.0], (2,), device);
    assert_close(&a.recip().unwrap().to_vec().unwrap(), &[0.5, 0.25], 1e-10, 1e-10);
}

#[allow(dead_code)]
pub fn test_f64_floor(device: &impl Device) {
    let a = tensor_f64_dev(&[1.7, -1.3], (2,), device);
    assert_close(&a.floor().unwrap().to_vec().unwrap(), &[1.0, -2.0], 1e-10, 1e-10);
}

#[allow(dead_code)]
pub fn test_f64_ceil(device: &impl Device) {
    let a = tensor_f64_dev(&[1.2, -1.7], (2,), device);
    assert_close(&a.ceil().unwrap().to_vec().unwrap(), &[2.0, -1.0], 1e-10, 1e-10);
}

#[allow(dead_code)]
pub fn test_f64_sign(device: &impl Device) {
    let a = tensor_f64_dev(&[5.0, -3.0], (2,), device);
    assert_close(&a.sign().unwrap().to_vec().unwrap(), &[1.0, -1.0], 1e-10, 1e-10);
}

#[allow(dead_code)]
pub fn test_f64_pow(device: &impl Device) {
    let a = tensor_f64_dev(&[2.0, 3.0], (2,), device);
    assert_close(&a.pow(2.0).unwrap().to_vec().unwrap(), &[4.0, 9.0], 1e-10, 1e-10);
}

#[allow(dead_code)]
pub fn test_f64_affine(device: &impl Device) {
    let a = tensor_f64_dev(&[1.0, 2.0], (2,), device);
    assert_close(&a.affine(2.0, 3.0).unwrap().to_vec().unwrap(), &[5.0, 7.0], 1e-10, 1e-10);
}

// ---- cmp ----

#[allow(dead_code)]
pub fn test_f64_eq(device: &impl Device) {
    let a = tensor_f64_dev(&[1.0, 2.0], (2,), device);
    let b = tensor_f64_dev(&[1.0, 3.0], (2,), device);
    assert_eq!(a.eq(&b).unwrap().to_vec().unwrap(), vec![true, false]);
}

#[allow(dead_code)]
pub fn test_f64_lt(device: &impl Device) {
    let a = tensor_f64_dev(&[1.0, 5.0], (2,), device);
    let b = tensor_f64_dev(&[2.0, 3.0], (2,), device);
    assert_eq!(a.lt(&b).unwrap().to_vec().unwrap(), vec![true, false]);
}

#[allow(dead_code)]
pub fn test_f64_gt(device: &impl Device) {
    let a = tensor_f64_dev(&[3.0, 1.0], (2,), device);
    let b = tensor_f64_dev(&[2.0, 2.0], (2,), device);
    assert_eq!(a.gt(&b).unwrap().to_vec().unwrap(), vec![true, false]);
}

#[allow(dead_code)]
pub fn test_f64_le(device: &impl Device) {
    let a = tensor_f64_dev(&[2.0, 3.0], (2,), device);
    let b = tensor_f64_dev(&[2.0, 2.0], (2,), device);
    assert_eq!(a.le(&b).unwrap().to_vec().unwrap(), vec![true, false]);
}

#[allow(dead_code)]
pub fn test_f64_ge(device: &impl Device) {
    let a = tensor_f64_dev(&[2.0, 1.0], (2,), device);
    let b = tensor_f64_dev(&[2.0, 2.0], (2,), device);
    assert_eq!(a.ge(&b).unwrap().to_vec().unwrap(), vec![true, false]);
}

#[allow(dead_code)]
pub fn test_f64_ne(device: &impl Device) {
    let a = tensor_f64_dev(&[1.0, 2.0], (2,), device);
    let b = tensor_f64_dev(&[1.0, 3.0], (2,), device);
    assert_eq!(a.ne(&b).unwrap().to_vec().unwrap(), vec![false, true]);
}

// ---- scalar ----

#[allow(dead_code)]
pub fn test_f64_add_scalar(device: &impl Device) {
    let a = tensor_f64_dev(&[1.0, 2.0], (2,), device);
    assert_close(&a.add_scalar(5.0).unwrap().to_vec().unwrap(), &[6.0, 7.0], 1e-10, 1e-10);
}

#[allow(dead_code)]
pub fn test_f64_sub_scalar(device: &impl Device) {
    let a = tensor_f64_dev(&[10.0, 20.0], (2,), device);
    assert_close(&a.sub_scalar(5.0).unwrap().to_vec().unwrap(), &[5.0, 15.0], 1e-10, 1e-10);
}

#[allow(dead_code)]
pub fn test_f64_sub_scalar_lhs(device: &impl Device) {
    let a = tensor_f64_dev(&[2.0, 3.0], (2,), device);
    assert_close(&a.sub_scalar_lhs(10.0).unwrap().to_vec().unwrap(), &[8.0, 7.0], 1e-10, 1e-10);
}

#[allow(dead_code)]
pub fn test_f64_mul_scalar(device: &impl Device) {
    let a = tensor_f64_dev(&[3.0, 4.0], (2,), device);
    assert_close(&a.mul_scalar(2.0).unwrap().to_vec().unwrap(), &[6.0, 8.0], 1e-10, 1e-10);
}

#[allow(dead_code)]
pub fn test_f64_div_scalar(device: &impl Device) {
    let a = tensor_f64_dev(&[10.0, 20.0], (2,), device);
    assert_close(&a.div_scalar(2.0).unwrap().to_vec().unwrap(), &[5.0, 10.0], 1e-10, 1e-10);
}

#[allow(dead_code)]
pub fn test_f64_div_scalar_lhs(device: &impl Device) {
    let a = tensor_f64_dev(&[2.0, 4.0], (2,), device);
    assert_close(&a.div_scalar_lhs(20.0).unwrap().to_vec().unwrap(), &[10.0, 5.0], 1e-10, 1e-10);
}

// ---- reduce ----

#[allow(dead_code)]
pub fn test_f64_sum_dim(device: &impl Device) {
    let t = tensor_f64_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let s = t.sum(0usize).unwrap();
    assert_close(&s.to_vec().unwrap(), &[4.0, 6.0], 1e-10, 1e-10);
}

#[allow(dead_code)]
pub fn test_f64_max_all(device: &impl Device) {
    let t = tensor_f64_dev(&[1.5, 3.7, 2.1], (3,), device);
    assert!((t.max_all().unwrap().to_scalar().unwrap() - 3.7).abs() < 1e-10);
}

// ---- grad ----

#[allow(dead_code)]
pub fn test_f64_grad_add(device: &impl Device) {
    let x1 = tensor_f64_dev(&[1.0, 2.0], (2,), device);
    let x2 = tensor_f64_dev(&[3.0, 4.0], (2,), device);
    x1.set_requires_grad(true);
    x2.set_requires_grad(true);
    let y = x1.add(&x2).unwrap();
    let loss = y.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    assert_close(&grads.get(&x1).unwrap().to_vec().unwrap(), &[1.0, 1.0], 1e-10, 1e-10);
    assert_close(&grads.get(&x2).unwrap().to_vec().unwrap(), &[1.0, 1.0], 1e-10, 1e-10);
}

#[allow(dead_code)]
pub fn test_f64_grad_mul(device: &impl Device) {
    let x1 = tensor_f64_dev(&[2.0, 3.0], (2,), device);
    let x2 = tensor_f64_dev(&[4.0, 5.0], (2,), device);
    x1.set_requires_grad(true);
    x2.set_requires_grad(true);
    let y = x1.mul(&x2).unwrap();
    let loss = y.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    assert_close(&grads.get(&x1).unwrap().to_vec().unwrap(), &[4.0, 5.0], 1e-10, 1e-10);
    assert_close(&grads.get(&x2).unwrap().to_vec().unwrap(), &[2.0, 3.0], 1e-10, 1e-10);
}
