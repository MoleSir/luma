//! Tests for numeric ops: binary, unary, cmp, broadcast, scalar variants.

mod common;
use common::*;
use luma_tensor::dtype::FloatDType;
use luma_tensor::{Cpu, Tensor};

// ---- Binary (elementwise, same-shape) ----

#[test]
fn add_f32() {
    let a = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (2, 2));
    let b = tensor_f32(&[5.0, 6.0, 7.0, 8.0], (2, 2));
    let c = a.add(&b).unwrap();
    assert_close(&c.to_vec().unwrap(), &[6.0, 8.0, 10.0, 12.0], 1e-7, 1e-7);
}

#[test]
fn sub_f32() {
    let a = tensor_f32(&[5.0, 6.0, 7.0, 8.0], (2, 2));
    let b = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (2, 2));
    let c = a.sub(&b).unwrap();
    assert_close(&c.to_vec().unwrap(), &[4.0, 4.0, 4.0, 4.0], 1e-7, 1e-7);
}

#[test]
fn mul_f32() {
    let a = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (2, 2));
    let b = tensor_f32(&[2.0, 3.0, 4.0, 5.0], (2, 2));
    let c = a.mul(&b).unwrap();
    assert_close(&c.to_vec().unwrap(), &[2.0, 6.0, 12.0, 20.0], 1e-7, 1e-7);
}

#[test]
fn div_f32() {
    let a = tensor_f32(&[6.0, 8.0, 10.0, 12.0], (2, 2));
    let b = tensor_f32(&[2.0, 2.0, 2.0, 3.0], (2, 2));
    let c = a.div(&b).unwrap();
    assert_close(&c.to_vec().unwrap(), &[3.0, 4.0, 5.0, 4.0], 1e-7, 1e-7);
}

#[test]
fn neg_f32() {
    let a = tensor_f32(&[1.0, -2.0, 3.0, -4.0], (2, 2));
    let c = a.neg().unwrap();
    assert_close(&c.to_vec().unwrap(), &[-1.0, 2.0, -3.0, 4.0], 1e-7, 1e-7);
}

#[test]
fn abs_f32() {
    let a = tensor_f32(&[-1.0, 2.0, -3.0, 4.0], (2, 2));
    let c = a.abs().unwrap();
    assert_close(&c.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0], 1e-7, 1e-7);
}

// ---- Unary float ops ----

#[test]
fn relu_f32() {
    let a = tensor_f32(&[-1.0, 0.0, 2.0, -3.0], (2, 2));
    let c = a.relu().unwrap();
    assert_close(&c.to_vec().unwrap(), &[0.0, 0.0, 2.0, 0.0], 1e-7, 1e-7);
}

#[test]
fn exp_f32() {
    let a = tensor_f32(&[0.0, 1.0, 2.0], (3,));
    let c = a.exp().unwrap();
    let v = c.to_vec().unwrap();
    let expected: Vec<f64> = [0.0f64, 1.0, 2.0].iter().map(|&x| x.exp()).collect();
    assert_close(&v, &expected, 1e-5, 1e-5);
}

#[test]
fn sigmoid_f32() {
    let a = tensor_f32(&[0.0], (1,));
    let c = a.sigmoid().unwrap();
    assert!((c.to_vec().unwrap()[0] - 0.5).abs() < 1e-5);
}

#[test]
fn tanh_f32() {
    let a = tensor_f32(&[0.0], (1,));
    let c = a.tanh().unwrap();
    assert!(c.to_vec().unwrap()[0].abs() < 1e-7);
}

// ---- Scalar binary ops ----

#[test]
fn add_scalar_f32() {
    let a = tensor_f32(&[1.0, 2.0, 3.0], (3,));
    let c = a.add_scalar(10.0).unwrap();
    assert_close(&c.to_vec().unwrap(), &[11.0, 12.0, 13.0], 1e-7, 1e-7);
}

#[test]
fn mul_scalar_f32() {
    let a = tensor_f32(&[1.0, 2.0, 3.0], (3,));
    let c = a.mul_scalar(3.0).unwrap();
    assert_close(&c.to_vec().unwrap(), &[3.0, 6.0, 9.0], 1e-7, 1e-7);
}

// ---- Broadcast binary ops ----

#[test]
fn broadcast_add_f32() {
    let a = tensor_f32(&[1.0, 2.0, 3.0], (1, 3)); // shape (1,3)
    let b = tensor_f32(&[10.0, 20.0], (2, 1)); // shape (2,1)
    let c = a.broadcast_add(&b).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_close(
        &c.to_vec().unwrap(),
        &[
            11.0, 12.0, 13.0,
            21.0, 22.0, 23.0,
        ],
        1e-7,
        1e-7,
    );
}

#[test]
fn broadcast_mul_f32() {
    let a = tensor_f32(&[2.0, 3.0], (1, 2)); // (1,2)
    let b = tensor_f32(&[10.0, 100.0], (2, 1)); // (2,1)
    let c = a.broadcast_mul(&b).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    assert_close(&c.to_vec().unwrap(), &[20.0, 30.0, 200.0, 300.0], 1e-7, 1e-7);
}

// ---- Cmp ops ----

#[test]
fn eq_f32() {
    let a = tensor_f32(&[1.0, 2.0, 3.0], (3,));
    let b = tensor_f32(&[1.0, 0.0, 3.0], (3,));
    let c = a.eq(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![true, false, true]);
}

#[test]
fn lt_f32() {
    let a = tensor_f32(&[1.0, 5.0, 3.0], (3,));
    let b = tensor_f32(&[2.0, 1.0, 3.0], (3,));
    let c = a.lt(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![true, false, false]);
}

// ---- Display ----

#[test]
fn display_scalar() {
    let t = Tensor::<Cpu>::full((), 3.14, FloatDType::F32).unwrap();
    let s = format!("{}", t);
    assert!(s.contains("Tensor<f32>"));
    assert!(s.contains("3.14"));
}

#[test]
fn display_1d() {
    let t = tensor_f32(&[1.0, 2.0, 3.0], (3,));
    let s = format!("{}", t);
    assert!(s.contains('['));
    assert!(s.contains("1.000"));
}

// ---- More unary float ops ----

#[test]
fn ln_f32() {
    let a = tensor_f32(&[1.0, std::f64::consts::E], (2,));
    let c = a.ln().unwrap();
    let v = c.to_vec().unwrap();
    assert!((v[0] - 0.0).abs() < 1e-5);
    assert!((v[1] - 1.0).abs() < 1e-5);
}

#[test]
fn sin_f32() {
    use std::f64::consts::PI;
    let a = tensor_f32(&[0.0, PI / 2.0], (2,));
    let c = a.sin().unwrap();
    let v = c.to_vec().unwrap();
    assert!(v[0].abs() < 1e-5);
    assert!((v[1] - 1.0).abs() < 1e-5);
}

#[test]
fn cos_f32() {
    let a = tensor_f32(&[0.0], (1,));
    let c = a.cos().unwrap();
    assert!((c.to_vec().unwrap()[0] - 1.0).abs() < 1e-5);
}

#[test]
fn sqr_f32() {
    let a = tensor_f32(&[2.0, -3.0], (2,));
    let c = a.sqr().unwrap();
    assert_close(&c.to_vec().unwrap(), &[4.0, 9.0], 1e-7, 1e-7);
}

#[test]
fn sqrt_f32() {
    let a = tensor_f32(&[4.0, 9.0], (2,));
    let c = a.sqrt().unwrap();
    assert_close(&c.to_vec().unwrap(), &[2.0, 3.0], 1e-5, 1e-5);
}

#[test]
fn recip_f32() {
    let a = tensor_f32(&[2.0, 4.0], (2,));
    let c = a.recip().unwrap();
    assert_close(&c.to_vec().unwrap(), &[0.5, 0.25], 1e-7, 1e-7);
}

#[test]
fn gelu_f32() {
    let a = tensor_f32(&[0.0, 1.0, -1.0], (3,));
    let c = a.gelu().unwrap();
    let v = c.to_vec().unwrap();
    // gelu(0) ≈ 0
    assert!(v[0].abs() < 1e-5);
    // gelu(x) > 0 for x > 0
    assert!(v[1] > 0.0);
    // gelu(x) < 0 for x < 0 (but close to zero)
    assert!(v[2] <= 0.0);
}

#[test]
fn silu_f32() {
    let a = tensor_f32(&[0.0, 2.0], (2,));
    let c = a.silu().unwrap();
    let v = c.to_vec().unwrap();
    // silu(0) = 0 * sigmoid(0) = 0
    assert!(v[0].abs() < 1e-5);
    // silu(2) > 1 (sigmoid(2) ≈ 0.88, 2*0.88 ≈ 1.76)
    assert!(v[1] > 1.0);
}

#[test]
fn floor_f32() {
    let a = tensor_f32(&[1.7, -2.3], (2,));
    let c = a.floor().unwrap();
    assert_close(&c.to_vec().unwrap(), &[1.0, -3.0], 1e-7, 1e-7);
}

#[test]
fn ceil_f32() {
    let a = tensor_f32(&[1.2, -2.7], (2,));
    let c = a.ceil().unwrap();
    assert_close(&c.to_vec().unwrap(), &[2.0, -2.0], 1e-7, 1e-7);
}

#[test]
fn sign_f32() {
    let a = tensor_f32(&[3.0, -1.0, 0.0], (3,));
    let c = a.sign().unwrap();
    // f32::signum(0.0) = 1.0 (IEEE 754)
    assert_close(&c.to_vec().unwrap(), &[1.0, -1.0, 1.0], 1e-7, 1e-7);
}

#[test]
fn leaky_relu_f32() {
    let a = tensor_f32(&[-1.0, 2.0], (2,));
    let c = a.leaky_relu(0.1).unwrap();
    // leaky_relu: x if x>0, 0.1*x otherwise
    let v = c.to_vec().unwrap();
    assert!((v[0] - (-0.1)).abs() < 1e-5); // -1 * 0.1
    assert!((v[1] - 2.0).abs() < 1e-5);
}

#[test]
fn pow_f32() {
    let a = tensor_f32(&[2.0, 3.0], (2,));
    let c = a.pow(2.0).unwrap();
    assert_close(&c.to_vec().unwrap(), &[4.0, 9.0], 1e-5, 1e-5);
}

#[test]
fn affine_f32() {
    let a = tensor_f32(&[1.0, 2.0], (2,));
    let c = a.affine(2.0, 1.0).unwrap(); // 2*x + 1
    assert_close(&c.to_vec().unwrap(), &[3.0, 5.0], 1e-7, 1e-7);
}

// ---- Maximum / Minimum ----

#[test]
fn maximum_f32() {
    let a = tensor_f32(&[1.0, 5.0, 3.0], (3,));
    let b = tensor_f32(&[4.0, 2.0, 6.0], (3,));
    let c = a.maximum(&b).unwrap();
    assert_close(&c.to_vec().unwrap(), &[4.0, 5.0, 6.0], 1e-7, 1e-7);
}

#[test]
fn minimum_f32() {
    let a = tensor_f32(&[1.0, 5.0, 3.0], (3,));
    let b = tensor_f32(&[4.0, 2.0, 6.0], (3,));
    let c = a.minimum(&b).unwrap();
    assert_close(&c.to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-7, 1e-7);
}

// ---- More cmp ops ----

#[test]
fn ne_f32() {
    let a = tensor_f32(&[1.0, 2.0, 3.0], (3,));
    let b = tensor_f32(&[1.0, 0.0, 3.0], (3,));
    let c = a.ne(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![false, true, false]);
}

#[test]
fn ge_f32() {
    let a = tensor_f32(&[1.0, 5.0, 3.0], (3,));
    let b = tensor_f32(&[1.0, 1.0, 3.0], (3,));
    let c = a.ge(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![true, true, true]);
}

#[test]
fn gt_f32() {
    let a = tensor_f32(&[1.0, 5.0, 3.0], (3,));
    let b = tensor_f32(&[2.0, 1.0, 3.0], (3,));
    let c = a.gt(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![false, true, false]);
}

#[test]
fn broadcast_cmp_f32() {
    let a = tensor_f32(&[1.0], (1,)); // scalar broadcast
    let b = tensor_f32(&[1.0, 2.0, 0.0], (3,));
    let c = a.broadcast_eq(&b).unwrap();
    assert_eq!(c.dims(), &[3]);
    assert_eq!(c.to_vec().unwrap(), vec![true, false, false]);
}

// ---- Int arithmetic ----

#[test]
fn add_i32() {
    let a = tensor_i32(&[10, 20, 30], (3,));
    let b = tensor_i32(&[1, 2, 3], (3,));
    let c = a.add(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![11, 22, 33]);
}

#[test]
fn neg_i32() {
    let a = tensor_i32(&[1, -2, 3], (3,));
    let c = a.neg().unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![-1, 2, -3]);
}

#[test]
fn abs_i32() {
    let a = tensor_i32(&[-1, 2, -3], (3,));
    let c = a.abs().unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![1, 2, 3]);
}

// ---- f64 precision ----

#[test]
fn f64_zeros() {
    let t = Tensor::<Cpu>::zeros((2, 2), FloatDType::F64).unwrap();
    let v = t.to_vec().unwrap();
    assert_eq!(v.len(), 4);
    assert!(v.iter().all(|&x| x == 0.0));
}

#[test]
fn f64_add() {
    let a = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0], (3,), FloatDType::F64).unwrap();
    let b = Tensor::<Cpu>::from_slice(&[4.0, 5.0, 6.0], (3,), FloatDType::F64).unwrap();
    let c = a.add(&b).unwrap();
    assert_close(&c.to_vec().unwrap(), &[5.0, 7.0, 9.0], 1e-15, 1e-15);
}

// ---- Scalar LHS ----

#[test]
fn add_scalar_lhs_f32() {
    let a = tensor_f32(&[1.0, 2.0], (2,));
    let c = a.add_scalar_lhs(10.0).unwrap();
    assert_close(&c.to_vec().unwrap(), &[11.0, 12.0], 1e-7, 1e-7);
}

#[test]
fn div_scalar_f32() {
    let a = tensor_f32(&[6.0, 8.0, 10.0], (3,));
    let c = a.div_scalar(2.0).unwrap();
    assert_close(&c.to_vec().unwrap(), &[3.0, 4.0, 5.0], 1e-7, 1e-7);
}

// ---- clamp ----

#[test]
fn clamp_both() {
    let a = tensor_f32(&[-1.0, 0.0, 2.0, 5.0, 10.0], (5,));
    let c = a.clamp(Some(0.0), Some(6.0)).unwrap();
    assert_close(&c.to_vec().unwrap(), &[0.0, 0.0, 2.0, 5.0, 6.0], 1e-7, 1e-7);
}

#[test]
fn clamp_min_only() {
    let a = tensor_f32(&[-1.0, 0.0, 2.0], (3,));
    let c = a.clamp(Some(0.0), None).unwrap();
    assert_close(&c.to_vec().unwrap(), &[0.0, 0.0, 2.0], 1e-7, 1e-7);
}

#[test]
fn clamp_max_only() {
    let a = tensor_f32(&[1.0, 5.0, 10.0], (3,));
    let c = a.clamp(None, Some(6.0)).unwrap();
    assert_close(&c.to_vec().unwrap(), &[1.0, 5.0, 6.0], 1e-7, 1e-7);
}
