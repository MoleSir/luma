//! Device-generic tests for the half-precision float dtypes (F16 / BF16).
//!
//! Every function takes the target `FloatDType` and a tolerance so a single
//! body can drive both f16 and bf16 on any backend.
#![allow(dead_code)]

use super::*;
use crate::Device;
use crate::dtype::FloatDType;

fn t<D: Device, S: Into<Shape>>(data: &[f64], shape: S, device: &D, dtype: FloatDType) -> Tensor<D> {
    Tensor::<D>::from_slice(data, shape, (device, dtype)).unwrap()
}

// ---- binary ----

pub fn test_half_binary(device: &impl Device, dtype: FloatDType, tol: f64) {
    let a = t(&[1.0, 2.0, 3.0], (3,), device, dtype);
    let b = t(&[4.0, 5.0, 6.0], (3,), device, dtype);
    assert_close(&a.add(&b).unwrap().to_vec().unwrap(), &[5.0, 7.0, 9.0], tol, tol);
    assert_close(&a.sub(&b).unwrap().to_vec().unwrap(), &[-3.0, -3.0, -3.0], tol, tol);
    assert_close(&a.mul(&b).unwrap().to_vec().unwrap(), &[4.0, 10.0, 18.0], tol, tol);
    assert_close(&b.div(&a).unwrap().to_vec().unwrap(), &[4.0, 2.5, 2.0], tol, tol);
}

// ---- unary ----

pub fn test_half_unary(device: &impl Device, dtype: FloatDType, tol: f64) {
    let a = t(&[1.0, -2.0, 3.0], (3,), device, dtype);
    assert_close(&a.neg().unwrap().to_vec().unwrap(), &[-1.0, 2.0, -3.0], tol, tol);
    assert_close(&a.abs().unwrap().to_vec().unwrap(), &[1.0, 2.0, 3.0], tol, tol);
    assert_close(&a.sqr().unwrap().to_vec().unwrap(), &[1.0, 4.0, 9.0], tol, tol);

    let r = t(&[-1.0, 0.0, 2.0], (3,), device, dtype);
    assert_close(&r.relu().unwrap().to_vec().unwrap(), &[0.0, 0.0, 2.0], tol, tol);

    let inv = t(&[2.0, 4.0], (2,), device, dtype);
    assert_close(&inv.recip().unwrap().to_vec().unwrap(), &[0.5, 0.25], tol, tol);

    let f = t(&[1.7, -1.3], (2,), device, dtype);
    assert_close(&f.floor().unwrap().to_vec().unwrap(), &[1.0, -2.0], tol, tol);
    assert_close(&f.sign().unwrap().to_vec().unwrap(), &[1.0, -1.0], tol, tol);

    let p = t(&[2.0, 3.0], (2,), device, dtype);
    assert_close(&p.pow(2.0).unwrap().to_vec().unwrap(), &[4.0, 9.0], tol, tol);
    assert_close(&p.affine(2.0, 3.0).unwrap().to_vec().unwrap(), &[7.0, 9.0], tol, tol);

    let s = t(&[4.0, 9.0], (2,), device, dtype);
    assert_close(&s.sqrt().unwrap().to_vec().unwrap(), &[2.0, 3.0], tol, tol);

    let e = t(&[0.0, 1.0], (2,), device, dtype);
    let ev = e.exp().unwrap().to_vec().unwrap();
    assert!((ev[0] - 1.0).abs() < tol);
    assert!((ev[1] - std::f64::consts::E).abs() < tol);

    let sig = t(&[0.0], (1,), device, dtype);
    assert!((sig.sigmoid().unwrap().to_vec().unwrap()[0] - 0.5).abs() < tol);
    assert!(sig.tanh().unwrap().to_vec().unwrap()[0].abs() < tol);

    let lr = t(&[-1.0, 2.0], (2,), device, dtype);
    assert_close(&lr.leaky_relu(0.1).unwrap().to_vec().unwrap(), &[-0.1, 2.0], tol, tol);

    let g = t(&[0.0, 1.0], (2,), device, dtype);
    let gv = g.gelu().unwrap().to_vec().unwrap();
    assert!(gv[0].abs() < tol);
    assert!((gv[1] - 0.8412).abs() < 5e-2);
}

// ---- cmp ----

pub fn test_half_cmp(device: &impl Device, dtype: FloatDType, _tol: f64) {
    let a = t(&[1.0, 5.0], (2,), device, dtype);
    let b = t(&[2.0, 3.0], (2,), device, dtype);
    assert_eq!(a.eq(&b).unwrap().to_vec().unwrap(), vec![false, false]);
    assert_eq!(a.lt(&b).unwrap().to_vec().unwrap(), vec![true, false]);
    assert_eq!(a.gt(&b).unwrap().to_vec().unwrap(), vec![false, true]);
    assert_eq!(a.le(&b).unwrap().to_vec().unwrap(), vec![true, false]);
    assert_eq!(a.ge(&b).unwrap().to_vec().unwrap(), vec![false, true]);
    assert_eq!(a.ne(&b).unwrap().to_vec().unwrap(), vec![true, true]);
}

// ---- scalar ----

pub fn test_half_scalar(device: &impl Device, dtype: FloatDType, tol: f64) {
    let a = t(&[1.0, 2.0], (2,), device, dtype);
    assert_close(&a.add_scalar(5.0).unwrap().to_vec().unwrap(), &[6.0, 7.0], tol, tol);
    assert_close(&a.sub_scalar(1.0).unwrap().to_vec().unwrap(), &[0.0, 1.0], tol, tol);
    assert_close(&a.sub_scalar_lhs(10.0).unwrap().to_vec().unwrap(), &[9.0, 8.0], tol, tol);
    assert_close(&a.mul_scalar(2.0).unwrap().to_vec().unwrap(), &[2.0, 4.0], tol, tol);
    assert_close(&a.div_scalar(2.0).unwrap().to_vec().unwrap(), &[0.5, 1.0], tol, tol);
    assert_close(&a.div_scalar_lhs(4.0).unwrap().to_vec().unwrap(), &[4.0, 2.0], tol, tol);
}

// ---- reduce ----

pub fn test_half_reduce(device: &impl Device, dtype: FloatDType, tol: f64) {
    let m = t(&[1.0, 2.0, 3.0, 4.0], (2, 2), device, dtype);
    assert_close(&m.sum(0usize).unwrap().to_vec().unwrap(), &[4.0, 6.0], tol, tol);
    assert_close(&m.max(1usize).unwrap().to_vec().unwrap(), &[2.0, 4.0], tol, tol);
    assert_close(&m.mean_all().unwrap().to_vec().unwrap(), &[2.5], tol, tol);

    let mx = t(&[1.5, 3.7, 2.1], (3,), device, dtype);
    assert!((mx.max_all().unwrap().to_scalar().unwrap() - 3.7).abs() < 0.05);
    // argmax must run through the half reduce path too.
    let am = mx.argmax(0usize).unwrap();
    assert_eq!(am.to_vec().unwrap(), vec![1]);
}

// ---- nn ----

pub fn test_half_softmax(device: &impl Device, dtype: FloatDType, _tol: f64) {
    let x = t(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], (2, 3), device, dtype);
    let out = x.softmax(1usize).unwrap();
    let v = out.to_vec().unwrap();
    for row in 0..2 {
        let s: f64 = v[row * 3..row * 3 + 3].iter().sum();
        assert!((s - 1.0).abs() < 0.05, "softmax row {row} sum {s}");
    }
    // argmax of the softmax row is the largest logit.
    assert_eq!(out.argmax(1usize).unwrap().to_vec().unwrap(), vec![2, 2]);
}

pub fn test_half_rms_norm(device: &impl Device, dtype: FloatDType, tol: f64) {
    let x = t(&[1.0, 2.0, 3.0, 4.0], (2, 2), device, dtype);
    let w = t(&[1.0, 1.0], (2,), device, dtype);
    let out = x.rms_norm(&w, 1e-5).unwrap();
    let v = out.to_vec().unwrap();
    // Row 0: x / sqrt(mean(x^2)+eps) = [1,2]/sqrt(2.5) since eps is negligible.
    let inv = 1.0 / (2.5f64 + 1e-5).sqrt();
    assert!((v[0] - 1.0 * inv).abs() < tol, "v0={}", v[0]);
    assert!((v[1] - 2.0 * inv).abs() < tol, "v1={}", v[1]);
}

pub fn test_half_matmul(device: &impl Device, dtype: FloatDType, _tol: f64) {
    let a = t(&[1.0, 2.0, 3.0, 4.0], (2, 2), device, dtype);
    let b = t(&[5.0, 6.0, 7.0, 8.0], (2, 2), device, dtype);
    let y = a.matmul(&b).unwrap();
    // cublas f16/bf16 accumulate in f32, so error tracks the input rounding.
    assert_close(&y.to_vec().unwrap(), &[19.0, 22.0, 43.0, 50.0], 0.5, 0.5);
}

// ---- cast ----

pub fn test_half_cast(device: &impl Device, dtype: FloatDType, tol: f64) {
    let a = t(&[1.0, 2.0, 3.0, 4.0], (4,), device, dtype);
    let wide = a.cast(FloatDType::F32).unwrap();
    assert_eq!(wide.dtype(), FloatDType::F32);
    assert_close(&wide.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0], tol, tol);

    let back = wide.cast(dtype).unwrap();
    assert_eq!(back.dtype(), dtype);
    assert_close(&back.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0], tol, tol);
}

// ---- grad ----

pub fn test_half_grad(device: &impl Device, dtype: FloatDType, tol: f64) {
    let x1 = t(&[2.0, 3.0], (2,), device, dtype);
    let x2 = t(&[4.0, 5.0], (2,), device, dtype);
    x1.set_requires_grad(true);
    x2.set_requires_grad(true);
    let y = x1.mul(&x2).unwrap();
    let loss = y.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    assert_eq!(grads.get(&x1).unwrap().dtype(), dtype);
    assert_close(&grads.get(&x1).unwrap().to_vec().unwrap(), &[4.0, 5.0], tol, tol);
    assert_close(&grads.get(&x2).unwrap().to_vec().unwrap(), &[2.0, 3.0], tol, tol);
}
