//! Tests for type casting between kinds and precisions.

mod common;
use common::*;
use luma_tensor::dtype::{BoolDType, FloatDType, IntDType};
use luma_tensor::{Bool, Cpu, Int, Tensor};

// ---- Float -> Float (precision change within kind) ----

#[test]
fn cast_f32_to_f64() {
    let a = tensor_f32(&[1.0, 2.0, 3.0], (3,));
    let b = a.cast(FloatDType::F64).unwrap();
    assert_close(&b.to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-7, 1e-7);
}

#[test]
fn cast_f64_to_f32() {
    let a = Tensor::<Cpu>::from_slice(&[1.5, 2.5], (2,), FloatDType::F64).unwrap();
    let b = a.cast(FloatDType::F32).unwrap();
    assert_close(&b.to_vec().unwrap(), &[1.5, 2.5], 1e-5, 1e-5);
}

// ---- Float -> Int ----

#[test]
fn cast_f32_to_i32() {
    let a = tensor_f32(&[1.7, 2.3, -3.8], (3,));
    let b: Tensor<Cpu, Int> = a.cast(IntDType::I32).unwrap();
    assert_eq!(b.to_vec().unwrap(), vec![1, 2, -3]); // truncation via `as i32`
}

// ---- Float -> Bool ----

#[test]
fn cast_f32_to_bool() {
    let a = tensor_f32(&[0.0, 1.0, -2.0, 0.0], (4,));
    let b = a.cast_bool(BoolDType::Bool).unwrap();
    assert_eq!(b.to_vec().unwrap(), vec![false, true, true, false]);
}

// ---- Int -> Float ----

#[test]
fn cast_i32_to_f32() {
    let a = tensor_i32(&[1, 2, 3], (3,));
    let b = a.cast_float(FloatDType::F32).unwrap();
    assert_close(&b.to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-7, 1e-7);
}

// ---- Int -> Int ----

#[test]
fn cast_i32_to_u32() {
    let a = tensor_i32(&[10, 20], (2,));
    let b: Tensor<Cpu, Int> = a.cast(IntDType::U32).unwrap();
    assert_eq!(b.to_vec().unwrap(), vec![10, 20]);
}

// ---- Int -> Bool ----

#[test]
fn cast_i32_to_bool() {
    let a = tensor_i32(&[0, 1, -5, 0], (4,));
    let b = a.cast(BoolDType::Bool).unwrap();
    assert_eq!(b.to_vec().unwrap(), vec![false, true, true, false]);
}

// ---- Bool -> Float ----

#[test]
fn cast_bool_to_f32() {
    let a = Tensor::<Cpu, Bool>::from_slice(&[true, false, true], (3,), ()).unwrap();
    let b = a.cast(FloatDType::F32).unwrap();
    assert_close(&b.to_vec().unwrap(), &[1.0, 0.0, 1.0], 1e-7, 1e-7);
}

// ---- Bool -> Int ----

#[test]
fn cast_bool_to_i32() {
    let a = Tensor::<Cpu, Bool>::from_slice(&[true, false, true], (3,), ()).unwrap();
    let b: Tensor<Cpu, Int> = a.cast(IntDType::I32).unwrap();
    assert_eq!(b.to_vec().unwrap(), vec![1, 0, 1]);
}

// ---- Bool -> Bool ----

#[test]
fn cast_bool_to_bool() {
    let a = Tensor::<Cpu, Bool>::from_slice(&[true, false], (2,), ()).unwrap();
    let b = a.cast_bool(BoolDType::Bool).unwrap();
    assert_eq!(b.to_vec().unwrap(), vec![true, false]);
}

// ---- generic cast method ----

#[test]
fn generic_cast_float_to_int() {
    let a = tensor_f32(&[1.0, 2.0], (2,));
    let b: Tensor<Cpu, Int> = a.cast(IntDType::I32).unwrap();
    assert_eq!(b.to_vec().unwrap(), vec![1, 2]);
}

// ---- Grad through cast (Float -> Float) ----

#[test]
fn grad_cast_f32_to_f64() {
    let x = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0], (3,), FloatDType::F32).unwrap();
    x.set_requires_grad(true);

    let y = x.cast(FloatDType::F64).unwrap();
    let loss = y.sum_all().unwrap();
    let grads = loss.backward().unwrap();

    let g = grads.get(&x).unwrap().to_vec().unwrap();
    assert_close(&g, &[1.0, 1.0, 1.0], 1e-7, 1e-7);
}
