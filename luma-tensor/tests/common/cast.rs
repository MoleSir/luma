use luma_tensor::Device;
use luma_tensor::dtype::{BoolDType, FloatDType, IntDType};
use super::*;

fn tensor_f64_dev<D: luma_tensor::Device, S: Into<luma_tensor::Shape>>(data: &[f64], shape: S, device: &D) -> luma_tensor::Tensor<D> {
    luma_tensor::Tensor::<D>::from_slice(data, shape, (device, FloatDType::F64)).unwrap()
}

#[allow(dead_code)]
pub fn test_cast_f32_to_f64(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0], (3,), device);
    let b = a.cast(FloatDType::F64).unwrap();
    assert_close(&b.to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_cast_f32_to_i32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.7, 2.3, -3.8], (3,), device);
    let b = a.cast(IntDType::I32).unwrap();
    let v: Vec<i64> = b.to_vec().unwrap();
    assert_eq!(v, vec![1, 2, -3]);
}

#[allow(dead_code)]
pub fn test_cast_f32_to_bool(device: &impl Device) {
    let a = tensor_f32_dev(&[0.0, 1.0, -2.0, 0.0], (4,), device);
    let b = a.cast_bool(BoolDType::Bool).unwrap();
    assert_eq!(b.to_vec().unwrap(), vec![false, true, true, false]);
}

#[allow(dead_code)]
pub fn test_cast_i32_to_f32(device: &impl Device) {
    let a = tensor_i32_dev(&[1, 2, 3], (3,), device);
    let b = a.cast_float(FloatDType::F32).unwrap();
    assert_close(&b.to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_cast_bool_to_f32(device: &impl Device) {
    let a = tensor_bool_dev(&[true, false, true], (3,), device);
    let b = a.cast(FloatDType::F32).unwrap();
    assert_close(&b.to_vec().unwrap(), &[1.0, 0.0, 1.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_cast_bool_to_i32(device: &impl Device) {
    let a = tensor_bool_dev(&[true, false, true], (3,), device);
    let b = a.cast(IntDType::I32).unwrap();
    let v: Vec<i64> = b.to_vec().unwrap();
    assert_eq!(v, vec![1, 0, 1]);
}

#[allow(dead_code)]
pub fn test_cast_i32_to_bool(device: &impl Device) {
    let a = tensor_i32_dev(&[0, 1, -5, 0], (4,), device);
    let b = a.cast_bool(BoolDType::Bool).unwrap();
    assert_eq!(b.to_vec().unwrap(), vec![false, true, true, false]);
}

#[allow(dead_code)]
pub fn test_cast_i32_to_u32(device: &impl Device) {
    let a = tensor_i32_dev(&[10, 20], (2,), device);
    let b = a.cast(IntDType::U32).unwrap();
    let v: Vec<i64> = b.to_vec().unwrap();
    assert_eq!(v, vec![10, 20]);
}

#[allow(dead_code)]
pub fn test_cast_bool_to_bool(device: &impl Device) {
    let a = tensor_bool_dev(&[true, false], (2,), device);
    let b = a.cast_bool(BoolDType::Bool).unwrap();
    assert_eq!(b.to_vec().unwrap(), vec![true, false]);
}

#[allow(dead_code)]
pub fn test_cast_f64_to_f32(device: &impl Device) {
    let a = tensor_f64_dev(&[1.5, 2.5], (2,), device);
    let b = a.cast(FloatDType::F32).unwrap();
    assert_close(&b.to_vec().unwrap(), &[1.5, 2.5], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_f64_zeros(device: &impl Device) {
    let t = tensor_f64_dev(&[0.0f64, 0.0, 0.0, 0.0], (2, 2), device);
    assert!(t.to_vec().unwrap().iter().all(|&x| x == 0.0));
}

#[allow(dead_code)]
pub fn test_f64_add(device: &impl Device) {
    let a = tensor_f64_dev(&[1.0, 2.0, 3.0], (3,), device);
    let b = tensor_f64_dev(&[4.0, 5.0, 6.0], (3,), device);
    assert_close(&a.add(&b).unwrap().to_vec().unwrap(), &[5.0, 7.0, 9.0], 1e-10, 1e-10);
}
