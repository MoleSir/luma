#![allow(dead_code)]

use luma_tensor::Device;
use luma_tensor::dtype::{FloatDType, IntDType};
use super::*;

// ---- u8 ops ----

#[allow(dead_code)]
pub fn test_u8_construct(device: &impl Device) {
    let t = tensor_u8_dev(&[1, 2, 3, 4], (2, 2), device);
    assert_eq!(t.to_vec().unwrap(), vec![1i64, 2, 3, 4]);
}

#[allow(dead_code)]
pub fn test_u8_add(device: &impl Device) {
    let a = tensor_u8_dev(&[1, 2, 3], (3,), device);
    let b = tensor_u8_dev(&[4, 5, 6], (3,), device);
    let c = a.add(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![5i64, 7, 9]);
}

#[allow(dead_code)]
pub fn test_u8_sub(device: &impl Device) {
    let a = tensor_u8_dev(&[10, 20, 30], (3,), device);
    let b = tensor_u8_dev(&[1, 2, 3], (3,), device);
    let c = a.sub(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![9i64, 18, 27]);
}

#[allow(dead_code)]
pub fn test_u8_clamp(device: &impl Device) {
    let a = tensor_u8_dev(&[0, 100, 200, 50], (4,), device);
    let c = a.clamp(Some(10i64), Some(150i64)).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![10i64, 100, 150, 50]);
}

#[allow(dead_code)]
pub fn test_u8_cast_to_i32(device: &impl Device) {
    let a = tensor_u8_dev(&[1, 2, 255], (3,), device);
    let b = a.cast(IntDType::I32).unwrap();
    assert_eq!(b.to_vec().unwrap(), vec![1i64, 2, 255]);
}

#[allow(dead_code)]
pub fn test_u8_cast_to_f32(device: &impl Device) {
    let a = tensor_u8_dev(&[0, 128, 255], (3,), device);
    let b = a.cast_float(FloatDType::F32).unwrap();
    assert_close(&b.to_vec().unwrap(), &[0.0, 128.0, 255.0], 1e-5, 1e-5);
}

// ---- u32 ops ----

#[allow(dead_code)]
pub fn test_u32_construct(device: &impl Device) {
    let t = tensor_u32_dev(&[1, 2, 3, 4], (2, 2), device);
    assert_eq!(t.to_vec().unwrap(), vec![1i64, 2, 3, 4]);
}

#[allow(dead_code)]
pub fn test_u32_add(device: &impl Device) {
    let a = tensor_u32_dev(&[100, 200, 300], (3,), device);
    let b = tensor_u32_dev(&[50, 60, 70], (3,), device);
    let c = a.add(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![150i64, 260, 370]);
}

#[allow(dead_code)]
pub fn test_u32_mul(device: &impl Device) {
    let a = tensor_u32_dev(&[2, 3, 4], (3,), device);
    let b = tensor_u32_dev(&[5, 6, 7], (3,), device);
    let c = a.mul(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![10i64, 18, 28]);
}
