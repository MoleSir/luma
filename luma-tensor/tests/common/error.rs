#![allow(dead_code)]

use luma_tensor::Device;
use super::*;

#[allow(dead_code)]
pub fn test_binary_shape_mismatch(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0], (3,), device);
    let b = tensor_f32_dev(&[1.0, 2.0], (2,), device);
    assert!(a.add(&b).is_err());
    assert!(a.sub(&b).is_err());
    assert!(a.mul(&b).is_err());
}

#[allow(dead_code)]
pub fn test_matmul_shape_mismatch(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let b = tensor_f32_dev(&[1.0, 2.0, 3.0], (3,), device);
    assert!(a.matmul(&b).is_err());
}

#[allow(dead_code)]
pub fn test_narrow_out_of_range(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    assert!(a.narrow(0, 3, 1).is_err());
    assert!(a.narrow(1, 0, 5).is_err());
}

#[allow(dead_code)]
pub fn test_dim_out_of_range(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    assert!(a.sum(2usize).is_err());
    assert!(a.max(5usize).is_err());
}

#[allow(dead_code)]
pub fn test_allclose_shape_mismatch(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0], (2,), device);
    let b = tensor_f32_dev(&[1.0, 2.0, 3.0], (3,), device);
    assert!(!a.allclose(&b, 1e-5, 1e-5).unwrap());
}

#[allow(dead_code)]
pub fn test_f64_to_f32_add(device: &impl Device) {
    let a = tensor_f64_dev(&[1.0, 2.0], (2,), device);
    let b = tensor_f32_dev(&[3.0, 4.0], (2,), device);
    assert!(a.add(&b).is_err());
}

#[allow(dead_code)]
pub fn test_reshape_wrong_elements(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    assert!(a.reshape((3, 2)).is_err());
}
