#![allow(dead_code)]

use super::*;
use luma_tensor::Device;

/// Same-device `to_device` returns the identical handle (shared `Arc`, same
/// `TensorId`) — the O(1) no-op fast path generic code relies on.
#[allow(dead_code)]
pub fn test_to_device_identity_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let moved = t.to_device(device).unwrap();
    assert_eq!(moved.id(), t.id(), "same-device to_device must be a no-op");
    assert_eq!(moved.dtype(), t.dtype());
    assert_eq!(moved.shape(), t.shape());
}

#[allow(dead_code)]
pub fn test_to_device_identity_int(device: &impl Device) {
    let t = tensor_i32_dev(&[1, 2, 3], (3,), device);
    let moved = t.to_device(device).unwrap();
    assert_eq!(moved.id(), t.id());
    assert_eq!(moved.to_vec().unwrap(), vec![1, 2, 3]);
}

#[allow(dead_code)]
pub fn test_to_device_identity_bool(device: &impl Device) {
    let t = tensor_bool_dev(&[true, false, true], (3,), device);
    let moved = t.to_device(device).unwrap();
    assert_eq!(moved.id(), t.id());
    assert_eq!(moved.to_vec().unwrap(), vec![true, false, true]);
}

/// Same-device identity preserves the `requires_grad` flag and the graph.
#[allow(dead_code)]
pub fn test_to_device_identity_requires_grad(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0], (2,), device);
    t.set_requires_grad(true);
    let moved = t.to_device(device).unwrap();
    assert!(moved.requires_grad());
    assert_eq!(moved.id(), t.id());
}
