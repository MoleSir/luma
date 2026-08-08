use luma_tensor::Device;
use super::*;

#[allow(dead_code)]
pub fn test_display_scalar(device: &impl Device) {
    let vals = vec![3.14];
    let t = tensor_f32_dev(&vals, (1,), device);
    let s = format!("{}", t);
    assert!(!s.is_empty());
    assert!(s.contains('3'));
}

#[allow(dead_code)]
pub fn test_display_1d(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0], (3,), device);
    let s = format!("{}", t);
    assert!(s.contains('['));
}
