use luma_tensor::Device;
use super::*;

#[allow(dead_code)]
pub fn test_zeros_like_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    let z = t.zeros_like().unwrap();
    assert_eq!(z.dims(), t.dims());
    assert!(z.to_vec().unwrap().iter().all(|&x| x == 0.0));
}

#[allow(dead_code)]
pub fn test_ones_like_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[0.0, 0.0, 0.0, 0.0], (2, 2), device);
    let o = t.ones_like().unwrap();
    assert_eq!(o.dims(), t.dims());
    assert!(o.to_vec().unwrap().iter().all(|&x| (x - 1.0).abs() < 1e-5));
}

#[allow(dead_code)]
pub fn test_from_slice_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    assert_eq!(t.dims(), &[2, 3]);
    assert_close(&t.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_rand_like_shape(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let r = t.rand_like(0.0, 1.0).unwrap();
    assert_eq!(r.dims(), t.dims());
}

#[allow(dead_code)]
pub fn test_randn_like_shape(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let r = t.randn_like(0.0, 1.0).unwrap();
    assert_eq!(r.dims(), t.dims());
}

#[allow(dead_code)]
pub fn test_full_scalar(device: &impl Device) {
    let t = tensor_f32_dev(&[42.0], (1,), device);
    let v = t.to_vec().unwrap();
    assert!((v[0] - 42.0).abs() < 1e-5);
}
