use luma_tensor::{Device, Shape, Tensor};
use luma_tensor::dtype::FloatDType;
use super::*;

#[allow(dead_code)]
pub fn test_matmul_2x2(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let b = tensor_f32_dev(&[5.0, 6.0, 7.0, 8.0], (2, 2), device);
    let y = a.matmul(&b).unwrap();
    assert_close(&y.to_vec().unwrap(), &[19.0, 22.0, 43.0, 50.0], 1e-3, 1e-3);
}

#[allow(dead_code)]
pub fn test_matmul_2x3_3x2(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    let b = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), device);
    let y = a.matmul(&b).unwrap();
    assert_eq!(y.dims(), &[2, 2]);
    assert_close(&y.to_vec().unwrap(), &[22.0, 28.0, 49.0, 64.0], 1e-3, 1e-3);
}

fn tensor_f64_dev<D: Device, S: Into<Shape>>(data: &[f64], shape: S, device: &D) -> Tensor<D> {
    Tensor::<D>::from_slice(data, shape, (device, FloatDType::F64)).unwrap()
}

#[allow(dead_code)]
pub fn test_matmul_f64(device: &impl Device) {
    let a = tensor_f64_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let b = tensor_f64_dev(&[5.0, 6.0, 7.0, 8.0], (2, 2), device);
    let y = a.matmul(&b).unwrap();
    assert_close(&y.to_vec().unwrap(), &[19.0, 22.0, 43.0, 50.0], 1e-8, 1e-8);
}

// ---- large tensor stress ----

#[allow(dead_code)]
pub fn test_large_matmul_f32(device: &impl Device) {
    let n = 100usize;
    let data: Vec<f64> = (0..(n * n)).map(|i| (i % 10) as f64).collect();
    let a = tensor_f32_dev(&data, (n, n), device);
    let b = tensor_f32_dev(&data, (n, n), device);
    let y = a.matmul(&b).unwrap();
    assert_eq!(y.dims(), &[n, n]);
    let v = y.to_vec().unwrap();
    assert!(v.iter().all(|&x| x.is_finite()), "matmul result not finite");
}
