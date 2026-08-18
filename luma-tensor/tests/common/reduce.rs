use super::*;
use luma_tensor::{Device, Shape, Tensor};

#[allow(dead_code)]
pub fn test_sum_dim_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    let s = t.sum(1usize).unwrap(); // sum over cols → (2,)
    assert_eq!(s.dims(), &[2]);
    assert_close(&s.to_vec().unwrap(), &[6.0, 15.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_sum_keepdim_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let s = t.sum_keepdim(0usize).unwrap();
    assert_eq!(s.dims(), &[1, 2]);
    assert_close(&s.to_vec().unwrap(), &[4.0, 6.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_sum_all_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let s = t.sum_all().unwrap();
    assert_eq!(s.element_count(), 1);
    assert!((s.to_scalar().unwrap() - 10.0).abs() < 1e-5);
}

#[allow(dead_code)]
pub fn test_sum_dims_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let s = t.sum_dims([0usize, 1usize], false).unwrap();
    assert_eq!(s.element_count(), 1);
}

#[allow(dead_code)]
pub fn test_max_dim_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 5.0, 3.0, 2.0, 4.0, 6.0], (2, 3), device);
    let m = t.max(1usize).unwrap();
    assert_close(&m.to_vec().unwrap(), &[5.0, 6.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_max_keepdim_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let m = t.max_keepdim(0usize).unwrap();
    assert_eq!(m.dims(), &[1, 2]);
    assert_close(&m.to_vec().unwrap(), &[3.0, 4.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_max_all_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 5.0, 3.0, 2.0], (2, 2), device);
    let m = t.max_all().unwrap();
    assert!((m.to_scalar().unwrap() - 5.0).abs() < 1e-5);
}

#[allow(dead_code)]
pub fn test_min_dim_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 5.0, 3.0, 2.0, 4.0, 6.0], (2, 3), device);
    let m = t.min(1usize).unwrap();
    assert_close(&m.to_vec().unwrap(), &[1.0, 2.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_min_all_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 5.0, 3.0, 2.0], (2, 2), device);
    let m = t.min_all().unwrap();
    assert!((m.to_scalar().unwrap() - 1.0).abs() < 1e-5);
}

#[allow(dead_code)]
pub fn test_mean_dim_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    let m = t.mean(1usize).unwrap();
    assert_close(&m.to_vec().unwrap(), &[2.0, 5.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_mean_all_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let m = t.mean_all().unwrap();
    assert!((m.to_scalar().unwrap() - 2.5).abs() < 1e-5);
}

#[allow(dead_code)]
pub fn test_prod_dim_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let p = t.prod(0usize).unwrap();
    assert_eq!(p.dims(), &[2]);
    assert_close(&p.to_vec().unwrap(), &[3.0, 8.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_prod_all_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (4,), device);
    let p = t.prod_all().unwrap();
    assert!((p.to_scalar().unwrap() - 24.0).abs() < 1e-5);
}

#[allow(dead_code)]
pub fn test_argmax_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 5.0, 3.0, 2.0], (4,), device);
    let idx = t.argmax(0usize).unwrap();
    let v: Vec<i64> = idx.to_vec().unwrap();
    assert_eq!(v[0], 1);
}

#[allow(dead_code)]
pub fn test_argmin_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[3.0, 1.0, 5.0, 2.0], (4,), device);
    let idx = t.argmin(0usize).unwrap();
    let v: Vec<i64> = idx.to_vec().unwrap();
    assert_eq!(v[0], 1);
}

#[allow(dead_code)]
pub fn test_argmax_keepdim(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 5.0, 3.0, 2.0], (4,), device);
    let idx = t.argmax_keepdim(0usize).unwrap();
    assert_eq!(idx.dims(), &[1]);
}

#[allow(dead_code)]
pub fn test_var_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (4,), device);
    let v = t.var(0usize).unwrap();
    assert!((v.to_scalar().unwrap() - 1.25).abs() < 1e-4);
}

#[allow(dead_code)]
pub fn test_var_unbiased_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (4,), device);
    let v = t.var_unbiased(0usize).unwrap();
    assert!((v.to_scalar().unwrap() - 1.666666).abs() < 1e-4);
}

#[allow(dead_code)]
pub fn test_std_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (4,), device);
    let s = t.std(0usize).unwrap();
    assert!((s.to_scalar().unwrap() - 1.118034).abs() < 1e-4);
}

#[allow(dead_code)]
pub fn test_std_all_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let s = t.std_all().unwrap();
    let expected = 1.25f64.sqrt();
    assert!((s.to_scalar().unwrap() - expected).abs() < 1e-4);
}

#[allow(dead_code)]
pub fn test_logsumexp_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[0.0, 1.0, 2.0], (3,), device);
    let s = t.logsumexp(0usize).unwrap();
    let expected = (1.0f64 + 2.7182818 + 7.389056).ln();
    assert!((s.to_scalar().unwrap() - expected).abs() < 1e-4);
}

#[allow(dead_code)]
pub fn test_logsumexp_keepdim(device: &impl Device) {
    let t = tensor_f32_dev(&[0.0, 1.0, 2.0], (3,), device);
    let s = t.logsumexp_keepdim(0usize).unwrap();
    assert_eq!(s.dims(), &[1]);
}

// ---- i32 reduce ----

#[allow(dead_code)]
pub fn test_sum_i32(device: &impl Device) {
    let t = tensor_i32_dev(&[1, 2, 3, 4], (2, 2), device);
    let s = t.sum(0usize).unwrap();
    assert_eq!(s.to_vec().unwrap(), vec![4i64, 6]);
}

#[allow(dead_code)]
pub fn test_sum_all_i32(device: &impl Device) {
    let t = tensor_i32_dev(&[1, 2, 3, 4], (4,), device);
    let s = t.sum_all().unwrap();
    assert_eq!(s.to_vec().unwrap(), vec![10i64]);
}

#[allow(dead_code)]
pub fn test_max_dim_i32(device: &impl Device) {
    let t = tensor_i32_dev(&[1, 5, 3, 2, 4, 6], (2, 3), device);
    let m = t.max(1usize).unwrap();
    assert_eq!(m.to_vec().unwrap(), vec![5i64, 6]);
}

#[allow(dead_code)]
pub fn test_min_dim_i32(device: &impl Device) {
    let t = tensor_i32_dev(&[1, 5, 3, 2, 4, 6], (2, 3), device);
    let m = t.min(1usize).unwrap();
    assert_eq!(m.to_vec().unwrap(), vec![1i64, 2]);
}

// ---- f64 ops ----

#[allow(dead_code)]
pub fn test_sum_f64(device: &impl Device) {
    let t = tensor_f64_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let s = t.sum(0usize).unwrap();
    assert_close(&s.to_vec().unwrap(), &[4.0, 6.0], 1e-10, 1e-10);
}

#[allow(dead_code)]
pub fn test_mean_f64(device: &impl Device) {
    let t = tensor_f64_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let m = t.mean_all().unwrap();
    assert!((m.to_scalar().unwrap() - 2.5).abs() < 1e-10);
}

// ---- u8 reduce ----

#[allow(dead_code)]
pub fn test_sum_u8(device: &impl Device) {
    let t = tensor_u8_dev(&[1, 2, 3, 4], (2, 2), device);
    let s = t.sum(0usize).unwrap();
    assert_eq!(s.to_vec().unwrap(), vec![4i64, 6]);
}

#[allow(dead_code)]
pub fn test_max_u8(device: &impl Device) {
    let t = tensor_u8_dev(&[10, 5, 30, 20], (4,), device);
    let m = t.max_all().unwrap();
    assert_eq!(m.to_vec().unwrap(), vec![30i64]);
}

// ---- u32 reduce ----

#[allow(dead_code)]
pub fn test_sum_u32(device: &impl Device) {
    let t = tensor_u32_dev(&[100, 200, 300, 400], (2, 2), device);
    let s = t.sum_all().unwrap();
    assert_eq!(s.to_vec().unwrap(), vec![1000i64]);
}

#[allow(dead_code)]
pub fn test_min_u32(device: &impl Device) {
    let t = tensor_u32_dev(&[100, 5, 300, 50], (4,), device);
    let m = t.min_all().unwrap();
    assert_eq!(m.to_vec().unwrap(), vec![5i64]);
}

// ---- large tensor stress ----

#[allow(dead_code)]
pub fn test_large_sum_f32(device: &impl Device) {
    let n = 100_000usize;
    let data: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let t = tensor_f32_dev(&data, (n,), device);
    let s = t.sum_all().unwrap();
    let expected = n as f64 * (n as f64 + 1.0) / 2.0;
    assert!((s.to_scalar().unwrap() - expected).abs() < (expected * 1e-3));
}

fn tensor_f64_dev<D: Device, S: Into<Shape>>(data: &[f64], shape: S, device: &D) -> Tensor<D> {
    use luma_tensor::dtype::FloatDType;
    Tensor::<D>::from_slice(data, shape, (device, FloatDType::F64)).unwrap()
}
