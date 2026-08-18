#![allow(dead_code)]

use super::*;
use luma_tensor::Device;

// ---- transpose + elementwise ----

#[allow(dead_code)]
pub fn test_transpose_add(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    let b = tensor_f32_dev(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], (2, 3), device);
    let at = a.transpose(0usize, 1usize).unwrap();
    let bt = b.transpose(0usize, 1usize).unwrap();
    assert!(!at.is_contiguous());
    let c = at.add(&bt).unwrap();
    assert_eq!(c.dims(), &[3, 2]);
    let v = c.to_vec().unwrap();
    assert_close(&v, &[11.0, 44.0, 22.0, 55.0, 33.0, 66.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_transpose_sub(device: &impl Device) {
    let a = tensor_f32_dev(&[5.0, 6.0, 7.0, 8.0], (2, 2), device);
    let at = a.transpose(0, 1).unwrap();
    let b = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let bt = b.transpose(0, 1).unwrap();
    let c = at.sub(&bt).unwrap();
    assert_close(&c.to_vec().unwrap(), &[4.0, 4.0, 4.0, 4.0], 1e-5, 1e-5);
}

// ---- transpose + reduce ----

#[allow(dead_code)]
pub fn test_transpose_sum(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    let at = a.transpose(0, 1).unwrap();
    assert!(!at.is_contiguous());
    let s = at.sum(0usize).unwrap();
    assert_close(&s.to_vec().unwrap(), &[6.0, 15.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_transpose_max(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 5.0, 2.0, 3.0, 1.0, 4.0], (3, 2), device);
    let at = a.transpose(0, 1).unwrap();
    let m = at.max_all().unwrap();
    assert!((m.to_scalar().unwrap() - 5.0).abs() < 1e-5);
}

// ---- permute + binary ----

#[allow(dead_code)]
pub fn test_permute_add(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let ap = a.permute([1, 0]).unwrap();
    assert_eq!(ap.dims(), &[2, 2]);
    let b = tensor_f32_dev(&[5.0, 6.0, 7.0, 8.0], (2, 2), device);
    let bp = b.permute([1, 0]).unwrap();
    let c = ap.add(&bp).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    let v = c.to_vec().unwrap();
    assert_close(&v, &[6.0, 10.0, 8.0, 12.0], 1e-5, 1e-5);
}

// ---- slice + reduce ----

#[allow(dead_code)]
pub fn test_slice_sum(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    let s = a.slice(1, 0, 2, 1).unwrap();
    let r = s.sum(1usize).unwrap();
    assert_close(&r.to_vec().unwrap(), &[3.0, 9.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_narrow_add(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    let b = tensor_f32_dev(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], (2, 3), device);
    let an = a.narrow(0, 0, 1).unwrap();
    let bn = b.narrow(0, 0, 1).unwrap();
    assert_eq!(an.dims(), &[1, 3]);
    let c = an.add(&bn).unwrap();
    assert_close(&c.to_vec().unwrap(), &[11.0, 22.0, 33.0], 1e-5, 1e-5);
}

// ---- contiguous after permute + matmul ----

#[allow(dead_code)]
pub fn test_permute_contiguous_add(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let ap = a.permute([1, 0]).unwrap();
    assert!(!ap.is_contiguous());
    let ac = ap.contiguous().unwrap();
    assert!(ac.is_contiguous());
    let b = tensor_f32_dev(&[10.0, 20.0, 30.0, 40.0], (2, 2), device);
    let c = ac.add(&b).unwrap();
    assert_close(&c.to_vec().unwrap(), &[11.0, 23.0, 32.0, 44.0], 1e-5, 1e-5);
}

// ---- broadcast + reduce ----

#[allow(dead_code)]
pub fn test_broadcast_sum(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0], (1, 3), device);
    let b = tensor_f32_dev(&[10.0, 20.0], (2, 1), device);
    let c = a.broadcast_add(&b).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    let s = c.sum(0usize).unwrap();
    assert_close(&s.to_vec().unwrap(), &[32.0, 34.0, 36.0], 1e-5, 1e-5);
}
