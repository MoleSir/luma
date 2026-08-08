#![allow(dead_code)]

use luma_tensor::Device;
use super::*;

#[allow(dead_code)]
pub fn test_softmax_dim0(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], (2, 3), device);
    let out = t.softmax(1usize).unwrap();
    assert_eq!(out.dims(), &[2, 3]);
    let v = out.to_vec().unwrap();
    for row in v.chunks(3) {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax row sum {} != 1", sum);
    }
}

#[allow(dead_code)]
pub fn test_softmax_dim1(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], (2, 3), device);
    let out = t.softmax(0usize).unwrap();
    assert_eq!(out.dims(), &[2, 3]);
    let v = out.to_vec().unwrap();
    assert!((v[0] + v[3] - 1.0).abs() < 1e-5, "col 0 sum {}", v[0] + v[3]);
    assert!((v[1] + v[4] - 1.0).abs() < 1e-5, "col 1 sum {}", v[1] + v[4]);
    assert!((v[2] + v[5] - 1.0).abs() < 1e-5, "col 2 sum {}", v[2] + v[5]);
}

#[allow(dead_code)]
pub fn test_softmax_numerical_stability(device: &impl Device) {
    let t = tensor_f32_dev(&[1000.0, 1000.0, 1000.0], (3,), device);
    let out = t.softmax(0usize).unwrap();
    let v = out.to_vec().unwrap();
    let expected = 1.0 / 3.0;
    for &x in &v {
        assert!((x - expected).abs() < 1e-4, "large values: {} vs {}", x, expected);
    }
}

#[allow(dead_code)]
pub fn test_softmax_grad(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (4,), device);
    t.set_requires_grad(true);
    let out = t.softmax(0usize).unwrap();
    let loss = out.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let gv = grads.get(&t).unwrap().to_vec().unwrap();
    assert_close(&gv, &[0.0, 0.0, 0.0, 0.0], 1e-4, 1e-4);
}

#[allow(dead_code)]
pub fn test_rms_norm_f32(device: &impl Device) {
    let x = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    let w = tensor_f32_dev(&[1.0, 1.0, 1.0], (3,), device);
    let out = x.rms_norm(&w, 1e-5).unwrap();
    assert_eq!(out.dims(), &[2, 3]);
    let v = out.to_vec().unwrap();
    for row in v.chunks(3) {
        let mean_sq = row.iter().map(|&x| x * x).sum::<f64>() / 3.0;
        assert!((mean_sq - 1.0).abs() < 0.1, "row mean_sq {}", mean_sq);
    }
}

#[allow(dead_code)]
pub fn test_rms_norm_weighted(device: &impl Device) {
    let x = tensor_f32_dev(&[1.0, 2.0, 3.0], (3,), device);
    let w = tensor_f32_dev(&[2.0, 0.0, 1.0], (3,), device);
    let out = x.rms_norm(&w, 0.0).unwrap();
    let v = out.to_vec().unwrap();
    let inv_rms = 1.0_f64 / (14.0_f64 / 3.0_f64).sqrt();
    let e0 = 1.0 * inv_rms * 2.0;
    let e2 = 3.0 * inv_rms * 1.0;
    assert_close(&[v[0]], &[e0], 1e-4, 1e-4);
    assert!((v[1] - 0.0).abs() < 1e-5, "expected 0, got {}", v[1]);
    assert_close(&[v[2]], &[e2], 1e-4, 1e-4);
}

#[allow(dead_code)]
pub fn test_large_softmax(device: &impl Device) {
    let n = 5000usize;
    let data: Vec<f64> = (0..n).map(|i| (i as f64 % 100.0) - 50.0).collect();
    let t = tensor_f32_dev(&data, (n,), device);
    let out = t.softmax(0usize).unwrap();
    let v = out.to_vec().unwrap();
    let sum: f64 = v.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "softmax sum for {} elems: {}", n, sum);
    assert!(v.iter().all(|&x| x >= 0.0 && x <= 1.0));
}
