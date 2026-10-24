#![allow(dead_code)]

use super::*;
use crate::Device;
use crate::dtype::FloatDType;
use crate::ops::NnOpDefault;

fn tensor_float_dev<D: Device, S: Into<Shape>>(data: &[f64], shape: S, device: &D, dtype: FloatDType) -> Tensor<D> {
    Tensor::<D>::from_slice(data, shape, (device, dtype)).unwrap()
}

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
    let out = x.rms_norm(Some(&w), None, 1e-5).unwrap();
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
    let out = x.rms_norm(Some(&w), None, 0.0).unwrap();
    let v = out.to_vec().unwrap();
    let inv_rms = 1.0_f64 / (14.0_f64 / 3.0_f64).sqrt();
    let e0 = 1.0 * inv_rms * 2.0;
    let e2 = 3.0 * inv_rms * 1.0;
    assert_close(&[v[0]], &[e0], 1e-4, 1e-4);
    assert!((v[1] - 0.0).abs() < 1e-5, "expected 0, got {}", v[1]);
    assert_close(&[v[2]], &[e2], 1e-4, 1e-4);
}

#[allow(dead_code)]
pub fn test_rms_norm_bias(device: &impl Device, dtype: FloatDType, tol: f64) {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let wdata = [0.5, 1.0, 2.0];
    let bdata = [0.1, -0.2, 0.3];

    let x = tensor_float_dev(&data, (2, 3), device, dtype);
    let w = tensor_float_dev(&wdata, (3,), device, dtype);
    let b = tensor_float_dev(&bdata, (3,), device, dtype);
    x.set_requires_grad(true);
    w.set_requires_grad(true);
    b.set_requires_grad(true);
    let loss = x.rms_norm(Some(&w), Some(&b), 1e-5).unwrap().sum_all().unwrap();
    let grads = loss.backward().unwrap();

    let x2 = tensor_float_dev(&data, (2, 3), device, dtype);
    let w2 = tensor_float_dev(&wdata, (3,), device, dtype);
    let b2 = tensor_float_dev(&bdata, (3,), device, dtype);
    x2.set_requires_grad(true);
    w2.set_requires_grad(true);
    b2.set_requires_grad(true);
    let ms = x2.sqr().unwrap().mean_keepdim(1).unwrap();
    let rms = (ms + 1e-5).sqrt().unwrap();
    let out = x2.broadcast_div(&rms).unwrap().broadcast_mul(&w2).unwrap().broadcast_add(&b2).unwrap();
    let loss2 = out.sum_all().unwrap();
    let grads2 = loss2.backward().unwrap();

    for (t1, t2) in [(&x, &x2), (&w, &w2), (&b, &b2)] {
        assert_close(&grads.get(t1).unwrap().to_vec().unwrap(), &grads2.get(t2).unwrap().to_vec().unwrap(), tol, tol);
    }
}

#[allow(dead_code)]
pub fn test_rms_norm_grad(device: &impl Device, dtype: FloatDType, tol: f64) {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let wdata = [0.5, 1.0, 2.0];

    // Fused path (NnProvide override / composition fallback).
    let x = tensor_float_dev(&data, (2, 3), device, dtype);
    let w = tensor_float_dev(&wdata, (3,), device, dtype);
    x.set_requires_grad(true);
    w.set_requires_grad(true);
    let loss = x.rms_norm(Some(&w), None, 1e-5).unwrap().sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let gx = grads.get(&x).unwrap().to_vec().unwrap();
    let gw = grads.get(&w).unwrap().to_vec().unwrap();

    // Reference: the primitive composition, differentiated by autograd.
    let x2 = tensor_float_dev(&data, (2, 3), device, dtype);
    let w2 = tensor_float_dev(&wdata, (3,), device, dtype);
    x2.set_requires_grad(true);
    w2.set_requires_grad(true);
    let ms = x2.sqr().unwrap().mean_keepdim(1).unwrap();
    let rms = (ms + 1e-5).sqrt().unwrap();
    let out = x2.broadcast_div(&rms).unwrap().broadcast_mul(&w2).unwrap();
    let loss2 = out.sum_all().unwrap();
    let grads2 = loss2.backward().unwrap();
    let gx2 = grads2.get(&x2).unwrap().to_vec().unwrap();
    let gw2 = grads2.get(&w2).unwrap().to_vec().unwrap();

    assert_close(&gx, &gx2, tol, tol);
    assert_close(&gw, &gw2, tol, tol);
}

#[allow(dead_code)]
pub fn test_default_softmax(device: &impl Device, dtype: FloatDType, tol: f64) {
    let data = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0];

    let x_default = tensor_float_dev(&data, (2, 3), device, dtype);
    let x_override = tensor_float_dev(&data, (2, 3), device, dtype);
    let out_default = NnOpDefault::softmax_forward(&x_default, 1, true).unwrap();
    let out_override = x_override.softmax(1usize).unwrap();
    assert_close(&out_default.to_vec().unwrap(), &out_override.to_vec().unwrap(), tol, tol);

    x_default.set_requires_grad(true);
    x_override.set_requires_grad(true);

    let g_default = NnOpDefault::softmax_forward(&x_default, 1, true).unwrap().sum_all().unwrap().backward().unwrap();
    let g_override = x_override.softmax(1usize).unwrap().sum_all().unwrap().backward().unwrap();
    assert_close(&g_default.get(&x_default).unwrap().to_vec().unwrap(), &g_override.get(&x_override).unwrap().to_vec().unwrap(), tol, tol);
}

#[allow(dead_code)]
pub fn test_log_softmax(device: &impl Device, dtype: FloatDType, tol: f64) {
    let data = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    let t = tensor_float_dev(&data, (2, 3), device, dtype);
    let out = t.log_softmax(1usize).unwrap();
    let expected = t.broadcast_sub(&t.logsumexp_keepdim(1).unwrap()).unwrap();
    assert_close(&out.to_vec().unwrap(), &expected.to_vec().unwrap(), tol, tol);
    assert_eq!(out.dims(), &[2, 3]);

    let x = tensor_float_dev(&data, (2, 3), device, dtype);
    x.set_requires_grad(true);
    let loss = x.log_softmax(1usize).unwrap().sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let g = grads.get(&x).unwrap().to_vec().unwrap();
    let sm = x.softmax(1usize).unwrap().to_vec().unwrap();
    let expected_grad: Vec<f64> = sm.iter().map(|v| 1.0 - 3.0 * v).collect();
    assert_close(&g, &expected_grad, tol, tol);
}

#[allow(dead_code)]
pub fn test_default_rms_norm(device: &impl Device, dtype: FloatDType, tol: f64) {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let wdata = [0.5, 1.0, 2.0];
    let eps = 1e-5;

    let x_default = tensor_float_dev(&data, (2, 3), device, dtype);
    let w_default = tensor_float_dev(&wdata, (3,), device, dtype);
    let x_override = tensor_float_dev(&data, (2, 3), device, dtype);
    let w_override = tensor_float_dev(&wdata, (3,), device, dtype);

    let out_default = NnOpDefault::rms_norm_forward(&x_default, Some(&w_default), None, eps).unwrap();
    let out_override = x_override.rms_norm(Some(&w_override), None, eps).unwrap();
    assert_close(&out_default.to_vec().unwrap(), &out_override.to_vec().unwrap(), tol, tol);

    x_default.set_requires_grad(true);
    w_default.set_requires_grad(true);
    x_override.set_requires_grad(true);
    w_override.set_requires_grad(true);

    let g_default = NnOpDefault::rms_norm_forward(&x_default, Some(&w_default), None, eps).unwrap().sum_all().unwrap().backward().unwrap();
    let g_override = x_override.rms_norm(Some(&w_override), None, eps).unwrap().sum_all().unwrap().backward().unwrap();
    assert_close(&g_default.get(&x_default).unwrap().to_vec().unwrap(), &g_override.get(&x_override).unwrap().to_vec().unwrap(), tol, tol);
    assert_close(&g_default.get(&w_default).unwrap().to_vec().unwrap(), &g_override.get(&w_override).unwrap().to_vec().unwrap(), tol, tol);
}

#[allow(dead_code)]
pub fn test_layer_norm(device: &impl Device, dtype: FloatDType, tol: f64) {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let wdata = [0.5, 1.0, 2.0];
    let bdata = [0.1, -0.2, 0.3];
    let eps = 1e-5;

    let x = tensor_float_dev(&data, (2, 3), device, dtype);
    let w = tensor_float_dev(&wdata, (3,), device, dtype);
    let b = tensor_float_dev(&bdata, (3,), device, dtype);
    x.set_requires_grad(true);
    w.set_requires_grad(true);
    b.set_requires_grad(true);
    let y = NnOpDefault::layer_norm_forward(&x, Some(&w), Some(&b), eps).unwrap();
    let grads = y.sum_all().unwrap().backward().unwrap();

    let (dx, dweight, dbias) = NnOpDefault::layer_norm_backward(&x, Some(&w), Some(&b), &y, &y.ones_like().unwrap(), eps).unwrap();
    assert_close(&dx.to_vec().unwrap(), &grads.get(&x).unwrap().to_vec().unwrap(), tol, tol);
    assert_close(&dweight.unwrap().to_vec().unwrap(), &grads.get(&w).unwrap().to_vec().unwrap(), tol, tol);
    assert_close(&dbias.unwrap().to_vec().unwrap(), &grads.get(&b).unwrap().to_vec().unwrap(), tol, tol);
}

#[allow(dead_code)]
pub fn test_group_norm(device: &impl Device, dtype: FloatDType, tol: f64) {
    let data: Vec<f64> = (1..=24).map(|v| v as f64).collect();
    let wdata = [0.5, 1.0, 2.0, -1.0];
    let bdata = [0.1, -0.2, 0.3, 0.4];
    let eps = 1e-5;

    let x = tensor_float_dev(&data, (2, 4, 3), device, dtype);
    let w = tensor_float_dev(&wdata, (4,), device, dtype);
    let b = tensor_float_dev(&bdata, (4,), device, dtype);
    x.set_requires_grad(true);
    w.set_requires_grad(true);
    b.set_requires_grad(true);
    let y = NnOpDefault::group_norm_forward(&x, 2, Some(&w), Some(&b), eps).unwrap();
    assert_eq!(y.dims(), &[2, 4, 3]);
    let grads = y.sum_all().unwrap().backward().unwrap();

    let (dx, dweight, dbias) = NnOpDefault::group_norm_backward(&x, 2, Some(&w), Some(&b), &y, &y.ones_like().unwrap(), eps).unwrap();
    assert_close(&dx.to_vec().unwrap(), &grads.get(&x).unwrap().to_vec().unwrap(), tol, tol);
    assert_close(&dweight.unwrap().to_vec().unwrap(), &grads.get(&w).unwrap().to_vec().unwrap(), tol, tol);
    assert_close(&dbias.unwrap().to_vec().unwrap(), &grads.get(&b).unwrap().to_vec().unwrap(), tol, tol);
}

#[allow(dead_code)]
pub fn test_conv2d(device: &impl Device, dtype: FloatDType, tol: f64) {
    // 3x3 input, 2x2 all-ones kernel, stride 1, no padding.
    let x = tensor_float_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], (1, 1, 3, 3), device, dtype);
    let w = tensor_float_dev(&[1.0, 1.0, 1.0, 1.0], (1, 1, 2, 2), device, dtype);
    let y = NnOpDefault::conv2d_forward(&x, &w, None, (1, 1), (0, 0), (1, 1)).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    assert_close(&y.to_vec().unwrap(), &[12.0, 16.0, 24.0, 28.0], tol, tol);

    x.set_requires_grad(true);
    w.set_requires_grad(true);
    let y = NnOpDefault::conv2d_forward(&x, &w, None, (1, 1), (0, 0), (1, 1)).unwrap();
    let grads = y.sum_all().unwrap().backward().unwrap();
    let (dx, dweight, dbias) = NnOpDefault::conv2d_backward(&x, &w, None, &y, &y.ones_like().unwrap(), (1, 1), (0, 0), (1, 1)).unwrap();
    assert!(dbias.is_none());
    assert_close(&dx.to_vec().unwrap(), &grads.get(&x).unwrap().to_vec().unwrap(), tol, tol);
    assert_close(&dweight.to_vec().unwrap(), &grads.get(&w).unwrap().to_vec().unwrap(), tol, tol);

    // 4x4 input, stride 2, padding 1.
    let data: Vec<f64> = (1..=16).map(|v| v as f64).collect();
    let x = tensor_float_dev(&data, (1, 1, 4, 4), device, dtype);
    let w = tensor_float_dev(&[1.0, 2.0, 3.0, 4.0], (1, 1, 2, 2), device, dtype);
    x.set_requires_grad(true);
    w.set_requires_grad(true);
    let y = NnOpDefault::conv2d_forward(&x, &w, None, (2, 2), (1, 1), (1, 1)).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3, 3]);
    let grads = y.sum_all().unwrap().backward().unwrap();
    let (dx, dweight, _) = NnOpDefault::conv2d_backward(&x, &w, None, &y, &y.ones_like().unwrap(), (2, 2), (1, 1), (1, 1)).unwrap();
    assert_close(&dx.to_vec().unwrap(), &grads.get(&x).unwrap().to_vec().unwrap(), tol, tol);
    assert_close(&dweight.to_vec().unwrap(), &grads.get(&w).unwrap().to_vec().unwrap(), tol, tol);
}

#[allow(dead_code)]
pub fn test_attention(device: &impl Device, dtype: FloatDType, tol: f64) {
    let qdata = [1.0, 0.0, 0.0, 1.0];
    let kdata = [1.0, 0.0, 0.0, 1.0];
    let vdata = [0.0, 1.0, 2.0, 0.0];

    let q = tensor_float_dev(&qdata, (1, 2, 1, 2), device, dtype);
    let k = tensor_float_dev(&kdata, (1, 2, 1, 2), device, dtype);
    let v = tensor_float_dev(&vdata, (1, 2, 1, 2), device, dtype);
    let y = NnOpDefault::scaled_dot_product_attention_forward(&q, &k, &v, None, false, 0).unwrap();
    assert_eq!(y.dims(), &[1, 2, 1, 2]);

    let s = 1.0 / 2.0f64.sqrt();
    let a = s.exp();
    let e0 = a / (a + 1.0);
    let e1 = 1.0 / (a + 1.0);
    assert_close(&y.to_vec().unwrap(), &[2.0 * e1, e0, 2.0 * e0, e1], tol, tol);

    for causal in [false, true] {
        let q = tensor_float_dev(&qdata, (1, 2, 1, 2), device, dtype);
        let k = tensor_float_dev(&kdata, (1, 2, 1, 2), device, dtype);
        let v = tensor_float_dev(&vdata, (1, 2, 1, 2), device, dtype);
        q.set_requires_grad(true);
        k.set_requires_grad(true);
        v.set_requires_grad(true);
        let y = NnOpDefault::scaled_dot_product_attention_forward(&q, &k, &v, None, causal, 0).unwrap();
        let grads = y.sum_all().unwrap().backward().unwrap();
        let (dq, dk, dv) =
            NnOpDefault::scaled_dot_product_attention_backward(&q, &k, &v, &y, &y.ones_like().unwrap(), None, causal, 0).unwrap();
        assert_close(&dq.to_vec().unwrap(), &grads.get(&q).unwrap().to_vec().unwrap(), tol, tol);
        assert_close(&dk.to_vec().unwrap(), &grads.get(&k).unwrap().to_vec().unwrap(), tol, tol);
        assert_close(&dv.to_vec().unwrap(), &grads.get(&v).unwrap().to_vec().unwrap(), tol, tol);
    }

    // GQA: 2 query heads share 1 KV head.
    let qdata = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, -1.0];
    let q = tensor_float_dev(&qdata, (1, 2, 2, 2), device, dtype);
    let k = tensor_float_dev(&kdata, (1, 2, 1, 2), device, dtype);
    let v = tensor_float_dev(&vdata, (1, 2, 1, 2), device, dtype);
    q.set_requires_grad(true);
    k.set_requires_grad(true);
    v.set_requires_grad(true);
    let y = NnOpDefault::scaled_dot_product_attention_forward(&q, &k, &v, None, false, 0).unwrap();
    let grads = y.sum_all().unwrap().backward().unwrap();
    let (dq, dk, dv) = NnOpDefault::scaled_dot_product_attention_backward(&q, &k, &v, &y, &y.ones_like().unwrap(), None, false, 0).unwrap();
    assert_close(&dq.to_vec().unwrap(), &grads.get(&q).unwrap().to_vec().unwrap(), tol, tol);
    assert_close(&dk.to_vec().unwrap(), &grads.get(&k).unwrap().to_vec().unwrap(), tol, tol);
    assert_close(&dv.to_vec().unwrap(), &grads.get(&v).unwrap().to_vec().unwrap(), tol, tol);
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

// ---- cross entropy (manual chain: softmax → ln → gather → neg → mean_all) ----

#[allow(dead_code)]
pub fn test_cross_entropy_chain_f32(device: &impl Device) {
    // (batch=4, classes=3) logits
    let logits = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], (4, 3), device);
    // target class indices: shape (4, 1)
    let targets = tensor_i32_dev(&[0, 1, 2, 0], (4, 1), device);

    // step 1: softmax on last dim
    let softmax_out = logits.softmax(1usize).unwrap();
    assert_eq!(softmax_out.dims(), &[4, 3]);
    let sm = softmax_out.to_vec().unwrap();
    for row in sm.chunks(3) {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax row sum {}", sum);
    }

    // step 2: ln (log)
    let log_probs = softmax_out.ln().unwrap();

    // step 3: gather (simulates NLL: select log-prob at target class)
    let selected = log_probs.gather(&targets, 1usize).unwrap();
    assert_eq!(selected.dims(), &[4, 1]);

    // step 4: neg + mean_all
    let loss = (-&selected).mean_all().unwrap();
    let v = loss.to_vec().unwrap();
    assert_eq!(v.len(), 1);

    // verify positivity: cross entropy should be > 0 for these non-trivial logits
    assert!(v[0] > 0.0, "cross entropy should be positive, got {}", v[0]);
}

#[allow(dead_code)]
pub fn test_cross_entropy_basic_f32(device: &impl Device) {
    // (batch=2, classes=3) — class 0 should be more confident than class 2
    let logits = tensor_f32_dev(&[2.0, 1.0, 0.1, 0.1, 1.0, 2.0], (2, 3), device);
    // target: class 0 for first sample, class 2 for second
    let targets = tensor_i32_dev(&[0, 2], (2, 1), device);

    let log_probs = logits.softmax(1usize).unwrap().ln().unwrap();
    let selected = log_probs.gather(&targets, 1usize).unwrap();
    let loss = (-&selected).mean_all().unwrap();

    let v = loss.to_vec().unwrap();
    assert!(v[0] > 0.0, "loss positive, got {}", v[0]);
    // rough sanity: loss should be relatively small since logits favor correct class
    assert!(v[0] < 1.5, "loss {} too high for easy case", v[0]);
}

#[allow(dead_code)]
pub fn test_cross_entropy_mnist_shape_f32(device: &impl Device) {
    // exact MNIST shape: batch=64, classes=10
    let n = 64 * 10;
    let logits_data: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let logits = tensor_f32_dev(&logits_data, (64, 10), device);

    let target_data: Vec<i64> = (0..64).map(|i| (i % 10) as i64).collect();
    let targets = tensor_i32_dev(&target_data, (64, 1), device);

    let log_probs = logits.softmax(1usize).unwrap().ln().unwrap();
    let selected = log_probs.gather(&targets, 1usize).unwrap();
    assert_eq!(selected.dims(), &[64, 1]);
    let loss = (-&selected).mean_all().unwrap();
    assert!(loss.to_scalar().unwrap() > 0.0);
}

// ---- matmul with transposed weight (simulates Linear::forward) ----

#[allow(dead_code)]
pub fn test_matmul_transposed_weight_small_f32(device: &impl Device) {
    // small: (batch=2, 3) @ (4, 3)^T = (2, 3) @ (3, 4)
    let input = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    let weight = tensor_f32_dev(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], (4, 3), device);
    let weight_t = weight.transpose_last().unwrap();
    assert_eq!(weight_t.dims(), &[3, 4]);
    let output = input.matmul(&weight_t).unwrap();
    assert_eq!(output.dims(), &[2, 4]);
    let v = output.to_vec().unwrap();
    assert!(v.iter().all(|&x| x.is_finite()));
}

#[allow(dead_code)]
pub fn test_matmul_transposed_weight_f32(device: &impl Device) {
    // simulate: F::linear does input.matmul(&weight.transpose_last()?)
    // for fc1: (batch=64, 784) @ (512, 784)^T = (64, 784) @ (784, 512)
    let in_features = 784usize;
    let out_features = 512usize;
    let batch = 64usize;

    let input_data: Vec<f64> = (0..(batch * in_features)).map(|i| (i % 128) as f64).collect();
    let input = tensor_f32_dev(&input_data, (batch, in_features), device);

    // weight: (out_features=512, in_features=784), stored contiguously
    let weight_data: Vec<f64> = (0..(out_features * in_features)).map(|i| (i % 256) as f64 * 0.01).collect();
    let weight = tensor_f32_dev(&weight_data, (out_features, in_features), device);

    // transpose_last: swap last two dims → (in_features, out_features) with strides (1, in_features)
    let weight_t = weight.transpose_last().unwrap();
    assert_eq!(weight_t.dims(), &[in_features, out_features]);

    let output = input.matmul(&weight_t).unwrap();
    assert_eq!(output.dims(), &[batch, out_features]);

    let v = output.to_vec().unwrap();
    assert_eq!(v.len(), batch * out_features);
    // sanity: all values finite
    assert!(v.iter().all(|&x| x.is_finite()), "output contains NaN or Inf");
}

#[allow(dead_code)]
pub fn test_matmul_add_bias_f32(device: &impl Device) {
    // Simulates Linear::forward: matmul then add bias with broadcasting
    let batch = 4usize;
    let in_features = 3usize;
    let out_features = 5usize;

    let input_data: Vec<f64> = (1..=(batch * in_features)).map(|i| i as f64).collect();
    let input = tensor_f32_dev(&input_data, (batch, in_features), device);

    let weight_data: Vec<f64> = (0..(out_features * in_features)).map(|i| (i as f64 + 1.0) * 0.1).collect();
    let weight = tensor_f32_dev(&weight_data, (out_features, in_features), device);
    let weight_t = weight.transpose_last().unwrap();

    let output = input.matmul(&weight_t).unwrap();
    assert_eq!(output.dims(), &[batch, out_features]);

    let bias = tensor_f32_dev(&[0.1, 0.2, 0.3, 0.4, 0.5], (out_features,), device);
    // Use broadcast_add (same as fixed Linear::forward)
    let output = output.broadcast_add(&bias).unwrap();
    assert_eq!(output.dims(), &[batch, out_features]);
    let v = output.to_vec().unwrap();
    assert!(v.iter().all(|&x| x.is_finite()), "output has NaN/Inf");

    // Verify: each row should have bias added
    let matmul_only = input.matmul(&weight_t).unwrap();
    let mv = matmul_only.to_vec().unwrap();
    let bv = bias.to_vec().unwrap();
    for r in 0..batch {
        for c in 0..out_features {
            let idx = r * out_features + c;
            assert!(
                (v[idx] - (mv[idx] + bv[c])).abs() < 1e-4,
                "row {} col {}: expected {} + {} = {}, got {}",
                r,
                c,
                mv[idx],
                bv[c],
                mv[idx] + bv[c],
                v[idx]
            );
        }
    }
}

// ---- broadcast_add correctness (bias broadcasting pattern in Linear) ----

#[allow(dead_code)]
pub fn test_broadcast_add_precision(device: &impl Device) {
    // Recreate the exact Linear::forward pattern: (batch, out) + (out,)
    let batch = 4usize;
    let out = 5usize;

    let a_data: Vec<f64> = vec![0.5, 1.2, 3.4, 5.6, 7.8, 1.1, 2.3, 4.5, 6.7, 8.9, 2.2, 3.4, 5.6, 7.8, 9.0, 3.3, 4.5, 6.7, 8.9, 0.1];
    let a = tensor_f32_dev(&a_data, (batch, out), device);

    let b_data: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let b = tensor_f32_dev(&b_data, (out,), device);

    let result = a.broadcast_add(&b).unwrap();
    let v = result.to_vec().unwrap();

    // Expected: a[i][j] + b[j]
    for i in 0..batch {
        for j in 0..out {
            let expected = a_data[i * out + j] + b_data[j];
            let actual = v[i * out + j];
            assert!((actual - expected).abs() < 1e-5, "broadcast_add mismatch at ({},{}): expected {}, got {}", i, j, expected, actual);
        }
    }
}

// ---- cross-entropy chain precision check ----

#[allow(dead_code)]
pub fn test_cross_entropy_precision(device: &impl Device) {
    // Fixed logits and targets — compare against manually computed f64 reference
    let logits_data = vec![2.0_f64, 1.0, 0.1, 0.1, 2.0, 1.0, 1.0, 0.1, 2.0];
    let logits = tensor_f32_dev(&logits_data, (3, 3), device);
    let targets = tensor_i32_dev(&[0_i64, 1, 2], (3, 1), device);

    // Compute softmax in f64 for reference
    let f64_logits: Vec<Vec<f64>> = logits_data.chunks(3).map(|r| r.to_vec()).collect();
    let target_classes = [0usize, 1usize, 2usize];
    let mut ref_values = vec![0.0_f64; 3];
    for (row_idx, row) in f64_logits.iter().enumerate() {
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = row.iter().map(|&v| (v - max_val).exp()).sum();
        let log_prob = ((row[target_classes[row_idx]] - max_val).exp() / exp_sum).ln();
        ref_values[row_idx] = -log_prob;
    }
    let ref_loss = ref_values.iter().sum::<f64>() / 3.0;

    let log_probs = logits.softmax(1usize).unwrap().ln().unwrap();
    let selected = log_probs.gather(&targets, 1usize).unwrap();
    let loss = (-&selected).mean_all().unwrap();
    let actual = loss.to_scalar().unwrap();

    assert!(
        (actual - ref_loss).abs() < 1e-4,
        "cross entropy mismatch: expected {}, got {} (diff: {:.2e})",
        ref_loss,
        actual,
        (actual - ref_loss).abs()
    );
}

// ---- gradient through broadcast_add (Linear bias pattern) ----

#[allow(dead_code)]
pub fn test_broadcast_add_grad_f32(device: &impl Device) {
    let batch = 4usize;
    let in_features = 3usize;
    let out_features = 2usize;

    let input = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], (batch, in_features), device);

    let weight = tensor_f32_dev(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], (out_features, in_features), device);
    weight.set_requires_grad(true);

    let bias = tensor_f32_dev(&[0.01, 0.02], (out_features,), device);
    bias.set_requires_grad(true);

    let weight_t = weight.transpose_last().unwrap();
    let matmul_out = input.matmul(&weight_t).unwrap();
    let output = matmul_out.broadcast_add(&bias).unwrap();

    assert_eq!(output.dims(), &[batch, out_features]);

    let loss = output.sum_all().unwrap();
    let grads = loss.backward().unwrap();

    let wg = grads.get(&weight).unwrap();
    assert_eq!(wg.dims(), &[out_features, in_features]);
    let wgv = wg.to_vec().unwrap();
    assert!(wgv.iter().all(|&x| x.is_finite()), "weight grad has NaN/Inf");

    let bg = grads.get(&bias).unwrap();
    assert_eq!(bg.dims(), &[out_features]);
    let bgv = bg.to_vec().unwrap();
    assert!(bgv.iter().all(|&x| x.is_finite()), "bias grad has NaN/Inf");
    // For loss=sum(output), bias gradient = batch for each output class
    for &bg_val in &bgv {
        assert!((bg_val - batch as f64).abs() < 1e-3, "bias grad expected {}, got {}", batch, bg_val);
    }

    // Simulate one SGD step: var -= lr * grad
    let lr = 0.01;
    let bias_before = bias.to_vec().unwrap();
    let scaled = bg.mul_scalar(lr).unwrap();
    bias.sub_(&scaled).unwrap();
    let bias_after = bias.to_vec().unwrap();
    for (i, (&before, &after)) in bias_before.iter().zip(bias_after.iter()).enumerate() {
        let expected = before - lr * bgv[i];
        assert!(
            (after - expected).abs() < 1e-5,
            "SGD step mismatch at {}: {} - {}*{} = {}, got {}",
            i,
            before,
            lr,
            bgv[i],
            expected,
            after
        );
    }
}

// ---- sum_keepdim + squeeze (used in broadcast backward) ----

#[allow(dead_code)]
pub fn test_broadcast_reduce_backward_f32(device: &impl Device) {
    let batch = 4usize;
    let out = 3usize;

    let grad = tensor_f32_dev(&[0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3], (batch, out), device);

    let summed = grad.sum_keepdim(0usize).unwrap();
    assert_eq!(summed.dims(), &[1, out]);

    let squeezed = summed.squeeze(0usize).unwrap();
    assert_eq!(squeezed.dims(), &[out]);

    let v = squeezed.to_vec().unwrap();
    for i in 0..out {
        assert!(
            (v[i] - batch as f64 * 0.1 * (i + 1) as f64).abs() < 1e-5,
            "reduce mismatch at {}: expected {}, got {}",
            i,
            batch as f64 * 0.1 * (i + 1) as f64,
            v[i]
        );
    }
}

// ---- argmax + eq + true_count (evaluation pipeline) ----

#[allow(dead_code)]
pub fn test_argmax_eval_pipeline_f32(device: &impl Device) {
    // Simulates the test evaluation: argmax_keepdim(1) → eq → true_count
    let batch = 4usize;
    let classes = 3usize;

    // Model output: row 0→class 0, row 1→class 1, row 2→class 2, row 3→class 0
    let logits = tensor_f32_dev(
        &[
            3.0, 1.0, 2.0, // max at col 0
            1.0, 5.0, 2.0, // max at col 1
            2.0, 1.0, 3.0, // max at col 2
            4.0, 2.0, 1.0, // max at col 0
        ],
        (batch, classes),
        device,
    );

    let predictions = logits.argmax_keepdim(1usize).unwrap();
    assert_eq!(predictions.dims(), &[batch, 1]);
    let pred_vals = predictions.to_vec().unwrap();
    // argmax should give u32 indices
    assert_eq!(pred_vals, vec![0i64, 1, 2, 0]);

    // Target labels (u32 to match argmax output type)
    let target = tensor_u32_dev(&[0i64, 1, 2, 0], (batch, 1), device);
    let correct = predictions.eq(&target).unwrap();
    let count = correct.true_count().unwrap();
    assert_eq!(count, 4, "all 4 predictions should be correct, got {}", count);

    // Test with partially wrong predictions
    let target2 = tensor_u32_dev(&[0i64, 0, 2, 1], (batch, 1), device);
    let correct2 = predictions.eq(&target2).unwrap();
    let count2 = correct2.true_count().unwrap();
    assert_eq!(count2, 2, "2 predictions should be correct, got {}", count2);
}

// ---- mini training loop (forward + backward + SGD step + re-forward) ----

#[allow(dead_code)]
pub fn test_mini_training_step_f32(device: &impl Device) {
    // Simulate one SGD step: loss should decrease after parameter update
    let batch = 8usize;
    let in_features = 3usize;
    let out_features = 2usize;

    let input = tensor_f32_dev(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
        (batch, in_features),
        device,
    );

    let weight = tensor_f32_dev(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], (out_features, in_features), device);
    weight.set_requires_grad(true);

    let bias = tensor_f32_dev(&[0.01, 0.02], (out_features,), device);
    bias.set_requires_grad(true);

    // Forward 1
    let w_t = weight.transpose_last().unwrap();
    let m = input.matmul(&w_t).unwrap();
    let out1 = m.broadcast_add(&bias).unwrap();
    let loss1 = out1.sum_all().unwrap().to_scalar().unwrap();

    // Backward
    let grads = out1.sum_all().unwrap().backward().unwrap();
    let lr = 0.1;

    // SGD step
    if let Some(wg) = grads.get(&weight) {
        let scaled = wg.mul_scalar(lr).unwrap();
        weight.sub_(&scaled).unwrap();
    }
    if let Some(bg) = grads.get(&bias) {
        let scaled = bg.mul_scalar(lr).unwrap();
        bias.sub_(&scaled).unwrap();
    }

    // Forward 2 with same input
    let w_t2 = weight.transpose_last().unwrap();
    let m2 = input.matmul(&w_t2).unwrap();
    let out2 = m2.broadcast_add(&bias).unwrap();
    let loss2 = out2.sum_all().unwrap().to_scalar().unwrap();

    // Loss should decrease: gradient descent moves params in -grad direction
    // which should reduce sum(output) since grad of sum is all ones
    assert!(loss2 < loss1, "SGD step should reduce loss: before={}, after={}", loss1, loss2);
}

#[allow(dead_code)]
pub fn test_sum_keepdim_nonuniform_f32(device: &impl Device) {
    // Non-uniform data — different values per row
    // sum over dim=0 should produce row-wise sum
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], (3, 3), device);

    let summed = t.sum_keepdim(0usize).unwrap();
    assert_eq!(summed.dims(), &[1, 3]);
    let v = summed.to_vec().unwrap();
    assert_close(&v, &[12.0, 15.0, 18.0], 1e-5, 1e-5);

    let summed1 = t.sum_keepdim(1usize).unwrap();
    assert_eq!(summed1.dims(), &[3, 1]);
    let v1 = summed1.to_vec().unwrap();
    assert_close(&v1, &[6.0, 15.0, 24.0], 1e-5, 1e-5);
}

// ---- argmax + eq + true_count at MNIST test scale (batch=1000, classes=10) ----

#[allow(dead_code)]
pub fn test_argmax_eval_large_f32(device: &impl Device) {
    let batch = 1000usize;
    let classes = 10usize;

    // Generate logits: row i gets max at position i%10
    let mut logits_data = vec![0.0f64; batch * classes];
    let mut target_data = vec![0i64; batch];
    for i in 0..batch {
        let correct_class = i % classes;
        logits_data[i * classes + correct_class] = 100.0;
        // Make every 100th row wrong (rows 0, 100, 200, ...)
        target_data[i] = if i % 100 == 0 { ((correct_class + 1) % classes) as i64 } else { correct_class as i64 };
    }

    let logits = tensor_f32_dev(&logits_data, (batch, classes), device);
    let target = tensor_u32_dev(&target_data, (batch, 1), device);

    // Step 1: argmax_keepdim
    let preds = logits.argmax_keepdim(1usize).unwrap();
    assert_eq!(preds.dims(), &[batch, 1]);
    let pred_vals = preds.to_vec().unwrap();
    for i in 0..batch {
        assert_eq!(pred_vals[i], (i % classes) as i64, "argmax wrong at row {}: expected {}, got {}", i, i % classes, pred_vals[i]);
    }

    // Step 2: eq (compare predictions against target)
    let correct_mask = preds.eq(&target).unwrap();
    assert_eq!(correct_mask.dims(), &[batch, 1]);

    // Step 3: true_count — expect 990 correct (10 wrong rows at 0, 100, ..., 900)
    let count = correct_mask.true_count().unwrap();
    let expected = batch - batch / 100; // 1000 - 10 = 990
    assert_eq!(count, expected, "true_count mismatch: expected {}, got {}", expected, count);
}

#[allow(dead_code)]
pub fn test_max_pool2d(device: &impl Device, dtype: FloatDType, tol: f64) {
    // Basic 2x2/stride 2.
    let data: Vec<f64> = (1..=16).map(|v| v as f64).collect();
    let x = tensor_float_dev(&data, (1, 1, 4, 4), device, dtype);
    let y = x.max_pool2d((2, 2), (2, 2), (0, 0), (1, 1), false).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    assert_close(&y.to_vec().unwrap(), &[6.0, 8.0, 14.0, 16.0], tol, tol);

    // Padding is -inf.
    let x = tensor_float_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], (1, 1, 3, 3), device, dtype);
    let y = x.max_pool2d((2, 2), (1, 1), (1, 1), (1, 1), false).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4, 4]);
    assert_close(&y.to_vec().unwrap(), &[1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 6.0, 7.0, 8.0, 9.0, 9.0, 7.0, 8.0, 9.0, 9.0], tol, tol);

    // ceil_mode=True keeps the last partial window.
    let data: Vec<f64> = (1..=25).map(|v| v as f64).collect();
    let x = tensor_float_dev(&data, (1, 1, 5, 5), device, dtype);
    let y = x.max_pool2d((2, 2), (2, 2), (0, 0), (1, 1), true).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3, 3]);
    assert_close(&y.to_vec().unwrap(), &[7.0, 9.0, 10.0, 17.0, 19.0, 20.0, 22.0, 24.0, 25.0], tol, tol);

    // Dilation.
    let y = x.max_pool2d((2, 2), (1, 1), (0, 0), (2, 2), false).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3, 3]);
    assert_close(&y.to_vec().unwrap(), &[13.0, 14.0, 15.0, 18.0, 19.0, 20.0, 23.0, 24.0, 25.0], tol, tol);

    // Padding + dilation + ceil_mode (matches PyTorch).
    let y = x.max_pool2d((3, 3), (2, 2), (1, 1), (2, 2), true).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    assert_close(&y.to_vec().unwrap(), &[19.0, 19.0, 19.0, 19.0], tol, tol);

    // Padding + ceil_mode extends the last row/column.
    let y = x.max_pool2d((2, 2), (1, 1), (1, 1), (1, 1), true).unwrap();
    assert_eq!(y.dims(), &[1, 1, 6, 6]);
    assert_close(
        &y.to_vec().unwrap(),
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
            20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 25.0, 21.0, 22.0, 23.0, 24.0, 25.0, 25.0,
        ],
        tol,
        tol,
    );
}

#[allow(dead_code)]
pub fn test_avg_pool2d(device: &impl Device, dtype: FloatDType, tol: f64) {
    // Basic 2x2/stride 2.
    let data: Vec<f64> = (1..=16).map(|v| v as f64).collect();
    let x = tensor_float_dev(&data, (1, 1, 4, 4), device, dtype);
    let y = x.avg_pool2d((2, 2), (2, 2), (0, 0), false, true).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    assert_close(&y.to_vec().unwrap(), &[3.5, 5.5, 11.5, 13.5], tol, tol);

    let x = tensor_float_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], (1, 1, 3, 3), device, dtype);

    // Count padding in the divisor.
    let y = x.avg_pool2d((2, 2), (1, 1), (1, 1), false, true).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4, 4]);
    assert_close(
        &y.to_vec().unwrap(),
        &[0.25, 0.75, 1.25, 0.75, 1.25, 3.0, 4.0, 2.25, 2.75, 6.0, 7.0, 3.75, 1.75, 3.75, 4.25, 2.25],
        tol,
        tol,
    );

    // Exclude padding from the divisor.
    let y = x.avg_pool2d((2, 2), (1, 1), (1, 1), false, false).unwrap();
    assert_close(&y.to_vec().unwrap(), &[1.0, 1.5, 2.5, 3.0, 2.5, 3.0, 4.0, 4.5, 5.5, 6.0, 7.0, 7.5, 7.0, 7.5, 8.5, 9.0], tol, tol);

    // ceil_mode=True: the last partial window is averaged over real elements.
    let data: Vec<f64> = (1..=25).map(|v| v as f64).collect();
    let x = tensor_float_dev(&data, (1, 1, 5, 5), device, dtype);
    let y = x.avg_pool2d((2, 2), (2, 2), (0, 0), true, true).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3, 3]);
    assert_close(&y.to_vec().unwrap(), &[4.0, 6.0, 7.5, 14.0, 16.0, 17.5, 21.5, 23.5, 25.0], tol, tol);

    // Padding + ceil_mode, excluding padding from the divisor.
    let y = x.avg_pool2d((3, 3), (2, 2), (1, 1), true, false).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3, 3]);
    assert_close(&y.to_vec().unwrap(), &[4.0, 5.5, 7.0, 11.5, 13.0, 14.5, 19.0, 20.5, 22.0], tol, tol);

    // Padding + ceil_mode, counting padding in the divisor. The extra cells
    // added for ceil_mode are still excluded, matching PyTorch.
    let y = x.avg_pool2d((3, 3), (2, 2), (1, 1), true, true).unwrap();
    assert_close(
        &y.to_vec().unwrap(),
        &[
            1.7777777777777777,
            3.6666666666666665,
            3.111111111111111,
            7.666666666666667,
            13.0,
            9.666666666666666,
            8.444444444444445,
            13.666666666666666,
            9.777777777777779,
        ],
        tol,
        tol,
    );
}

#[allow(dead_code)]
pub fn test_adaptive_avg_pool2d(device: &impl Device, dtype: FloatDType, tol: f64) {
    let data: Vec<f64> = (1..=16).map(|v| v as f64).collect();
    let x = tensor_float_dev(&data, (1, 1, 4, 4), device, dtype);

    let y = x.adaptive_avg_pool2d((2, 2)).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    assert_close(&y.to_vec().unwrap(), &[3.5, 5.5, 11.5, 13.5], tol, tol);

    let y = x.adaptive_avg_pool2d((1, 1)).unwrap();
    assert_close(&y.to_vec().unwrap(), &[8.5], tol, tol);

    // Non-divisible input: windows may overlap.
    let data: Vec<f64> = (1..=25).map(|v| v as f64).collect();
    let x = tensor_float_dev(&data, (1, 1, 5, 5), device, dtype);
    let y = x.adaptive_avg_pool2d((2, 2)).unwrap();
    assert_close(&y.to_vec().unwrap(), &[7.0, 9.0, 17.0, 19.0], tol, tol);

    let y = x.adaptive_avg_pool2d((4, 4)).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4, 4]);
    assert_close(
        &y.to_vec().unwrap(),
        &[4.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0, 17.0, 19.0, 20.0, 21.0, 22.0],
        tol,
        tol,
    );

    // Output larger than input: overlapping windows.
    let y = x.adaptive_avg_pool2d((7, 7)).unwrap();
    assert_eq!(y.dims(), &[1, 1, 7, 7]);
    assert_close(
        &y.to_vec().unwrap(),
        &[
            1.0, 1.5, 2.5, 3.0, 3.5, 4.5, 5.0, 3.5, 4.0, 5.0, 5.5, 6.0, 7.0, 7.5, 8.5, 9.0, 10.0, 10.5, 11.0, 12.0, 12.5, 11.0, 11.5, 12.5,
            13.0, 13.5, 14.5, 15.0, 13.5, 14.0, 15.0, 15.5, 16.0, 17.0, 17.5, 18.5, 19.0, 20.0, 20.5, 21.0, 22.0, 22.5, 21.0, 21.5, 22.5,
            23.0, 23.5, 24.5, 25.0,
        ],
        tol,
        tol,
    );
}

#[allow(dead_code)]
pub fn test_max_pool2d_grad(device: &impl Device, dtype: FloatDType, tol: f64) {
    let data: Vec<f64> = (1..=25).map(|v| v as f64).collect();
    let x = tensor_float_dev(&data, (1, 1, 5, 5), device, dtype);
    x.set_requires_grad(true);
    let y = NnOpDefault::max_pool2d_forward(&x, (2, 2), (2, 2), (0, 0), (1, 1), true).unwrap();
    let grads = y.sum_all().unwrap().backward().unwrap();
    let gx = grads.get(&x).unwrap().clone();

    let dx = NnOpDefault::max_pool2d_backward(&x, &y, &y.ones_like().unwrap(), (2, 2), (2, 2), (0, 0), (1, 1), true).unwrap();
    assert_close(&gx.to_vec().unwrap(), &dx.to_vec().unwrap(), tol, tol);
}

#[allow(dead_code)]
pub fn test_avg_pool2d_grad(device: &impl Device, dtype: FloatDType, tol: f64) {
    let data: Vec<f64> = (1..=25).map(|v| v as f64).collect();
    let x = tensor_float_dev(&data, (1, 1, 5, 5), device, dtype);
    x.set_requires_grad(true);
    let y = NnOpDefault::avg_pool2d_forward(&x, (3, 3), (1, 1), (1, 1), true, false).unwrap();
    let grads = y.sum_all().unwrap().backward().unwrap();
    let gx = grads.get(&x).unwrap().clone();

    let dx = NnOpDefault::avg_pool2d_backward(&x, &y, &y.ones_like().unwrap(), (3, 3), (1, 1), (1, 1), true, false).unwrap();
    assert_close(&gx.to_vec().unwrap(), &dx.to_vec().unwrap(), tol, tol);
}

#[allow(dead_code)]
pub fn test_adaptive_avg_pool2d_grad(device: &impl Device, dtype: FloatDType, tol: f64) {
    let data: Vec<f64> = (1..=25).map(|v| v as f64).collect();
    let x = tensor_float_dev(&data, (1, 1, 5, 5), device, dtype);
    x.set_requires_grad(true);
    let y = NnOpDefault::adaptive_avg_pool2d_forward(&x, (2, 2)).unwrap();
    let grads = y.sum_all().unwrap().backward().unwrap();
    let gx = grads.get(&x).unwrap().clone();

    let dx = NnOpDefault::adaptive_avg_pool2d_backward(&x, &y, &y.ones_like().unwrap(), (2, 2)).unwrap();
    assert_close(&gx.to_vec().unwrap(), &dx.to_vec().unwrap(), tol, tol);
}

#[allow(dead_code)]
pub fn test_linear(device: &impl Device, dtype: FloatDType, tol: f64) {
    let x = tensor_float_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device, dtype);
    let w = tensor_float_dev(&[1.0, 0.0, -1.0, 0.5, 1.0, 0.0], (2, 3), device, dtype);
    let b = tensor_float_dev(&[0.1, -0.2], (2,), device, dtype);

    let y = x.linear(&w, Some(&b)).unwrap();
    let expected = x.matmul(&w.transpose_last().unwrap()).unwrap().broadcast_add(&b).unwrap();
    assert_close(&y.to_vec().unwrap(), &expected.to_vec().unwrap(), tol, tol);

    x.set_requires_grad(true);
    w.set_requires_grad(true);
    b.set_requires_grad(true);
    let loss = x.linear(&w, Some(&b)).unwrap().sum_all().unwrap();
    let grads = loss.backward().unwrap();
    assert!(grads.get(&x).is_some());
    assert!(grads.get(&w).is_some());
    assert!(grads.get(&b).is_some());
}

#[allow(dead_code)]
pub fn test_embedding(device: &impl Device, dtype: FloatDType, tol: f64) {
    let weight = tensor_float_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), device, dtype);
    let indices = tensor_i32_dev(&[2, 0, 1, 2], (2, 2), device);

    let y = weight.embedding(&indices).unwrap();
    assert_eq!(y.dims(), &[2, 2, 2]);
    assert_close(&y.to_vec().unwrap(), &[5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], tol, tol);

    weight.set_requires_grad(true);
    let grads = weight.embedding(&indices).unwrap().sum_all().unwrap().backward().unwrap();
    let gw = grads.get(&weight).unwrap().to_vec().unwrap();
    assert_close(&gw, &[1.0, 1.0, 1.0, 1.0, 2.0, 2.0], tol, tol);
}

#[allow(dead_code)]
pub fn test_dropout(device: &impl Device) {
    let data: Vec<f64> = (0..1024).map(|i| (i % 7) as f64).collect();
    let x = tensor_f32_dev(&data, (1024,), device);

    let y = x.dropout(0.5, false).unwrap();
    assert_close(&y.to_vec().unwrap(), &x.to_vec().unwrap(), 1e-9, 1e-9);

    let y = x.dropout(0.0, true).unwrap();
    assert_close(&y.to_vec().unwrap(), &x.to_vec().unwrap(), 1e-9, 1e-9);

    let y = x.dropout(0.5, true).unwrap();
    let xv = x.to_vec().unwrap();
    let yv = y.to_vec().unwrap();
    assert_eq!(xv.len(), yv.len());
    let mut zeros = 0;
    let mut kept = 0;
    for (&a, &b) in xv.iter().zip(yv.iter()) {
        if b == 0.0 {
            zeros += 1;
        } else {
            assert!((b - 2.0 * a).abs() < 1e-6, "got {b}, want {}", 2.0 * a);
            kept += 1;
        }
    }
    assert!(zeros > 0 && kept > 0, "expected both dropped and kept elements");
}

#[allow(dead_code)]
pub fn test_losses(device: &impl Device, dtype: FloatDType, tol: f64) {
    let p = tensor_float_dev(&[1.0, 2.0, 3.0], (3,), device, dtype);
    let t = tensor_float_dev(&[0.0, 0.0, 0.0], (3,), device, dtype);
    let mse = p.mse_loss(&t).unwrap().to_scalar().unwrap();
    assert_close(&[mse], &[14.0 / 3.0], tol, tol);

    let log_probs = tensor_float_dev(&[-0.1, -1.0, -2.0, -2.0, -1.0, -0.1], (2, 3), device, dtype);
    let targets = tensor_i32_dev(&[0, 2], (2,), device);
    let nll = log_probs.nll_loss(&targets).unwrap().to_scalar().unwrap();
    assert_close(&[nll], &[0.1], tol, tol);

    let logits = tensor_float_dev(&[1.0, 2.0, 3.0, 0.0, 0.0, 0.0], (2, 3), device, dtype);
    let ce = logits.cross_entropy_loss(&targets).unwrap().to_scalar().unwrap();
    let manual = logits.log_softmax(1usize).unwrap().nll_loss(&targets).unwrap().to_scalar().unwrap();
    assert_close(&[ce], &[manual], tol, tol);

    let bpred = tensor_float_dev(&[0.2, 0.8], (2,), device, dtype);
    let btarget = tensor_float_dev(&[0.0, 1.0], (2,), device, dtype);
    let bce = bpred.bce_loss(&btarget).unwrap().to_scalar().unwrap();
    let expected = -((1.0_f64 - 0.2).ln() + 0.8_f64.ln()) / 2.0;
    assert_close(&[bce], &[expected], tol, tol);
}

#[allow(dead_code)]
pub fn test_loss_grads(device: &impl Device, dtype: FloatDType, tol: f64) {
    // MSE: d/dp mean((p-t)^2) = 2(p-t)/n
    let p = tensor_float_dev(&[1.0, 2.0, 3.0], (3,), device, dtype);
    let t = tensor_float_dev(&[0.0, 0.0, 0.0], (3,), device, dtype);
    p.set_requires_grad(true);
    let grads = p.mse_loss(&t).unwrap().backward().unwrap();
    let gp = grads.get(&p).unwrap().to_vec().unwrap();
    assert_close(&gp, &[2.0 / 3.0, 4.0 / 3.0, 6.0 / 3.0], tol, tol);

    // NLL: -1/n at the target class.
    let log_probs = tensor_float_dev(&[-0.1, -1.0, -2.0, -2.0, -1.0, -0.1], (2, 3), device, dtype);
    let targets = tensor_i32_dev(&[0, 2], (2,), device);
    log_probs.set_requires_grad(true);
    let grads = log_probs.nll_loss(&targets).unwrap().backward().unwrap();
    let gl = grads.get(&log_probs).unwrap().to_vec().unwrap();
    assert_close(&gl, &[-0.5, 0.0, 0.0, 0.0, 0.0, -0.5], tol, tol);

    // CrossEntropy: (softmax - onehot) / batch.
    let logits = tensor_float_dev(&[1.0, 2.0, 3.0, 0.0, 0.0, 0.0], (2, 3), device, dtype);
    logits.set_requires_grad(true);
    let grads = logits.cross_entropy_loss(&targets).unwrap().backward().unwrap();
    let gx = grads.get(&logits).unwrap().to_vec().unwrap();
    let mut expected = logits.softmax(1usize).unwrap().to_vec().unwrap();
    expected[0] -= 1.0;
    expected[5] -= 1.0;
    for v in &mut expected {
        *v /= 2.0;
    }
    assert_close(&gx, &expected, tol, tol);

    // BCE: (-t/p + (1-t)/(1-p)) / n
    let bpred = tensor_float_dev(&[0.2, 0.8], (2,), device, dtype);
    let btarget = tensor_float_dev(&[0.0, 1.0], (2,), device, dtype);
    bpred.set_requires_grad(true);
    let grads = bpred.bce_loss(&btarget).unwrap().backward().unwrap();
    let gb = grads.get(&bpred).unwrap().to_vec().unwrap();
    let expected = [1.0 / 0.8 / 2.0, -1.0 / 0.8 / 2.0];
    assert_close(&gb, &expected, tol, tol);
}
