use super::*;
use luma_tensor::Device;

#[allow(dead_code)]
pub fn test_grad_add(device: &impl Device) {
    let x1 = tensor_f32_dev(&[1.0, 2.0, 3.0], (3,), device);
    let x2 = tensor_f32_dev(&[4.0, 5.0, 6.0], (3,), device);
    x1.set_requires_grad(true);
    x2.set_requires_grad(true);
    let y = x1.add(&x2).unwrap();
    let loss = y.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    assert_close(&grads.get(&x1).unwrap().to_vec().unwrap(), &[1.0, 1.0, 1.0], 1e-5, 1e-5);
    assert_close(&grads.get(&x2).unwrap().to_vec().unwrap(), &[1.0, 1.0, 1.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_grad_sub(device: &impl Device) {
    let x1 = tensor_f32_dev(&[5.0], (1,), device);
    let x2 = tensor_f32_dev(&[2.0], (1,), device);
    x1.set_requires_grad(true);
    x2.set_requires_grad(true);
    let loss = x1.sub(&x2).unwrap().sum_all().unwrap();
    let grads = loss.backward().unwrap();
    assert!((grads.get(&x1).unwrap().to_vec().unwrap()[0] - 1.0).abs() < 1e-5);
    assert!((grads.get(&x2).unwrap().to_vec().unwrap()[0] + 1.0).abs() < 1e-5);
}

#[allow(dead_code)]
pub fn test_grad_mul(device: &impl Device) {
    let x1 = tensor_f32_dev(&[2.0, 3.0], (2,), device);
    let x2 = tensor_f32_dev(&[4.0, 5.0], (2,), device);
    x1.set_requires_grad(true);
    x2.set_requires_grad(true);
    let y = x1.mul(&x2).unwrap();
    let loss = y.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    assert_close(&grads.get(&x1).unwrap().to_vec().unwrap(), &[4.0, 5.0], 1e-4, 1e-4);
    assert_close(&grads.get(&x2).unwrap().to_vec().unwrap(), &[2.0, 3.0], 1e-4, 1e-4);
}

#[allow(dead_code)]
pub fn test_grad_div(device: &impl Device) {
    let x1 = tensor_f32_dev(&[6.0, 8.0], (2,), device);
    let x2 = tensor_f32_dev(&[2.0, 4.0], (2,), device);
    x1.set_requires_grad(true);
    x2.set_requires_grad(true);
    let y = x1.div(&x2).unwrap();
    let loss = y.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    assert_close(&grads.get(&x1).unwrap().to_vec().unwrap(), &[0.5, 0.25], 1e-4, 1e-4);
    assert_close(&grads.get(&x2).unwrap().to_vec().unwrap(), &[-1.5, -0.5], 1e-4, 1e-4);
}

#[allow(dead_code)]
pub fn test_grad_relu(device: &impl Device) {
    let x = tensor_f32_dev(&[-1.0, 0.5, 2.0, -3.0], (4,), device);
    x.set_requires_grad(true);
    let y = x.relu().unwrap();
    let loss = y.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    assert_close(&grads.get(&x).unwrap().to_vec().unwrap(), &[0.0, 1.0, 1.0, 0.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_grad_sum(device: &impl Device) {
    let x = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (4,), device);
    x.set_requires_grad(true);
    let loss = x.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    assert_close(&grads.get(&x).unwrap().to_vec().unwrap(), &[1.0, 1.0, 1.0, 1.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_grad_mean(device: &impl Device) {
    let x = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (4,), device);
    x.set_requires_grad(true);
    let loss = x.mean_all().unwrap();
    let grads = loss.backward().unwrap();
    assert_close(&grads.get(&x).unwrap().to_vec().unwrap(), &[0.25, 0.25, 0.25, 0.25], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_grad_exp(device: &impl Device) {
    let x = tensor_f32_dev(&[0.0, 1.0], (2,), device);
    x.set_requires_grad(true);
    let y = x.exp().unwrap();
    let loss = y.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    assert_close(&grads.get(&x).unwrap().to_vec().unwrap(), &[1.0, std::f64::consts::E], 1e-4, 1e-4);
}

#[allow(dead_code)]
pub fn test_grad_sigmoid(device: &impl Device) {
    let x = tensor_f32_dev(&[0.0, 1.0], (2,), device);
    x.set_requires_grad(true);
    let y = x.sigmoid().unwrap();
    let loss = y.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let s0 = 0.5;
    let expected0 = s0 * (1.0 - s0);
    let s1 = 1.0 / (1.0 + (-1.0f64).exp());
    let expected1 = s1 * (1.0 - s1);
    assert_close(&grads.get(&x).unwrap().to_vec().unwrap(), &[expected0, expected1], 1e-4, 1e-4);
}

#[allow(dead_code)]
pub fn test_grad_clamp(device: &impl Device) {
    let x = tensor_f32_dev(&[-1.0, 0.0, 2.0, 5.0], (4,), device);
    x.set_requires_grad(true);
    let y = x.clamp(Some(0.0), Some(3.0)).unwrap();
    let loss = y.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    assert_close(&grads.get(&x).unwrap().to_vec().unwrap(), &[0.0, 0.0, 1.0, 0.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_grad_clamp_min(device: &impl Device) {
    let x = tensor_f32_dev(&[-1.0, 0.0, 2.0], (3,), device);
    x.set_requires_grad(true);
    let y = x.clamp(Some(0.0), None).unwrap();
    let loss = y.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    assert_close(&grads.get(&x).unwrap().to_vec().unwrap(), &[0.0, 0.0, 1.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_grad_prod(device: &impl Device) {
    let x = tensor_f32_dev(&[2.0, 3.0, 4.0], (3,), device);
    x.set_requires_grad(true);
    let y = x.prod_all().unwrap();
    let loss = y.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    assert_close(&grads.get(&x).unwrap().to_vec().unwrap(), &[12.0, 8.0, 6.0], 1e-4, 1e-4);
}

#[allow(dead_code)]
pub fn test_grad_matmul(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let b = tensor_f32_dev(&[5.0, 6.0, 7.0, 8.0], (2, 2), device);
    a.set_requires_grad(true);
    b.set_requires_grad(true);
    let y = a.matmul(&b).unwrap();
    let loss = y.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    assert_close(&grads.get(&a).unwrap().to_vec().unwrap(), &[11.0, 15.0, 11.0, 15.0], 1e-4, 1e-4);
    assert_close(&grads.get(&b).unwrap().to_vec().unwrap(), &[4.0, 4.0, 6.0, 6.0], 1e-4, 1e-4);
}

#[allow(dead_code)]
pub fn test_grad_reshape(device: &impl Device) {
    let x = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    x.set_requires_grad(true);
    let r = x.reshape((4,)).unwrap();
    let loss = r.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    assert_close(&grads.get(&x).unwrap().to_vec().unwrap(), &[1.0, 1.0, 1.0, 1.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_grad_transpose(device: &impl Device) {
    let x = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    x.set_requires_grad(true);
    let t = x.transpose(0usize, 1usize).unwrap();
    let loss = t.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    assert_close(&grads.get(&x).unwrap().to_vec().unwrap(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_grad_accumulate(device: &impl Device) {
    let x = tensor_f32_dev(&[1.0, 2.0, 3.0], (3,), device);
    x.set_requires_grad(true);

    // Two micro-batches accumulated into the same store.
    let loss1 = x.mul_scalar(2.0).unwrap().sum_all().unwrap();
    let loss2 = x.mul_scalar(3.0).unwrap().sum_all().unwrap();
    let mut store = luma_tensor::GradStore::new();
    loss1.backward_into(&mut store).unwrap();
    loss2.backward_into(&mut store).unwrap();
    assert_close(&store.get(&x).unwrap().to_vec().unwrap(), &[5.0, 5.0, 5.0], 1e-5, 1e-5);

    // Equivalent to a single backward of the summed loss.
    let x2 = tensor_f32_dev(&[1.0, 2.0, 3.0], (3,), device);
    x2.set_requires_grad(true);
    let combined = x2.mul_scalar(2.0).unwrap().add(&x2.mul_scalar(3.0).unwrap()).unwrap().sum_all().unwrap();
    let grads = combined.backward().unwrap();
    assert_close(&grads.get(&x2).unwrap().to_vec().unwrap(), &[5.0, 5.0, 5.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_no_grad_disabled(device: &impl Device) {
    let x = tensor_f32_dev(&[1.0, 2.0], (2,), device);
    x.set_requires_grad(true);
    let _guard = luma_tensor::NoGradGuard::new();
    let y = x.mul_scalar(2.0).unwrap();
    assert!(!y.requires_grad());
}
