use super::*;
use crate::{BinaryOp, Device, FloatOps, Layout, Shape};

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
    let mut store = crate::GradStore::new();
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
    let _guard = crate::NoGradGuard::new();
    let y = x.mul_scalar(2.0).unwrap();
    assert!(!y.requires_grad());
}

// ---------------------------------------------------------------------------
// Custom ops: `Tensor::custom_op*` runs the op's `forward` (as a black box) and
// records it for backward. All inputs must receive their gradients.
// ---------------------------------------------------------------------------

struct Mul3Op;

impl<Dev: Device> crate::CustomOp3<Dev> for Mul3Op {
    fn name(&self) -> String {
        "mul3".to_string()
    }

    fn forward(&self, a: &Tensor<Dev>, b: &Tensor<Dev>, c: &Tensor<Dev>) -> Result<(Dev::FloatStorage, Shape), crate::CustomOpError> {
        // A custom op returns raw `Storage` + `Shape`; the framework assembles
        // the output tensor. Here we just use the device's elementwise mul.
        let a_g = a.storage_read()?;
        let b_g = b.storage_read()?;
        let c_g = c.storage_read()?;
        let ab = <Dev as FloatOps<Dev>>::f_binary(&*a_g, a.layout(), &*b_g, b.layout(), BinaryOp::Mul)?;
        let ab_layout = Layout::contiguous(a.shape().clone());
        let abc = <Dev as FloatOps<Dev>>::f_binary(&ab, &ab_layout, &*c_g, c.layout(), BinaryOp::Mul)?;
        Ok((abc, a.shape().clone()))
    }

    fn backward(
        &self,
        a: &Tensor<Dev>,
        b: &Tensor<Dev>,
        c: &Tensor<Dev>,
        _ret: &Tensor<Dev>,
        g: &Tensor<Dev>,
    ) -> Result<(Tensor<Dev>, Tensor<Dev>, Tensor<Dev>), crate::CustomOpError> {
        // y = a*b*c  =>  da = g*b*c, db = g*a*c, dc = g*a*b
        let ga = g.mul(b)?.mul(c)?;
        let gb = g.mul(a)?.mul(c)?;
        let gc = g.mul(a)?.mul(b)?;
        Ok((ga, gb, gc))
    }
}

#[allow(dead_code)]
pub fn test_custom_op3_forward_backward(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0], (2,), device);
    let b = tensor_f32_dev(&[3.0, 4.0], (2,), device);
    let c = tensor_f32_dev(&[5.0, 6.0], (2,), device);
    a.set_requires_grad(true);
    b.set_requires_grad(true);
    c.set_requires_grad(true);

    let y = a.custom_op3(&b, &c, Box::new(Mul3Op)).unwrap();
    assert!(y.requires_grad());
    assert_eq!(y.shape().dims(), &[2]);

    // forward actually ran
    assert_close(&y.to_vec().unwrap(), &[15.0, 48.0], 1e-5, 1e-5);

    // backward routes gradients to all three inputs (arg3 included)
    let loss = y.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    assert_close(&grads.get(&a).unwrap().to_vec().unwrap(), &[15.0, 24.0], 1e-4, 1e-4);
    assert_close(&grads.get(&b).unwrap().to_vec().unwrap(), &[5.0, 12.0], 1e-4, 1e-4);
    assert_close(&grads.get(&c).unwrap().to_vec().unwrap(), &[3.0, 8.0], 1e-4, 1e-4);
}
