//! Device-generic reduction / arg-reduction executor tests.

use super::*;
use luma_jit::Traced;
use luma_tensor::dtype::FloatDType;
use luma_tensor::{Device, Float, Tensor};

pub fn test_sum_dim<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0, 3.0, 4.0], (2, 2), |a| a.sum(0usize), |a| a.sum(0usize));
}
pub fn test_sum_keepdim<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0, 3.0, 4.0], (2, 2), |a| a.sum_keepdim(1usize), |a| a.sum_keepdim(1usize));
}
pub fn test_sum_all<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0, 3.0, 4.0], (2, 2), |a| a.sum_all(), |a| a.sum_all());
}
pub fn test_max_dim<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 4.0, 3.0, 2.0], (2, 2), |a| a.max(0usize), |a| a.max(0usize));
}
pub fn test_max_keepdim<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 4.0, 3.0, 2.0], (2, 2), |a| a.max_keepdim(1usize), |a| a.max_keepdim(1usize));
}
pub fn test_max_all<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 4.0, 3.0, 2.0], (2, 2), |a| a.max_all(), |a| a.max_all());
}
pub fn test_min_dim<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 4.0, 3.0, 2.0], (2, 2), |a| a.min(0usize), |a| a.min(0usize));
}
pub fn test_min_keepdim<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 4.0, 3.0, 2.0], (2, 2), |a| a.min_keepdim(1usize), |a| a.min_keepdim(1usize));
}
pub fn test_min_all<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 4.0, 3.0, 2.0], (2, 2), |a| a.min_all(), |a| a.min_all());
}
pub fn test_prod_dim<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0, 3.0, 4.0], (2, 2), |a| a.prod(0usize), |a| a.prod(0usize));
}
pub fn test_prod_keepdim<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0, 3.0, 4.0], (2, 2), |a| a.prod_keepdim(1usize), |a| a.prod_keepdim(1usize));
}
pub fn test_prod_all<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0, 3.0, 4.0], (2, 2), |a| a.prod_all(), |a| a.prod_all());
}
pub fn test_mean_dim<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0, 3.0, 4.0], (2, 2), |a| a.mean(0usize), |a| a.mean(0usize));
}
pub fn test_mean_keepdim<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0, 3.0, 4.0], (2, 2), |a| a.mean_keepdim(1usize), |a| a.mean_keepdim(1usize));
}
pub fn test_mean_all<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0, 3.0, 4.0], (2, 2), |a| a.mean_all(), |a| a.mean_all());
}

// ---- arg-reduce (int output) -------------------------------------------------

fn arg_check<D, F, G>(dev: &D, data: &[f64], shape: (usize, usize), trace_op: F, real_op: G)
where
    D: Device,
    F: FnOnce(&Tensor<Trace, Float>) -> luma_tensor::Result<Tensor<Trace, Int>>,
    G: FnOnce(&Tensor<D>) -> luma_tensor::Result<Tensor<D, Int>>,
{
    let a = tensor_f32(dev, data, shape);
    let expected = real_op(&a).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(&[shape.0, shape.1], 0.0, (t, FloatDType::F32)).unwrap();
            let in_id = ta.trace_id();
            let o = trace_op(&ta).unwrap();
            (vec![in_id], o.trace_id())
        },
        vec![a.clone().into()],
    );
    assert_eq!(as_i64s(&out[0]), expected);
}

pub fn test_argmax<D: Device>(dev: &D) {
    arg_check(dev, &[1.0, 3.0, 2.0, 5.0, 4.0, 6.0], (2, 3), |a| a.argmax(1usize), |a| a.argmax(1usize));
}
pub fn test_argmin<D: Device>(dev: &D) {
    arg_check(dev, &[1.0, 3.0, 2.0, 5.0, 4.0, 6.0], (2, 3), |a| a.argmin(1usize), |a| a.argmin(1usize));
}
pub fn test_argmax_keepdim<D: Device>(dev: &D) {
    arg_check(dev, &[1.0, 3.0, 2.0, 5.0, 4.0, 6.0], (2, 3), |a| a.argmax_keepdim(1usize), |a| a.argmax_keepdim(1usize));
}
pub fn test_argmin_keepdim<D: Device>(dev: &D) {
    arg_check(dev, &[1.0, 3.0, 2.0, 5.0, 4.0, 6.0], (2, 3), |a| a.argmin_keepdim(1usize), |a| a.argmin_keepdim(1usize));
}
