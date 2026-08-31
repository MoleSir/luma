//! Device-generic elementwise / unary / comparison executor tests.

use super::*;
use luma_compile::Traced;
use luma_tensor::dtype::{FloatDType, IntDType};
use luma_tensor::{Device, Float, Int, Tensor};

// ---- binary ------------------------------------------------------------------

pub fn test_add<D: Device>(dev: &D) {
    binary_check(dev, &[1.0, 2.0, 3.0, 4.0], (2, 2), &[5.0, 6.0, 7.0, 8.0], (2, 2), |a, b| a.add(b), |a, b| a.add(b));
}
pub fn test_sub<D: Device>(dev: &D) {
    binary_check(dev, &[1.0, 2.0, 3.0, 4.0], (2, 2), &[5.0, 6.0, 7.0, 8.0], (2, 2), |a, b| a.sub(b), |a, b| a.sub(b));
}
pub fn test_mul<D: Device>(dev: &D) {
    binary_check(dev, &[1.0, 2.0, 3.0, 4.0], (2, 2), &[5.0, 6.0, 7.0, 8.0], (2, 2), |a, b| a.mul(b), |a, b| a.mul(b));
}
pub fn test_div<D: Device>(dev: &D) {
    binary_check(dev, &[1.0, 2.0, 3.0, 4.0], (2, 2), &[5.0, 6.0, 7.0, 8.0], (2, 2), |a, b| a.div(b), |a, b| a.div(b));
}
pub fn test_maximum<D: Device>(dev: &D) {
    binary_check(dev, &[1.0, 9.0, 3.0, 4.0], (2, 2), &[5.0, 6.0, 7.0, 8.0], (2, 2), |a, b| a.maximum(b), |a, b| a.maximum(b));
}
pub fn test_minimum<D: Device>(dev: &D) {
    binary_check(dev, &[1.0, 9.0, 3.0, 4.0], (2, 2), &[5.0, 6.0, 7.0, 8.0], (2, 2), |a, b| a.minimum(b), |a, b| a.minimum(b));
}

// ---- scalar (rhs) ------------------------------------------------------------

pub fn test_add_scalar<D: Device>(dev: &D) {
    scalar_rhs_check(dev, &[1.0, 2.0, 3.0, 4.0], (2, 2), 2.0, |a, s| a.add_scalar(s), |a, s| a.add_scalar(s));
}
pub fn test_sub_scalar<D: Device>(dev: &D) {
    scalar_rhs_check(dev, &[1.0, 2.0, 3.0, 4.0], (2, 2), 2.0, |a, s| a.sub_scalar(s), |a, s| a.sub_scalar(s));
}
pub fn test_mul_scalar<D: Device>(dev: &D) {
    scalar_rhs_check(dev, &[1.0, 2.0, 3.0, 4.0], (2, 2), 2.0, |a, s| a.mul_scalar(s), |a, s| a.mul_scalar(s));
}
pub fn test_div_scalar<D: Device>(dev: &D) {
    scalar_rhs_check(dev, &[1.0, 2.0, 3.0, 4.0], (2, 2), 2.0, |a, s| a.div_scalar(s), |a, s| a.div_scalar(s));
}
pub fn test_maximum_scalar<D: Device>(dev: &D) {
    scalar_rhs_check(dev, &[1.0, 2.0, 3.0, 4.0], (2, 2), 2.5, |a, s| a.maximum_scalar(s), |a, s| a.maximum_scalar(s));
}
pub fn test_minimum_scalar<D: Device>(dev: &D) {
    scalar_rhs_check(dev, &[1.0, 2.0, 3.0, 4.0], (2, 2), 2.5, |a, s| a.minimum_scalar(s), |a, s| a.minimum_scalar(s));
}

// ---- scalar (lhs) ------------------------------------------------------------

pub fn test_sub_scalar_lhs<D: Device>(dev: &D) {
    scalar_lhs_check(dev, &[1.0, 2.0, 3.0, 4.0], (2, 2), 10.0, |a, s| a.sub_scalar_lhs(s), |a, s| a.sub_scalar_lhs(s));
}
pub fn test_div_scalar_lhs<D: Device>(dev: &D) {
    scalar_lhs_check(dev, &[1.0, 2.0, 3.0, 4.0], (2, 2), 12.0, |a, s| a.div_scalar_lhs(s), |a, s| a.div_scalar_lhs(s));
}

// ---- unary (float, no arg) ---------------------------------------------------

pub fn test_neg<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, -2.0, 3.0, -4.0], (2, 2), |a| a.neg(), |a| a.neg());
}
pub fn test_abs<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, -2.0, 3.0, -4.0], (2, 2), |a| a.abs(), |a| a.abs());
}
pub fn test_sign<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, -2.0, 3.0, 0.0], (2, 2), |a| a.sign(), |a| a.sign());
}
pub fn test_exp<D: Device>(dev: &D) {
    unary_check(dev, &[0.0, 1.0, 2.0], (3,), |a| a.exp(), |a| a.exp());
}
pub fn test_ln<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0, 4.0], (3,), |a| a.ln(), |a| a.ln());
}
pub fn test_sin<D: Device>(dev: &D) {
    unary_check(dev, &[0.0, 1.0, 2.0], (3,), |a| a.sin(), |a| a.sin());
}
pub fn test_cos<D: Device>(dev: &D) {
    unary_check(dev, &[0.0, 1.0, 2.0], (3,), |a| a.cos(), |a| a.cos());
}
pub fn test_tanh<D: Device>(dev: &D) {
    unary_check(dev, &[0.0, 1.0, -1.0], (3,), |a| a.tanh(), |a| a.tanh());
}
pub fn test_sqr<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0, 3.0], (3,), |a| a.sqr(), |a| a.sqr());
}
pub fn test_sqrt<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 4.0, 9.0], (3,), |a| a.sqrt(), |a| a.sqrt());
}
pub fn test_recip<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0, 4.0], (3,), |a| a.recip(), |a| a.recip());
}
pub fn test_relu<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, -2.0, 3.0], (3,), |a| a.relu(), |a| a.relu());
}
pub fn test_sigmoid<D: Device>(dev: &D) {
    unary_check(dev, &[0.0, 1.0, -1.0], (3,), |a| a.sigmoid(), |a| a.sigmoid());
}
pub fn test_silu<D: Device>(dev: &D) {
    unary_check(dev, &[0.0, 1.0, -1.0], (3,), |a| a.silu(), |a| a.silu());
}
pub fn test_gelu<D: Device>(dev: &D) {
    unary_check(dev, &[0.0, 1.0, -1.0], (3,), |a| a.gelu(), |a| a.gelu());
}
pub fn test_gelu_erf<D: Device>(dev: &D) {
    unary_check(dev, &[0.0, 1.0, -1.0], (3,), |a| a.gelu_erf(), |a| a.gelu_erf());
}
pub fn test_erf<D: Device>(dev: &D) {
    unary_check(dev, &[0.0, 1.0, -1.0], (3,), |a| a.erf(), |a| a.erf());
}
pub fn test_floor<D: Device>(dev: &D) {
    unary_check(dev, &[1.5, -1.5, 2.0], (3,), |a| a.floor(), |a| a.floor());
}
pub fn test_ceil<D: Device>(dev: &D) {
    unary_check(dev, &[1.5, -1.5, 2.0], (3,), |a| a.ceil(), |a| a.ceil());
}
pub fn test_round<D: Device>(dev: &D) {
    unary_check(dev, &[1.4, 2.6, -1.4], (3,), |a| a.round(), |a| a.round());
}

// ---- unary with scalar parameter (float) -------------------------------------

pub fn test_affine<D: Device>(dev: &D) {
    let data = [1.0, 2.0, 3.0];
    let real = tensor_f32(dev, &data, (3,));
    let expected = real.affine(2.0, 1.0).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(&[3], 0.0, (t, FloatDType::F32)).unwrap();
            let in_id = ta.trace_id();
            let o = ta.affine(2.0, 1.0).unwrap();
            (vec![in_id], o.trace_id())
        },
        vec![real.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}
pub fn test_pow<D: Device>(dev: &D) {
    let data = [1.0, 2.0, 3.0];
    let real = tensor_f32(dev, &data, (3,));
    let expected = real.pow(2.0).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(&[3], 0.0, (t, FloatDType::F32)).unwrap();
            let in_id = ta.trace_id();
            let o = ta.pow(2.0).unwrap();
            (vec![in_id], o.trace_id())
        },
        vec![real.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}
pub fn test_clamp<D: Device>(dev: &D) {
    let data = [1.0, -2.0, 5.0];
    let real = tensor_f32(dev, &data, (3,));
    let expected = real.clamp(Some(0.0), Some(3.0)).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(&[3], 0.0, (t, FloatDType::F32)).unwrap();
            let in_id = ta.trace_id();
            let o = ta.clamp(Some(0.0), Some(3.0)).unwrap();
            (vec![in_id], o.trace_id())
        },
        vec![real.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}
pub fn test_leaky_relu<D: Device>(dev: &D) {
    let data = [1.0, -2.0, 3.0];
    let real = tensor_f32(dev, &data, (3,));
    let expected = real.leaky_relu(0.1).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(&[3], 0.0, (t, FloatDType::F32)).unwrap();
            let in_id = ta.trace_id();
            let o = ta.leaky_relu(0.1).unwrap();
            (vec![in_id], o.trace_id())
        },
        vec![real.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}

// ---- unary (int) -------------------------------------------------------------

pub fn test_neg_i32<D: Device>(dev: &D) {
    let a = tensor_i32(dev, &[1, -2, 3], (3,));
    let expected = a.neg().unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Int>::full(&[3], 0, (t, IntDType::I32)).unwrap();
            let in_id = ta.trace_id();
            let o = ta.neg().unwrap();
            (vec![in_id], o.trace_id())
        },
        vec![a.clone().into()],
    );
    assert_eq!(as_i64s(&out[0]), expected);
}
pub fn test_abs_i32<D: Device>(dev: &D) {
    let a = tensor_i32(dev, &[1, -2, 3], (3,));
    let expected = a.abs().unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Int>::full(&[3], 0, (t, IntDType::I32)).unwrap();
            let in_id = ta.trace_id();
            let o = ta.abs().unwrap();
            (vec![in_id], o.trace_id())
        },
        vec![a.clone().into()],
    );
    assert_eq!(as_i64s(&out[0]), expected);
}
pub fn test_sign_i32<D: Device>(dev: &D) {
    let a = tensor_i32(dev, &[1, -2, 0], (3,));
    let expected = a.sign().unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Int>::full(&[3], 0, (t, IntDType::I32)).unwrap();
            let in_id = ta.trace_id();
            let o = ta.sign().unwrap();
            (vec![in_id], o.trace_id())
        },
        vec![a.clone().into()],
    );
    assert_eq!(as_i64s(&out[0]), expected);
}

// ---- comparison --------------------------------------------------------------

pub fn test_eq<D: Device>(dev: &D) {
    cmp_check(dev, &[1.0, 2.0, 3.0], (3,), &[1.0, 0.0, 3.0], (3,), |a, b| a.eq(b), |a, b| a.eq(b));
}
pub fn test_ne<D: Device>(dev: &D) {
    cmp_check(dev, &[1.0, 2.0, 3.0], (3,), &[1.0, 0.0, 3.0], (3,), |a, b| a.ne(b), |a, b| a.ne(b));
}
pub fn test_lt<D: Device>(dev: &D) {
    cmp_check(dev, &[1.0, 2.0, 3.0], (3,), &[2.0, 2.0, 2.0], (3,), |a, b| a.lt(b), |a, b| a.lt(b));
}
pub fn test_gt<D: Device>(dev: &D) {
    cmp_check(dev, &[1.0, 2.0, 3.0], (3,), &[2.0, 2.0, 2.0], (3,), |a, b| a.gt(b), |a, b| a.gt(b));
}
pub fn test_le<D: Device>(dev: &D) {
    cmp_check(dev, &[1.0, 2.0, 3.0], (3,), &[2.0, 2.0, 2.0], (3,), |a, b| a.le(b), |a, b| a.le(b));
}
pub fn test_ge<D: Device>(dev: &D) {
    cmp_check(dev, &[1.0, 2.0, 3.0], (3,), &[2.0, 2.0, 2.0], (3,), |a, b| a.ge(b), |a, b| a.ge(b));
}

pub fn test_gt_scalar<D: Device>(dev: &D) {
    cmp_scalar_check(dev, &[1.0, 2.0, 3.0], (3,), 2.0, |a, s| a.gt_scalar(s), |a, s| a.gt_scalar(s));
}
