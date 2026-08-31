//! Device-generic cast executor tests.

use super::*;
use luma_compile::Traced;
use luma_tensor::dtype::{BoolDType, FloatDType, IntDType};
use luma_tensor::{Device, Float, Int, Tensor};

pub fn test_cast_f32_f64<D: Device>(dev: &D) {
    let a = tensor_f32(dev, &[1.5, 2.5, 3.5], (3,));
    let expected = a.cast(FloatDType::F64).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(&[3], 0.0, (t, FloatDType::F32)).unwrap();
            let in_id = ta.trace_id();
            let o = ta.cast(FloatDType::F64).unwrap();
            (vec![in_id], o.trace_id())
        },
        vec![a.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-6);
}

pub fn test_cast_f32_i32<D: Device>(dev: &D) {
    let a = tensor_f32(dev, &[1.9, -2.9, 3.0], (3,));
    let expected = a.cast(IntDType::I32).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(&[3], 0.0, (t, FloatDType::F32)).unwrap();
            let in_id = ta.trace_id();
            let o = ta.cast(IntDType::I32).unwrap();
            (vec![in_id], o.trace_id())
        },
        vec![a.clone().into()],
    );
    assert_eq!(as_i64s(&out[0]), expected);
}

pub fn test_cast_f32_bool<D: Device>(dev: &D) {
    let a = tensor_f32(dev, &[1.0, 0.0, -2.0], (3,));
    let expected = a.cast(BoolDType::Bool).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(&[3], 0.0, (t, FloatDType::F32)).unwrap();
            let in_id = ta.trace_id();
            let o = ta.cast(BoolDType::Bool).unwrap();
            (vec![in_id], o.trace_id())
        },
        vec![a.clone().into()],
    );
    assert_eq!(as_bools(&out[0]), expected);
}

pub fn test_cast_i32_f32<D: Device>(dev: &D) {
    let a = tensor_i32(dev, &[1, 2, 3], (3,));
    let expected = a.cast(FloatDType::F32).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Int>::full(&[3], 0, (t, IntDType::I32)).unwrap();
            let in_id = ta.trace_id();
            let o = ta.cast(FloatDType::F32).unwrap();
            (vec![in_id], o.trace_id())
        },
        vec![a.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}

pub fn test_cast_i32_bool<D: Device>(dev: &D) {
    let a = tensor_i32(dev, &[1, 0, -2], (3,));
    let expected = a.cast(BoolDType::Bool).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Int>::full(&[3], 0, (t, IntDType::I32)).unwrap();
            let in_id = ta.trace_id();
            let o = ta.cast(BoolDType::Bool).unwrap();
            (vec![in_id], o.trace_id())
        },
        vec![a.clone().into()],
    );
    assert_eq!(as_bools(&out[0]), expected);
}

pub fn test_cast_bool_f32<D: Device>(dev: &D) {
    let a = tensor_bool(dev, &[true, false, true], (3,));
    let expected = a.cast(FloatDType::F32).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Bool>::falses(&[3], (t, BoolDType::Bool)).unwrap();
            let in_id = ta.trace_id();
            let o = ta.cast(FloatDType::F32).unwrap();
            (vec![in_id], o.trace_id())
        },
        vec![a.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}

pub fn test_cast_bool_i32<D: Device>(dev: &D) {
    let a = tensor_bool(dev, &[true, false, true], (3,));
    let expected = a.cast(IntDType::I32).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Bool>::falses(&[3], (t, BoolDType::Bool)).unwrap();
            let in_id = ta.trace_id();
            let o = ta.cast(IntDType::I32).unwrap();
            (vec![in_id], o.trace_id())
        },
        vec![a.clone().into()],
    );
    assert_eq!(as_i64s(&out[0]), expected);
}
