//! Device-generic nn / arange executor tests.

use super::*;
use luma_compile::Traced;
use luma_tensor::dtype::{FloatDType, IntDType};
use luma_tensor::{Device, Float, Int, Tensor};

pub fn test_softmax<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], (2, 3), |a| a.softmax(1usize), |a| a.softmax(1usize));
}

pub fn test_rms_norm<D: Device>(dev: &D) {
    let a = tensor_f32(dev, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
    let w = tensor_f32(dev, &[1.0, 1.0, 1.0], (3,));
    let expected = a.rms_norm(Some(&w), None, 1e-5).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(&[2, 3], 0.0, (t, FloatDType::F32)).unwrap();
            let tw = Tensor::<Trace, Float>::full(&[3], 0.0, (t, FloatDType::F32)).unwrap();
            let ia = ta.trace_id();
            let iw = tw.trace_id();
            let o = ta.rms_norm(Some(&tw), None, 1e-5).unwrap();
            (vec![ia, iw], o.trace_id())
        },
        vec![a.clone().into(), w.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}

pub fn test_arange<D: Device>(dev: &D) {
    let expected = Tensor::<D, Int>::arange(0, 5, 1, (dev, IntDType::I32)).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let o = Tensor::<Trace, Int>::arange(0, 5, 1, (t, IntDType::I32)).unwrap();
            (vec![], o.trace_id())
        },
        vec![],
    );
    assert_eq!(as_i64s(&out[0]), expected);
}

pub fn test_max_pool2d<D: Device>(dev: &D) {
    let data: Vec<f64> = (1..=16).map(|v| v as f64).collect();
    let a = tensor_f32(dev, &data, (1, 1, 4, 4));
    let expected = a.max_pool2d((2, 2), (2, 2), (0, 0), (1, 1), false).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(&[1, 1, 4, 4], 0.0, (t, FloatDType::F32)).unwrap();
            let ia = ta.trace_id();
            let o = ta.max_pool2d((2, 2), (2, 2), (0, 0), (1, 1), false).unwrap();
            (vec![ia], o.trace_id())
        },
        vec![a.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}

pub fn test_avg_pool2d<D: Device>(dev: &D) {
    let data: Vec<f64> = (1..=16).map(|v| v as f64).collect();
    let a = tensor_f32(dev, &data, (1, 1, 4, 4));
    let expected = a.avg_pool2d((2, 2), (2, 2), (0, 0), false, true).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(&[1, 1, 4, 4], 0.0, (t, FloatDType::F32)).unwrap();
            let ia = ta.trace_id();
            let o = ta.avg_pool2d((2, 2), (2, 2), (0, 0), false, true).unwrap();
            (vec![ia], o.trace_id())
        },
        vec![a.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}

pub fn test_adaptive_avg_pool2d<D: Device>(dev: &D) {
    let data: Vec<f64> = (1..=25).map(|v| v as f64).collect();
    let a = tensor_f32(dev, &data, (1, 1, 5, 5));
    let expected = a.adaptive_avg_pool2d((2, 2)).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(&[1, 1, 5, 5], 0.0, (t, FloatDType::F32)).unwrap();
            let ia = ta.trace_id();
            let o = ta.adaptive_avg_pool2d((2, 2)).unwrap();
            (vec![ia], o.trace_id())
        },
        vec![a.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}

pub fn test_mse_loss<D: Device>(dev: &D) {
    let pred = tensor_f32(dev, &[1.0, 2.0, 3.0, 4.0], (2, 2));
    let target = tensor_f32(dev, &[0.0, 0.0, 0.0, 0.0], (2, 2));
    let expected = pred.mse_loss(&target).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let tp = Tensor::<Trace, Float>::full(&[2, 2], 0.0, (t, FloatDType::F32)).unwrap();
            let tt = Tensor::<Trace, Float>::full(&[2, 2], 0.0, (t, FloatDType::F32)).unwrap();
            let ip = tp.trace_id();
            let it = tt.trace_id();
            let o = tp.mse_loss(&tt).unwrap();
            (vec![ip, it], o.trace_id())
        },
        vec![pred.clone().into(), target.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}

pub fn test_cross_entropy_loss<D: Device>(dev: &D) {
    let logits = tensor_f32(dev, &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0], (2, 3));
    let target = tensor_i32(dev, &[0, 2], (2,));
    let expected = logits.cross_entropy_loss(&target).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let tl = Tensor::<Trace, Float>::full(&[2, 3], 0.0, (t, FloatDType::F32)).unwrap();
            let tt = Tensor::<Trace, Int>::full(&[2], 0, (t, IntDType::I32)).unwrap();
            let il = tl.trace_id();
            let it = tt.trace_id();
            let o = tl.cross_entropy_loss(&tt).unwrap();
            (vec![il, it], o.trace_id())
        },
        vec![logits.clone().into(), target.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}
