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
    let expected = a.rms_norm(&w, 1e-5).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(&[2, 3], 0.0, (t, FloatDType::F32)).unwrap();
            let tw = Tensor::<Trace, Float>::full(&[3], 0.0, (t, FloatDType::F32)).unwrap();
            let ia = ta.trace_id();
            let iw = tw.trace_id();
            let o = ta.rms_norm(&tw, 1e-5).unwrap();
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
