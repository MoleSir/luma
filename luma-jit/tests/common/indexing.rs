//! Device-generic indexing / cat executor tests.

use super::*;
use luma_jit::Traced;
use luma_tensor::dtype::{FloatDType, IntDType};
use luma_tensor::{Device, Float, Int, Tensor};

pub fn test_index_select<D: Device>(dev: &D) {
    let a = tensor_f32(dev, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2));
    let idx = tensor_i32(dev, &[2, 0], (2,));
    let expected = a.index_select(&idx, 0usize).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(&[3, 2], 0.0, (t, FloatDType::F32)).unwrap();
            let ti = Tensor::<Trace, Int>::full(&[2], 0, (t, IntDType::I32)).unwrap();
            let ia = ta.trace_id();
            let ii = ti.trace_id();
            let o = ta.index_select(&ti, 0usize).unwrap();
            (vec![ia, ii], o.trace_id())
        },
        vec![a.clone().into(), idx.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}

pub fn test_gather<D: Device>(dev: &D) {
    let a = tensor_f32(dev, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], (3, 3));
    let idx = tensor_i32(dev, &[0, 2, 1, 2, 0, 1, 1, 1, 0], (3, 3));
    let expected = a.gather(&idx, 1usize).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(&[3, 3], 0.0, (t, FloatDType::F32)).unwrap();
            let ti = Tensor::<Trace, Int>::full(&[3, 3], 0, (t, IntDType::I32)).unwrap();
            let ia = ta.trace_id();
            let ii = ti.trace_id();
            let o = ta.gather(&ti, 1usize).unwrap();
            (vec![ia, ii], o.trace_id())
        },
        vec![a.clone().into(), idx.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}

pub fn test_index_add<D: Device>(dev: &D) {
    let a = tensor_f32(dev, &[1.0, 2.0, 3.0, 4.0], (4,));
    let idx = tensor_i32(dev, &[0, 2], (2,));
    let src = tensor_f32(dev, &[10.0, 20.0], (2,));
    let expected = a.index_add(&idx, &src, 0usize).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(&[4], 0.0, (t, FloatDType::F32)).unwrap();
            let ti = Tensor::<Trace, Int>::full(&[2], 0, (t, IntDType::I32)).unwrap();
            let ts = Tensor::<Trace, Float>::full(&[2], 0.0, (t, FloatDType::F32)).unwrap();
            let ia = ta.trace_id();
            let ii = ti.trace_id();
            let is = ts.trace_id();
            let o = ta.index_add(&ti, &ts, 0usize).unwrap();
            (vec![ia, ii, is], o.trace_id())
        },
        vec![a.clone().into(), idx.clone().into(), src.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}

pub fn test_scatter_add<D: Device>(dev: &D) {
    let a = tensor_f32(dev, &[1.0, 2.0, 3.0, 4.0], (4,));
    let idx = tensor_i32(dev, &[0, 2], (2,));
    let src = tensor_f32(dev, &[10.0, 20.0], (2,));
    let expected = a.scatter_add(&idx, &src, 0usize).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(&[4], 0.0, (t, FloatDType::F32)).unwrap();
            let ti = Tensor::<Trace, Int>::full(&[2], 0, (t, IntDType::I32)).unwrap();
            let ts = Tensor::<Trace, Float>::full(&[2], 0.0, (t, FloatDType::F32)).unwrap();
            let ia = ta.trace_id();
            let ii = ti.trace_id();
            let is = ts.trace_id();
            let o = ta.scatter_add(&ti, &ts, 0usize).unwrap();
            (vec![ia, ii, is], o.trace_id())
        },
        vec![a.clone().into(), idx.clone().into(), src.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}

pub fn test_cat<D: Device>(dev: &D) {
    let a = tensor_f32(dev, &[1.0, 2.0], (2,));
    let b = tensor_f32(dev, &[3.0, 4.0, 5.0], (3,));
    let expected = Tensor::<D>::cat(&[&a, &b], 0usize).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(&[2], 0.0, (t, FloatDType::F32)).unwrap();
            let tb = Tensor::<Trace, Float>::full(&[3], 0.0, (t, FloatDType::F32)).unwrap();
            let ia = ta.trace_id();
            let ib = tb.trace_id();
            let o = Tensor::<Trace, Float>::cat(&[&ta, &tb], 0usize).unwrap();
            (vec![ia, ib], o.trace_id())
        },
        vec![a.clone().into(), b.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}
