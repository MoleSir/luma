//! Device-generic boolean-logic / pick executor tests.

use super::*;
use luma_compile::Traced;
use luma_tensor::dtype::{BoolDType, FloatDType, IntDType};
use luma_tensor::{Bool, Device, Float, Int, Tensor};

pub fn test_and<D: Device>(dev: &D) {
    let a = tensor_bool(dev, &[true, false, true], (3,));
    let b = tensor_bool(dev, &[true, true, false], (3,));
    let expected = a.and(&b).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Bool>::falses(&[3], (t, BoolDType::Bool)).unwrap();
            let tb = Tensor::<Trace, Bool>::falses(&[3], (t, BoolDType::Bool)).unwrap();
            let ia = ta.trace_id();
            let ib = tb.trace_id();
            let o = ta.and(&tb).unwrap();
            (vec![ia, ib], o.trace_id())
        },
        vec![a.clone().into(), b.clone().into()],
    );
    assert_eq!(as_bools(&out[0]), expected);
}

pub fn test_or<D: Device>(dev: &D) {
    let a = tensor_bool(dev, &[true, false, true], (3,));
    let b = tensor_bool(dev, &[true, true, false], (3,));
    let expected = a.or(&b).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Bool>::falses(&[3], (t, BoolDType::Bool)).unwrap();
            let tb = Tensor::<Trace, Bool>::falses(&[3], (t, BoolDType::Bool)).unwrap();
            let ia = ta.trace_id();
            let ib = tb.trace_id();
            let o = ta.or(&tb).unwrap();
            (vec![ia, ib], o.trace_id())
        },
        vec![a.clone().into(), b.clone().into()],
    );
    assert_eq!(as_bools(&out[0]), expected);
}

pub fn test_xor<D: Device>(dev: &D) {
    let a = tensor_bool(dev, &[true, false, true], (3,));
    let b = tensor_bool(dev, &[true, true, false], (3,));
    let expected = a.xor(&b).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Bool>::falses(&[3], (t, BoolDType::Bool)).unwrap();
            let tb = Tensor::<Trace, Bool>::falses(&[3], (t, BoolDType::Bool)).unwrap();
            let ia = ta.trace_id();
            let ib = tb.trace_id();
            let o = ta.xor(&tb).unwrap();
            (vec![ia, ib], o.trace_id())
        },
        vec![a.clone().into(), b.clone().into()],
    );
    assert_eq!(as_bools(&out[0]), expected);
}

pub fn test_not<D: Device>(dev: &D) {
    let a = tensor_bool(dev, &[true, false, true], (3,));
    let expected = a.not().unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Bool>::falses(&[3], (t, BoolDType::Bool)).unwrap();
            let in_id = ta.trace_id();
            let o = ta.not().unwrap();
            (vec![in_id], o.trace_id())
        },
        vec![a.clone().into()],
    );
    assert_eq!(as_bools(&out[0]), expected);
}

// ---- pick (3-input) ----------------------------------------------------------

pub fn test_pick_f32<D: Device>(dev: &D) {
    let m = tensor_bool(dev, &[true, false, true], (3,));
    let t = tensor_f32(dev, &[1.0, 2.0, 3.0], (3,));
    let f = tensor_f32(dev, &[10.0, 20.0, 30.0], (3,));
    let expected = m.pick(&t, &f).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |d| {
            let tm = Tensor::<Trace, Bool>::falses(&[3], (d, BoolDType::Bool)).unwrap();
            let tt = Tensor::<Trace, Float>::full(&[3], 0.0, (d, FloatDType::F32)).unwrap();
            let tf = Tensor::<Trace, Float>::full(&[3], 0.0, (d, FloatDType::F32)).unwrap();
            let im = tm.trace_id();
            let it = tt.trace_id();
            let iff = tf.trace_id();
            let o = tm.pick(&tt, &tf).unwrap();
            (vec![im, it, iff], o.trace_id())
        },
        vec![m.clone().into(), t.clone().into(), f.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}

pub fn test_pick_i32<D: Device>(dev: &D) {
    let m = tensor_bool(dev, &[true, false, true], (3,));
    let t = tensor_i32(dev, &[1, 2, 3], (3,));
    let f = tensor_i32(dev, &[10, 20, 30], (3,));
    let expected = m.pick(&t, &f).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |d| {
            let tm = Tensor::<Trace, Bool>::falses(&[3], (d, BoolDType::Bool)).unwrap();
            let tt = Tensor::<Trace, Int>::full(&[3], 0, (d, IntDType::I32)).unwrap();
            let tf = Tensor::<Trace, Int>::full(&[3], 0, (d, IntDType::I32)).unwrap();
            let im = tm.trace_id();
            let it = tt.trace_id();
            let iff = tf.trace_id();
            let o = tm.pick(&tt, &tf).unwrap();
            (vec![im, it, iff], o.trace_id())
        },
        vec![m.clone().into(), t.clone().into(), f.clone().into()],
    );
    assert_eq!(as_i64s(&out[0]), expected);
}

pub fn test_pick_bool<D: Device>(dev: &D) {
    let m = tensor_bool(dev, &[true, false], (2,));
    let t = tensor_bool(dev, &[true, false], (2,));
    let f = tensor_bool(dev, &[false, true], (2,));
    let expected = m.pick(&t, &f).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |d| {
            let tm = Tensor::<Trace, Bool>::falses(&[2], (d, BoolDType::Bool)).unwrap();
            let tt = Tensor::<Trace, Bool>::falses(&[2], (d, BoolDType::Bool)).unwrap();
            let tf = Tensor::<Trace, Bool>::falses(&[2], (d, BoolDType::Bool)).unwrap();
            let im = tm.trace_id();
            let it = tt.trace_id();
            let iff = tf.trace_id();
            let o = tm.pick(&tt, &tf).unwrap();
            (vec![im, it, iff], o.trace_id())
        },
        vec![m.clone().into(), t.clone().into(), f.clone().into()],
    );
    assert_eq!(as_bools(&out[0]), expected);
}

// ---- pick_true / pick_false (scalar branch) ----------------------------------

pub fn test_pick_true_f32<D: Device>(dev: &D) {
    let m = tensor_bool(dev, &[true, false, true], (3,));
    let f = tensor_f32(dev, &[10.0, 20.0, 30.0], (3,));
    let expected = m.pick_true(5.0, &f).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |d| {
            let tm = Tensor::<Trace, Bool>::falses(&[3], (d, BoolDType::Bool)).unwrap();
            let tf = Tensor::<Trace, Float>::full(&[3], 0.0, (d, FloatDType::F32)).unwrap();
            let im = tm.trace_id();
            let iff = tf.trace_id();
            let o = tm.pick_true(5.0, &tf).unwrap();
            (vec![im, iff], o.trace_id())
        },
        vec![m.clone().into(), f.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}

pub fn test_pick_true_i32<D: Device>(dev: &D) {
    let m = tensor_bool(dev, &[true, false, true], (3,));
    let f = tensor_i32(dev, &[10, 20, 30], (3,));
    let expected = m.pick_true(5, &f).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |d| {
            let tm = Tensor::<Trace, Bool>::falses(&[3], (d, BoolDType::Bool)).unwrap();
            let tf = Tensor::<Trace, Int>::full(&[3], 0, (d, IntDType::I32)).unwrap();
            let im = tm.trace_id();
            let iff = tf.trace_id();
            let o = tm.pick_true(5, &tf).unwrap();
            (vec![im, iff], o.trace_id())
        },
        vec![m.clone().into(), f.clone().into()],
    );
    assert_eq!(as_i64s(&out[0]), expected);
}

pub fn test_pick_true_bool<D: Device>(dev: &D) {
    let m = tensor_bool(dev, &[true, false], (2,));
    let f = tensor_bool(dev, &[false, true], (2,));
    let expected = m.pick_true(true, &f).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |d| {
            let tm = Tensor::<Trace, Bool>::falses(&[2], (d, BoolDType::Bool)).unwrap();
            let tf = Tensor::<Trace, Bool>::falses(&[2], (d, BoolDType::Bool)).unwrap();
            let im = tm.trace_id();
            let iff = tf.trace_id();
            let o = tm.pick_true(true, &tf).unwrap();
            (vec![im, iff], o.trace_id())
        },
        vec![m.clone().into(), f.clone().into()],
    );
    assert_eq!(as_bools(&out[0]), expected);
}

pub fn test_pick_false_f32<D: Device>(dev: &D) {
    let m = tensor_bool(dev, &[true, false, true], (3,));
    let t = tensor_f32(dev, &[1.0, 2.0, 3.0], (3,));
    let expected = m.pick_false(&t, 5.0).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |d| {
            let tm = Tensor::<Trace, Bool>::falses(&[3], (d, BoolDType::Bool)).unwrap();
            let tt = Tensor::<Trace, Float>::full(&[3], 0.0, (d, FloatDType::F32)).unwrap();
            let im = tm.trace_id();
            let it = tt.trace_id();
            let o = tm.pick_false(&tt, 5.0).unwrap();
            (vec![im, it], o.trace_id())
        },
        vec![m.clone().into(), t.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}

pub fn test_pick_false_i32<D: Device>(dev: &D) {
    let m = tensor_bool(dev, &[true, false, true], (3,));
    let t = tensor_i32(dev, &[1, 2, 3], (3,));
    let expected = m.pick_false(&t, 5).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |d| {
            let tm = Tensor::<Trace, Bool>::falses(&[3], (d, BoolDType::Bool)).unwrap();
            let tt = Tensor::<Trace, Int>::full(&[3], 0, (d, IntDType::I32)).unwrap();
            let im = tm.trace_id();
            let it = tt.trace_id();
            let o = tm.pick_false(&tt, 5).unwrap();
            (vec![im, it], o.trace_id())
        },
        vec![m.clone().into(), t.clone().into()],
    );
    assert_eq!(as_i64s(&out[0]), expected);
}

pub fn test_pick_false_bool<D: Device>(dev: &D) {
    let m = tensor_bool(dev, &[true, false], (2,));
    let t = tensor_bool(dev, &[true, false], (2,));
    let expected = m.pick_false(&t, false).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |d| {
            let tm = Tensor::<Trace, Bool>::falses(&[2], (d, BoolDType::Bool)).unwrap();
            let tt = Tensor::<Trace, Bool>::falses(&[2], (d, BoolDType::Bool)).unwrap();
            let im = tm.trace_id();
            let it = tt.trace_id();
            let o = tm.pick_false(&tt, false).unwrap();
            (vec![im, it], o.trace_id())
        },
        vec![m.clone().into(), t.clone().into()],
    );
    assert_eq!(as_bools(&out[0]), expected);
}
