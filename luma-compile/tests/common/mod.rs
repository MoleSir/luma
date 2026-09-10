//! Shared, device-generic executor tests: trace a (set of) op(s) on the
//! `Trace` device, compile the graph for a concrete device, run it, and compare
//! against the direct op result on that same device. Each function takes a `&D`
//! so the same closed loop runs on `Cpu` and `Cuda`.
#![allow(dead_code)]

use luma_compile::{Trace, Traced, ValueId, trace};
use luma_nn::Linear;
use luma_nn::loss::CrossEntropyLoss;
use luma_tensor::dtype::{BoolDType, FloatDType, IntDType};
use luma_tensor::{Bool, Device, DynTensor, Float, Int, Shape, Tensor};

pub mod boolean;
pub mod cast;
pub mod indexing;
pub mod module;
pub mod nn;
pub mod numeric;
pub mod reduce;
pub mod shape;

/// Assert two `f64` slices match elementwise within an absolute tolerance.
pub fn assert_close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(diff <= tol, "mismatch at index {}: {} vs {} (diff {:.2e})", i, x, y, diff);
    }
}

// ---- input constructors (real tensors on the target device) ------------------

pub fn tensor_f32<D: Device, S: Into<Shape>>(dev: &D, data: &[f64], shape: S) -> Tensor<D> {
    Tensor::<D>::from_slice(data, shape, (dev, FloatDType::F32)).unwrap()
}

pub fn tensor_i32<D: Device, S: Into<Shape>>(dev: &D, data: &[i64], shape: S) -> Tensor<D, Int> {
    Tensor::<D, Int>::from_slice(data, shape, (dev, IntDType::I32)).unwrap()
}

pub fn tensor_bool<D: Device, S: Into<Shape>>(dev: &D, data: &[bool], shape: S) -> Tensor<D, Bool> {
    Tensor::<D, Bool>::from_slice(data, shape, (dev, BoolDType::Bool)).unwrap()
}

// ---- output accessors --------------------------------------------------------

pub fn as_f64s<D: Device>(out: &DynTensor<D>) -> Vec<f64> {
    out.as_float().unwrap().to_vec().unwrap()
}

pub fn as_i64s<D: Device>(out: &DynTensor<D>) -> Vec<i64> {
    out.as_int().unwrap().to_vec().unwrap()
}

pub fn as_bools<D: Device>(out: &DynTensor<D>) -> Vec<bool> {
    out.as_bool().unwrap().to_vec().unwrap()
}

// ---- the trace → compile → run closed loop -----------------------------------

/// Trace `build` on the `Trace` device, compile the graph for `dev`, run it on
/// the given `inputs`, and return the graph outputs. `build` returns the tuple
/// `(input value ids, output value id)`; the ids are the `trace_id()`s of the
/// traced leaves and the final output.
pub fn execute<D: Device>(dev: &D, build: impl FnOnce(&Trace) -> (Vec<ValueId>, ValueId), inputs: Vec<DynTensor<D>>) -> Vec<DynTensor<D>> {
    let trace_dev = Trace::new();
    let (ins, out) = build(&trace_dev);
    let graph = trace_dev.graph();
    {
        let mut g = graph.lock().unwrap();
        for id in ins {
            g.mark_input(id);
        }
        g.mark_output(out);
    }
    let mut exec = graph.lock().unwrap().compile(dev).unwrap();
    exec.run(&inputs).unwrap()
}

// ---- one-line op checks (trace → run → compare against the direct op) --------

/// One float input → one float output.
pub fn unary_check<D, S, F, G>(dev: &D, data: &[f64], shape: S, trace_op: F, real_op: G)
where
    D: Device,
    S: Into<Shape> + Clone,
    F: FnOnce(&Tensor<Trace, Float>) -> luma_tensor::Result<Tensor<Trace, Float>>,
    G: FnOnce(&Tensor<D>) -> luma_tensor::Result<Tensor<D>>,
{
    let real = tensor_f32(dev, data, shape.clone());
    let expected = real_op(&real).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(shape.clone(), 0.0, (t, FloatDType::F32)).unwrap();
            let in_id = ta.trace_id();
            let o = trace_op(&ta).unwrap();
            (vec![in_id], o.trace_id())
        },
        vec![real.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}

/// Two float inputs → one float output.
pub fn binary_check<D, S1, S2, F, G>(dev: &D, a: &[f64], sa: S1, b: &[f64], sb: S2, trace_op: F, real_op: G)
where
    D: Device,
    S1: Into<Shape> + Clone,
    S2: Into<Shape> + Clone,
    F: FnOnce(&Tensor<Trace, Float>, &Tensor<Trace, Float>) -> luma_tensor::Result<Tensor<Trace, Float>>,
    G: FnOnce(&Tensor<D>, &Tensor<D>) -> luma_tensor::Result<Tensor<D>>,
{
    let ra = tensor_f32(dev, a, sa.clone());
    let rb = tensor_f32(dev, b, sb.clone());
    let expected = real_op(&ra, &rb).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(sa.clone(), 0.0, (t, FloatDType::F32)).unwrap();
            let tb = Tensor::<Trace, Float>::full(sb.clone(), 0.0, (t, FloatDType::F32)).unwrap();
            let ia = ta.trace_id();
            let ib = tb.trace_id();
            let o = trace_op(&ta, &tb).unwrap();
            (vec![ia, ib], o.trace_id())
        },
        vec![ra.clone().into(), rb.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}

/// One float input + scalar → one float output.
pub fn scalar_rhs_check<D, S, F, G>(dev: &D, data: &[f64], shape: S, scalar: f64, trace_op: F, real_op: G)
where
    D: Device,
    S: Into<Shape> + Clone,
    F: FnOnce(&Tensor<Trace, Float>, f64) -> luma_tensor::Result<Tensor<Trace, Float>>,
    G: FnOnce(&Tensor<D>, f64) -> luma_tensor::Result<Tensor<D>>,
{
    let real = tensor_f32(dev, data, shape.clone());
    let expected = real_op(&real, scalar).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(shape.clone(), 0.0, (t, FloatDType::F32)).unwrap();
            let in_id = ta.trace_id();
            let o = trace_op(&ta, scalar).unwrap();
            (vec![in_id], o.trace_id())
        },
        vec![real.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}

/// One float input with a scalar on the LHS → one float output.
pub fn scalar_lhs_check<D, S, F, G>(dev: &D, data: &[f64], shape: S, scalar: f64, trace_op: F, real_op: G)
where
    D: Device,
    S: Into<Shape> + Clone,
    F: FnOnce(&Tensor<Trace, Float>, f64) -> luma_tensor::Result<Tensor<Trace, Float>>,
    G: FnOnce(&Tensor<D>, f64) -> luma_tensor::Result<Tensor<D>>,
{
    let real = tensor_f32(dev, data, shape.clone());
    let expected = real_op(&real, scalar).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(shape.clone(), 0.0, (t, FloatDType::F32)).unwrap();
            let in_id = ta.trace_id();
            let o = trace_op(&ta, scalar).unwrap();
            (vec![in_id], o.trace_id())
        },
        vec![real.clone().into()],
    );
    assert_close(&as_f64s(&out[0]), &expected, 1e-5);
}

/// Two float inputs → one bool output.
pub fn cmp_check<D, S1, S2, F, G>(dev: &D, a: &[f64], sa: S1, b: &[f64], sb: S2, trace_op: F, real_op: G)
where
    D: Device,
    S1: Into<Shape> + Clone,
    S2: Into<Shape> + Clone,
    F: FnOnce(&Tensor<Trace, Float>, &Tensor<Trace, Float>) -> luma_tensor::Result<Tensor<Trace, Bool>>,
    G: FnOnce(&Tensor<D>, &Tensor<D>) -> luma_tensor::Result<Tensor<D, Bool>>,
{
    let ra = tensor_f32(dev, a, sa.clone());
    let rb = tensor_f32(dev, b, sb.clone());
    let expected = real_op(&ra, &rb).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(sa.clone(), 0.0, (t, FloatDType::F32)).unwrap();
            let tb = Tensor::<Trace, Float>::full(sb.clone(), 0.0, (t, FloatDType::F32)).unwrap();
            let ia = ta.trace_id();
            let ib = tb.trace_id();
            let o = trace_op(&ta, &tb).unwrap();
            (vec![ia, ib], o.trace_id())
        },
        vec![ra.clone().into(), rb.clone().into()],
    );
    assert_eq!(&as_bools(&out[0]), &expected, "cmp mismatch");
}

/// One float input + scalar → one bool output.
pub fn cmp_scalar_check<D, S, F, G>(dev: &D, data: &[f64], shape: S, scalar: f64, trace_op: F, real_op: G)
where
    D: Device,
    S: Into<Shape> + Clone,
    F: FnOnce(&Tensor<Trace, Float>, f64) -> luma_tensor::Result<Tensor<Trace, Bool>>,
    G: FnOnce(&Tensor<D>, f64) -> luma_tensor::Result<Tensor<D, Bool>>,
{
    let real = tensor_f32(dev, data, shape.clone());
    let expected = real_op(&real, scalar).unwrap().to_vec().unwrap();
    let out = execute(
        dev,
        |t| {
            let ta = Tensor::<Trace, Float>::full(shape.clone(), 0.0, (t, FloatDType::F32)).unwrap();
            let in_id = ta.trace_id();
            let o = trace_op(&ta, scalar).unwrap();
            (vec![in_id], o.trace_id())
        },
        vec![real.clone().into()],
    );
    assert_eq!(&as_bools(&out[0]), &expected, "cmp scalar mismatch");
}
