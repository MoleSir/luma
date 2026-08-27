//! Device-generic module-level tests: trace a real module, compile, run, and
//! compare against its own `forward`.

use super::*;
use luma_tensor::dtype::{FloatDType, IntDType};

pub fn test_traced_linear_matches_forward<D: Device>(dev: &D) {
    let linear = Linear::new(3, 4, true, dev).unwrap();
    let x = Tensor::<D>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), (dev, FloatDType::F32)).unwrap();

    let graph = trace(&linear, &x).unwrap();
    let mut exec = graph.lock().unwrap().compile(dev).unwrap();

    let expected = linear.forward(&x).unwrap();
    let out = exec.run(&[x.clone().into()]).unwrap();
    assert_eq!(out.len(), 1);
    let got = out[0].as_float().unwrap();
    let (e, g) = (expected.to_vec().unwrap(), got.to_vec().unwrap());
    assert_eq!(e.len(), g.len());
    for (a, b) in e.iter().zip(&g) {
        assert!((a - b).abs() < 1e-5, "graph execution must match the module forward");
    }
}

pub fn test_repeated_runs_different_inputs<D: Device>(dev: &D) {
    let linear = Linear::new(3, 2, true, dev).unwrap();
    let x = Tensor::<D>::from_slice(&[1.0, 1.0, 1.0], (1, 3), (dev, FloatDType::F32)).unwrap();

    let graph = trace(&linear, &x).unwrap();
    let mut exec = graph.lock().unwrap().compile(dev).unwrap();

    // Different values against the same compiled graph (constants reused).
    // Note: a traced graph is shape-specialised — every batch must match the
    // example input's shape (1, 3).
    for batch in [&[0.5, -1.0, 2.0][..], &[3.0, 3.0, 3.0][..]] {
        let input = Tensor::<D>::from_slice(batch, (1, 3), (dev, FloatDType::F32)).unwrap();
        let expected = linear.forward(&input).unwrap().to_vec().unwrap();
        let out = exec.run(&[input.clone().into()]).unwrap();
        let got = out[0].as_float().unwrap().to_vec().unwrap();
        for (a, b) in expected.iter().zip(&got) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}

pub fn test_bool_ops<D: Device>(dev: &D) {
    let trace_dev = Trace::new();
    let a = Tensor::<Trace, Float>::full(&[3], 0.0, (&trace_dev, FloatDType::F32)).unwrap();
    let b = Tensor::<Trace, Float>::full(&[3], 0.0, (&trace_dev, FloatDType::F32)).unwrap();
    let m = a.gt(&b).unwrap();
    let n = m.not().unwrap();
    let graph = trace_dev.graph();
    {
        let mut g = graph.lock().unwrap();
        g.mark_input(a.trace_id());
        g.mark_input(b.trace_id());
        g.mark_output(n.trace_id());
    }

    let mut exec = graph.lock().unwrap().compile(dev).unwrap();
    let ra = Tensor::<D>::from_slice(&[1.0, 0.5, 2.0], (3,), (dev, FloatDType::F32)).unwrap();
    let rb = Tensor::<D>::from_slice(&[0.5, 0.5, 2.0], (3,), (dev, FloatDType::F32)).unwrap();
    let out = exec.run(&[ra.into(), rb.into()]).unwrap();
    let got = out[0].as_bool().unwrap().to_vec().unwrap();
    assert_eq!(got, vec![false, true, true], "not(a > b)");
}

pub fn test_traced_cross_entropy_matches_forward<D: Device>(dev: &D) {
    let loss = CrossEntropyLoss::<D>::new();
    let pred = Tensor::<D>::from_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), (dev, FloatDType::F32)).unwrap();
    let target = Tensor::<D, Int>::from_slice(&[0, 1], (2,), (dev, IntDType::I32)).unwrap();

    let graph = trace(&loss, &(pred.clone(), target.clone())).unwrap();
    let mut exec = graph.lock().unwrap().compile(dev).unwrap();

    let expected = loss.forward(&pred, &target).unwrap().to_vec().unwrap();
    let out = exec.run(&[pred.clone().into(), target.clone().into()]).unwrap();
    let got = out[0].as_float().unwrap().to_vec().unwrap();
    for (a, b) in expected.iter().zip(&got) {
        assert!((a - b).abs() < 1e-5);
    }
}
