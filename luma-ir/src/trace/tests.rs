use super::*;
use crate::graph::{NodeOp, Scalar};
use luma_nn::functional::linear;
use luma_tensor::dtype::{BoolDType, FloatDType, IntDType};
use luma_tensor::{Bool, Float, Int, Tensor};

#[test]
fn traces_linear_relu() {
    let trace = Trace::new();
    let opts = (&trace, FloatDType::F32);

    let x = Tensor::<Trace, Float>::full(&[2, 3], 1.0, opts).unwrap();
    let w = Tensor::<Trace, Float>::full(&[4, 3], 1.0, opts).unwrap();
    let b = Tensor::<Trace, Float>::full(&[4], 0.0, opts).unwrap();

    let y = linear(&x, &w, Some(&b)).unwrap();
    let y = y.relu().unwrap();

    let g = trace.graph();
    let g = g.lock().unwrap();

    // x, w, b are leaves; then transpose(w) -> matmul -> broadcast(matmul)
    // -> broadcast(b) -> add -> relu.
    let kinds: Vec<_> = g.nodes.iter().map(|n| &n.op).collect();
    assert!(matches!(kinds[0], NodeOp::Transpose(1, 0) | NodeOp::Transpose(0, 1)));
    assert!(matches!(kinds[1], NodeOp::Matmul));
    assert!(matches!(kinds[2], NodeOp::Broadcast));
    assert!(matches!(kinds[3], NodeOp::Broadcast));
    assert!(matches!(kinds[4], NodeOp::Binary(luma_tensor::BinaryOp::Add)));
    assert!(matches!(kinds[5], NodeOp::FloatUnary(luma_tensor::FloatUnaryOp::Relu)));
    assert_eq!(g.nodes.len(), 6);

    // The final value is the relu output.
    assert_eq!(g.nodes[5].outputs[0], y.trace_id());
    // matmul consumes the transposed weight (not the original).
    let matmul_in = &g.nodes[1].inputs;
    assert_eq!(matmul_in[0], x.trace_id());
    assert_eq!(matmul_in[1], g.nodes[0].outputs[0]);
}

#[test]
fn traces_int_and_bool_ops() {
    let trace = Trace::new();

    let idx = Tensor::<Trace, Int>::full(&[4], 0, (&trace, IntDType::I32)).unwrap();
    let shifted = idx.add_scalar(3).unwrap();

    let a = Tensor::<Trace, Bool>::trues(&[2, 2], (&trace, BoolDType::Bool)).unwrap();
    let b = Tensor::<Trace, Bool>::falses(&[2, 2], (&trace, BoolDType::Bool)).unwrap();
    let c = a.and(&b).unwrap();

    let g = trace.graph();
    let g = g.lock().unwrap();

    assert_eq!(g.nodes.len(), 2);
    assert!(matches!(g.nodes[0].op, NodeOp::BinaryScalarRhs(Scalar::I64(3), luma_tensor::BinaryOp::Add)));
    assert!(matches!(g.nodes[1].op, NodeOp::And));
    assert_eq!(g.nodes[0].outputs[0], shifted.trace_id());
    assert_eq!(g.nodes[1].outputs[0], c.trace_id());
}
