//! Graph-structure tracing tests: the `Trace` device records the expected SSA
//! nodes for a handful of modules without running any kernel. Device-independent.

use luma_jit::{NodeOp, Scalar, Trace, Traced, trace};
use luma_nn::Linear;
use luma_nn::functional::linear;
use luma_nn::loss::CrossEntropyLoss;
use luma_tensor::dtype::{BoolDType, FloatDType, IntDType};
use luma_tensor::{BinaryOp, Bool, Cpu, DType, Float, FloatUnaryOp, Int, Tensor};

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
    assert!(matches!(kinds[4], NodeOp::Binary(BinaryOp::Add)));
    assert!(matches!(kinds[5], NodeOp::FloatUnary(FloatUnaryOp::Relu)));
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
    assert!(matches!(g.nodes[0].op, NodeOp::BinaryScalarRhs(Scalar::I64(3), BinaryOp::Add)));
    assert!(matches!(g.nodes[1].op, NodeOp::And));
    assert_eq!(g.nodes[0].outputs[0], shifted.trace_id());
    assert_eq!(g.nodes[1].outputs[0], c.trace_id());
}

// ---- module tracing: state capture through to_device ------------------------

#[test]
fn trace_module_captures_state() {
    let cpu = Cpu;
    let linear = Linear::new(3, 4, true, cpu).unwrap();
    let x = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), FloatDType::F32).unwrap();

    let graph = trace(&linear, &x).unwrap();
    let g = graph.lock().unwrap();

    // Every parameter is captured as a constant leaf carrying its data.
    let consts: Vec<_> = g.values.iter().filter(|v| v.data.is_some()).collect();
    assert_eq!(consts.len(), 2, "weight + bias");
    let weight = &g.values[consts[0].id];
    let bias = &g.values[consts[1].id];
    assert_eq!(weight.shape.dims(), &[4, 3]);
    assert_eq!(bias.shape.dims(), &[4]);
    assert_eq!(weight.data.as_ref().unwrap().0.len(), 4 * 3 * 4);
    assert_eq!(bias.data.as_ref().unwrap().0.len(), 4 * 4);

    // The captured bytes round-trip through `Tensor::from_bytes` on a real
    // device — the exact path a future executor will use to materialise them.
    let w_restored = Tensor::<Cpu>::from_bytes(&weight.data.as_ref().unwrap().0, weight.shape.clone(), (cpu, FloatDType::F32)).unwrap();
    let w_orig = linear.weight.to_vec().unwrap();
    let w_back = w_restored.to_vec().unwrap();
    assert_eq!(w_orig.len(), w_back.len());
    for (a, b) in w_orig.iter().zip(&w_back) {
        assert!((a - b).abs() < 1e-7, "captured constant must reproduce the original weight");
    }

    // The example input is a data-less input leaf.
    assert_eq!(g.inputs.len(), 1);
    let input = &g.values[g.inputs[0]];
    assert_eq!(input.shape.dims(), &[2, 3]);
    assert_eq!(input.dtype, DType::F32);
    assert!(input.data.is_none());

    // Forward recorded: transpose(w) -> matmul -> broadcast -> broadcast -> add.
    let matmul_idx = g.nodes.iter().position(|n| matches!(n.op, NodeOp::Matmul)).expect("matmul node");
    assert_eq!(g.nodes[matmul_idx].inputs[0], g.inputs[0], "matmul consumes the graph input");
    assert_ne!(g.nodes[matmul_idx].inputs[1], weight.id, "matmul consumes the transposed weight, not the constant");
    assert!(matches!(g.nodes[matmul_idx - 1].op, NodeOp::Transpose(1, 0) | NodeOp::Transpose(0, 1)));
    assert!(matches!(g.nodes.last().unwrap().op, NodeOp::Binary(BinaryOp::Add)));

    // Output registered.
    assert_eq!(g.outputs.len(), 1);
    assert_eq!(g.nodes.last().unwrap().outputs[0], g.outputs[0]);
}

#[test]
fn trace_multi_input_module() {
    let loss = CrossEntropyLoss::new();
    let pred = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), FloatDType::F32).unwrap();
    let target = Tensor::<Cpu, Int>::from_slice(&[0, 1], (2,), IntDType::I32).unwrap();

    let graph = trace(&loss, &(pred, target)).unwrap();
    let g = graph.lock().unwrap();

    assert_eq!(g.inputs.len(), 2, "pred + target");
    assert!(!g.nodes.is_empty());
    assert_eq!(g.outputs.len(), 1);
    assert_eq!(g.values.iter().filter(|v| v.data.is_some()).count(), 0, "a loss has no parameters");
}
