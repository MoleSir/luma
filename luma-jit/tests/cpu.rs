//! CPU tests — execution (trace → compile → run) delegates to `common`, plus
//! CPU-specific validation. Run with: cargo test --test cpu

mod common;

use luma_jit::{Graph, NodeOp};
use luma_tensor::dtype::FloatDType;
use luma_tensor::{Cpu, DType, Shape, Tensor};

// ---- module-level (trace a real module) --------------------------------------

#[test]
fn cpu_traced_linear_matches_forward() {
    common::module::test_traced_linear_matches_forward(&Cpu);
}

#[test]
fn cpu_repeated_runs_different_inputs() {
    common::module::test_repeated_runs_different_inputs(&Cpu);
}

#[test]
fn cpu_bool_ops() {
    common::module::test_bool_ops(&Cpu);
}

#[test]
fn cpu_traced_cross_entropy_matches_forward() {
    common::module::test_traced_cross_entropy_matches_forward(&Cpu);
}

// Int matmul is CPU-only: the Cuda backend does not support integer matmul.
#[test]
fn cpu_int_matmul() {
    use luma_jit::{Trace, Traced};
    use luma_tensor::Int;
    use luma_tensor::dtype::IntDType;

    let trace_dev = Trace::new();
    let a = Tensor::<Trace, Int>::full(&[2, 3], 0, (&trace_dev, IntDType::I32)).unwrap();
    let b = Tensor::<Trace, Int>::full(&[3, 2], 0, (&trace_dev, IntDType::I32)).unwrap();
    let c = a.matmul(&b).unwrap();
    let graph = trace_dev.graph();
    {
        let mut g = graph.lock().unwrap();
        g.mark_input(a.trace_id());
        g.mark_input(b.trace_id());
        g.mark_output(c.trace_id());
    }

    let mut exec = graph.lock().unwrap().compile(&Cpu).unwrap();
    let ra = Tensor::<Cpu, Int>::from_slice(&[1, 2, 3, 4, 5, 6], (2, 3), IntDType::I32).unwrap();
    let rb = Tensor::<Cpu, Int>::from_slice(&[1, 0, 0, 1, 1, 1], (3, 2), IntDType::I32).unwrap();
    let out = exec.run(&[ra.clone().into(), rb.clone().into()]).unwrap();
    let got = out[0].as_int().unwrap().to_vec().unwrap();
    let expected = ra.matmul(&rb).unwrap().to_vec().unwrap();
    assert_eq!(got, expected);
}

// ---- op-family coverage ------------------------------------------------------

#[test]
fn cpu_binary() {
    common::numeric::test_add(&Cpu);
    common::numeric::test_sub(&Cpu);
    common::numeric::test_mul(&Cpu);
    common::numeric::test_div(&Cpu);
    common::numeric::test_maximum(&Cpu);
    common::numeric::test_minimum(&Cpu);
}

#[test]
fn cpu_scalar() {
    common::numeric::test_add_scalar(&Cpu);
    common::numeric::test_sub_scalar(&Cpu);
    common::numeric::test_mul_scalar(&Cpu);
    common::numeric::test_div_scalar(&Cpu);
    common::numeric::test_maximum_scalar(&Cpu);
    common::numeric::test_minimum_scalar(&Cpu);
    common::numeric::test_sub_scalar_lhs(&Cpu);
    common::numeric::test_div_scalar_lhs(&Cpu);
}

#[test]
fn cpu_unary() {
    common::numeric::test_neg(&Cpu);
    common::numeric::test_abs(&Cpu);
    common::numeric::test_sign(&Cpu);
    common::numeric::test_exp(&Cpu);
    common::numeric::test_ln(&Cpu);
    common::numeric::test_sin(&Cpu);
    common::numeric::test_cos(&Cpu);
    common::numeric::test_tanh(&Cpu);
    common::numeric::test_sqr(&Cpu);
    common::numeric::test_sqrt(&Cpu);
    common::numeric::test_recip(&Cpu);
    common::numeric::test_relu(&Cpu);
    common::numeric::test_sigmoid(&Cpu);
    common::numeric::test_silu(&Cpu);
    common::numeric::test_gelu(&Cpu);
    common::numeric::test_gelu_erf(&Cpu);
    common::numeric::test_erf(&Cpu);
    common::numeric::test_floor(&Cpu);
    common::numeric::test_ceil(&Cpu);
    common::numeric::test_round(&Cpu);
    common::numeric::test_affine(&Cpu);
    common::numeric::test_pow(&Cpu);
    common::numeric::test_clamp(&Cpu);
    common::numeric::test_leaky_relu(&Cpu);
}

#[test]
fn cpu_int_unary() {
    common::numeric::test_neg_i32(&Cpu);
    common::numeric::test_abs_i32(&Cpu);
    common::numeric::test_sign_i32(&Cpu);
}

#[test]
fn cpu_cmp() {
    common::numeric::test_eq(&Cpu);
    common::numeric::test_ne(&Cpu);
    common::numeric::test_lt(&Cpu);
    common::numeric::test_gt(&Cpu);
    common::numeric::test_le(&Cpu);
    common::numeric::test_ge(&Cpu);
    common::numeric::test_gt_scalar(&Cpu);
}

#[test]
fn cpu_reduce() {
    common::reduce::test_sum_dim(&Cpu);
    common::reduce::test_sum_keepdim(&Cpu);
    common::reduce::test_sum_all(&Cpu);
    common::reduce::test_max_dim(&Cpu);
    common::reduce::test_max_keepdim(&Cpu);
    common::reduce::test_max_all(&Cpu);
    common::reduce::test_min_dim(&Cpu);
    common::reduce::test_min_keepdim(&Cpu);
    common::reduce::test_min_all(&Cpu);
    common::reduce::test_prod_dim(&Cpu);
    common::reduce::test_prod_keepdim(&Cpu);
    common::reduce::test_prod_all(&Cpu);
    common::reduce::test_mean_dim(&Cpu);
    common::reduce::test_mean_keepdim(&Cpu);
    common::reduce::test_mean_all(&Cpu);
}

#[test]
fn cpu_argreduce() {
    common::reduce::test_argmax(&Cpu);
    common::reduce::test_argmin(&Cpu);
    common::reduce::test_argmax_keepdim(&Cpu);
    common::reduce::test_argmin_keepdim(&Cpu);
}

#[test]
fn cpu_shape() {
    common::shape::test_reshape(&Cpu);
    common::shape::test_transpose(&Cpu);
    common::shape::test_permute(&Cpu);
    common::shape::test_narrow(&Cpu);
    common::shape::test_slice(&Cpu);
    common::shape::test_squeeze(&Cpu);
    common::shape::test_unsqueeze(&Cpu);
    common::shape::test_squeeze_ambiguous_dim(&Cpu);
    common::shape::test_unsqueeze_ambiguous_dim(&Cpu);
    common::shape::test_broadcast_as(&Cpu);
}

#[test]
fn cpu_indexing() {
    common::indexing::test_index_select(&Cpu);
    common::indexing::test_gather(&Cpu);
    common::indexing::test_index_add(&Cpu);
    common::indexing::test_scatter_add(&Cpu);
    common::indexing::test_cat(&Cpu);
}

#[test]
fn cpu_nn() {
    common::nn::test_softmax(&Cpu);
    common::nn::test_rms_norm(&Cpu);
    common::nn::test_arange(&Cpu);
}

#[test]
fn cpu_cast() {
    common::cast::test_cast_f32_f64(&Cpu);
    common::cast::test_cast_f32_i32(&Cpu);
    common::cast::test_cast_f32_bool(&Cpu);
    common::cast::test_cast_i32_f32(&Cpu);
    common::cast::test_cast_i32_bool(&Cpu);
    common::cast::test_cast_bool_f32(&Cpu);
    common::cast::test_cast_bool_i32(&Cpu);
}

#[test]
fn cpu_boolean() {
    common::boolean::test_and(&Cpu);
    common::boolean::test_or(&Cpu);
    common::boolean::test_xor(&Cpu);
    common::boolean::test_not(&Cpu);
    common::boolean::test_pick_f32(&Cpu);
    common::boolean::test_pick_i32(&Cpu);
    common::boolean::test_pick_bool(&Cpu);
    common::boolean::test_pick_true_f32(&Cpu);
    common::boolean::test_pick_true_i32(&Cpu);
    common::boolean::test_pick_true_bool(&Cpu);
    common::boolean::test_pick_false_f32(&Cpu);
    common::boolean::test_pick_false_i32(&Cpu);
    common::boolean::test_pick_false_bool(&Cpu);
}

// ---- error paths --------------------------------------------------------------

#[test]
fn compile_rejects_wrong_input_metadata() {
    use luma_nn::Linear;

    let cpu = Cpu;
    let linear = Linear::new(3, 4, true, cpu).unwrap();
    let x = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0], (1, 3), FloatDType::F32).unwrap();

    let graph = luma_jit::trace(&linear, &x).unwrap();
    let mut exec = graph.lock().unwrap().compile(&cpu).unwrap();

    // Wrong shape → rejected at run.
    let bad = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), FloatDType::F32).unwrap();
    assert!(exec.run(&[bad.into()]).is_err());
    // Wrong count → rejected.
    assert!(exec.run(&[]).is_err());
}

#[test]
fn compile_rejects_malformed_graphs() {
    // A dangling value: no constant, no input, no producer.
    let mut g = Graph::default();
    g.add_value(DType::F32, Shape::from((2, 2)));
    assert!(g.compile(&Cpu).is_err());

    // Bool fed into Matmul → rejected by kind inference.
    let mut g = Graph::default();
    let m = g.add_value(DType::Bool, Shape::from((2, 2)));
    g.mark_input(m);
    let _ = g.add_node(NodeOp::Matmul, vec![m, m], DType::Bool, Shape::from((2, 2)));
    assert!(g.compile(&Cpu).is_err(), "Bool Matmul must be rejected");
}
