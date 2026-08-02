//! Tests for boolean ops and if_else.

mod common;
use common::*;
use luma_tensor::{Bool, Cpu, Tensor};

// ---- Boolean logical ops ----

#[test]
fn bool_and() {
    let a = Tensor::<Cpu, Bool>::from_slice(&[true, true, false, false], (4,), ()).unwrap();
    let b = Tensor::<Cpu, Bool>::from_slice(&[true, false, true, false], (4,), ()).unwrap();
    let c = a.and(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![true, false, false, false]);
}

#[test]
fn bool_or() {
    let a = Tensor::<Cpu, Bool>::from_slice(&[true, true, false, false], (4,), ()).unwrap();
    let b = Tensor::<Cpu, Bool>::from_slice(&[true, false, true, false], (4,), ()).unwrap();
    let c = a.or(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![true, true, true, false]);
}

#[test]
fn bool_xor() {
    let a = Tensor::<Cpu, Bool>::from_slice(&[true, true, false, false], (4,), ()).unwrap();
    let b = Tensor::<Cpu, Bool>::from_slice(&[true, false, true, false], (4,), ()).unwrap();
    let c = a.xor(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![false, true, true, false]);
}

#[test]
fn bool_not() {
    let a = Tensor::<Cpu, Bool>::from_slice(&[true, false, true], (3,), ()).unwrap();
    let c = a.not().unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![false, true, false]);
}

// ---- All / Any / Count ----

#[test]
fn bool_all_all() {
    let t = Tensor::<Cpu, Bool>::from_slice(&[true, true, true], (3,), ()).unwrap();
    assert!(t.all_all().unwrap());
    let f = Tensor::<Cpu, Bool>::from_slice(&[true, false, true], (3,), ()).unwrap();
    assert!(!f.all_all().unwrap());
}

#[test]
fn bool_any_all() {
    let t = Tensor::<Cpu, Bool>::from_slice(&[false, false, false], (3,), ()).unwrap();
    assert!(!t.any_all().unwrap());
    let f = Tensor::<Cpu, Bool>::from_slice(&[false, true, false], (3,), ()).unwrap();
    assert!(f.any_all().unwrap());
}

#[test]
fn bool_true_count() {
    let t = Tensor::<Cpu, Bool>::from_slice(&[true, false, true, true, false], (5,), ()).unwrap();
    assert_eq!(t.true_count().unwrap(), 3);
}

#[test]
fn bool_false_count() {
    let t = Tensor::<Cpu, Bool>::from_slice(&[true, false, true], (3,), ()).unwrap();
    assert_eq!(t.false_count().unwrap(), 1);
}

// ---- If Else (masked select) ----

#[test]
fn if_else_f32() {
    let mask = Tensor::<Cpu, Bool>::from_slice(&[true, false, true, false], (4,), ()).unwrap();
    let on_true = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (4,));
    let on_false = tensor_f32(&[10.0, 20.0, 30.0, 40.0], (4,));
    let result = mask.if_else(&on_true, &on_false).unwrap();
    assert_close(&result.to_vec().unwrap(), &[1.0, 20.0, 3.0, 40.0], 1e-7, 1e-7);
}

#[test]
fn if_else_scalar_true() {
    let mask = Tensor::<Cpu, Bool>::from_slice(&[true, false, true], (3,), ()).unwrap();
    let on_false = tensor_f32(&[10.0, 20.0, 30.0], (3,));
    let result = mask.if_else_scalar_true(5.0, &on_false).unwrap();
    assert_close(&result.to_vec().unwrap(), &[5.0, 20.0, 5.0], 1e-7, 1e-7);
}

#[test]
fn if_else_scalar_false() {
    let mask = Tensor::<Cpu, Bool>::from_slice(&[true, false, true], (3,), ()).unwrap();
    let on_true = tensor_f32(&[1.0, 2.0, 3.0], (3,));
    let result = mask.if_else_scalar_false(&on_true, 0.0).unwrap();
    assert_close(&result.to_vec().unwrap(), &[1.0, 0.0, 3.0], 1e-7, 1e-7);
}

// ---- Allclose ----

#[test]
fn allclose_exact() {
    let a = tensor_f32(&[1.0, 2.0, 3.0], (3,));
    let b = tensor_f32(&[1.0, 2.0, 3.0], (3,));
    assert!(a.allclose(&b, 1e-7, 1e-7).unwrap());
}

#[test]
fn allclose_false() {
    let a = tensor_f32(&[1.0, 2.0], (2,));
    let b = tensor_f32(&[1.0, 100.0], (2,));
    assert!(!a.allclose(&b, 1e-5, 1e-5).unwrap());
}

// ---- matmul (2D) ----

#[test]
fn matmul_2x2() {
    let a = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (2, 2));
    let b = tensor_f32(&[5.0, 6.0, 7.0, 8.0], (2, 2));
    let y = a.matmul(&b).unwrap();
    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    assert_close(&y.to_vec().unwrap(), &[19.0, 22.0, 43.0, 50.0], 1e-5, 1e-5);
}

#[test]
fn matmul_2x3_3x2() {
    let a = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
    let b = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2));
    let y = a.matmul(&b).unwrap();
    assert_eq!(y.dims(), &[2, 2]);
    // [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]]
    // row0: [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
    // row1: [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
    assert_close(&y.to_vec().unwrap(), &[22.0, 28.0, 49.0, 64.0], 1e-5, 1e-5);
}
