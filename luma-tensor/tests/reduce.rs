//! Tests for reduce ops: sum, max, min, mean, var, argmin, argmax.

mod common;
use common::*;

// ---- Sum ----

#[test]
fn sum_dim_f32() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
    let s = t.sum(1usize).unwrap(); // sum over cols → (2,)
    assert_eq!(s.dims(), &[2]);
    assert_close(&s.to_vec().unwrap(), &[6.0, 15.0], 1e-7, 1e-5); // 1+2+3=6, 4+5+6=15
}

#[test]
fn sum_keepdim_f32() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (2, 2));
    let s = t.sum_keepdim(0usize).unwrap();
    assert_eq!(s.dims(), &[1, 2]);
    assert_close(&s.to_vec().unwrap(), &[4.0, 6.0], 1e-7, 1e-7); // [1,2;3,4] sum dim0 → [1+3, 2+4]
}

#[test]
fn sum_all_f32() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (2, 2));
    let s = t.sum_all().unwrap();
    assert_eq!(s.element_count(), 1);
    assert!((s.to_scalar().unwrap() - 10.0).abs() < 1e-7);
}

#[test]
fn sum_dims_f32() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (2, 2));
    let s = t.sum_dims([0usize, 1usize], false).unwrap();
    assert_eq!(s.element_count(), 1);
}

// ---- Max / Min ----

#[test]
fn max_dim_f32() {
    let t = tensor_f32(&[1.0, 5.0, 3.0, 2.0, 4.0, 6.0], (2, 3));
    let m = t.max(1usize).unwrap();
    assert_close(&m.to_vec().unwrap(), &[5.0, 6.0], 1e-7, 1e-7);
}

#[test]
fn max_keepdim_f32() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (2, 2));
    let m = t.max_keepdim(0usize).unwrap();
    assert_eq!(m.dims(), &[1, 2]);
    assert_close(&m.to_vec().unwrap(), &[3.0, 4.0], 1e-7, 1e-7);
}

#[test]
fn max_all_f32() {
    let t = tensor_f32(&[1.0, 5.0, 3.0, 2.0], (2, 2));
    let m = t.max_all().unwrap();
    assert!((m.to_scalar().unwrap() - 5.0).abs() < 1e-7);
}

#[test]
fn min_dim_f32() {
    let t = tensor_f32(&[1.0, 5.0, 3.0, 2.0, 4.0, 6.0], (2, 3));
    let m = t.min(1usize).unwrap();
    assert_close(&m.to_vec().unwrap(), &[1.0, 2.0], 1e-7, 1e-7);
}

#[test]
fn min_all_f32() {
    let t = tensor_f32(&[1.0, 5.0, 3.0, 2.0], (2, 2));
    let m = t.min_all().unwrap();
    assert!((m.to_scalar().unwrap() - 1.0).abs() < 1e-7);
}

// ---- Mean ----

#[test]
fn mean_dim_f32() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
    let m = t.mean(1usize).unwrap();
    assert_close(&m.to_vec().unwrap(), &[2.0, 5.0], 1e-7, 1e-5); // mean of [1,2,3]=2, [4,5,6]=5
}

#[test]
fn mean_all_f32() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (2, 2));
    let m = t.mean_all().unwrap();
    assert!((m.to_scalar().unwrap() - 2.5).abs() < 1e-7);
}

// ---- Var ----

#[test]
fn var_f32() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (4,));
    // biased var: mean((x - mean)^2); mean=2.5, squared diffs [2.25, 0.25, 0.25, 2.25], mean=1.25
    let v = t.var(0usize).unwrap();
    assert!((v.to_scalar().unwrap() - 1.25).abs() < 1e-5);
}

#[test]
fn var_unbiased_f32() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (4,));
    // unbiased var: 1.25 * 4/3 = 1.666...
    let v = t.var_unbiased(0usize).unwrap();
    assert!((v.to_scalar().unwrap() - 1.666666).abs() < 1e-4);
}

// ---- Argmin / Argmax ----

#[test]
fn argmax_f32() {
    let t = tensor_f32(&[1.0, 5.0, 3.0, 2.0], (4,));
    let idx = t.argmax(0usize).unwrap();
    let v: Vec<i64> = idx.to_vec().unwrap();
    assert_eq!(v[0], 1); // index of max = 5.0
}

#[test]
fn argmin_f32() {
    let t = tensor_f32(&[3.0, 1.0, 5.0, 2.0], (4,));
    let idx = t.argmin(0usize).unwrap();
    let v: Vec<i64> = idx.to_vec().unwrap();
    assert_eq!(v[0], 1); // index of min = 1.0
}

#[test]
fn argmax_keepdim() {
    let t = tensor_f32(&[1.0, 5.0, 3.0, 2.0], (4,));
    let idx = t.argmax_keepdim(0usize).unwrap();
    assert_eq!(idx.dims(), &[1]);
}

// ---- Int reduce ----

#[test]
fn sum_i32() {
    let a = tensor_i32(&[1, 2, 3, 4], (4,));
    let s = a.sum_all().unwrap();
    assert_eq!(s.to_vec().unwrap(), vec![10]);
}

// ---- Prod ----

#[test]
fn prod_dim_f32() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (2, 2));
    let p = t.prod(0usize).unwrap();
    assert_eq!(p.dims(), &[2]);
    assert_close(&p.to_vec().unwrap(), &[3.0, 8.0], 1e-5, 1e-5); // [1*3, 2*4]
}

#[test]
fn prod_all_f32() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (4,));
    let p = t.prod_all().unwrap();
    assert!((p.to_scalar().unwrap() - 24.0).abs() < 1e-5);
}

#[test]
fn prod_i32() {
    let a = tensor_i32(&[1, 2, 3, 4], (4,));
    let s = a.prod_all().unwrap();
    assert_eq!(s.to_vec().unwrap(), vec![24]);
}

// ---- Std ----

#[test]
fn std_f32() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (4,));
    // biased var = 1.25, std = sqrt(1.25) ≈ 1.118
    let s = t.std(0usize).unwrap();
    assert!((s.to_scalar().unwrap() - 1.118034).abs() < 1e-4);
}

#[test]
fn std_all_f32() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (2, 2));
    let s = t.std_all().unwrap();
    let expected = 1.25f64.sqrt(); // var_all = mean((x-2.5)^2)
    assert!((s.to_scalar().unwrap() - expected).abs() < 1e-4);
}

// ---- LogSumExp ----

#[test]
fn logsumexp_f32() {
    let t = tensor_f32(&[0.0, 1.0, 2.0], (3,));
    let s = t.logsumexp(0usize).unwrap();
    // log(exp(0) + exp(1) + exp(2)) ≈ log(1 + 2.718 + 7.389) ≈ log(11.107) ≈ 2.4076
    let expected = (1.0f64 + 2.7182818 + 7.389056).ln();
    assert!((s.to_scalar().unwrap() - expected).abs() < 1e-4);
}

#[test]
fn logsumexp_keepdim() {
    let t = tensor_f32(&[0.0, 1.0, 2.0], (3,));
    let s = t.logsumexp_keepdim(0usize).unwrap();
    assert_eq!(s.dims(), &[1]);
}
