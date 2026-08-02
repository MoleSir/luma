//! Tests for tensor construction: zeros, ones, full, from_slice, arange, randn, falses, trues.

mod common;
use common::*;
use luma_tensor::dtype::{FloatDType, IntDType};
use luma_tensor::{Bool, Cpu, Int, Tensor};

// ---- Float construction ----

#[test]
fn zeros_f32() {
    let t = Tensor::<Cpu>::zeros((2, 3), FloatDType::F32).unwrap();
    assert_eq!(t.shape().dims(), &[2, 3]);
    let v = t.to_vec().unwrap();
    assert_eq!(v.len(), 6);
    assert!(v.iter().all(|&x| x == 0.0));
}

#[test]
fn ones_f32() {
    let t = Tensor::<Cpu>::ones((2, 3), FloatDType::F32).unwrap();
    let v = t.to_vec().unwrap();
    assert!(v.iter().all(|&x| (x - 1.0).abs() < 1e-7));
}

#[test]
fn full_f32() {
    let t = Tensor::<Cpu>::full((2, 2), 3.14, FloatDType::F32).unwrap();
    let v = t.to_vec().unwrap();
    assert_eq!(v.len(), 4);
    for &x in &v {
        assert!((x - 3.14).abs() < 1e-5);
    }
}

#[test]
fn from_slice_f32() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::<Cpu>::from_slice(&data, (2, 3), FloatDType::F32).unwrap();
    assert_eq!(t.shape().dims(), &[2, 3]);
    let v = t.to_vec().unwrap();
    assert_close(&v, &data, 1e-7, 1e-7);
}

#[test]
fn zeros_like_f32() {
    let t = Tensor::<Cpu>::ones((2, 3), FloatDType::F32).unwrap();
    let z = t.zeros_like().unwrap();
    assert_eq!(z.shape(), t.shape());
    assert!(z.to_vec().unwrap().iter().all(|&x| x == 0.0));
}

#[test]
fn ones_like_f32() {
    let t = Tensor::<Cpu>::zeros((2, 2), FloatDType::F32).unwrap();
    let o = t.ones_like().unwrap();
    assert_eq!(o.shape(), t.shape());
    assert!(o.to_vec().unwrap().iter().all(|&x| (x - 1.0).abs() < 1e-7));
}

// ---- Int construction ----

#[test]
fn arange_i32() {
    let t = Tensor::<Cpu, Int>::arange(0, 5, 1, IntDType::I32).unwrap();
    assert_eq!(t.dims(), &[5]);
    let v = t.to_vec().unwrap();
    assert_eq!(v, vec![0, 1, 2, 3, 4]);
}

#[test]
fn arange_step_2() {
    let t = Tensor::<Cpu, Int>::arange(0, 10, 2, IntDType::I32).unwrap();
    let v = t.to_vec().unwrap();
    assert_eq!(v, vec![0, 2, 4, 6, 8]);
}

#[test]
fn arange_reverse() {
    let t = Tensor::<Cpu, Int>::arange(5, 0, -1, IntDType::I32).unwrap();
    let v = t.to_vec().unwrap();
    assert_eq!(v, vec![5, 4, 3, 2, 1]);
}

#[test]
fn from_slice_i32() {
    let data = vec![10i64, 20, 30, 40];
    let t = Tensor::<Cpu, Int>::from_slice(&data, (2, 2), IntDType::I32).unwrap();
    assert_eq!(t.shape().dims(), &[2, 2]);
    let v = t.to_vec().unwrap();
    assert_eq!(v, data);
}

// ---- Bool construction ----

#[test]
fn falses_bool() {
    let t = Tensor::<Cpu, Bool>::falses((2, 3), ()).unwrap();
    assert_eq!(t.shape().dims(), &[2, 3]);
    let v = t.to_vec().unwrap();
    assert_eq!(v.len(), 6);
    assert!(v.iter().all(|&x| !x));
}

#[test]
fn trues_bool() {
    let t = Tensor::<Cpu, Bool>::trues((2, 3), ()).unwrap();
    let v = t.to_vec().unwrap();
    assert!(v.iter().all(|&x| x));
}

#[test]
fn from_slice_bool() {
    let data = vec![true, false, true, false];
    let t = Tensor::<Cpu, Bool>::from_slice(&data, (2, 2), ()).unwrap();
    assert_eq!(t.to_vec().unwrap(), data);
}

#[test]
fn from_slice_size_mismatch() {
    let result = Tensor::<Cpu>::from_slice(&[1.0, 2.0], (2, 2), FloatDType::F32);
    assert!(result.is_err());
}

// ---- Scalar tensor ----

#[test]
fn scalar_to_scalar_f32() {
    let t = Tensor::<Cpu>::full((), 42.0, FloatDType::F32).unwrap();
    assert_eq!(t.element_count(), 1);
    assert!((t.to_scalar().unwrap() - 42.0).abs() < 1e-7);
}

// ---- Tensor::new() (IntoTensor) ----

#[test]
fn new_scalar_f32() {
    let t = Tensor::<Cpu>::new(3.14).unwrap();
    assert_eq!(t.element_count(), 1);
    assert!((t.to_scalar().unwrap() - 3.14).abs() < 1e-5);
}

#[test]
fn new_slice_f32() {
    let t = Tensor::<Cpu>::new(&[1.0, 2.0, 3.0][..]).unwrap();
    assert_eq!(t.dims(), &[3]);
    assert_close(&t.to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-7, 1e-7);
}

#[test]
fn new_array_f32() {
    let t = Tensor::<Cpu>::new([1.0, 2.0, 3.0]).unwrap();
    assert_eq!(t.dims(), &[3]);
    assert_close(&t.to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-7, 1e-7);
}

#[test]
fn new_array_ref_f32() {
    let t = Tensor::<Cpu>::new(&[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(t.dims(), &[3]);
    assert_close(&t.to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-7, 1e-7);
}

#[test]
fn new_vec_f32() {
    let t = Tensor::<Cpu>::new(vec![1.0, 2.0, 3.0]).unwrap();
    assert_eq!(t.dims(), &[3]);
    assert_close(&t.to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-7, 1e-7);
}

#[test]
fn new_2d_f32() {
    let t = Tensor::<Cpu>::new(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
    assert_eq!(t.dims(), &[2, 3]);
    assert_close(&t.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-7, 1e-7);
}

#[test]
fn new_3d_f32() {
    let t = Tensor::<Cpu>::new(&[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).unwrap();
    assert_eq!(t.dims(), &[2, 2, 2]);
    assert_close(&t.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 1e-7, 1e-7);
}

#[test]
fn new_scalar_i64() {
    let t = Tensor::<Cpu, Int>::new(42i64).unwrap();
    assert_eq!(t.to_vec().unwrap(), vec![42]);
}

#[test]
fn new_slice_i64() {
    let t = Tensor::<Cpu, Int>::new(&[1i64, 2, 3][..]).unwrap();
    assert_eq!(t.to_vec().unwrap(), vec![1, 2, 3]);
}

#[test]
fn new_2d_i64() {
    let t = Tensor::<Cpu, Int>::new(&[[1i64, 2], [3, 4]]).unwrap();
    assert_eq!(t.dims(), &[2, 2]);
    assert_eq!(t.to_vec().unwrap(), vec![1, 2, 3, 4]);
}

#[test]
fn new_scalar_bool() {
    let t = Tensor::<Cpu, Bool>::new(true).unwrap();
    assert_eq!(t.to_vec().unwrap(), vec![true]);
}

// ---- eye / diag / tril / triu / linspace / rand_like ----

#[test]
fn eye_f32() {
    let t = Tensor::<Cpu>::eye(3).unwrap();
    assert_eq!(t.dims(), &[3, 3]);
    assert_close(&t.to_vec().unwrap(), &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 1e-7, 1e-7);
}

#[test]
fn eye_i32() {
    let t = Tensor::<Cpu, Int>::eye(2).unwrap();
    assert_eq!(t.to_vec().unwrap(), vec![1, 0, 0, 1]);
}

#[test]
fn diag_f32() {
    let t = Tensor::<Cpu>::diag(&[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(t.dims(), &[3, 3]);
    assert_close(&t.to_vec().unwrap(), &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], 1e-7, 1e-7);
}

#[test]
fn tril_f32() {
    let t = Tensor::<Cpu>::tril(3, false).unwrap();
    assert_eq!(t.dims(), &[3, 3]);
    // lower triangular without diagonal
    assert_close(&t.to_vec().unwrap(), &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], 1e-7, 1e-7);
}

#[test]
fn tril_diagonal() {
    let t = Tensor::<Cpu>::tril(3, true).unwrap();
    // lower triangular with diagonal
    assert_close(&t.to_vec().unwrap(), &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0], 1e-7, 1e-7);
}

#[test]
fn triu_f32() {
    let t = Tensor::<Cpu>::triu(3, false).unwrap();
    // upper triangular without diagonal
    assert_close(&t.to_vec().unwrap(), &[0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 1e-7, 1e-7);
}

#[test]
fn triu_diagonal() {
    let t = Tensor::<Cpu>::triu(3, true).unwrap();
    // upper triangular with diagonal
    assert_close(&t.to_vec().unwrap(), &[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], 1e-7, 1e-7);
}

#[test]
fn tril_i32() {
    let t = Tensor::<Cpu, Int>::tril(3, true).unwrap();
    assert_eq!(t.to_vec().unwrap(), vec![1, 0, 0, 1, 1, 0, 1, 1, 1]);
}

#[test]
fn linspace_f32() {
    let t = Tensor::<Cpu>::linspace(0.0, 1.0, 5).unwrap();
    assert_eq!(t.dims(), &[5]);
    let v = t.to_vec().unwrap();
    assert!((v[0] - 0.0).abs() < 1e-5);
    assert!((v[1] - 0.25).abs() < 1e-5);
    assert!((v[2] - 0.5).abs() < 1e-5);
    assert!((v[3] - 0.75).abs() < 1e-5);
    assert!((v[4] - 1.0).abs() < 1e-5);
}

#[test]
fn linspace_single() {
    let t = Tensor::<Cpu>::linspace(3.0, 3.0, 1).unwrap();
    assert_eq!(t.element_count(), 1);
    assert!((t.to_scalar().unwrap() - 3.0).abs() < 1e-5);
}

#[test]
fn rand_like_shape() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (2, 2));
    let r = t.rand_like(0.0, 1.0).unwrap();
    assert_eq!(r.dims(), t.dims());
    assert!((0.0..=1.0).contains(&r.to_vec().unwrap()[0]));
}

#[test]
fn randn_like_shape() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (2, 2));
    let r = t.randn_like(0.0, 1.0).unwrap();
    assert_eq!(r.dims(), t.dims());
}

#[test]
fn new_slice_bool() {
    let t = Tensor::<Cpu, Bool>::new(&[true, false, true]).unwrap();
    assert_eq!(t.to_vec().unwrap(), vec![true, false, true]);
}

#[test]
fn new_2d_bool() {
    let t = Tensor::<Cpu, Bool>::new(&[[true, false], [false, true]]).unwrap();
    assert_eq!(t.dims(), &[2, 2]);
    assert_eq!(t.to_vec().unwrap(), vec![true, false, false, true]);
}
