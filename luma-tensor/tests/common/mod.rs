//! Common test utilities and generic test functions for luma-tensor.
#![allow(dead_code)]

use luma_tensor::Device;
use luma_tensor::dtype::{BoolDType, FloatDType, IntDType};
use luma_tensor::{Bool, Cpu, Int, Shape, Tensor};

pub mod numeric;
pub mod reduce;
pub mod boolean;
pub mod cast;
pub mod shape;
pub mod matmul;
pub mod indexing;
pub mod construct;
pub mod display;
pub mod grad;
pub mod nn;
pub mod dtype;
pub mod f64;
pub mod cross;
pub mod error;

/// Create a Float tensor (f32) from a slice of f64 values.
pub fn tensor_f32<S: Into<Shape>>(data: &[f64], shape: S) -> Tensor<Cpu> {
    Tensor::<Cpu>::from_slice(data, shape, FloatDType::F32).unwrap()
}

/// Create a Float tensor (f32) on a specific device.
pub fn tensor_f32_dev<D: Device, S: Into<Shape>>(data: &[f64], shape: S, device: &D) -> Tensor<D> {
    Tensor::<D>::from_slice(data, shape, (device, FloatDType::F32)).unwrap()
}

/// Create an Int tensor (i32) from a slice of i64 values.
pub fn tensor_i32<S: Into<Shape>>(data: &[i64], shape: S) -> Tensor<Cpu, Int> {
    Tensor::<Cpu, Int>::from_slice(data, shape, IntDType::I32).unwrap()
}

/// Create an Int tensor (i32) on a specific device.
pub fn tensor_i32_dev<D: Device, S: Into<Shape>>(data: &[i64], shape: S, device: &D) -> Tensor<D, Int> {
    Tensor::<D, Int>::from_slice(data, shape, (device, IntDType::I32)).unwrap()
}

/// Create a Bool tensor on a specific device.
pub fn tensor_bool_dev<D: Device, S: Into<Shape>>(data: &[bool], shape: S, device: &D) -> Tensor<D, Bool> {
    Tensor::<D, Bool>::from_slice(data, shape, (device, BoolDType::Bool)).unwrap()
}

/// Create an Int tensor (u8) on a specific device.
pub fn tensor_u8_dev<D: Device, S: Into<Shape>>(data: &[i64], shape: S, device: &D) -> Tensor<D, Int> {
    Tensor::<D, Int>::from_slice(data, shape, (device, IntDType::U8)).unwrap()
}

/// Create an Int tensor (u32) on a specific device.
pub fn tensor_u32_dev<D: Device, S: Into<Shape>>(data: &[i64], shape: S, device: &D) -> Tensor<D, Int> {
    Tensor::<D, Int>::from_slice(data, shape, (device, IntDType::U32)).unwrap()
}

/// Create a Float tensor (f64) on a specific device.
pub fn tensor_f64_dev<D: Device, S: Into<Shape>>(data: &[f64], shape: S, device: &D) -> Tensor<D> {
    Tensor::<D>::from_slice(data, shape, (device, FloatDType::F64)).unwrap()
}

/// Assert two f64 slices match elementwise within tolerance.
pub fn assert_close(a: &[f64], b: &[f64], rtol: f64, atol: f64) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * y.abs();
        assert!(diff <= tol, "mismatch at index {}: {} vs {}, diff={:.2e}, tol={:.2e}", i, x, y, diff, tol);
    }
}
