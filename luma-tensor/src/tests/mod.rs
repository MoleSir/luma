//! Shared, device-generic test templates.
//!
//! Every `test_*` function verifies a piece of tensor semantics against any
//! [`Device`]. Backends enable the `test-utils` feature and drive the suite
//! from their own integration tests, e.g. for the CPU backend:
//!
//! ```ignore
//! # #[cfg(feature = "test-utils")]
//! use luma_tensor::testutil::numeric::test_add_f32;
//! # luma_cpu::Cpu::default()
//! ```
#![allow(dead_code)]

use crate::dtype::{BoolDType, FloatDType, IntDType};
use crate::Device;
use crate::{Bool, Int, Shape, Tensor};

pub mod boolean;
pub mod cast;
pub mod construct;
pub mod cross;
pub mod display;
pub mod dtype;
pub mod error;
pub mod f64;
pub mod grad;
pub mod indexing;
pub mod matmul;
pub mod nn;
pub mod numeric;
pub mod reduce;
pub mod shape;
pub mod transfer;

/// Create a Float tensor (f32) from a slice of f64 values on a specific device.
pub fn tensor_f32_dev<D: Device, S: Into<Shape>>(data: &[f64], shape: S, device: &D) -> Tensor<D> {
    Tensor::<D>::from_slice(data, shape, (device, FloatDType::F32)).unwrap()
}

/// Create an Int tensor (i32) from a slice of i64 values on a specific device.
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
