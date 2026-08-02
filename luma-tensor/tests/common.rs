//! Common test utilities for luma-tensor.

use luma_tensor::dtype::{FloatDType, IntDType};
use luma_tensor::{Cpu, Int, Shape, Tensor};

/// Create a Float tensor (f32) from a slice of f64 values.
#[allow(dead_code)]
pub fn tensor_f32<S: Into<Shape>>(data: &[f64], shape: S) -> Tensor<Cpu> {
    Tensor::<Cpu>::from_slice(data, shape, FloatDType::F32).unwrap()
}

/// Create an Int tensor (i32) from a slice of i64 values.
#[allow(dead_code)]
pub fn tensor_i32<S: Into<Shape>>(data: &[i64], shape: S) -> Tensor<Cpu, Int> {
    Tensor::<Cpu, Int>::from_slice(data, shape, IntDType::I32).unwrap()
}

/// Assert two f64 slices match elementwise within tolerance.
pub fn assert_close(a: &[f64], b: &[f64], rtol: f64, atol: f64) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * y.abs();
        assert!(
            diff <= tol,
            "mismatch at index {}: {} vs {}, diff={:.2e}, tol={:.2e}",
            i,
            x,
            y,
            diff,
            tol
        );
    }
}
