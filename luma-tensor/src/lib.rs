//! **luma-tensor** — a tensor computation library with:
//! - Compile-time **kind** separation (`Float`, `Int`, `Bool`) so a `Float`
//!   tensor cannot accidentally participate in a `Bool` operation.
//! - Runtime **precision** (`f32`, `f64`, `i32`, …) within each kind, decided
//!   at construction time via [`DType`].
//! - A **device** abstraction ([`Device`]) that lets the same code run on any
//!   backend implementing it. Backends live in separate crates: `luma-cpu`
//!   (`Cpu`) and `luma-cuda` (`Cuda`).
//! - A tape-based **autograd** engine that tracks only `Float`-kind tensors.
pub mod device;
pub mod dtype;
pub mod dynamic;
pub mod error;
pub mod grad;
pub mod ops;
pub mod scalar;
pub mod tensor;
#[cfg(feature = "test-utils")]
pub mod tests;

pub use device::{BoolOps, Device, FloatOps, IntOps};
pub use dtype::{Bool, BoolDType, DType, DTypeKind, Float, FloatDType, Int, IntDType, KindTag, Storage};
pub use dynamic::DynTensor;
pub use error::{Error, Result};
pub use grad::{FloatMeta, GradStore, NoGradGuard, TensorMeta, is_grad_enabled, set_grad_enabled};
pub use ops::{BinaryOp, CmpOp, FloatUnaryOp, Op, ReduceOp, TransferDTypeKind, UnaryOp, ViewOp, CustomOp1, CustomOp2, CustomOp3, CustomOp, CustomOpError};
pub use ops::{IndexOp, Indexer, Slice};
pub use scalar::Scalar;
pub use tensor::{D, Dim, Dims, Layout, Shape, StorageIndices, Tensor, TensorId, TensorImpl};

/// Half-precision scalar types backing [`FloatDType::F16`] / [`FloatDType::BF16`].
pub use half::{bf16, f16};
