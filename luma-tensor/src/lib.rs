//! **luma-tensor** — a tensor computation library with:
//! - Compile-time **kind** separation (`Float`, `Int`, `Bool`) so a `Float`
//!   tensor cannot accidentally participate in a `Bool` operation.
//! - Runtime **precision** (`f32`, `f64`, `i32`, …) within each kind, decided
//!   at construction time via [`DType`].
//! - A **device** abstraction ([`Device`]) that lets the same code run on `Cpu`
//!   or (in the future) `Cuda`.
//! - A tape-based **autograd** engine that tracks only `Float`-kind tensors.

pub mod device;
pub mod dtype;
pub mod error;
pub mod grad;
pub mod ops;
pub mod tensor;

// ---- convenience re-exports ----
pub use device::{
    BoolOps, Cpu, Device, FloatOps, IntOps,
    cpu::{CpuBoolStorage, CpuFloatStorage, CpuIntStorage},
};
pub use dtype::{Bool, DType, DTypeKind, Float, Int, KindTag, Storage};
pub use error::{Error, Result};
pub use grad::{TensorMeta, FloatMeta, GradStore, NoGradGuard, is_grad_enabled, set_grad_enabled};
pub use ops::{BinaryOp, CmpOp, Op, ReduceOp, UnaryOp, FloatDTypeKind};
pub use ops::indexer::{IndexOp, Indexer, Slice};
pub use tensor::{D, Dim, Dims, Layout, Shape, StorageIndices, Tensor, TensorId, TensorImpl};