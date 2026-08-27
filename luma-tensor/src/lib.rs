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
pub mod dynamic;
pub mod error;
pub mod grad;
pub mod ops;
pub mod scalar;
pub mod tensor;

// convenience re-exports
pub use device::cpu::{Cpu, CpuBoolStorage, CpuFloatStorage, CpuIntStorage};
#[cfg(feature = "cuda")]
pub use device::cuda::{Cuda, CudaBoolStorage, CudaFloatStorage, CudaIntStorage};
pub use device::{BoolOps, Device, FloatOps, IntOps};

pub use dtype::{Bool, DType, DTypeKind, Float, Int, KindTag, Storage, FloatDType, IntDType, BoolDType};
pub use dynamic::DynTensor;
pub use error::{Error, Result};
pub use grad::{FloatMeta, GradStore, NoGradGuard, TensorMeta, is_grad_enabled, set_grad_enabled};
pub use ops::{BinaryOp, CmpOp, FloatUnaryOp, Op, ReduceOp, TransferDTypeKind, UnaryOp, ViewOp};
pub use ops::{IndexOp, Indexer, Slice};
pub use scalar::Scalar;
pub use tensor::{D, Dim, Dims, Layout, Shape, StorageIndices, Tensor, TensorId, TensorImpl};
