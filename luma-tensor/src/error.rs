use std::str::Utf8Error;

use crate::{DType, Shape};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    // === DType Errors ===
    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType { msg: &'static str, expected: DType, got: DType },

    #[error("dtype mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    DTypeMismatchBinaryOp { lhs: DType, rhs: DType, op: &'static str },

    /// A `Float`/`Int` tensor holds a concrete precision (`f32`/`f64`, ...) decided at
    /// runtime; two operands must share the same precision. Use `.cast(..)` to align them.
    #[error("dtype mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?} (align them with `cast`)")]
    DTypeMismatch { lhs: DType, rhs: DType, op: &'static str },

    #[error("unsupported dtype {0:?} for op {1}")]
    UnsupportedDTypeForOp(DType, &'static str),

    // === Dimension Index Errors ===
    #[error("Index '{index}' out of range at storage({storage_len}) in take method")]
    IndexOutOfRangeTake { storage_len: usize, index: usize },

    #[error("index '{index}' out of range range({max_size}) in {op}")]
    IndexOutOfRange { max_size: usize, index: usize, op: &'static str },

    #[error("{op}: dimension index {dim} out of range for shape {shape:?}")]
    DimOutOfRange { shape: Shape, dim: i32, op: &'static str },

    #[error("{op}: dim size out of range in size {size}")]
    DimSizeOutOfRange { size: usize, op: &'static str },

    #[error("{op}: duplicate dim index {dims:?} for shape {shape:?}")]
    DuplicateDimIndex { shape: Shape, dims: Vec<usize>, op: &'static str },

    #[error("try to repeat {repeats} for shape {shape}")]
    RepeatRankOutOfRange { repeats: Shape, shape: Shape },

    // === Shape Errors ===
    #[error("element count mismatch in reshape, try reshape {origin} to {target}")]
    ElementCountMismatchInReshape { origin: Shape, target: Shape },

    #[error("unexpected element size in {op}, expected: {expected}, got: {got}")]
    ElementSizeMismatch { expected: usize, got: usize, op: &'static str },

    #[error("unexpected rank, expected: {expected}, got: {got} ({shape:?})")]
    UnexpectedNumberOfDims { expected: usize, got: usize, shape: Shape },

    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedShape { msg: String, expected: Shape, got: Shape },

    #[error("requires contiguous {op}")]
    RequiresContiguous { op: &'static str },

    #[error("invalid index in {op}")]
    InvalidIndex { index: usize, size: usize, op: &'static str },

    #[error("shape mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    ShapeMismatchBinaryOp { lhs: Shape, rhs: Shape, op: &'static str },

    #[error("device mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    DeviceMismatchBinaryOp { lhs: String, rhs: String, op: &'static str },

    #[error("device mismatch, {lhs:?} and {rhs:?}")]
    DeviceMismatch { lhs: String, rhs: String },

    #[error("shape mismatch in cat for dim {dim}, shape for arg1: {first_shape:?} shape for arg {n}: {nth_shape:?}")]
    ShapeMismatchCat { dim: usize, first_shape: Shape, n: usize, nth_shape: Shape },

    #[error("source Tensor shape {src:?} mismatch with condition shape {condition:?}")]
    ShapeMismatchMaskedSelect { src: Shape, condition: Shape },

    #[error("mask Tensor shape {mask:?} mismatch with {who} shape")]
    ShapeMismatchSelect { mask: Shape, who: &'static str },

    #[error("dst Tensor shape {dst:?} mismatch with src Tensor {src} shape")]
    ShapeMismatchCopyFrom { dst: Shape, src: Shape },

    // === Op Specific Errors ===
    #[error("narrow invalid args {msg}: {shape:?}, dim: {dim}, start: {start}, len:{len}")]
    NarrowInvalidArgs { shape: Shape, dim: usize, start: usize, len: usize, msg: &'static str },

    #[error("can squeeze {dim} dim of {shape:?}(not 1)")]
    SqueezeDimNot1 { shape: Shape, dim: usize },

    #[error("cannot broadcast {src_shape:?} to {dst_shape:?}")]
    BroadcastIncompatibleShapes { src_shape: Shape, dst_shape: Shape },

    #[error("{op} expects at least one tensor")]
    OpRequiresAtLeastOneTensor { op: &'static str },

    #[error("rand error because {0}")]
    Rand(String),

    #[error("Tensor is not a scalar")]
    NotScalar,

    #[error("backward not support '{0}'")]
    BackwardNotSupported(&'static str),

    /// Integer parse error.
    #[error(transparent)]
    ParseInt(#[from] std::num::ParseIntError),

    /// Utf8 parse error.
    #[error(transparent)]
    FromUtf8(#[from] std::string::FromUtf8Error),

    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Cuda(#[from] crate::device::cuda::CudaError),

    #[error(transparent)]
    Utf8(#[from] Utf8Error),

    /// Storage error
    #[error("visit a meta tensor!")]
    MetaTensor,

    /// User generated error message
    #[error("{0}")]
    Msg(String),

    #[error("unwrap none")]
    UnwrapNone,
}

pub type Result<T> = std::result::Result<T, Error>;

#[macro_export]
macro_rules! bail {
    ($msg:literal $(,)?) => {
        return Err($crate::Error::Msg(format!($msg).into()))?
    };
    ($err:expr $(,)?) => {
        return Err($crate::Error::Msg(format!($err).into()))?
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err($crate::Error::Msg(format!($fmt, $($arg)*).into()))?
    };
}
