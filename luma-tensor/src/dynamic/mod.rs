//! Type-erased tensor enum for runtime kind dispatch.
//!
//! [`DynTensor<D>`] erases the compile-time kind parameter (`Float`/`Int`/`Bool`)
//! into a three-variant enum, enabling heterogeneous collections like
//! `HashMap<String, DynTensor<D>>` for model serialization (safetensors I/O).

use crate::dtype::{BoolDType, DType, KindTag};
use crate::error::Result;
use crate::tensor::{Shape, Tensor};
use crate::{Bool, Device, Float, Int};

/// A tensor whose kind (`Float`/`Int`/`Bool`) is determined at runtime.
///
/// This is the type-erased counterpart to [`Tensor<D, K>`]. Use it when you need
/// to store or pass tensors of different kinds together — most importantly for
/// disk I/O ([`from_bytes`](Self::from_bytes) / [`to_bytes`](Self::to_bytes)).
pub enum DynTensor<D: Device> {
    Float(Tensor<D, Float>),
    Int(Tensor<D, Int>),
    Bool(Tensor<D, Bool>),
}

// ---- accessors ----

impl<D: Device> DynTensor<D> {
    /// Runtime element type.
    pub fn dtype(&self) -> DType {
        match self {
            Self::Float(t) => t.dtype().into(),
            Self::Int(t) => t.dtype().into(),
            Self::Bool(_) => DType::Bool,
        }
    }

    /// Logical shape.
    pub fn shape(&self) -> &Shape {
        match self {
            Self::Float(t) => t.shape(),
            Self::Int(t) => t.shape(),
            Self::Bool(t) => t.shape(),
        }
    }

    /// Dimensions as a slice.
    pub fn dims(&self) -> &[usize] {
        self.shape().dims()
    }

    /// Device that owns the storage.
    pub fn device(&self) -> &D {
        match self {
            Self::Float(t) => t.device(),
            Self::Int(t) => t.device(),
            Self::Bool(t) => t.device(),
        }
    }
}

// ---- checked conversions ----

impl<D: Device> DynTensor<D> {
    /// Borrow as a Float tensor, or `None`.
    pub fn as_float(&self) -> Option<&Tensor<D, Float>> {
        match self {
            Self::Float(t) => Some(t),
            _ => None,
        }
    }

    /// Borrow as an Int tensor, or `None`.
    pub fn as_int(&self) -> Option<&Tensor<D, Int>> {
        match self {
            Self::Int(t) => Some(t),
            _ => None,
        }
    }

    /// Borrow as a Bool tensor, or `None`.
    pub fn as_bool(&self) -> Option<&Tensor<D, Bool>> {
        match self {
            Self::Bool(t) => Some(t),
            _ => None,
        }
    }

    /// Unwrap into a Float tensor, or error.
    pub fn into_float(self) -> Result<Tensor<D, Float>> {
        match self {
            Self::Float(t) => Ok(t),
            other => Err(crate::Error::Msg(format!("expected Float DynTensor, got {:?}", other.dtype()))),
        }
    }

    /// Unwrap into an Int tensor, or error.
    pub fn into_int(self) -> Result<Tensor<D, Int>> {
        match self {
            Self::Int(t) => Ok(t),
            other => Err(crate::Error::Msg(format!("expected Int DynTensor, got {:?}", other.dtype()))),
        }
    }

    /// Unwrap into a Bool tensor, or error.
    pub fn into_bool(self) -> Result<Tensor<D, Bool>> {
        match self {
            Self::Bool(t) => Ok(t),
            other => Err(crate::Error::Msg(format!("expected Bool DynTensor, got {:?}", other.dtype()))),
        }
    }
}

// ---- I/O ----

impl<D: Device> DynTensor<D> {
    /// Create a `DynTensor` from raw little-endian bytes and a runtime [`DType`].
    ///
    /// This is the primary entry point for deserialization (safetensors, NPY, …).
    /// The byte slice length must equal `shape.element_count() * dtype.size_in_bytes()`.
    pub fn from_bytes(bytes: &[u8], dtype: DType, shape: impl Into<Shape>, device: &D) -> Result<Self> {
        let shape = shape.into();
        match dtype.kind() {
            KindTag::Float => Ok(DynTensor::Float(Tensor::<D, Float>::from_bytes(bytes, shape, (device, dtype.as_float()))?)),
            KindTag::Int => Ok(DynTensor::Int(Tensor::<D, Int>::from_bytes(bytes, shape, (device, dtype.as_int()))?)),
            KindTag::Bool => Ok(DynTensor::Bool(Tensor::<D, Bool>::from_bytes(bytes, shape, (device, BoolDType::Bool))?)),
        }
    }

    /// Serialize to raw little-endian bytes in logical order.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        match self {
            Self::Float(t) => t.to_bytes(),
            Self::Int(t) => t.to_bytes(),
            Self::Bool(t) => t.to_bytes(),
        }
    }
}

// ---- From impls ----

impl<D: Device> From<Tensor<D, Float>> for DynTensor<D> {
    fn from(t: Tensor<D, Float>) -> Self {
        Self::Float(t)
    }
}

impl<D: Device> From<Tensor<D, Int>> for DynTensor<D> {
    fn from(t: Tensor<D, Int>) -> Self {
        Self::Int(t)
    }
}

impl<D: Device> From<Tensor<D, Bool>> for DynTensor<D> {
    fn from(t: Tensor<D, Bool>) -> Self {
        Self::Bool(t)
    }
}

// ---- Display ----

impl<D: Device> std::fmt::Display for DynTensor<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Float(t) => write!(f, "{t}"),
            Self::Int(t) => write!(f, "{t}"),
            Self::Bool(t) => write!(f, "{t}"),
        }
    }
}

// ---- Clone (Tensor is Clone via Arc) ----

impl<D: Device> Clone for DynTensor<D> {
    fn clone(&self) -> Self {
        match self {
            Self::Float(t) => Self::Float(t.clone()),
            Self::Int(t) => Self::Int(t.clone()),
            Self::Bool(t) => Self::Bool(t.clone()),
        }
    }
}
