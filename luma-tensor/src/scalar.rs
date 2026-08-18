//! Scalar values mirroring [`crate::DType`] variants.
//!
//! Used for optimizer hyperparameters and other non-tensor data in checkpoint files.

use crate::DType;

/// A typed scalar value corresponding to a [`DType`] variant.
///
/// Unlike a 0-d tensor this carries no device, layout, or autograd metadata —
/// it is just a plain value with a known element type.
#[derive(Clone, Debug, PartialEq)]
pub enum Scalar {
    F32(f32),
    F64(f64),
    I32(i32),
    U32(u32),
    U8(u8),
    Bool(bool),
}

impl Scalar {
    /// The [`DType`] of this scalar.
    pub fn dtype(&self) -> DType {
        match self {
            Scalar::F32(_) => DType::F32,
            Scalar::F64(_) => DType::F64,
            Scalar::I32(_) => DType::I32,
            Scalar::U32(_) => DType::U32,
            Scalar::U8(_) => DType::U8,
            Scalar::Bool(_) => DType::Bool,
        }
    }

    /// View as `f64` if the scalar is a float variant.
    pub fn to_f64(&self) -> Option<f64> {
        match self {
            Scalar::F32(v) => Some(*v as f64),
            Scalar::F64(v) => Some(*v),
            _ => None,
        }
    }

    /// View as `i64` if the scalar is an integer variant.
    pub fn to_i64(&self) -> Option<i64> {
        match self {
            Scalar::I32(v) => Some(*v as i64),
            Scalar::U32(v) => Some(*v as i64),
            Scalar::U8(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// View as `bool` if the scalar is the bool variant.
    pub fn to_bool(&self) -> Option<bool> {
        match self {
            Scalar::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

// ---- Convenience From impls ----

impl From<f64> for Scalar {
    fn from(v: f64) -> Self {
        Scalar::F64(v)
    }
}

impl From<f32> for Scalar {
    fn from(v: f32) -> Self {
        Scalar::F32(v)
    }
}

impl From<i64> for Scalar {
    fn from(v: i64) -> Self {
        Scalar::I32(v as i32)
    }
}

impl From<i32> for Scalar {
    fn from(v: i32) -> Self {
        Scalar::I32(v)
    }
}

impl From<bool> for Scalar {
    fn from(v: bool) -> Self {
        Scalar::Bool(v)
    }
}

impl std::fmt::Display for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scalar::F32(v) => write!(f, "{}", v),
            Scalar::F64(v) => write!(f, "{}", v),
            Scalar::I32(v) => write!(f, "{}", v),
            Scalar::U32(v) => write!(f, "{}", v),
            Scalar::U8(v) => write!(f, "{}", v),
            Scalar::Bool(v) => write!(f, "{}", v),
        }
    }
}
