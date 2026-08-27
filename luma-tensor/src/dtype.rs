use crate::{Device, FloatMeta, grad::TensorMeta};
use std::fmt::Debug;

/// Binds a tensor kind to its per-device storage type and its autograd metadata.
///
/// This is the bridge between the compile-time kind marker and the device's
/// runtime storage: `Tensor<D, K>` stores `K::Storage` and `K::Meta`.
pub trait DTypeKind<D: Device>: Sized {
    type Scalar: Send + Sync + Clone + Copy + 'static;
    type Storage: Storage<D, Self>;
    type Meta: TensorMeta<D, Self>;
    type DType: Send + Sync + Clone + Copy + 'static + PartialEq + Eq + Debug + Default;

    /// Runtime discriminant for this kind.
    const KIND: KindTag;
}

pub trait Storage<D: Device, K: DTypeKind<D>>: Send + Sync + 'static {
    fn dtype(&self) -> K::DType;
    fn device(&self) -> &D;
}

/// Runtime element type. The tensor *kind* (`Float`/`Int`/`Bool`) is a compile-time
/// generic; the concrete precision inside a kind is this enum, decided at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    // Float kind
    F32,
    F64,
    // Int kind
    I32,
    U8,
    U32,
    // Bool kind
    Bool,
}

impl DType {
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F32 | DType::F64)
    }

    pub fn is_int(&self) -> bool {
        matches!(self, DType::I32 | DType::U8 | DType::U32)
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, DType::Bool)
    }

    /// The tensor kind this dtype belongs to.
    pub fn kind(&self) -> KindTag {
        match self {
            DType::F32 | DType::F64 => KindTag::Float,
            DType::I32 | DType::U8 | DType::U32 => KindTag::Int,
            DType::Bool => KindTag::Bool,
        }
    }

    /// Size of a single element in bytes.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::I32 => 4,
            DType::U8 => 1,
            DType::U32 => 4,
            DType::Bool => 1,
        }
    }

    /// Convert to `FloatDType` if this is a float dtype, otherwise panic.
    pub fn as_float(&self) -> FloatDType {
        match self {
            DType::F32 => FloatDType::F32,
            DType::F64 => FloatDType::F64,
            _ => panic!("DType::{:?} is not a float dtype", self),
        }
    }

    /// Convert to `IntDType` if this is an int dtype, otherwise panic.
    pub fn as_int(&self) -> IntDType {
        match self {
            DType::I32 => IntDType::I32,
            DType::U8 => IntDType::U8,
            DType::U32 => IntDType::U32,
            _ => panic!("DType::{:?} is not an int dtype", self),
        }
    }
}

/// Runtime discriminant mirroring the compile-time kind markers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KindTag {
    Float,
    Int,
    Bool,
}

/// The three tensor kinds, used only as zero-sized compile-time markers on
/// `Tensor<D, K>`. They never hold data; behaviour lives in the device ops traits.
pub struct Float;
pub struct Int;
pub struct Bool;

impl<D: Device> DTypeKind<D> for Float {
    type Scalar = f64;
    type Storage = D::FloatStorage;
    type Meta = FloatMeta<D>;
    type DType = FloatDType;
    const KIND: KindTag = KindTag::Float;
}

impl<D: Device> DTypeKind<D> for Int {
    type Scalar = i64;
    type Storage = D::IntStorage;
    type Meta = ();
    type DType = IntDType;
    const KIND: KindTag = KindTag::Int;
}

impl<D: Device> DTypeKind<D> for Bool {
    type Scalar = bool;
    type Storage = D::BoolStorage;
    type Meta = ();
    type DType = BoolDType;
    const KIND: KindTag = KindTag::Bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FloatDType {
    #[default]
    F32,
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum IntDType {
    #[default]
    I32,
    U8,
    U32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BoolDType {
    #[default]
    Bool,
}

impl Into<DType> for FloatDType {
    fn into(self) -> DType {
        match self {
            FloatDType::F32 => DType::F32,
            FloatDType::F64 => DType::F64,
        }
    }
}

impl Into<DType> for IntDType {
    fn into(self) -> DType {
        match self {
            IntDType::I32 => DType::I32,
            IntDType::U8 => DType::U8,
            IntDType::U32 => DType::U32,
        }
    }
}

impl Into<DType> for BoolDType {
    fn into(self) -> DType {
        match self {
            BoolDType::Bool => DType::Bool,
        }
    }
}
