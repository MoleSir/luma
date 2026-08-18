//! CPU storage: one flat `Vec<T>` per precision, wrapped in an enum so the
//! concrete precision of a `Float`/`Int` tensor is a runtime choice.

use crate::{
    Bool, DType, Float, Int, Storage,
    dtype::{BoolDType, FloatDType, IntDType},
};

use super::Cpu;

/// Backing buffer for a `Float`-kind tensor on the CPU.
#[derive(Debug, Clone)]
pub enum CpuFloatStorage {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl Storage<Cpu, Float> for CpuFloatStorage {
    fn dtype(&self) -> FloatDType {
        match self {
            Self::F32(_) => FloatDType::F32,
            Self::F64(_) => FloatDType::F64,
        }
    }

    fn device(&self) -> &Cpu {
        &Cpu
    }
}

impl CpuFloatStorage {
    pub fn dtype(&self) -> DType {
        match self {
            CpuFloatStorage::F32(_) => DType::F32,
            CpuFloatStorage::F64(_) => DType::F64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            CpuFloatStorage::F32(v) => v.len(),
            CpuFloatStorage::F64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Backing buffer for an `Int`-kind tensor on the CPU.
#[derive(Debug, Clone)]
pub enum CpuIntStorage {
    I32(Vec<i32>),
    U32(Vec<u32>),
    U8(Vec<u8>),
}

impl Storage<Cpu, Int> for CpuIntStorage {
    fn dtype(&self) -> IntDType {
        match self {
            Self::I32(_) => IntDType::I32,
            Self::U32(_) => IntDType::U32,
            Self::U8(_) => IntDType::U8,
        }
    }

    fn device(&self) -> &Cpu {
        &Cpu
    }
}

impl CpuIntStorage {
    pub fn dtype(&self) -> DType {
        match self {
            CpuIntStorage::I32(_) => DType::I32,
            CpuIntStorage::U32(_) => DType::U32,
            CpuIntStorage::U8(_) => DType::U8,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            CpuIntStorage::I32(v) => v.len(),
            CpuIntStorage::U32(v) => v.len(),
            CpuIntStorage::U8(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Backing buffer for a `Bool`-kind tensor on the CPU.
#[derive(Debug, Clone)]
pub struct CpuBoolStorage(pub Vec<bool>);

impl CpuBoolStorage {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Storage<Cpu, Bool> for CpuBoolStorage {
    fn dtype(&self) -> BoolDType {
        BoolDType::Bool
    }

    fn device(&self) -> &Cpu {
        &Cpu
    }
}
