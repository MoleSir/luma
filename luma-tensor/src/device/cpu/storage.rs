//! CPU storage: one flat `Vec<T>` per precision, wrapped in an enum so the
//! concrete precision of a `Float`/`Int` tensor is a runtime choice.
//!
//! Every storage carries the [`Cpu`] instance that created it, so
//! `Storage::device()` returns the *real* instance (which will later hold the
//! allocator) instead of a promoted static. Result tensors inherit it via
//! `from_storage`, so an allocator configured on one device instance flows
//! through the whole computation chain.

use crate::{
    Bool, DType, Float, Int, Storage,
    dtype::{BoolDType, FloatDType, IntDType},
};

use super::Cpu;

/// Backing buffer for a `Float`-kind tensor on the CPU.
#[derive(Debug, Clone)]
pub enum CpuFloatStorage {
    F32(Vec<f32>, Cpu),
    F64(Vec<f64>, Cpu),
}

impl Storage<Cpu, Float> for CpuFloatStorage {
    fn dtype(&self) -> FloatDType {
        match self {
            Self::F32(_, _) => FloatDType::F32,
            Self::F64(_, _) => FloatDType::F64,
        }
    }

    fn device(&self) -> &Cpu {
        match self {
            Self::F32(_, device) => device,
            Self::F64(_, device) => device,
        }
    }
}

impl Drop for CpuFloatStorage {
    fn drop(&mut self) {
        // 在 &mut self 上就地取走字段：真 Vec 交给 allocator（System 下 = drop，
        // 池化下 = 复用），原位留下零分配空 Vec——对象释放时 drop 只跑一次，
        // 空占位不会二次分配，也没有递归。
        let alloc = match self {
            CpuFloatStorage::F32(_, d) | CpuFloatStorage::F64(_, d) => d.allocator().clone(),
        };
        let alloc = alloc.read().expect("allocator poisoned");
        match self {
            CpuFloatStorage::F32(v, _) => alloc.dealloc_f32(std::mem::replace(v, Vec::new())),
            CpuFloatStorage::F64(v, _) => alloc.dealloc_f64(std::mem::replace(v, Vec::new())),
        }
    }
}

impl CpuFloatStorage {
    pub fn dtype(&self) -> DType {
        match self {
            CpuFloatStorage::F32(_, _) => DType::F32,
            CpuFloatStorage::F64(_, _) => DType::F64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            CpuFloatStorage::F32(v, _) => v.len(),
            CpuFloatStorage::F64(v, _) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Backing buffer for an `Int`-kind tensor on the CPU.
#[derive(Debug, Clone)]
pub enum CpuIntStorage {
    I32(Vec<i32>, Cpu),
    U32(Vec<u32>, Cpu),
    U8(Vec<u8>, Cpu),
}

impl Storage<Cpu, Int> for CpuIntStorage {
    fn dtype(&self) -> IntDType {
        match self {
            Self::I32(_, _) => IntDType::I32,
            Self::U32(_, _) => IntDType::U32,
            Self::U8(_, _) => IntDType::U8,
        }
    }

    fn device(&self) -> &Cpu {
        match self {
            Self::I32(_, device) => device,
            Self::U32(_, device) => device,
            Self::U8(_, device) => device,
        }
    }
}

impl Drop for CpuIntStorage {
    fn drop(&mut self) {
        let alloc = match self {
            CpuIntStorage::I32(_, d) | CpuIntStorage::U32(_, d) | CpuIntStorage::U8(_, d) => d.allocator().clone(),
        };
        let alloc = alloc.read().expect("allocator poisoned");
        match self {
            CpuIntStorage::I32(v, _) => alloc.dealloc_i32(std::mem::replace(v, Vec::new())),
            CpuIntStorage::U32(v, _) => alloc.dealloc_u32(std::mem::replace(v, Vec::new())),
            CpuIntStorage::U8(v, _) => alloc.dealloc_u8(std::mem::replace(v, Vec::new())),
        }
    }
}

impl CpuIntStorage {
    pub fn dtype(&self) -> DType {
        match self {
            CpuIntStorage::I32(_, _) => DType::I32,
            CpuIntStorage::U32(_, _) => DType::U32,
            CpuIntStorage::U8(_, _) => DType::U8,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            CpuIntStorage::I32(v, _) => v.len(),
            CpuIntStorage::U32(v, _) => v.len(),
            CpuIntStorage::U8(v, _) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Backing buffer for a `Bool`-kind tensor on the CPU.
#[derive(Debug, Clone)]
pub struct CpuBoolStorage(pub Vec<bool>, pub Cpu);

impl Drop for CpuBoolStorage {
    fn drop(&mut self) {
        let v = std::mem::replace(&mut self.0, Vec::new());
        self.1.allocator().read().expect("allocator poisoned").dealloc_bool(v);
    }
}

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
        &self.1
    }
}
