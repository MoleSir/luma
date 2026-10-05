use cudarc::driver::CudaSlice;

use crate::Cuda;
use luma_tensor::{BoolDType, FloatDType, IntDType};
use luma_tensor::{Bool, DType, Float, Int, Storage};

impl Storage<Cuda, Float> for CudaFloatStorage {
    fn dtype(&self) -> FloatDType {
        match &self.slice {
            CudaFloatSlice::F32(_) => FloatDType::F32,
            CudaFloatSlice::F64(_) => FloatDType::F64,
            CudaFloatSlice::F16(_) => FloatDType::F16,
            CudaFloatSlice::BF16(_) => FloatDType::BF16,
        }
    }

    fn device(&self) -> &Cuda {
        &self.device
    }
}

impl CudaFloatStorage {
    pub fn dtype(&self) -> DType {
        match &self.slice {
            CudaFloatSlice::F32(_) => DType::F32,
            CudaFloatSlice::F64(_) => DType::F64,
            CudaFloatSlice::F16(_) => DType::F16,
            CudaFloatSlice::BF16(_) => DType::BF16,
        }
    }

    pub fn len(&self) -> usize {
        match &self.slice {
            CudaFloatSlice::F32(s) => s.len(),
            CudaFloatSlice::F64(s) => s.len(),
            CudaFloatSlice::F16(s) => s.len(),
            CudaFloatSlice::BF16(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Storage<Cuda, Int> for CudaIntStorage {
    fn dtype(&self) -> IntDType {
        match &self.slice {
            CudaIntSlice::I32(_) => IntDType::I32,
            CudaIntSlice::U32(_) => IntDType::U32,
            CudaIntSlice::U8(_) => IntDType::U8,
        }
    }

    fn device(&self) -> &Cuda {
        &self.device
    }
}

impl CudaIntStorage {
    pub fn dtype(&self) -> DType {
        match &self.slice {
            CudaIntSlice::I32(_) => DType::I32,
            CudaIntSlice::U32(_) => DType::U32,
            CudaIntSlice::U8(_) => DType::U8,
        }
    }

    pub fn len(&self) -> usize {
        match &self.slice {
            CudaIntSlice::I32(s) => s.len(),
            CudaIntSlice::U32(s) => s.len(),
            CudaIntSlice::U8(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Storage<Cuda, Bool> for CudaBoolStorage {
    fn dtype(&self) -> BoolDType {
        BoolDType::Bool
    }

    fn device(&self) -> &Cuda {
        &self.device
    }
}

impl CudaBoolStorage {
    pub fn dtype(&self) -> DType {
        DType::Bool
    }

    pub fn len(&self) -> usize {
        self.slice.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct CudaFloatStorage {
    pub slice: CudaFloatSlice,
    pub device: Cuda,
}

pub enum CudaFloatSlice {
    F32(CudaSlice<f32>),
    F64(CudaSlice<f64>),
    F16(CudaSlice<half::f16>),
    BF16(CudaSlice<half::bf16>),
}

impl CudaFloatSlice {
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::F16(_) => DType::F16,
            Self::BF16(_) => DType::BF16,
        }
    }
}

pub struct CudaIntStorage {
    pub slice: CudaIntSlice,
    pub device: Cuda,
}

pub enum CudaIntSlice {
    I32(CudaSlice<i32>),
    U32(CudaSlice<u32>),
    U8(CudaSlice<u8>),
}

impl CudaIntSlice {
    pub fn dtype(&self) -> DType {
        match self {
            Self::I32(_) => DType::I32,
            Self::U32(_) => DType::U32,
            Self::U8(_) => DType::U8,
        }
    }
}

pub struct CudaBoolStorage {
    pub slice: CudaSlice<u8>,
    pub device: Cuda,
}
