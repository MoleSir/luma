use crate::{BinaryOp, CmpOp, IntOps, Layout, ReduceOp, Result, Shape, Storage, UnaryOp, device::cuda::CudaError, dtype::{BoolDType, DType, FloatDType, IntDType}};
use crate::device::cuda::{Cuda, CudaFloatSlice, CudaIntSlice, CudaIntStorage, CudaFloatStorage, CudaBoolStorage};
use cudarc::driver::CudaSlice;
use super::super::kernel;
use super::super::launch;

// ---- indexing dispatch helpers ----

macro_rules! _int_select {
    ($x:ident, $x_l:ident, $idx:ident, $idx_l:ident, $dim:ident, $launch_fn:ident, $op:literal, $out_shape:expr) => {
        match (&$idx.slice, &$x.slice) {
            (CudaIntSlice::I32(ids), CudaIntSlice::I32(v)) => {
                let out = launch::$launch_fn(&$x.device, "i32", "i32", &kernel::INDEXING, v, $x_l, ids, $idx_l, $dim)?;
                Ok((CudaIntStorage { slice: CudaIntSlice::I32(out), device: $x.device.clone() }, $out_shape))
            }
            (CudaIntSlice::I32(ids), CudaIntSlice::U32(v)) => {
                let out = launch::$launch_fn(&$x.device, "i32", "u32", &kernel::INDEXING, v, $x_l, ids, $idx_l, $dim)?;
                Ok((CudaIntStorage { slice: CudaIntSlice::U32(out), device: $x.device.clone() }, $out_shape))
            }
            (CudaIntSlice::I32(ids), CudaIntSlice::U8(v)) => {
                let out = launch::$launch_fn(&$x.device, "i32", "u8", &kernel::INDEXING, v, $x_l, ids, $idx_l, $dim)?;
                Ok((CudaIntStorage { slice: CudaIntSlice::U8(out), device: $x.device.clone() }, $out_shape))
            }
            (CudaIntSlice::U32(ids), CudaIntSlice::I32(v)) => {
                let out = launch::$launch_fn(&$x.device, "u32", "i32", &kernel::INDEXING, v, $x_l, ids, $idx_l, $dim)?;
                Ok((CudaIntStorage { slice: CudaIntSlice::I32(out), device: $x.device.clone() }, $out_shape))
            }
            (CudaIntSlice::U32(ids), CudaIntSlice::U32(v)) => {
                let out = launch::$launch_fn(&$x.device, "u32", "u32", &kernel::INDEXING, v, $x_l, ids, $idx_l, $dim)?;
                Ok((CudaIntStorage { slice: CudaIntSlice::U32(out), device: $x.device.clone() }, $out_shape))
            }
            (CudaIntSlice::U32(ids), CudaIntSlice::U8(v)) => {
                let out = launch::$launch_fn(&$x.device, "u32", "u8", &kernel::INDEXING, v, $x_l, ids, $idx_l, $dim)?;
                Ok((CudaIntStorage { slice: CudaIntSlice::U8(out), device: $x.device.clone() }, $out_shape))
            }
            _ => Err(crate::Error::DTypeMismatch { lhs: $x.slice.dtype(), rhs: $idx.slice.dtype(), op: $op })
        }
    };
}

macro_rules! _int_add {
    ($init:ident, $init_l:ident, $idx:ident, $idx_l:ident, $src:ident, $src_l:ident, $dim:ident, $launch_fn:ident, $op:literal) => {
        match (&$idx.slice, &$src.slice, &$init.slice) {
            (CudaIntSlice::I32(ids), CudaIntSlice::I32(s), CudaIntSlice::I32(d)) => {
                let out = launch::$launch_fn(&$init.device, "i32", "i32", &kernel::INDEXING, d, $init_l, ids, $idx_l, s, $src_l, $dim)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(out), device: $init.device.clone() })
            }
            (CudaIntSlice::I32(ids), CudaIntSlice::U32(s), CudaIntSlice::U32(d)) => {
                let out = launch::$launch_fn(&$init.device, "i32", "u32", &kernel::INDEXING, d, $init_l, ids, $idx_l, s, $src_l, $dim)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(out), device: $init.device.clone() })
            }
            (CudaIntSlice::I32(ids), CudaIntSlice::U8(s), CudaIntSlice::U8(d)) => {
                let out = launch::$launch_fn(&$init.device, "i32", "u8", &kernel::INDEXING, d, $init_l, ids, $idx_l, s, $src_l, $dim)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(out), device: $init.device.clone() })
            }
            (CudaIntSlice::U32(ids), CudaIntSlice::I32(s), CudaIntSlice::I32(d)) => {
                let out = launch::$launch_fn(&$init.device, "u32", "i32", &kernel::INDEXING, d, $init_l, ids, $idx_l, s, $src_l, $dim)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(out), device: $init.device.clone() })
            }
            (CudaIntSlice::U32(ids), CudaIntSlice::U32(s), CudaIntSlice::U32(d)) => {
                let out = launch::$launch_fn(&$init.device, "u32", "u32", &kernel::INDEXING, d, $init_l, ids, $idx_l, s, $src_l, $dim)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(out), device: $init.device.clone() })
            }
            (CudaIntSlice::U32(ids), CudaIntSlice::U8(s), CudaIntSlice::U8(d)) => {
                let out = launch::$launch_fn(&$init.device, "u32", "u8", &kernel::INDEXING, d, $init_l, ids, $idx_l, s, $src_l, $dim)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(out), device: $init.device.clone() })
            }
            _ => Err(crate::Error::DTypeMismatch { lhs: $init.slice.dtype(), rhs: $src.slice.dtype(), op: $op })
        }
    };
}

impl IntOps<Cuda> for Cuda {
    fn i_zeros(shape: &Shape, device: &Cuda, dtype: IntDType) -> Result<CudaIntStorage> {
        let elem_count = shape.element_count();
        match dtype {
            IntDType::I32 => {
                let data = device.alloc_zeros::<i32>(elem_count)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(data), device: device.clone() })
            }
            IntDType::U32 => {
                let data = device.alloc_zeros::<u32>(elem_count)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(data), device: device.clone() })
            }
            IntDType::U8 => {
                let data = device.alloc_zeros::<u8>(elem_count)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(data), device: device.clone() })
            }
        }
    }
    
    fn i_ones(shape: &Shape, device: &Cuda, dtype: IntDType) -> Result<CudaIntStorage> {
        let elem_count = shape.element_count();
        match dtype {
            IntDType::I32 => {
                let host = vec![1i32; elem_count];
                let data = device.memcpy_stod(&host)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(data), device: device.clone() })
            }
            IntDType::U32 => {
                let host = vec![1u32; elem_count];
                let data = device.memcpy_stod(&host)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(data), device: device.clone() })
            }
            IntDType::U8 => {
                let host = vec![1u8; elem_count];
                let data = device.memcpy_stod(&host)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(data), device: device.clone() })
            }
        }
    }

    fn i_full(shape: &Shape, value: i64, device: &Cuda, dtype: IntDType) -> Result<CudaIntStorage> {
        let elem_count = shape.element_count();
        match dtype {
            IntDType::I32 => {
                let host = vec![value as i32; elem_count];
                let data = device.memcpy_stod(&host)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(data), device: device.clone() })
            }
            IntDType::U32 => {
                let host = vec![value as u32; elem_count];
                let data = device.memcpy_stod(&host)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(data), device: device.clone() })
            }
            IntDType::U8 => {
                let host = vec![value as u8; elem_count];
                let data = device.memcpy_stod(&host)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(data), device: device.clone() })
            }
        }
    }

    fn i_from_i64(data: &[i64], device: &Cuda, dtype: IntDType) -> Result<CudaIntStorage> {
        match dtype {
            IntDType::I32 => {
                let host: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                let slice = device.memcpy_stod(&host)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(slice), device: device.clone() })
            }
            IntDType::U32 => {
                let host: Vec<u32> = data.iter().map(|&v| v as u32).collect();
                let slice = device.memcpy_stod(&host)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(slice), device: device.clone() })
            }
            IntDType::U8 => {
                let host: Vec<u8> = data.iter().map(|&v| v as u8).collect();
                let slice = device.memcpy_stod(&host)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(slice), device: device.clone() })
            }
        }
    }

    fn i_arange(start: i64, end: i64, step: i64, device: &Cuda, dtype: IntDType) -> Result<(CudaIntStorage, usize)> {
        if step == 0 {
            return Err(crate::Error::Msg("arange step cannot be 0".into()));
        }
        let mut data = Vec::new();
        let mut v = start;
        if step > 0 {
            while v < end { data.push(v); v += step; }
        } else {
            while v > end { data.push(v); v += step; }
        }
        let n = data.len();
        match dtype {
            IntDType::I32 => {
                let host: Vec<i32> = data.iter().map(|&x| x as i32).collect();
                let slice = device.memcpy_stod(&host)?;
                Ok((CudaIntStorage { slice: CudaIntSlice::I32(slice), device: device.clone() }, n))
            }
            IntDType::U32 => {
                let host: Vec<u32> = data.iter().map(|&x| x as u32).collect();
                let slice = device.memcpy_stod(&host)?;
                Ok((CudaIntStorage { slice: CudaIntSlice::U32(slice), device: device.clone() }, n))
            }
            IntDType::U8 => {
                let host: Vec<u8> = data.iter().map(|&x| x as u8).collect();
                let slice = device.memcpy_stod(&host)?;
                Ok((CudaIntStorage { slice: CudaIntSlice::U8(slice), device: device.clone() }, n))
            }
        }
    }

    fn i_contiguous(x: &CudaIntStorage, layout: &Layout) -> Result<CudaIntStorage> {
        match &x.slice {
            CudaIntSlice::I32(data) => {
                let out = launch::launch_cast(&x.device, "i32", "i32", &kernel::CAST, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(out), device: x.device.clone() })
            }
            CudaIntSlice::U32(data) => {
                let out = launch::launch_cast(&x.device, "u32", "u32", &kernel::CAST, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(out), device: x.device.clone() })
            }
            CudaIntSlice::U8(data) => {
                let out = launch::launch_cast(&x.device, "u8", "u8", &kernel::CAST, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(out), device: x.device.clone() })
            }
        }
    }

    fn i_cast_float(x: &CudaIntStorage, layout: &Layout, to: FloatDType) -> Result<CudaFloatStorage> {
        let device = &x.device;
        match (&x.slice, to) {
            (CudaIntSlice::I32(data), FloatDType::F32) => {
                let out = launch::launch_cast(device, "i32", "f32", &kernel::CAST, data, layout)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: device.clone() })
            }
            (CudaIntSlice::I32(data), FloatDType::F64) => {
                let out = launch::launch_cast(device, "i32", "f64", &kernel::CAST, data, layout)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: device.clone() })
            }
            (CudaIntSlice::U32(data), FloatDType::F32) => {
                let out = launch::launch_cast(device, "u32", "f32", &kernel::CAST, data, layout)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: device.clone() })
            }
            (CudaIntSlice::U32(data), FloatDType::F64) => {
                let out = launch::launch_cast(device, "u32", "f64", &kernel::CAST, data, layout)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: device.clone() })
            }
            (CudaIntSlice::U8(data), FloatDType::F32) => {
                let out = launch::launch_cast(device, "u8", "f32", &kernel::CAST, data, layout)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: device.clone() })
            }
            (CudaIntSlice::U8(data), FloatDType::F64) => {
                let out = launch::launch_cast(device, "u8", "f64", &kernel::CAST, data, layout)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: device.clone() })
            }
        }
    }

    fn i_cast_int(x: &CudaIntStorage, layout: &Layout, to: IntDType) -> Result<CudaIntStorage> {
        let device = &x.device;
        match (&x.slice, to) {
            (CudaIntSlice::I32(data), IntDType::I32) => {
                let out = launch::launch_cast(device, "i32", "i32", &kernel::CAST, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(out), device: device.clone() })
            }
            (CudaIntSlice::I32(data), IntDType::U32) => {
                let out = launch::launch_cast(device, "i32", "u32", &kernel::CAST, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(out), device: device.clone() })
            }
            (CudaIntSlice::I32(data), IntDType::U8) => {
                let out = launch::launch_cast(device, "i32", "u8", &kernel::CAST, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(out), device: device.clone() })
            }
            (CudaIntSlice::U32(data), IntDType::I32) => {
                let out = launch::launch_cast(device, "u32", "i32", &kernel::CAST, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(out), device: device.clone() })
            }
            (CudaIntSlice::U32(data), IntDType::U32) => {
                let out = launch::launch_cast(device, "u32", "u32", &kernel::CAST, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(out), device: device.clone() })
            }
            (CudaIntSlice::U32(data), IntDType::U8) => {
                let out = launch::launch_cast(device, "u32", "u8", &kernel::CAST, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(out), device: device.clone() })
            }
            (CudaIntSlice::U8(data), IntDType::I32) => {
                let out = launch::launch_cast(device, "u8", "i32", &kernel::CAST, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(out), device: device.clone() })
            }
            (CudaIntSlice::U8(data), IntDType::U32) => {
                let out = launch::launch_cast(device, "u8", "u32", &kernel::CAST, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(out), device: device.clone() })
            }
            (CudaIntSlice::U8(data), IntDType::U8) => {
                let out = launch::launch_cast(device, "u8", "u8", &kernel::CAST, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(out), device: device.clone() })
            }
        }
    }

    fn i_cast_bool(x: &CudaIntStorage, layout: &Layout, _to: BoolDType) -> Result<CudaBoolStorage> {
        let device = &x.device;
        match &x.slice {
            CudaIntSlice::I32(data) => {
                let out = launch::launch_cast(device, "i32", "bool", &kernel::CAST, data, layout)?;
                Ok(CudaBoolStorage { slice: out, device: device.clone() })
            }
            CudaIntSlice::U32(data) => {
                let out = launch::launch_cast(device, "u32", "bool", &kernel::CAST, data, layout)?;
                Ok(CudaBoolStorage { slice: out, device: device.clone() })
            }
            CudaIntSlice::U8(data) => {
                let out = launch::launch_cast(device, "u8", "u8", &kernel::CAST, data, layout)?;
                Ok(CudaBoolStorage { slice: out, device: device.clone() })
            }
        }
    }

    fn i_to_vec(x: &CudaIntStorage, layout: &Layout) -> Result<Vec<i64>> {
        match &x.slice {
            CudaIntSlice::I32(data) => {
                let raw = x.device.memcpy_dtov(data)?;
                Ok(layout.storage_indices().map(|i| raw[i] as i64).collect())
            }
            CudaIntSlice::U32(data) => {
                let raw = x.device.memcpy_dtov(data)?;
                Ok(layout.storage_indices().map(|i| raw[i] as i64).collect())
            }
            CudaIntSlice::U8(data) => {
                let raw = x.device.memcpy_dtov(data)?;
                Ok(layout.storage_indices().map(|i| raw[i] as i64).collect())
            }
        }
    }
    
    fn i_binary(lhs: &CudaIntStorage, lhs_l: &Layout, rhs: &CudaIntStorage, rhs_l: &Layout, op: BinaryOp) -> Result<CudaIntStorage> {
        lhs.device.same_ordinal(&rhs.device, format!("Int {:?}", op))?;
        match (&lhs.slice, &rhs.slice) {
            (CudaIntSlice::I32(l), CudaIntSlice::I32(r)) => {
                let out = launch::launch_binary(&lhs.device, op, "i32", &kernel::BINARY, l, r, lhs_l, rhs_l)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(out), device: lhs.device.clone() })
            }
            (CudaIntSlice::U32(l), CudaIntSlice::U32(r)) => {
                let out = launch::launch_binary(&lhs.device, op, "u32", &kernel::BINARY, l, r, lhs_l, rhs_l)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(out), device: lhs.device.clone() })
            }
            (CudaIntSlice::U8(l), CudaIntSlice::U8(r)) => {
                let out = launch::launch_binary(&lhs.device, op, "u8", &kernel::BINARY, l, r, lhs_l, rhs_l)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(out), device: lhs.device.clone() })
            }
            _ => Err(crate::Error::DTypeMismatch { lhs: lhs.dtype(), rhs: rhs.dtype(), op: "binary" })
        }
    }

    fn i_binary_scalar(lhs: &CudaIntStorage, lhs_l: &Layout, rhs: i64, op: BinaryOp) -> Result<CudaIntStorage> {
        match &lhs.slice {
            CudaIntSlice::I32(data) => {
                let out = launch::launch_binary_scalar(&lhs.device, op, "i32", &kernel::BINARY_SCALAR, data, lhs_l, rhs as i32)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(out), device: lhs.device.clone() })
            }
            CudaIntSlice::U32(data) => {
                let out = launch::launch_binary_scalar(&lhs.device, op, "u32", &kernel::BINARY_SCALAR, data, lhs_l, rhs as u32)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(out), device: lhs.device.clone() })
            }
            CudaIntSlice::U8(data) => {
                let out = launch::launch_binary_scalar(&lhs.device, op, "u8", &kernel::BINARY_SCALAR, data, lhs_l, rhs as u8)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(out), device: lhs.device.clone() })
            }
        }
    }

    fn i_neg(x: &CudaIntStorage, layout: &Layout) -> Result<CudaIntStorage> {
        match &x.slice {
            CudaIntSlice::I32(data) => {
                let out = launch::launch_unary_raw_by_kernel_name(&x.device, &format!("uneg_i32"), &kernel::UNARY, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(out), device: x.device.clone() })
            }
            CudaIntSlice::U32(data) => {
                let out = launch::launch_unary_raw_by_kernel_name(&x.device, &format!("uneg_u32"), &kernel::UNARY, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(out), device: x.device.clone() })
            }
            CudaIntSlice::U8(data) => {
                let out = launch::launch_unary_raw_by_kernel_name(&x.device, &format!("uneg_u8"), &kernel::UNARY, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(out), device: x.device.clone() })
            }
        }
    }

    fn i_abs(x: &CudaIntStorage, layout: &Layout) -> Result<CudaIntStorage> {
        match &x.slice {
            CudaIntSlice::I32(data) => {
                let out = launch::launch_unary_raw_by_kernel_name(&x.device, &format!("uabs_i32"), &kernel::UNARY, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(out), device: x.device.clone() })
            }
            CudaIntSlice::U32(data) => {
                let out = launch::launch_unary_raw_by_kernel_name(&x.device, &format!("uabs_u32"), &kernel::UNARY, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(out), device: x.device.clone() })
            }
            CudaIntSlice::U8(data) => {
                let out = launch::launch_unary_raw_by_kernel_name(&x.device, &format!("uabs_u8"), &kernel::UNARY, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(out), device: x.device.clone() })
            }
        }
    }

    fn i_sign(x: &CudaIntStorage, layout: &Layout) -> Result<CudaIntStorage> {
        match &x.slice {
            CudaIntSlice::I32(data) => {
                let out = launch::launch_unary_raw_by_kernel_name(&x.device, &format!("usign_i32"), &kernel::UNARY, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(out), device: x.device.clone() })
            }
            CudaIntSlice::U32(data) => {
                let out = launch::launch_unary_raw_by_kernel_name(&x.device, &format!("usign_u32"), &kernel::UNARY, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(out), device: x.device.clone() })
            }
            CudaIntSlice::U8(data) => {
                let out = launch::launch_unary_raw_by_kernel_name(&x.device, &format!("usign_u8"), &kernel::UNARY, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(out), device: x.device.clone() })
            }
        }
    }

    fn i_affine(x: &CudaIntStorage, layout: &Layout, mul: i64, add: i64) -> Result<CudaIntStorage> {
        let device = &x.device;
        match &x.slice {
            CudaIntSlice::I32(data) => {
                let out = launch::launch_affine(device, "i32", &kernel::UNARY, data, layout, mul as i32, add as i32)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(out), device: device.clone() })
            }
            CudaIntSlice::U32(data) => {
                let out = launch::launch_affine(device, "u32", &kernel::UNARY, data, layout, mul as u32, add as u32)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(out), device: device.clone() })
            }
            CudaIntSlice::U8(data) => {
                let out = launch::launch_affine(device, "u8", &kernel::UNARY, data, layout, mul as u8, add as u8)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(out), device: device.clone() })
            }
        }
    }

    fn i_pow(x: &CudaIntStorage, layout: &Layout, exp: i64) -> Result<CudaIntStorage> {
        let device = &x.device;
        match &x.slice {
            CudaIntSlice::I32(data) => {
                let out = launch::launch_pow(device, "i32", &kernel::UNARY, data, layout, exp as i32)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(out), device: device.clone() })
            }
            CudaIntSlice::U32(data) => {
                let out = launch::launch_pow(device, "u32", &kernel::UNARY, data, layout, exp as u32)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(out), device: device.clone() })
            }
            CudaIntSlice::U8(data) => {
                let out = launch::launch_pow(device, "u8", &kernel::UNARY, data, layout, exp as u8)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(out), device: device.clone() })
            }
        }
    }

    fn i_clamp(x: &CudaIntStorage, layout: &Layout, min: Option<i64>, max: Option<i64>) -> Result<CudaIntStorage> {
        let device = &x.device;
        match &x.slice {
            CudaIntSlice::I32(data) => {
                let (has_min, min_val) = min.map_or((false, 0i32), |v| (true, v as i32));
                let (has_max, max_val) = max.map_or((false, 0i32), |v| (true, v as i32));
                let out = launch::launch_clamp(device, "i32", &kernel::UNARY, data, layout, has_min, min_val, has_max, max_val)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(out), device: device.clone() })
            }
            CudaIntSlice::U32(data) => {
                let (has_min, min_val) = min.map_or((false, 0u32), |v| (true, v.max(0) as u32));
                let (has_max, max_val) = max.map_or((false, 0u32), |v| (true, v as u32));
                let out = launch::launch_clamp(device, "u32", &kernel::UNARY, data, layout, has_min, min_val, has_max, max_val)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(out), device: device.clone() })
            }
            CudaIntSlice::U8(data) => {
                let (has_min, min_val) = min.map_or((false, 0u8), |v| (true, v.max(0) as u8));
                let (has_max, max_val) = max.map_or((false, 0u8), |v| (true, v as u8));
                let out = launch::launch_clamp(device, "u8", &kernel::UNARY, data, layout, has_min, min_val, has_max, max_val)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(out), device: device.clone() })
            }
        }
    }

    fn i_matmul(lhs: &CudaIntStorage, lhs_l: &Layout, rhs: &CudaIntStorage, rhs_l: &Layout) -> Result<(CudaIntStorage, Shape)> {
        Err(CudaError::UnsupportIntMatmul)?
    }

    fn i_cmp(lhs: &CudaIntStorage, lhs_l: &Layout, rhs: &CudaIntStorage, rhs_l: &Layout, op: CmpOp) -> Result<CudaBoolStorage> {
        lhs.device.same_ordinal(&rhs.device, format!("Cmp {:?}", op))?;
        match (&lhs.slice, &rhs.slice) {
            (CudaIntSlice::I32(l), CudaIntSlice::I32(r)) => {
                let out = launch::launch_cmp(&lhs.device, op, "i32", &kernel::BINARY, l, r, lhs_l, rhs_l)?;
                Ok(CudaBoolStorage { slice: out, device: lhs.device.clone() })
            }
            (CudaIntSlice::U32(l), CudaIntSlice::U32(r)) => {
                let out = launch::launch_cmp(&lhs.device, op, "u32", &kernel::BINARY, l, r, lhs_l, rhs_l)?;
                Ok(CudaBoolStorage { slice: out, device: lhs.device.clone() })
            }
            (CudaIntSlice::U8(l), CudaIntSlice::U8(r)) => {
                let out = launch::launch_cmp(&lhs.device, op, "u8", &kernel::BINARY, l, r, lhs_l, rhs_l)?;
                Ok(CudaBoolStorage { slice: out, device: lhs.device.clone() })
            }
            _ => Err(crate::Error::DTypeMismatch { lhs: lhs.dtype(), rhs: rhs.dtype(), op: "cmp" })
        }
    }

    fn i_reduce(x: &CudaIntStorage, layout: &Layout, dims: &[usize], keepdim: bool, op: ReduceOp) -> Result<(CudaIntStorage, Shape)> {
        if matches!(op, ReduceOp::Mean) {
            return Err(crate::Error::Msg("Mean reduce not supported for int types".into()));
        }
        let device = &x.device;
        match &x.slice {
            CudaIntSlice::I32(data) => {
                let (out, shape) = launch::launch_multi_reduce::<i32>(device, op, "i32", &kernel::REDUCE, data, layout, dims, keepdim)?;
                Ok((CudaIntStorage { slice: CudaIntSlice::I32(out), device: device.clone() }, shape))
            }
            CudaIntSlice::U32(data) => {
                let (out, shape) = launch::launch_multi_reduce::<u32>(device, op, "u32", &kernel::REDUCE, data, layout, dims, keepdim)?;
                Ok((CudaIntStorage { slice: CudaIntSlice::U32(out), device: device.clone() }, shape))
            }
            CudaIntSlice::U8(data) => {
                let (out, shape) = launch::launch_multi_reduce::<u8>(device, op, "u8", &kernel::REDUCE, data, layout, dims, keepdim)?;
                Ok((CudaIntStorage { slice: CudaIntSlice::U8(out), device: device.clone() }, shape))
            }
        }
    }

    fn i_arg_reduce(x: &CudaIntStorage, layout: &Layout, dim: usize, keepdim: bool, take_max: bool) -> Result<(CudaIntStorage, Shape)> {
        let dims = layout.dims().to_vec();
        let strides = layout.stride().to_vec();
        let reduce_size = dims[dim];
        let output_block_count: usize = dims.iter().enumerate()
            .filter(|&(i, _)| i != dim)
            .map(|(_, d)| *d)
            .product::<usize>()
            .max(1);

        let src = match &x.slice {
            CudaIntSlice::I32(s) => s,
            _ => return Err(crate::Error::DTypeMismatch {
                lhs: x.slice.dtype(), rhs: DType::I32, op: "arg_reduce",
            }),
        };

        let kernel_name = launch::arg_reduce_kernel_name(take_max, "i32");
        let indices = launch::launch_arg_reduce(&x.device, &kernel_name, &kernel::REDUCE,
            src, &dims, &strides, dim, reduce_size, output_block_count)?;

        let mut out_dims = dims;
        if keepdim {
            out_dims[dim] = 1;
        } else {
            out_dims.remove(dim);
        }
        if out_dims.is_empty() {
            out_dims = vec![1];
        }

        Ok((CudaIntStorage { slice: CudaIntSlice::I32(indices), device: x.device.clone() }, Shape::from(out_dims)))
    }

    fn i_index_select(x: &CudaIntStorage, x_l: &Layout, idx: &CudaIntStorage, idx_l: &Layout, dim: usize) -> Result<(CudaIntStorage, Shape)> {
        if !x_l.is_contiguous() || !idx_l.is_contiguous() {
            return Err(crate::Error::RequiresContiguous { op: "index_select" });
        }
        let mut out_dims = x_l.dims().to_vec();
        out_dims[dim] = idx_l.dims()[0];
        let out_shape = Shape::from(out_dims);
        _int_select!(x, x_l, idx, idx_l, dim, launch_index_select, "index_select", out_shape)
    }

    fn i_gather(x: &CudaIntStorage, x_l: &Layout, idx: &CudaIntStorage, idx_l: &Layout, dim: usize) -> Result<(CudaIntStorage, Shape)> {
        if !x_l.is_contiguous() || !idx_l.is_contiguous() {
            return Err(crate::Error::RequiresContiguous { op: "gather" });
        }
        let out_shape = Shape::from(idx_l.dims().to_vec());
        _int_select!(x, x_l, idx, idx_l, dim, launch_gather, "gather", out_shape)
    }

    fn i_index_add(init: &CudaIntStorage, init_l: &Layout, idx: &CudaIntStorage, idx_l: &Layout, src: &CudaIntStorage, src_l: &Layout, dim: usize) -> Result<CudaIntStorage> {
        if !init_l.is_contiguous() || !idx_l.is_contiguous() || !src_l.is_contiguous() {
            return Err(crate::Error::RequiresContiguous { op: "index_add" });
        }
        _int_add!(init, init_l, idx, idx_l, src, src_l, dim, launch_index_add, "index_add")
    }

    fn i_scatter_add(init: &CudaIntStorage, init_l: &Layout, idx: &CudaIntStorage, idx_l: &Layout, src: &CudaIntStorage, src_l: &Layout, dim: usize) -> Result<CudaIntStorage> {
        if !init_l.is_contiguous() || !idx_l.is_contiguous() || !src_l.is_contiguous() {
            return Err(crate::Error::RequiresContiguous { op: "scatter_add" });
        }
        _int_add!(init, init_l, idx, idx_l, src, src_l, dim, launch_scatter_add, "scatter_add")
    }

    fn i_cat(srcs: &[(&CudaIntStorage, &Layout)], dim: usize) -> Result<(CudaIntStorage, Shape)> {
        let layouts: Vec<&Layout> = srcs.iter().map(|(_, l)| *l).collect();
        let out_shape = super::cat_compute_shape(&layouts, dim)?;
        let device = &srcs[0].0.device;

        if dim == 0 {
            match &srcs[0].0.slice {
                CudaIntSlice::I32(_) => {
                    let mut out = device.alloc::<i32>(out_shape.element_count())?;
                    let mut offset = 0usize;
                    for (storage, layout) in srcs {
                        let CudaIntSlice::I32(data) = &storage.slice else { unreachable!() };
                        launch::launch_copy_offset(device, "ucopy_i32", &kernel::COPY, data, layout, &out, offset)?;
                        offset += layout.shape().element_count();
                    }
                    Ok((CudaIntStorage { slice: CudaIntSlice::I32(out), device: device.clone() }, out_shape))
                }
                CudaIntSlice::U32(_) => {
                    let mut out = device.alloc::<u32>(out_shape.element_count())?;
                    let mut offset = 0usize;
                    for (storage, layout) in srcs {
                        let CudaIntSlice::U32(data) = &storage.slice else { unreachable!() };
                        launch::launch_copy_offset(device, "ucopy_u32", &kernel::COPY, data, layout, &out, offset)?;
                        offset += layout.shape().element_count();
                    }
                    Ok((CudaIntStorage { slice: CudaIntSlice::U32(out), device: device.clone() }, out_shape))
                }
                CudaIntSlice::U8(_) => {
                    let mut out = device.alloc::<u8>(out_shape.element_count())?;
                    let mut offset = 0usize;
                    for (storage, layout) in srcs {
                        let CudaIntSlice::U8(data) = &storage.slice else { unreachable!() };
                        launch::launch_copy_offset(device, "ucopy_u8", &kernel::COPY, data, layout, &out, offset)?;
                        offset += layout.shape().element_count();
                    }
                    Ok((CudaIntStorage { slice: CudaIntSlice::U8(out), device: device.clone() }, out_shape))
                }
            }
        } else {
            let cat_size = out_shape.dims()[dim];
            let d1: usize = out_shape.dims()[..dim].iter().product();
            let block: usize = out_shape.dims()[dim + 1..].iter().product();
            let dst_s = block * cat_size;
            match &srcs[0].0.slice {
                CudaIntSlice::I32(_) => {
                    let mut out = device.alloc::<i32>(out_shape.element_count())?;
                    let mut saved: Vec<CudaSlice<i32>> = Vec::new();
                    for (storage, layout) in srcs {
                        let CudaIntSlice::I32(data) = &storage.slice else { unreachable!() };
                        let cat_dim_sz = layout.dims()[dim];
                        let d2 = block * cat_dim_sz;
                        if layout.is_contiguous() {
                            launch::launch_copy2d(device, "ucopy2d_i32", &kernel::COPY, d1, d2, d2, dst_s, data, layout.start_offset(), &out)?;
                        } else {
                            let contig = launch::launch_cast(device, "i32", "i32", &kernel::CAST, data, layout)?;
                            launch::launch_copy2d(device, "ucopy2d_i32", &kernel::COPY, d1, d2, d2, dst_s, &contig, 0, &out)?;
                            saved.push(contig);
                        }
                    }
                    Ok((CudaIntStorage { slice: CudaIntSlice::I32(out), device: device.clone() }, out_shape))
                }
                CudaIntSlice::U32(_) => {
                    let mut out = device.alloc::<u32>(out_shape.element_count())?;
                    let mut saved: Vec<CudaSlice<u32>> = Vec::new();
                    for (storage, layout) in srcs {
                        let CudaIntSlice::U32(data) = &storage.slice else { unreachable!() };
                        let cat_dim_sz = layout.dims()[dim];
                        let d2 = block * cat_dim_sz;
                        if layout.is_contiguous() {
                            launch::launch_copy2d(device, "ucopy2d_u32", &kernel::COPY, d1, d2, d2, dst_s, data, layout.start_offset(), &out)?;
                        } else {
                            let contig = launch::launch_cast(device, "u32", "u32", &kernel::CAST, data, layout)?;
                            launch::launch_copy2d(device, "ucopy2d_u32", &kernel::COPY, d1, d2, d2, dst_s, &contig, 0, &out)?;
                            saved.push(contig);
                        }
                    }
                    Ok((CudaIntStorage { slice: CudaIntSlice::U32(out), device: device.clone() }, out_shape))
                }
                CudaIntSlice::U8(_) => {
                    let mut out = device.alloc::<u8>(out_shape.element_count())?;
                    let mut saved: Vec<CudaSlice<u8>> = Vec::new();
                    for (storage, layout) in srcs {
                        let CudaIntSlice::U8(data) = &storage.slice else { unreachable!() };
                        let cat_dim_sz = layout.dims()[dim];
                        let d2 = block * cat_dim_sz;
                        if layout.is_contiguous() {
                            launch::launch_copy2d(device, "ucopy2d_u8", &kernel::COPY, d1, d2, d2, dst_s, data, layout.start_offset(), &out)?;
                        } else {
                            let contig = launch::launch_cast(device, "u8", "u8", &kernel::CAST, data, layout)?;
                            launch::launch_copy2d(device, "ucopy2d_u8", &kernel::COPY, d1, d2, d2, dst_s, &contig, 0, &out)?;
                            saved.push(contig);
                        }
                    }
                    Ok((CudaIntStorage { slice: CudaIntSlice::U8(out), device: device.clone() }, out_shape))
                }
            }
        }
    }
    
    fn i_pick(mask: &CudaBoolStorage, mask_l: &Layout, on_true: &CudaIntStorage, true_l: &Layout, on_false: &CudaIntStorage, false_l: &Layout) -> Result<CudaIntStorage> {
        let device = &mask.device;
        match (&on_true.slice, &on_false.slice) {
            (CudaIntSlice::I32(t), CudaIntSlice::I32(f)) => {
                let out = launch::launch_pick(device, "i32", &kernel::PICK, &mask.slice, mask_l, t, true_l, f, false_l)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(out), device: device.clone() })
            }
            (CudaIntSlice::U32(t), CudaIntSlice::U32(f)) => {
                let out = launch::launch_pick(device, "u32", &kernel::PICK, &mask.slice, mask_l, t, true_l, f, false_l)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(out), device: device.clone() })
            }
            (CudaIntSlice::U8(t), CudaIntSlice::U8(f)) => {
                let out = launch::launch_pick(device, "u8", &kernel::PICK, &mask.slice, mask_l, t, true_l, f, false_l)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(out), device: device.clone() })
            }
            _ => Err(crate::Error::DTypeMismatch { lhs: on_true.dtype(), rhs: on_false.dtype(), op: "pick" })
        }
    }

    fn i_pick_true(mask: &CudaBoolStorage, mask_l: &Layout, value: i64, on_false: &CudaIntStorage, false_l: &Layout) -> Result<CudaIntStorage> {
        let device = &mask.device;
        match &on_false.slice {
            CudaIntSlice::I32(f) => {
                let out = launch::launch_pick_true(device, "i32", &kernel::PICK, &mask.slice, mask_l, value as i32, f, false_l)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(out), device: device.clone() })
            }
            CudaIntSlice::U32(f) => {
                let out = launch::launch_pick_true(device, "u32", &kernel::PICK, &mask.slice, mask_l, value as u32, f, false_l)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(out), device: device.clone() })
            }
            CudaIntSlice::U8(f) => {
                let out = launch::launch_pick_true(device, "u8", &kernel::PICK, &mask.slice, mask_l, value as u8, f, false_l)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(out), device: device.clone() })
            }
        }
    }

    fn i_pick_false(mask: &CudaBoolStorage, mask_l: &Layout, on_true: &CudaIntStorage, true_l: &Layout, value: i64) -> Result<CudaIntStorage> {
        let device = &mask.device;
        match &on_true.slice {
            CudaIntSlice::I32(t) => {
                let out = launch::launch_pick_false(device, "i32", &kernel::PICK, &mask.slice, mask_l, t, true_l, value as i32)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(out), device: device.clone() })
            }
            CudaIntSlice::U32(t) => {
                let out = launch::launch_pick_false(device, "u32", &kernel::PICK, &mask.slice, mask_l, t, true_l, value as u32)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(out), device: device.clone() })
            }
            CudaIntSlice::U8(t) => {
                let out = launch::launch_pick_false(device, "u8", &kernel::PICK, &mask.slice, mask_l, t, true_l, value as u8)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(out), device: device.clone() })
            }
        }
    }

    fn i_allclose(a: &CudaIntStorage, a_l: &Layout, b: &CudaIntStorage, b_l: &Layout) -> Result<bool> {
        a.device.same_ordinal(&b.device, "allclose")?;
        match (&a.slice, &b.slice) {
            (CudaIntSlice::I32(ai), CudaIntSlice::I32(bi)) => {
                Ok(launch::launch_allclose_int(&a.device, "i32", ai, a_l, bi, b_l)?)
            }
            (CudaIntSlice::U32(ai), CudaIntSlice::U32(bi)) => {
                Ok(launch::launch_allclose_int(&a.device, "u32", ai, a_l, bi, b_l)?)
            }
            (CudaIntSlice::U8(ai), CudaIntSlice::U8(bi)) => {
                Ok(launch::launch_allclose_int(&a.device, "u8", ai, a_l, bi, b_l)?)
            }
            _ => Err(crate::Error::DTypeMismatch { lhs: a.dtype(), rhs: b.dtype(), op: "allclose" }),
        }
    }
}
