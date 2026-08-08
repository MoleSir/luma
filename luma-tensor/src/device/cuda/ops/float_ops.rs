use crate::{BinaryOp, CmpOp, FloatOps, Layout, ReduceOp, Result, Shape, UnaryOp, device::cuda::CudaFloatSlice, dtype::{BoolDType, FloatDType, IntDType}};
use crate::device::cuda::{Cuda, CudaIntStorage, CudaFloatStorage, CudaBoolStorage, CudaIntSlice, CudaError};
use cudarc::driver::{CudaSlice, PushKernelArg, LaunchConfig};
use super::super::kernel;
use super::super::launch;

// ---- indexing dispatch helpers ----

macro_rules! _float_select {
    ($x:ident, $x_l:ident, $idx:ident, $idx_l:ident, $dim:ident, $launch_fn:ident, $op:literal, $out_shape:expr) => {
        match (&$idx.slice, &$x.slice) {
            (CudaIntSlice::I32(ids), CudaFloatSlice::F32(v)) => {
                let out = launch::$launch_fn(&$x.device, "i32", "f32", &kernel::INDEXING, v, $x_l, ids, $idx_l, $dim)?;
                Ok((CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: $x.device.clone() }, $out_shape))
            }
            (CudaIntSlice::I32(ids), CudaFloatSlice::F64(v)) => {
                let out = launch::$launch_fn(&$x.device, "i32", "f64", &kernel::INDEXING, v, $x_l, ids, $idx_l, $dim)?;
                Ok((CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: $x.device.clone() }, $out_shape))
            }
            (CudaIntSlice::U32(ids), CudaFloatSlice::F32(v)) => {
                let out = launch::$launch_fn(&$x.device, "u32", "f32", &kernel::INDEXING, v, $x_l, ids, $idx_l, $dim)?;
                Ok((CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: $x.device.clone() }, $out_shape))
            }
            (CudaIntSlice::U32(ids), CudaFloatSlice::F64(v)) => {
                let out = launch::$launch_fn(&$x.device, "u32", "f64", &kernel::INDEXING, v, $x_l, ids, $idx_l, $dim)?;
                Ok((CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: $x.device.clone() }, $out_shape))
            }
            _ => Err(crate::Error::DTypeMismatch { lhs: $x.slice.dtype(), rhs: $idx.slice.dtype(), op: $op })
        }
    };
}

macro_rules! _float_add {
    ($init:ident, $init_l:ident, $idx:ident, $idx_l:ident, $src:ident, $src_l:ident, $dim:ident, $launch_fn:ident, $op:literal) => {
        match (&$idx.slice, &$src.slice, &$init.slice) {
            (CudaIntSlice::I32(ids), CudaFloatSlice::F32(s), CudaFloatSlice::F32(d)) => {
                let out = launch::$launch_fn(&$init.device, "i32", "f32", &kernel::INDEXING, d, $init_l, ids, $idx_l, s, $src_l, $dim)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: $init.device.clone() })
            }
            (CudaIntSlice::I32(ids), CudaFloatSlice::F64(s), CudaFloatSlice::F64(d)) => {
                let out = launch::$launch_fn(&$init.device, "i32", "f64", &kernel::INDEXING, d, $init_l, ids, $idx_l, s, $src_l, $dim)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: $init.device.clone() })
            }
            (CudaIntSlice::U32(ids), CudaFloatSlice::F32(s), CudaFloatSlice::F32(d)) => {
                let out = launch::$launch_fn(&$init.device, "u32", "f32", &kernel::INDEXING, d, $init_l, ids, $idx_l, s, $src_l, $dim)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: $init.device.clone() })
            }
            (CudaIntSlice::U32(ids), CudaFloatSlice::F64(s), CudaFloatSlice::F64(d)) => {
                let out = launch::$launch_fn(&$init.device, "u32", "f64", &kernel::INDEXING, d, $init_l, ids, $idx_l, s, $src_l, $dim)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: $init.device.clone() })
            }
            _ => Err(crate::Error::DTypeMismatch { lhs: $init.slice.dtype(), rhs: $src.slice.dtype(), op: $op })
        }
    };
}

impl FloatOps<Cuda> for Cuda {
    fn f_zeros(shape: &Shape, device: &Cuda, dtype: FloatDType) -> Result<CudaFloatStorage> {
        let elem_count = shape.element_count();
        match dtype {
            FloatDType::F32 => {
                let data = device.alloc_zeros::<f32>(elem_count)?;
                Ok(CudaFloatStorage { device: device.clone(), slice: CudaFloatSlice::F32(data) })
            }
            FloatDType::F64 => {
                let data = device.alloc_zeros::<f64>(elem_count)?;
                Ok(CudaFloatStorage { device: device.clone(), slice: CudaFloatSlice::F64(data) })
            }
        }
    }

    fn f_ones(shape: &Shape, device: &Cuda, dtype: FloatDType) -> Result<CudaFloatStorage> {
        let elem_count = shape.element_count();
        match dtype {
            FloatDType::F32 => {
                let host = vec![1.0f32; elem_count];
                let data = device.memcpy_stod(&host)?;
                Ok(CudaFloatStorage { device: device.clone(), slice: CudaFloatSlice::F32(data) })
            }
            FloatDType::F64 => {
                let host = vec![1.0f64; elem_count];
                let data = device.memcpy_stod(&host)?;
                Ok(CudaFloatStorage { device: device.clone(), slice: CudaFloatSlice::F64(data) })
            }
        }
    }

    fn f_full(shape: &Shape, value: f64, device: &Cuda, dtype: FloatDType) -> Result<CudaFloatStorage> {
        let elem_count = shape.element_count();
        match dtype {
            FloatDType::F32 => {
                let host = vec![value as f32; elem_count];
                let data = device.memcpy_stod(&host)?;
                Ok(CudaFloatStorage { device: device.clone(), slice: CudaFloatSlice::F32(data) })
            }
            FloatDType::F64 => {
                let host = vec![value; elem_count];
                let data = device.memcpy_stod(&host)?;
                Ok(CudaFloatStorage { device: device.clone(), slice: CudaFloatSlice::F64(data) })
            }
        }
    }

    fn f_from_f64(data: &[f64], dtype: FloatDType) -> Result<CudaFloatStorage> {
        let device = Cuda::default();
        match dtype {
            FloatDType::F32 => {
                let host: Vec<f32> = data.iter().map(|&v| v as f32).collect();
                let slice = device.memcpy_stod(&host)?;
                Ok(CudaFloatStorage { device, slice: CudaFloatSlice::F32(slice) })
            }
            FloatDType::F64 => {
                let slice = device.memcpy_stod(data)?;
                Ok(CudaFloatStorage { device, slice: CudaFloatSlice::F64(slice) })
            }
        }
    }

    fn f_rand_uniform(shape: &Shape, lo: f64, hi: f64, device: &Cuda, dtype: FloatDType) -> Result<CudaFloatStorage> {
        let elem_count = shape.element_count();
        let curand = device.0.curand.lock().unwrap();
        let contig = Layout::contiguous(shape.clone());
        match dtype {
            FloatDType::F32 => {
                let mut data = device.alloc::<f32>(elem_count)?;
                curand.fill_with_uniform(&mut data).map_err(CudaError::Curand)?;
                let out = launch::launch_affine(device, "f32", &kernel::UNARY, &data, &contig, (hi - lo) as f32, lo as f32)?;
                Ok(CudaFloatStorage { device: device.clone(), slice: CudaFloatSlice::F32(out) })
            }
            FloatDType::F64 => {
                let mut data = device.alloc::<f64>(elem_count)?;
                curand.fill_with_uniform(&mut data).map_err(CudaError::Curand)?;
                let out = launch::launch_affine(device, "f64", &kernel::UNARY, &data, &contig, hi - lo, lo)?;
                Ok(CudaFloatStorage { device: device.clone(), slice: CudaFloatSlice::F64(out) })
            }
        }
    }

    fn f_rand_normal(shape: &Shape, mean: f64, std: f64, device: &Cuda, dtype: FloatDType) -> Result<CudaFloatStorage> {
        let elem_count = shape.element_count();
        let curand = device.0.curand.lock().unwrap();
        match dtype {
            FloatDType::F32 => {
                let mut data = device.alloc::<f32>(elem_count)?;
                curand.fill_with_normal(&mut data, mean as f32, std as f32).map_err(CudaError::Curand)?;
                Ok(CudaFloatStorage { device: device.clone(), slice: CudaFloatSlice::F32(data) })
            }
            FloatDType::F64 => {
                let mut data = device.alloc::<f64>(elem_count)?;
                curand.fill_with_normal(&mut data, mean, std).map_err(CudaError::Curand)?;
                Ok(CudaFloatStorage { device: device.clone(), slice: CudaFloatSlice::F64(data) })
            }
        }
    }

    fn f_contiguous(x: &CudaFloatStorage, layout: &Layout) -> Result<CudaFloatStorage> {
        match &x.slice {
            CudaFloatSlice::F32(data) => {
                let out = launch::launch_cast(&x.device, "f32", "f32", &kernel::CAST, data, layout)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: x.device.clone() })
            }
            CudaFloatSlice::F64(data) => {
                let out = launch::launch_cast(&x.device, "f64", "f64", &kernel::CAST, data, layout)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: x.device.clone() })
            }
        }
    }

    fn f_cast_float(x: &CudaFloatStorage, layout: &Layout, to: FloatDType) -> Result<CudaFloatStorage> {
        let device = &x.device;
        match (&x.slice, to) {
            (CudaFloatSlice::F32(data), FloatDType::F32) => {
                let out = launch::launch_cast(device, "f32", "f32", &kernel::CAST, data, layout)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: device.clone() })
            }
            (CudaFloatSlice::F32(data), FloatDType::F64) => {
                let out = launch::launch_cast(device, "f32", "f64", &kernel::CAST, data, layout)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: device.clone() })
            }
            (CudaFloatSlice::F64(data), FloatDType::F32) => {
                let out = launch::launch_cast(device, "f64", "f32", &kernel::CAST, data, layout)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: device.clone() })
            }
            (CudaFloatSlice::F64(data), FloatDType::F64) => {
                let out = launch::launch_cast(device, "f64", "f64", &kernel::CAST, data, layout)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: device.clone() })
            }
        }
    }

    fn f_cast_int(x: &CudaFloatStorage, layout: &Layout, to: IntDType) -> Result<CudaIntStorage> {
        let device = &x.device;
        match (&x.slice, to) {
            (CudaFloatSlice::F32(data), IntDType::I32) => {
                let out = launch::launch_cast(device, "f32", "i32", &kernel::CAST, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(out), device: device.clone() })
            }
            (CudaFloatSlice::F32(data), IntDType::U32) => {
                let out = launch::launch_cast(device, "f32", "u32", &kernel::CAST, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(out), device: device.clone() })
            }
            (CudaFloatSlice::F32(data), IntDType::U8) => {
                let out = launch::launch_cast(device, "f32", "u8", &kernel::CAST, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(out), device: device.clone() })
            }
            (CudaFloatSlice::F64(data), IntDType::I32) => {
                let out = launch::launch_cast(device, "f64", "i32", &kernel::CAST, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(out), device: device.clone() })
            }
            (CudaFloatSlice::F64(data), IntDType::U32) => {
                let out = launch::launch_cast(device, "f64", "u32", &kernel::CAST, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(out), device: device.clone() })
            }
            (CudaFloatSlice::F64(data), IntDType::U8) => {
                let out = launch::launch_cast(device, "f64", "u8", &kernel::CAST, data, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(out), device: device.clone() })
            }
        }
    }

    fn f_cast_bool(x: &CudaFloatStorage, layout: &Layout, _to: BoolDType) -> Result<CudaBoolStorage> {
        let device = &x.device;
        match &x.slice {
            CudaFloatSlice::F32(data) => {
                let out = launch::launch_cast(device, "f32", "bool", &kernel::CAST, data, layout)?;
                Ok(CudaBoolStorage { slice: out, device: device.clone() })
            }
            CudaFloatSlice::F64(data) => {
                let out = launch::launch_cast(device, "f64", "bool", &kernel::CAST, data, layout)?;
                Ok(CudaBoolStorage { slice: out, device: device.clone() })
            }
        }
    }

    fn f_to_vec(x: &CudaFloatStorage, layout: &Layout) -> Result<Vec<f64>> {
        match &x.slice {
            CudaFloatSlice::F32(data) => {
                let raw = x.device.memcpy_dtov(data)?;
                Ok(layout.storage_indices().map(|i| raw[i] as f64).collect())
            }
            CudaFloatSlice::F64(data) => {
                let raw = x.device.memcpy_dtov(data)?;
                Ok(layout.storage_indices().map(|i| raw[i]).collect())
            }
        }
    }

    fn f_binary(lhs: &CudaFloatStorage, lhs_l: &Layout, rhs: &CudaFloatStorage, rhs_l: &Layout, op: BinaryOp) -> Result<CudaFloatStorage> {
        lhs.device.same_ordinal(&rhs.device, format!("Float {:?}", op))?;
        match (&lhs.slice, &rhs.slice) {
            (CudaFloatSlice::F32(l), CudaFloatSlice::F32(r)) => {
                let out = launch::launch_binary(&lhs.device, op, "f32", &kernel::BINARY, l, r, lhs_l, rhs_l)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: lhs.device.clone() })
            }
            (CudaFloatSlice::F64(l), CudaFloatSlice::F64(r)) => {
                let out = launch::launch_binary(&lhs.device, op, "f64", &kernel::BINARY, l, r, lhs_l, rhs_l)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: lhs.device.clone() })
            }
            _ => Err(crate::Error::DTypeMismatch { lhs: lhs.dtype(), rhs: rhs.dtype(), op: "binary" })
        }
    }

    fn f_binary_scalar(lhs: &CudaFloatStorage, lhs_l: &Layout, rhs: f64, op: BinaryOp) -> Result<CudaFloatStorage> {
        match &lhs.slice {
            CudaFloatSlice::F32(data) => {
                let out = launch::launch_binary_scalar(&lhs.device, op, "f32", &kernel::BINARY_SCALAR, data, lhs_l, rhs as f32)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: lhs.device.clone() })
            }
            CudaFloatSlice::F64(data) => {
                let out = launch::launch_binary_scalar(&lhs.device, op, "f64", &kernel::BINARY_SCALAR, data, lhs_l, rhs)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: lhs.device.clone() })
            }
        }
    }

    fn f_unary(x: &CudaFloatStorage, layout: &Layout, op: UnaryOp) -> Result<CudaFloatStorage> {
        let device = &x.device;
        match &x.slice {
            CudaFloatSlice::F32(data) => match op {
                UnaryOp::LeakyRelu(a) => {
                    let out = launch::launch_unary_param1(device, op, "f32", &kernel::UNARY, data, layout, a as f32)?;
                    Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: device.clone() })
                }
                _ => {
                    let out = launch::launch_unary(device, op, "f32", &kernel::UNARY, data, layout)?;
                    Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: device.clone() })
                }
            },
            CudaFloatSlice::F64(data) => match op {
                UnaryOp::LeakyRelu(a) => {
                    let out = launch::launch_unary_param1(device, op, "f64", &kernel::UNARY, data, layout, a)?;
                    Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: device.clone() })
                }
                _ => {
                    let out = launch::launch_unary(device, op, "f64", &kernel::UNARY, data, layout)?;
                    Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: device.clone() })
                }
            },
        }
    }

    fn f_neg(x: &CudaFloatStorage, layout: &Layout) -> Result<CudaFloatStorage> {
        match &x.slice {
            CudaFloatSlice::F32(data) => {
                let out = launch::launch_unary_raw_by_kernel_name(&x.device, &format!("uneg_f32"), &kernel::UNARY, data, layout)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: x.device.clone() })
            }
            CudaFloatSlice::F64(data) => {
                let out = launch::launch_unary_raw_by_kernel_name(&x.device, &format!("uneg_f64"), &kernel::UNARY, data, layout)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: x.device.clone() })
            }
        }
    }

    fn f_abs(x: &CudaFloatStorage, layout: &Layout) -> Result<CudaFloatStorage> {
        match &x.slice {
            CudaFloatSlice::F32(data) => {
                let out = launch::launch_unary_raw_by_kernel_name(&x.device, &format!("uabs_f32"), &kernel::UNARY, data, layout)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: x.device.clone() })
            }
            CudaFloatSlice::F64(data) => {
                let out = launch::launch_unary_raw_by_kernel_name(&x.device, &format!("uabs_f64"), &kernel::UNARY, data, layout)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: x.device.clone() })
            }
        }
    }

    fn f_sign(x: &CudaFloatStorage, layout: &Layout) -> Result<CudaFloatStorage> {
        match &x.slice {
            CudaFloatSlice::F32(data) => {
                let out = launch::launch_unary_raw_by_kernel_name(&x.device, &format!("usign_f32"), &kernel::UNARY, data, layout)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: x.device.clone() })
            }
            CudaFloatSlice::F64(data) => {
                let out = launch::launch_unary_raw_by_kernel_name(&x.device, &format!("usign_f64"), &kernel::UNARY, data, layout)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: x.device.clone() })
            }
        }
    }

    fn f_affine(x: &CudaFloatStorage, layout: &Layout, mul: f64, add: f64) -> Result<CudaFloatStorage> {
        let device = &x.device;
        match &x.slice {
            CudaFloatSlice::F32(data) => {
                let out = launch::launch_affine(device, "f32", &kernel::UNARY, data, layout, mul as f32, add as f32)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: device.clone() })
            }
            CudaFloatSlice::F64(data) => {
                let out = launch::launch_affine(device, "f64", &kernel::UNARY, data, layout, mul, add)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: device.clone() })
            }
        }
    }

    fn f_pow(x: &CudaFloatStorage, layout: &Layout, exp: f64) -> Result<CudaFloatStorage> {
        let device = &x.device;
        match &x.slice {
            CudaFloatSlice::F32(data) => {
                let out = launch::launch_pow(device, "f32", &kernel::UNARY, data, layout, exp as f32)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: device.clone() })
            }
            CudaFloatSlice::F64(data) => {
                let out = launch::launch_pow(device, "f64", &kernel::UNARY, data, layout, exp)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: device.clone() })
            }
        }
    }

    fn f_clamp(x: &CudaFloatStorage, layout: &Layout, min: Option<f64>, max: Option<f64>) -> Result<CudaFloatStorage> {
        let device = &x.device;
        match &x.slice {
            CudaFloatSlice::F32(data) => {
                let (has_min, min_val) = min.map_or((false, 0.0f32), |v| (true, v as f32));
                let (has_max, max_val) = max.map_or((false, 0.0f32), |v| (true, v as f32));
                let out = launch::launch_clamp(device, "f32", &kernel::UNARY, data, layout, has_min, min_val, has_max, max_val)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: device.clone() })
            }
            CudaFloatSlice::F64(data) => {
                let (has_min, min_val) = min.map_or((false, 0.0f64), |v| (true, v));
                let (has_max, max_val) = max.map_or((false, 0.0f64), |v| (true, v));
                let out = launch::launch_clamp(device, "f64", &kernel::UNARY, data, layout, has_min, min_val, has_max, max_val)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: device.clone() })
            }
        }
    }

    fn f_cmp(lhs: &CudaFloatStorage, lhs_l: &Layout, rhs: &CudaFloatStorage, rhs_l: &Layout, op: CmpOp) -> Result<CudaBoolStorage> {
        lhs.device.same_ordinal(&rhs.device, format!("Cmp {:?}", op))?;
        match (&lhs.slice, &rhs.slice) {
            (CudaFloatSlice::F32(l), CudaFloatSlice::F32(r)) => {
                let out = launch::launch_cmp(&lhs.device, op, "f32", &kernel::BINARY, l, r, lhs_l, rhs_l)?;
                Ok(CudaBoolStorage { slice: out, device: lhs.device.clone() })
            }
            (CudaFloatSlice::F64(l), CudaFloatSlice::F64(r)) => {
                let out = launch::launch_cmp(&lhs.device, op, "f64", &kernel::BINARY, l, r, lhs_l, rhs_l)?;
                Ok(CudaBoolStorage { slice: out, device: lhs.device.clone() })
            }
            _ => Err(crate::Error::DTypeMismatch { lhs: lhs.dtype(), rhs: rhs.dtype(), op: "cmp" })
        }
    }

    fn f_reduce(x: &CudaFloatStorage, layout: &Layout, dims: &[usize], keepdim: bool, op: ReduceOp) -> Result<(CudaFloatStorage, Shape)> {
        let device = &x.device;
        let kernel_op = match op {
            ReduceOp::Mean => ReduceOp::Sum,
            other => other,
        };
        let is_mean = matches!(op, ReduceOp::Mean);
        let total_factor: f64 = dims.iter().map(|&d| layout.dims()[d] as f64).product();

        match &x.slice {
            CudaFloatSlice::F32(data) => {
                let (out, shape) = launch::launch_multi_reduce::<f32>(device, kernel_op, "f32", &kernel::REDUCE, data, layout, dims, keepdim)?;
                let final_out = if is_mean {
                    let aff_layout = Layout::contiguous(shape.clone());
                    let mul = (1.0 / total_factor) as f32;
                    launch::launch_affine(device, "f32", &kernel::UNARY, &out, &aff_layout, mul, 0.0f32)?
                } else {
                    out
                };
                Ok((CudaFloatStorage { slice: CudaFloatSlice::F32(final_out), device: device.clone() }, shape))
            }
            CudaFloatSlice::F64(data) => {
                let (out, shape) = launch::launch_multi_reduce::<f64>(device, kernel_op, "f64", &kernel::REDUCE, data, layout, dims, keepdim)?;
                let final_out = if is_mean {
                    let aff_layout = Layout::contiguous(shape.clone());
                    let mul = 1.0 / total_factor;
                    launch::launch_affine(device, "f64", &kernel::UNARY, &out, &aff_layout, mul, 0.0f64)?
                } else {
                    out
                };
                Ok((CudaFloatStorage { slice: CudaFloatSlice::F64(final_out), device: device.clone() }, shape))
            }
        }
    }

    fn f_arg_reduce(x: &CudaFloatStorage, layout: &Layout, dim: usize, keepdim: bool, take_max: bool) -> Result<(CudaIntStorage, Shape)> {
        let dims = layout.dims().to_vec();
        let strides = layout.stride().to_vec();
        let reduce_size = dims[dim];
        let output_block_count: usize = dims.iter().enumerate()
            .filter(|&(i, _)| i != dim)
            .map(|(_, d)| *d)
            .product::<usize>()
            .max(1);

        let indices = match &x.slice {
            CudaFloatSlice::F32(s) => {
                let kn = launch::arg_reduce_kernel_name(take_max, "f32");
                launch::launch_arg_reduce(&x.device, &kn, &kernel::REDUCE,
                    s, &dims, &strides, dim, reduce_size, output_block_count)?
            }
            CudaFloatSlice::F64(s) => {
                let kn = launch::arg_reduce_kernel_name(take_max, "f64");
                launch::launch_arg_reduce(&x.device, &kn, &kernel::REDUCE,
                    s, &dims, &strides, dim, reduce_size, output_block_count)?
            }
        };

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

    fn f_matmul(lhs: &CudaFloatStorage, lhs_l: &Layout, rhs: &CudaFloatStorage, rhs_l: &Layout) -> Result<(CudaFloatStorage, Shape)> {
        lhs.device.same_ordinal(&rhs.device, "matmul")?;
        let lhs_dims = lhs_l.dims();
        let rhs_dims = rhs_l.dims();
        if lhs_dims.len() < 2 || rhs_dims.len() < 2 {
            return Err(crate::Error::UnexpectedNumberOfDims {
                expected: 2, got: lhs_dims.len().min(rhs_dims.len()),
                shape: lhs_l.shape().clone(),
            });
        }

        let m = lhs_dims[lhs_dims.len() - 2];
        let k = lhs_dims[lhs_dims.len() - 1];
        let n = rhs_dims[rhs_dims.len() - 1];

        let lhs_batch: Vec<usize> = lhs_dims[..lhs_dims.len() - 2].to_vec();
        let rhs_batch: Vec<usize> = rhs_dims[..rhs_dims.len() - 2].to_vec();
        let b: usize = lhs_batch.iter().chain(&rhs_batch).product::<usize>().max(1);

        let mut out_shape = if lhs_batch.len() >= rhs_batch.len() {
            lhs_batch.clone()
        } else {
            rhs_batch.clone()
        };
        out_shape.push(m);
        out_shape.push(n);
        let out_shape = Shape::from(out_shape);

        match (&lhs.slice, &rhs.slice) {
            (CudaFloatSlice::F32(l), CudaFloatSlice::F32(r)) => {
                let out = launch::launch_matmul(&lhs.device, 1.0f32, 0.0f32, (b, m, n, k), l, lhs_l, r, rhs_l)?;
                Ok((CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: lhs.device.clone() }, out_shape))
            }
            (CudaFloatSlice::F64(l), CudaFloatSlice::F64(r)) => {
                let out = launch::launch_matmul(&lhs.device, 1.0f64, 0.0f64, (b, m, n, k), l, lhs_l, r, rhs_l)?;
                Ok((CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: lhs.device.clone() }, out_shape))
            }
            _ => Err(crate::Error::DTypeMismatch { lhs: lhs.dtype(), rhs: rhs.dtype(), op: "matmul" }),
        }
    }

    fn f_add_matmul_(dst: &mut CudaFloatStorage, _dst_l: &Layout, lhs: &CudaFloatStorage, lhs_l: &Layout, rhs: &CudaFloatStorage, rhs_l: &Layout) -> Result<()> {
        lhs.device.same_ordinal(&rhs.device, "add_matmul")?;
        lhs.device.same_ordinal(&dst.device, "add_matmul")?;
        let lhs_dims = lhs_l.dims();
        let rhs_dims = rhs_l.dims();
        if lhs_dims.len() < 2 || rhs_dims.len() < 2 {
            return Err(crate::Error::UnexpectedNumberOfDims {
                expected: 2, got: lhs_dims.len().min(rhs_dims.len()),
                shape: lhs_l.shape().clone(),
            });
        }

        let m = lhs_dims[lhs_dims.len() - 2];
        let k = lhs_dims[lhs_dims.len() - 1];
        let n = rhs_dims[rhs_dims.len() - 1];

        let lhs_batch: Vec<usize> = lhs_dims[..lhs_dims.len() - 2].to_vec();
        let rhs_batch: Vec<usize> = rhs_dims[..rhs_dims.len() - 2].to_vec();
        let b: usize = lhs_batch.iter().chain(&rhs_batch).product::<usize>().max(1);

        match (&mut dst.slice, &lhs.slice, &rhs.slice) {
            (CudaFloatSlice::F32(d), CudaFloatSlice::F32(l), CudaFloatSlice::F32(r)) => {
                launch::launch_add_matmul_(&lhs.device, 1.0f32, 1.0f32, d, _dst_l, l, lhs_l, r, rhs_l, (b, m, n, k))?;
            }
            (CudaFloatSlice::F64(d), CudaFloatSlice::F64(l), CudaFloatSlice::F64(r)) => {
                launch::launch_add_matmul_(&lhs.device, 1.0f64, 1.0f64, d, _dst_l, l, lhs_l, r, rhs_l, (b, m, n, k))?;
            }
            _ => return Err(crate::Error::DTypeMismatch { lhs: dst.dtype(), rhs: lhs.dtype(), op: "add_matmul" }),
        }
        Ok(())
    }

    fn f_binary_(dst: &mut CudaFloatStorage, dst_l: &Layout, src: &CudaFloatStorage, src_l: &Layout, op: BinaryOp) -> Result<()> {
        dst.device.same_ordinal(&src.device, format!("Cmp {:?}", op))?;
        match (&dst.slice, &src.slice) {
            (CudaFloatSlice::F32(d), CudaFloatSlice::F32(s)) => {
                launch::launch_binary_inplace(&dst.device, op, "f32", &kernel::BINARY, d, s, dst_l, src_l)?;
                Ok(())
            }
            (CudaFloatSlice::F64(d), CudaFloatSlice::F64(s)) => {
                launch::launch_binary_inplace(&dst.device, op, "f64", &kernel::BINARY, d, s, dst_l, src_l)?;
                Ok(())
            }
            _ => Err(crate::Error::DTypeMismatch { lhs: dst.dtype(), rhs: src.dtype(), op: "binary_inplace" })
        }
    }

    fn f_index_select(x: &CudaFloatStorage, x_l: &Layout, idx: &CudaIntStorage, idx_l: &Layout, dim: usize) -> Result<(CudaFloatStorage, Shape)> {
        if !x_l.is_contiguous() || !idx_l.is_contiguous() {
            return Err(crate::Error::RequiresContiguous { op: "index_select" });
        }
        let mut out_dims = x_l.dims().to_vec();
        out_dims[dim] = idx_l.dims()[0];
        let out_shape = Shape::from(out_dims);
        _float_select!(x, x_l, idx, idx_l, dim, launch_index_select, "index_select", out_shape)
    }

    fn f_gather(x: &CudaFloatStorage, x_l: &Layout, idx: &CudaIntStorage, idx_l: &Layout, dim: usize) -> Result<(CudaFloatStorage, Shape)> {
        if !x_l.is_contiguous() || !idx_l.is_contiguous() {
            return Err(crate::Error::RequiresContiguous { op: "gather" });
        }
        let out_shape = Shape::from(idx_l.dims().to_vec());
        _float_select!(x, x_l, idx, idx_l, dim, launch_gather, "gather", out_shape)
    }

    fn f_index_add(init: &CudaFloatStorage, init_l: &Layout, idx: &CudaIntStorage, idx_l: &Layout, src: &CudaFloatStorage, src_l: &Layout, dim: usize) -> Result<CudaFloatStorage> {
        if !init_l.is_contiguous() || !idx_l.is_contiguous() || !src_l.is_contiguous() {
            return Err(crate::Error::RequiresContiguous { op: "index_add" });
        }
        _float_add!(init, init_l, idx, idx_l, src, src_l, dim, launch_index_add, "index_add")
    }

    fn f_scatter_add(init: &CudaFloatStorage, init_l: &Layout, idx: &CudaIntStorage, idx_l: &Layout, src: &CudaFloatStorage, src_l: &Layout, dim: usize) -> Result<CudaFloatStorage> {
        if !init_l.is_contiguous() || !idx_l.is_contiguous() || !src_l.is_contiguous() {
            return Err(crate::Error::RequiresContiguous { op: "scatter_add" });
        }
        _float_add!(init, init_l, idx, idx_l, src, src_l, dim, launch_scatter_add, "scatter_add")
    }

    fn f_cat(srcs: &[(&CudaFloatStorage, &Layout)], dim: usize) -> Result<(CudaFloatStorage, Shape)> {
        let layouts: Vec<&Layout> = srcs.iter().map(|(_, l)| *l).collect();
        let out_shape = super::cat_compute_shape(&layouts, dim)?;
        let device = &srcs[0].0.device;

        if dim == 0 {
            match &srcs[0].0.slice {
                CudaFloatSlice::F32(_) => {
                    let mut out = device.alloc::<f32>(out_shape.element_count())?;
                    let mut offset = 0usize;
                    for (storage, layout) in srcs {
                        let CudaFloatSlice::F32(data) = &storage.slice else { unreachable!() };
                        launch::launch_copy_offset(device, "ucopy_f32", &kernel::COPY, data, layout, &out, offset)?;
                        offset += layout.shape().element_count();
                    }
                    Ok((CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: device.clone() }, out_shape))
                }
                CudaFloatSlice::F64(_) => {
                    let mut out = device.alloc::<f64>(out_shape.element_count())?;
                    let mut offset = 0usize;
                    for (storage, layout) in srcs {
                        let CudaFloatSlice::F64(data) = &storage.slice else { unreachable!() };
                        launch::launch_copy_offset(device, "ucopy_f64", &kernel::COPY, data, layout, &out, offset)?;
                        offset += layout.shape().element_count();
                    }
                    Ok((CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: device.clone() }, out_shape))
                }
            }
        } else {
            match &srcs[0].0.slice {
                CudaFloatSlice::F32(_) => {
                    let cat_size = out_shape.dims()[dim];
                    let d1: usize = out_shape.dims()[..dim].iter().product();
                    let block: usize = out_shape.dims()[dim + 1..].iter().product();
                    let dst_s = block * cat_size;
                    let mut out = device.alloc::<f32>(out_shape.element_count())?;
                    let mut saved: Vec<CudaSlice<f32>> = Vec::new();
                    for (storage, layout) in srcs {
                        let CudaFloatSlice::F32(data) = &storage.slice else { unreachable!() };
                        let cat_dim_sz = layout.dims()[dim];
                        let d2 = block * cat_dim_sz;
                        if layout.is_contiguous() {
                            launch::launch_copy2d(device, "ucopy2d_f32", &kernel::COPY, d1, d2, d2, dst_s, data, layout.start_offset(), &out)?;
                        } else {
                            let contig = launch::launch_cast(device, "f32", "f32", &kernel::CAST, data, layout)?;
                            launch::launch_copy2d(device, "ucopy2d_f32", &kernel::COPY, d1, d2, d2, dst_s, &contig, 0, &out)?;
                            saved.push(contig);
                        }
                    }
                    Ok((CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: device.clone() }, out_shape))
                }
                CudaFloatSlice::F64(_) => {
                    let cat_size = out_shape.dims()[dim];
                    let d1: usize = out_shape.dims()[..dim].iter().product();
                    let block: usize = out_shape.dims()[dim + 1..].iter().product();
                    let dst_s = block * cat_size;
                    let mut out = device.alloc::<f64>(out_shape.element_count())?;
                    let mut saved: Vec<CudaSlice<f64>> = Vec::new();
                    for (storage, layout) in srcs {
                        let CudaFloatSlice::F64(data) = &storage.slice else { unreachable!() };
                        let cat_dim_sz = layout.dims()[dim];
                        let d2 = block * cat_dim_sz;
                        if layout.is_contiguous() {
                            launch::launch_copy2d(device, "ucopy2d_f64", &kernel::COPY, d1, d2, d2, dst_s, data, layout.start_offset(), &out)?;
                        } else {
                            let contig = launch::launch_cast(device, "f64", "f64", &kernel::CAST, data, layout)?;
                            launch::launch_copy2d(device, "ucopy2d_f64", &kernel::COPY, d1, d2, d2, dst_s, &contig, 0, &out)?;
                            saved.push(contig);
                        }
                    }
                    Ok((CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: device.clone() }, out_shape))
                }
            }
        }
    }

    fn f_softmax(x: &CudaFloatStorage, layout: &Layout, dim: usize) -> Result<CudaFloatStorage> {
        if !layout.is_contiguous() {
            return Err(crate::Error::RequiresContiguous { op: "softmax" });
        }
        let last_dim = layout.dims().len() - 1;
        if dim != last_dim {
            let data = Cuda::f_to_vec(x, layout)?;
            let dims = layout.dims();
            let reduce_size = dims[dim];
            let outer = dims[..dim].iter().product::<usize>();
            let inner = dims[dim + 1..].iter().product::<usize>();
            let mut out = vec![0f64; data.len()];
            for o in 0..outer {
                for k in 0..inner {
                    let mut row = vec![0f64; reduce_size];
                    for r in 0..reduce_size {
                        row[r] = data[o * reduce_size * inner + r * inner + k];
                    }
                    let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exp_sum: f64 = row.iter().map(|&v| (v - max_val).exp()).sum();
                    for r in 0..reduce_size {
                        out[o * reduce_size * inner + r * inner + k] =
                            ((row[r] - max_val).exp()) / exp_sum;
                    }
                }
            }
            return Cuda::f_from_f64(&out, x.dtype().as_float());
        }
        let device = &x.device;
        match &x.slice {
            CudaFloatSlice::F32(data) => {
                let slice = launch::launch_softmax_f32(device, data, layout, dim)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(slice), device: device.clone() })
            }
            CudaFloatSlice::F64(data) => {
                let slice = launch::launch_softmax_f64(device, data, layout, dim)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(slice), device: device.clone() })
            }
        }
    }

    fn f_rms_norm(x: &CudaFloatStorage, x_l: &Layout, weight: &CudaFloatStorage, weight_l: &Layout, eps: f64) -> Result<CudaFloatStorage> {
        if !x_l.is_contiguous() || !weight_l.is_contiguous() {
            return Err(crate::Error::RequiresContiguous { op: "rms_norm" });
        }
        let device = &x.device;
        match (&x.slice, &weight.slice) {
            (CudaFloatSlice::F32(xs), CudaFloatSlice::F32(ws)) => {
                let slice = launch::launch_rms_norm_f32(device, xs, ws, x_l, weight_l, eps as f32)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(slice), device: device.clone() })
            }
            (CudaFloatSlice::F64(xs), CudaFloatSlice::F64(ws)) => {
                let slice = launch::launch_rms_norm_f64(device, xs, ws, x_l, weight_l, eps)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(slice), device: device.clone() })
            }
            _ => Err(crate::Error::DTypeMismatch { lhs: x.dtype(), rhs: weight.dtype(), op: "rms_norm" }),
        }
    }

    fn f_pick(mask: &CudaBoolStorage, mask_l: &Layout, on_true: &CudaFloatStorage, true_l: &Layout, on_false: &CudaFloatStorage, false_l: &Layout) -> Result<CudaFloatStorage> {
        let device = &mask.device;
        match (&on_true.slice, &on_false.slice) {
            (CudaFloatSlice::F32(t), CudaFloatSlice::F32(f)) => {
                let out = launch::launch_pick(device, "f32", &kernel::PICK, &mask.slice, mask_l, t, true_l, f, false_l)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: device.clone() })
            }
            (CudaFloatSlice::F64(t), CudaFloatSlice::F64(f)) => {
                let out = launch::launch_pick(device, "f64", &kernel::PICK, &mask.slice, mask_l, t, true_l, f, false_l)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: device.clone() })
            }
            _ => Err(crate::Error::DTypeMismatch { lhs: on_true.dtype(), rhs: on_false.dtype(), op: "pick" })
        }
    }

    fn f_pick_true(mask: &CudaBoolStorage, mask_l: &Layout, value: f64, on_false: &CudaFloatStorage, false_l: &Layout) -> Result<CudaFloatStorage> {
        let device = &mask.device;
        match &on_false.slice {
            CudaFloatSlice::F32(f) => {
                let out = launch::launch_pick_true(device, "f32", &kernel::PICK, &mask.slice, mask_l, value as f32, f, false_l)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: device.clone() })
            }
            CudaFloatSlice::F64(f) => {
                let out = launch::launch_pick_true(device, "f64", &kernel::PICK, &mask.slice, mask_l, value, f, false_l)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: device.clone() })
            }
        }
    }

    fn f_pick_false(mask: &CudaBoolStorage, mask_l: &Layout, on_true: &CudaFloatStorage, true_l: &Layout, value: f64) -> Result<CudaFloatStorage> {
        let device = &mask.device;
        match &on_true.slice {
            CudaFloatSlice::F32(t) => {
                let out = launch::launch_pick_false(device, "f32", &kernel::PICK, &mask.slice, mask_l, t, true_l, value as f32)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: device.clone() })
            }
            CudaFloatSlice::F64(t) => {
                let out = launch::launch_pick_false(device, "f64", &kernel::PICK, &mask.slice, mask_l, t, true_l, value)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: device.clone() })
            }
        }
    }

    fn f_allclose(a: &CudaFloatStorage, a_l: &Layout, b: &CudaFloatStorage, b_l: &Layout, rtol: f64, atol: f64) -> Result<bool> {
        a.device.same_ordinal(&b.device, "allclose")?;
        match (&a.slice, &b.slice) {
            (CudaFloatSlice::F32(ai), CudaFloatSlice::F32(bi)) => {
                Ok(launch::launch_allclose_float(&a.device, "f32", ai, a_l, bi, b_l, rtol as f32, atol as f32)?)
            }
            (CudaFloatSlice::F64(ai), CudaFloatSlice::F64(bi)) => {
                Ok(launch::launch_allclose_float(&a.device, "f64", ai, a_l, bi, b_l, rtol, atol)?)
            }
            _ => Err(crate::Error::DTypeMismatch { lhs: a.dtype(), rhs: b.dtype(), op: "allclose" }),
        }
    }
}
