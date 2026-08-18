use std::borrow::Cow;

use super::super::kernel;
use super::super::launch;
use crate::device::cuda::{Cuda, CudaBoolStorage, CudaFloatSlice, CudaFloatStorage, CudaIntSlice, CudaIntStorage};
use crate::{
    BoolOps, Layout, Result, Shape,
    dtype::{BoolDType, FloatDType, IntDType},
};
use cudarc::driver::CudaSlice;

impl BoolOps<Cuda> for Cuda {
    fn b_falses(shape: &Shape, device: &Cuda, _dtype: BoolDType) -> Result<CudaBoolStorage> {
        let elem_count = shape.element_count();
        let data = device.alloc_zeros::<u8>(elem_count)?;
        Ok(CudaBoolStorage { slice: data, device: device.clone() })
    }

    fn b_trues(shape: &Shape, device: &Cuda, _dtype: BoolDType) -> Result<CudaBoolStorage> {
        let elem_count = shape.element_count();
        let host = vec![1u8; elem_count];
        let data = device.memcpy_stod(&host)?;
        Ok(CudaBoolStorage { slice: data, device: device.clone() })
    }

    fn b_from_bool<'a>(data: impl Into<Cow<'a, [bool]>>, device: &Cuda) -> Result<CudaBoolStorage> {
        let data = data.into();
        let host: Vec<u8> = data.iter().map(|&b| b as u8).collect();
        let slice = device.memcpy_stod(&host)?;
        Ok(CudaBoolStorage { slice, device: device.clone() })
    }

    fn b_from_bytes<'a>(bytes: impl Into<Cow<'a, [u8]>>, device: &Cuda, _dtype: BoolDType) -> Result<CudaBoolStorage> {
        let bytes = bytes.into();
        let slice = device.memcpy_stod(&*bytes)?;
        Ok(CudaBoolStorage { slice, device: device.clone() })
    }

    fn b_contiguous(x: &CudaBoolStorage, layout: &Layout) -> Result<CudaBoolStorage> {
        let out = launch::launch_cast(&x.device, "u8", "u8", &kernel::CAST, &x.slice, layout)?;
        Ok(CudaBoolStorage { slice: out, device: x.device.clone() })
    }

    fn b_cast_float(x: &CudaBoolStorage, layout: &Layout, to: FloatDType) -> Result<CudaFloatStorage> {
        let device = &x.device;
        match to {
            FloatDType::F32 => {
                let out = launch::launch_cast(device, "u8", "f32", &kernel::CAST, &x.slice, layout)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(out), device: device.clone() })
            }
            FloatDType::F64 => {
                let out = launch::launch_cast(device, "u8", "f64", &kernel::CAST, &x.slice, layout)?;
                Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(out), device: device.clone() })
            }
        }
    }

    fn b_cast_int(x: &CudaBoolStorage, layout: &Layout, to: IntDType) -> Result<CudaIntStorage> {
        let device = &x.device;
        match to {
            IntDType::I32 => {
                let out = launch::launch_cast(device, "u8", "i32", &kernel::CAST, &x.slice, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::I32(out), device: device.clone() })
            }
            IntDType::U32 => {
                let out = launch::launch_cast(device, "u8", "u32", &kernel::CAST, &x.slice, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U32(out), device: device.clone() })
            }
            IntDType::U8 => {
                let out = launch::launch_cast(device, "u8", "u8", &kernel::CAST, &x.slice, layout)?;
                Ok(CudaIntStorage { slice: CudaIntSlice::U8(out), device: device.clone() })
            }
        }
    }

    fn b_cast_bool(x: &CudaBoolStorage, layout: &Layout, _to: BoolDType) -> Result<CudaBoolStorage> {
        let out = launch::launch_cast(&x.device, "u8", "u8", &kernel::CAST, &x.slice, layout)?;
        Ok(CudaBoolStorage { slice: out, device: x.device.clone() })
    }

    fn b_to_vec(x: &CudaBoolStorage, layout: &Layout) -> Result<Vec<bool>> {
        let raw = x.device.memcpy_dtov(&x.slice)?;
        Ok(layout.storage_indices().map(|i| raw[i] != 0).collect())
    }

    fn b_to_bytes<'a>(x: &'a CudaBoolStorage, layout: &Layout) -> Result<Cow<'a, [u8]>> {
        let raw: Vec<u8> = x.device.memcpy_dtov(&x.slice)?;
        if layout.is_contiguous() {
            Ok(Cow::Owned(raw))
        } else {
            let gathered: Vec<u8> = layout.storage_indices().map(|i| raw[i]).collect();
            Ok(Cow::Owned(gathered))
        }
    }

    fn b_and(lhs: &CudaBoolStorage, lhs_l: &Layout, rhs: &CudaBoolStorage, rhs_l: &Layout) -> Result<CudaBoolStorage> {
        lhs.device.same_ordinal(&rhs.device, "and")?;
        let out = launch::launch_binary_by_kernel_name(&lhs.device, "band_u8", &kernel::BINARY, &lhs.slice, &rhs.slice, lhs_l, rhs_l)?;
        Ok(CudaBoolStorage { slice: out, device: lhs.device.clone() })
    }

    fn b_or(lhs: &CudaBoolStorage, lhs_l: &Layout, rhs: &CudaBoolStorage, rhs_l: &Layout) -> Result<CudaBoolStorage> {
        lhs.device.same_ordinal(&rhs.device, "or")?;
        let out = launch::launch_binary_by_kernel_name(&lhs.device, "bor_u8", &kernel::BINARY, &lhs.slice, &rhs.slice, lhs_l, rhs_l)?;
        Ok(CudaBoolStorage { slice: out, device: lhs.device.clone() })
    }

    fn b_xor(lhs: &CudaBoolStorage, lhs_l: &Layout, rhs: &CudaBoolStorage, rhs_l: &Layout) -> Result<CudaBoolStorage> {
        lhs.device.same_ordinal(&rhs.device, "xor")?;
        let out = launch::launch_binary_by_kernel_name(&lhs.device, "bxor_u8", &kernel::BINARY, &lhs.slice, &rhs.slice, lhs_l, rhs_l)?;
        Ok(CudaBoolStorage { slice: out, device: lhs.device.clone() })
    }

    fn b_not(x: &CudaBoolStorage, layout: &Layout) -> Result<CudaBoolStorage> {
        let out = launch::launch_unary_raw_by_kernel_name(&x.device, "unot_u8", &kernel::UNARY, &x.slice, layout)?;
        Ok(CudaBoolStorage { slice: out, device: x.device.clone() })
    }

    fn b_reduce_all(x: &CudaBoolStorage, layout: &Layout, dims: &[usize], keepdim: bool) -> Result<(CudaBoolStorage, Shape)> {
        let (out, shape) =
            launch::launch_multi_reduce_by_kernel_name::<u8>(&x.device, "sall_u8", &kernel::REDUCE, &x.slice, layout, dims, keepdim)?;
        Ok((CudaBoolStorage { slice: out, device: x.device.clone() }, shape))
    }

    fn b_reduce_any(x: &CudaBoolStorage, layout: &Layout, dims: &[usize], keepdim: bool) -> Result<(CudaBoolStorage, Shape)> {
        let (out, shape) =
            launch::launch_multi_reduce_by_kernel_name::<u8>(&x.device, "sany_u8", &kernel::REDUCE, &x.slice, layout, dims, keepdim)?;
        Ok((CudaBoolStorage { slice: out, device: x.device.clone() }, shape))
    }

    fn b_true_count(x: &CudaBoolStorage, layout: &Layout) -> Result<usize> {
        let data = x.device.memcpy_dtov(&x.slice)?;
        let count = if layout.is_contiguous() {
            data.iter().filter(|&&b| b != 0).count()
        } else {
            layout.storage_indices().filter(|&i| data[i] != 0).count()
        };
        Ok(count)
    }

    fn b_cat(srcs: &[(&CudaBoolStorage, &Layout)], dim: usize) -> Result<(CudaBoolStorage, Shape)> {
        let layouts: Vec<&Layout> = srcs.iter().map(|(_, l)| *l).collect();
        let out_shape = super::cat_compute_shape(&layouts, dim)?;
        let device = &srcs[0].0.device;
        for (storage, _) in srcs {
            storage.device.same_ordinal(device, "cat")?;
        }

        if dim == 0 {
            let mut out = device.alloc::<u8>(out_shape.element_count())?;
            let mut offset = 0usize;
            for (storage, layout) in srcs {
                launch::launch_copy_offset(device, "ucopy_u8", &kernel::COPY, &storage.slice, layout, &out, offset)?;
                offset += layout.shape().element_count();
            }
            Ok((CudaBoolStorage { slice: out, device: device.clone() }, out_shape))
        } else {
            let cat_size = out_shape.dims()[dim];
            let d1: usize = out_shape.dims()[..dim].iter().product();
            let block: usize = out_shape.dims()[dim + 1..].iter().product();
            let dst_s = block * cat_size;
            let mut out = device.alloc::<u8>(out_shape.element_count())?;
            let mut saved: Vec<CudaSlice<u8>> = Vec::new();
            let mut offset = 0usize;
            for (storage, layout) in srcs {
                let cat_dim_sz = layout.dims()[dim];
                let d2 = block * cat_dim_sz;
                if layout.is_contiguous() {
                    launch::launch_copy2d(
                        device,
                        "ucopy2d_u8",
                        &kernel::COPY,
                        d1,
                        d2,
                        d2,
                        dst_s,
                        &storage.slice,
                        layout.start_offset(),
                        &out,
                        offset,
                    )?;
                } else {
                    let contig = launch::launch_cast(device, "u8", "u8", &kernel::CAST, &storage.slice, layout)?;
                    launch::launch_copy2d(device, "ucopy2d_u8", &kernel::COPY, d1, d2, d2, dst_s, &contig, 0, &out, offset)?;
                    saved.push(contig);
                }
                offset += d2;
            }
            Ok((CudaBoolStorage { slice: out, device: device.clone() }, out_shape))
        }
    }

    fn b_pick(
        mask: &CudaBoolStorage,
        mask_l: &Layout,
        on_true: &CudaBoolStorage,
        true_l: &Layout,
        on_false: &CudaBoolStorage,
        false_l: &Layout,
    ) -> Result<CudaBoolStorage> {
        mask.device.same_ordinal(&on_true.device, "pick")?;
        mask.device.same_ordinal(&on_false.device, "pick")?;
        let device = &mask.device;
        let slice =
            launch::launch_pick(device, "u8", &kernel::PICK, &mask.slice, mask_l, &on_true.slice, true_l, &on_false.slice, false_l)?;
        Ok(CudaBoolStorage { slice, device: device.clone() })
    }

    fn b_pick_true(
        mask: &CudaBoolStorage,
        mask_l: &Layout,
        value: bool,
        on_false: &CudaBoolStorage,
        false_l: &Layout,
    ) -> Result<CudaBoolStorage> {
        mask.device.same_ordinal(&on_false.device, "pick_true")?;
        let device = &mask.device;
        let slice = launch::launch_pick_true(device, "u8", &kernel::PICK, &mask.slice, mask_l, value as u8, &on_false.slice, false_l)?;
        Ok(CudaBoolStorage { slice, device: device.clone() })
    }

    fn b_pick_false(
        mask: &CudaBoolStorage,
        mask_l: &Layout,
        on_true: &CudaBoolStorage,
        true_l: &Layout,
        value: bool,
    ) -> Result<CudaBoolStorage> {
        mask.device.same_ordinal(&on_true.device, "pick_false")?;
        let device = &mask.device;
        let slice = launch::launch_pick_false(device, "u8", &kernel::PICK, &mask.slice, mask_l, &on_true.slice, true_l, value as u8)?;
        Ok(CudaBoolStorage { slice, device: device.clone() })
    }

    fn b_allclose(a: &CudaBoolStorage, a_l: &Layout, b: &CudaBoolStorage, b_l: &Layout) -> Result<bool> {
        a.device.same_ordinal(&b.device, "allclose")?;
        Ok(launch::launch_allclose_int(&a.device, "u8", &a.slice, a_l, &b.slice, b_l)?)
    }
}
