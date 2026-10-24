use crate::launch::{buffer_start, layout_index_meta};
use crate::ops::BindGroupBuilder;
use crate::shaders::ShaderKind;
use crate::storage::WgpuStorage;
use crate::{WgpuError, WgpuFloatStorage, WgpuIntStorage};
use luma_tensor::{FloatDType, IntDType, Layout, Result};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CastParams {
    n: u32,
    rank: u32,
    start: u32,
    dims_base: u32,
    strides_base: u32,
    contiguous: u32,
    _pad0: u32,
    _pad1: u32,
}

pub(crate) fn launch_cast_float<S: WgpuStorage>(x: &S, layout: &Layout, dst: FloatDType) -> Result<WgpuFloatStorage> {
    let dst_elem_bytes = match dst {
        FloatDType::F32 => 4usize,
        FloatDType::F16 => 2usize,
        FloatDType::F64 | FloatDType::BF16 => {
            return Err(WgpuError::UnsupportedDType(dst.into(), "wgpu cast_float").into());
        }
    };

    let dtype = x.wgpu_dtype();
    let in_elem_bytes = x.wgpu_elem_bytes();
    let device = x.wgpu_device();
    let n = layout.shape().element_count();
    let out = device.alloc_storage_output(n * dst_elem_bytes)?;
    if n == 0 {
        return Ok(WgpuFloatStorage::new(out, dst, device));
    }

    let (start, offset) = buffer_start(layout, in_elem_bytes, device);
    let rank = layout.shape().rank();
    let params = device.upload_uniform(&CastParams {
        n: n as u32,
        rank: rank as u32,
        start,
        dims_base: 0,
        strides_base: rank as u32,
        contiguous: u32::from(layout.is_contiguous()),
        _pad0: 0,
        _pad1: 0,
    })?;
    let meta = device.upload_storage_u32(&layout_index_meta(layout))?;

    let shader = device.shader(ShaderKind::CastFloat(dtype.into(), dst))?;
    let bind_group = BindGroupBuilder::new()
        .storage_read_at(0, x.wgpu_buffer(), offset)
        .storage_rw(1, &out)
        .uniform(2, &params)
        .storage_read(3, &meta)
        .build(device, shader.layout());
    shader.run(device, &bind_group, shader.workgroups_1d(n));

    Ok(WgpuFloatStorage::new(out, dst, device))
}

pub(crate) fn launch_cast_int<S: WgpuStorage>(x: &S, layout: &Layout, dst: IntDType) -> Result<WgpuIntStorage> {
    if dst == IntDType::U8 {
        return Err(WgpuError::UnsupportedDType(dst.into(), "wgpu cast_int").into());
    }

    let dtype = x.wgpu_dtype();
    let in_elem_bytes = x.wgpu_elem_bytes();
    let device = x.wgpu_device();
    let n = layout.shape().element_count();
    let out = device.alloc_storage_output(n * std::mem::size_of::<u32>())?;
    if n == 0 {
        return Ok(WgpuIntStorage::new(out, dst, device));
    }

    let (start, offset) = buffer_start(layout, in_elem_bytes, device);
    let rank = layout.shape().rank();
    let params = device.upload_uniform(&CastParams {
        n: n as u32,
        rank: rank as u32,
        start,
        dims_base: 0,
        strides_base: rank as u32,
        contiguous: u32::from(layout.is_contiguous()),
        _pad0: 0,
        _pad1: 0,
    })?;
    let meta = device.upload_storage_u32(&layout_index_meta(layout))?;

    let shader = device.shader(ShaderKind::CastInt(dtype.into(), dst))?;
    let bind_group = BindGroupBuilder::new()
        .storage_read_at(0, x.wgpu_buffer(), offset)
        .storage_rw(1, &out)
        .uniform(2, &params)
        .storage_read(3, &meta)
        .build(device, shader.layout());
    shader.run(device, &bind_group, shader.workgroups_1d(n));

    Ok(WgpuIntStorage::new(out, dst, device))
}
