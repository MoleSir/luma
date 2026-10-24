use crate::launch::{buffer_start, layout_index_meta};
use crate::ops::float_ops::float_buffer;
use crate::ops::{BindGroupBuilder, float_unary_op_code};
use crate::shaders::ShaderKind;
use crate::storage::WgpuStorage;
use crate::{WgpuError, WgpuFloatStorage};
use luma_tensor::dtype::DTypeKindT;
use luma_tensor::{FloatDType, FloatUnaryOp, Layout, Result};

/// Uniform payload for `f_unary`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UnaryParams {
    n: u32,
    op: u32,
    arg0_bits: u32,
    arg1_bits: u32,
    arg0_opt: u32,
    arg1_opt: u32,
    rank: u32,
    start: u32,
    dims_base: u32,
    strides_base: u32,
    contiguous: u32,
    _pad1: u32,
}

/// Uniform payload for `f_float_unary` (activation functions).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FloatUnaryParams {
    n: u32,
    op: u32,
    arg0: f32,
    rank: u32,
    start: u32,
    dims_base: u32,
    strides_base: u32,
    contiguous: u32,
}

pub(crate) fn launch_unary<S: WgpuStorage>(
    input: &S,
    layout: &Layout,
    op_code: u32,
    arg0_bits: u32,
    arg1_bits: u32,
    arg0_opt: u32,
    arg1_opt: u32,
) -> Result<S> {
    let dtype = input.wgpu_dtype();
    let key = ShaderKind::Unary(dtype.into());
    let elem_bytes = dtype.size_of();

    let device = input.wgpu_device();
    let n = layout.shape().element_count();
    let out = device.alloc_storage_output(n * elem_bytes)?;
    if n == 0 {
        return Ok(S::wrap_wgpu(out, dtype, device));
    }

    let (start, offset) = buffer_start(layout, elem_bytes, device);
    let rank = layout.shape().rank();
    let params = device.upload_uniform(&UnaryParams {
        n: n as u32,
        op: op_code,
        arg0_bits,
        arg1_bits,
        arg0_opt,
        arg1_opt,
        rank: rank as u32,
        start,
        dims_base: 0,
        strides_base: rank as u32,
        contiguous: u32::from(layout.is_contiguous()),
        _pad1: 0,
    })?;
    let meta = device.upload_storage_u32(&layout_index_meta(layout))?;

    let shader = device.shader(key)?;
    let bind_group = BindGroupBuilder::new()
        .storage_read_at(0, input.wgpu_buffer(), offset)
        .storage_rw(1, &out)
        .uniform(2, &params)
        .storage_read(4, &meta)
        .build(device, shader.layout());
    shader.run(device, &bind_group, shader.workgroups_1d(n));

    Ok(S::wrap_wgpu(out, dtype, device))
}

pub(crate) fn launch_unary_inplace<S: WgpuStorage>(
    dst: &S,
    layout: &Layout,
    op_code: u32,
    arg0_bits: u32,
    arg1_bits: u32,
    arg0_opt: u32,
    arg1_opt: u32,
) -> Result<()> {
    let dtype = dst.wgpu_dtype();
    let key = ShaderKind::UnaryInplace(dtype.into());
    let elem_bytes = dtype.size_of();

    let device = dst.wgpu_device();
    let n = layout.shape().element_count();
    if n == 0 {
        return Ok(());
    }

    let (start, offset) = buffer_start(layout, elem_bytes, device);
    let rank = layout.shape().rank();
    let params = device.upload_uniform(&UnaryParams {
        n: n as u32,
        op: op_code,
        arg0_bits,
        arg1_bits,
        arg0_opt,
        arg1_opt,
        rank: rank as u32,
        start,
        dims_base: 0,
        strides_base: rank as u32,
        contiguous: u32::from(layout.is_contiguous()),
        _pad1: 0,
    })?;
    let meta = device.upload_storage_u32(&layout_index_meta(layout))?;

    let shader = device.shader(key)?;
    let bind_group = BindGroupBuilder::new()
        .storage_rw_at(0, dst.wgpu_buffer(), offset)
        .uniform(1, &params)
        .storage_read(2, &meta)
        .build(device, shader.layout());
    shader.run(device, &bind_group, shader.workgroups_1d(n));

    Ok(())
}

pub(crate) fn launch_float_unary(x: &WgpuFloatStorage, layout: &Layout, op: FloatUnaryOp) -> Result<WgpuFloatStorage> {
    let device = &x.device;
    let (buf, dtype) = float_buffer(x, "wgpu f_float_unary")?;
    let contiguous = layout.is_contiguous();
    let (key, elem_bytes) = match dtype {
        FloatDType::F32 => (ShaderKind::FloatUnary(FloatDType::F32), 4usize),
        FloatDType::F16 => (ShaderKind::FloatUnary(FloatDType::F16), 2usize),
        FloatDType::F64 | FloatDType::BF16 => {
            return Err(WgpuError::UnsupportedDType(dtype.into(), "wgpu f_float_unary").into());
        }
    };

    let n = layout.shape().element_count();
    let out = device.alloc_storage_output(n * elem_bytes)?;
    if n == 0 {
        return Ok(WgpuFloatStorage::new(out, dtype, device));
    }

    let (start, offset) = buffer_start(layout, elem_bytes, device);
    let rank = layout.shape().rank();
    let arg0 = match op {
        FloatUnaryOp::LeakyRelu(slope) => slope as f32,
        _ => 0.0,
    };
    let params = device.upload_uniform(&FloatUnaryParams {
        n: n as u32,
        op: float_unary_op_code(op),
        arg0,
        rank: rank as u32,
        start,
        dims_base: 0,
        strides_base: rank as u32,
        contiguous: u32::from(contiguous),
    })?;
    let meta = device.upload_storage_u32(&layout_index_meta(layout))?;

    let shader = device.shader(key)?;
    let bind_group = BindGroupBuilder::new()
        .storage_read_at(0, buf, offset)
        .storage_rw(1, &out)
        .uniform(2, &params)
        .storage_read(4, &meta)
        .build(device, shader.layout());
    shader.run(device, &bind_group, shader.workgroups_1d(n));

    Ok(WgpuFloatStorage::new(out, dtype, device))
}

pub(crate) fn launch_float_unary_inplace(x: &WgpuFloatStorage, layout: &Layout, op: FloatUnaryOp) -> Result<()> {
    let device = &x.device;
    let (buf, dtype) = float_buffer(x, "wgpu f_float_unary_")?;
    let key = match dtype {
        FloatDType::F32 => ShaderKind::FloatUnaryInplace(FloatDType::F32),
        FloatDType::F16 => ShaderKind::FloatUnaryInplace(FloatDType::F16),
        FloatDType::F64 | FloatDType::BF16 => {
            return Err(WgpuError::UnsupportedDType(dtype.into(), "wgpu f_float_unary_").into());
        }
    };
    let elem_bytes = dtype.size_of();

    let n = layout.shape().element_count();
    if n == 0 {
        return Ok(());
    }

    let (start, offset) = buffer_start(layout, elem_bytes, device);
    let rank = layout.shape().rank();
    let arg0 = match op {
        FloatUnaryOp::LeakyRelu(slope) => slope as f32,
        _ => 0.0,
    };
    let params = device.upload_uniform(&FloatUnaryParams {
        n: n as u32,
        op: float_unary_op_code(op),
        arg0,
        rank: rank as u32,
        start,
        dims_base: 0,
        strides_base: rank as u32,
        contiguous: u32::from(layout.is_contiguous()),
    })?;
    let meta = device.upload_storage_u32(&layout_index_meta(layout))?;

    let shader = device.shader(key)?;
    let bind_group = BindGroupBuilder::new()
        .storage_rw_at(0, buf, offset)
        .uniform(1, &params)
        .storage_read(2, &meta)
        .build(device, shader.layout());
    shader.run(device, &bind_group, shader.workgroups_1d(n));

    Ok(())
}
