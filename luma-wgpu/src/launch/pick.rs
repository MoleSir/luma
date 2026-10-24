use crate::WgpuBoolStorage;
use crate::launch::buffer_start;
use crate::ops::BindGroupBuilder;
use crate::shaders::ShaderKind;
use crate::storage::WgpuStorage;
use luma_tensor::{Error, Layout, Result};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PickParams {
    n: u32,
    rank: u32,
    mask_start: u32,
    true_start: u32,
    false_start: u32,
    dims_base: u32,
    mask_strides_base: u32,
    true_strides_base: u32,
    false_strides_base: u32,
    mask_contiguous: u32,
    true_contiguous: u32,
    false_contiguous: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PickScalarParams {
    n: u32,
    rank: u32,
    mask_start: u32,
    other_start: u32,
    dims_base: u32,
    mask_strides_base: u32,
    other_strides_base: u32,
    mask_contiguous: u32,
    other_contiguous: u32,
    scalar_bits: u32,
    scalar_is_true: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
}

fn pick_meta(mask_l: &Layout, true_l: &Layout, false_l: &Layout) -> Vec<u32> {
    let rank = mask_l.shape().rank();
    let mut meta = Vec::with_capacity(4 * rank);
    meta.extend(mask_l.dims().iter().map(|&v| v as u32));
    meta.extend(mask_l.stride().iter().map(|&v| v as u32));
    meta.extend(true_l.stride().iter().map(|&v| v as u32));
    meta.extend(false_l.stride().iter().map(|&v| v as u32));
    meta
}

fn pick_scalar_meta(mask_l: &Layout, other_l: &Layout) -> Vec<u32> {
    let rank = mask_l.shape().rank();
    let mut meta = Vec::with_capacity(3 * rank);
    meta.extend(mask_l.dims().iter().map(|&v| v as u32));
    meta.extend(mask_l.stride().iter().map(|&v| v as u32));
    meta.extend(other_l.stride().iter().map(|&v| v as u32));
    meta
}

pub(crate) fn launch_pick<S: WgpuStorage>(
    mask: &WgpuBoolStorage,
    mask_l: &Layout,
    on_true: &S,
    true_l: &Layout,
    on_false: &S,
    false_l: &Layout,
) -> Result<S> {
    if mask_l.shape() != true_l.shape() || mask_l.shape() != false_l.shape() {
        return Err(Error::ShapeMismatchBinaryOp { lhs: mask_l.shape().clone(), rhs: true_l.shape().clone(), op: "pick" });
    }

    let dtype = on_true.wgpu_dtype();
    if dtype != on_false.wgpu_dtype() {
        return Err(Error::DTypeMismatch { lhs: dtype.into(), rhs: on_false.wgpu_dtype().into(), op: "pick" });
    }

    let device = on_true.wgpu_device();
    let key = ShaderKind::Pick(dtype.into());
    let elem_bytes = on_true.wgpu_elem_bytes();
    let n = mask_l.shape().element_count();
    let out = device.alloc_storage_output(n * elem_bytes)?;
    if n == 0 {
        return Ok(S::wrap_wgpu(out, dtype, device));
    }

    let (mask_start, mask_offset) = buffer_start(mask_l, 4, device);
    let (true_start, true_offset) = buffer_start(true_l, elem_bytes, device);
    let (false_start, false_offset) = buffer_start(false_l, elem_bytes, device);
    let rank = mask_l.shape().rank();
    let params = device.upload_uniform(&PickParams {
        n: n as u32,
        rank: rank as u32,
        mask_start,
        true_start,
        false_start,
        dims_base: 0,
        mask_strides_base: rank as u32,
        true_strides_base: (2 * rank) as u32,
        false_strides_base: (3 * rank) as u32,
        mask_contiguous: u32::from(mask_l.is_contiguous()),
        true_contiguous: u32::from(true_l.is_contiguous()),
        false_contiguous: u32::from(false_l.is_contiguous()),
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    })?;
    let meta = device.upload_storage_u32(&pick_meta(mask_l, true_l, false_l))?;

    let shader = device.shader(key)?;
    let bind_group = BindGroupBuilder::new()
        .storage_read_at(0, &mask.buffer, mask_offset)
        .storage_read_at(1, on_true.wgpu_buffer(), true_offset)
        .storage_read_at(2, on_false.wgpu_buffer(), false_offset)
        .storage_rw(3, &out)
        .uniform(4, &params)
        .storage_read(5, &meta)
        .build(device, shader.layout());
    shader.run(device, &bind_group, shader.workgroups_1d(n));

    Ok(S::wrap_wgpu(out, dtype, device))
}

pub(crate) fn launch_pick_scalar<S: WgpuStorage>(
    mask: &WgpuBoolStorage,
    mask_l: &Layout,
    other: &S,
    other_l: &Layout,
    scalar_bits: u32,
    scalar_is_true: bool,
) -> Result<S> {
    if mask_l.shape() != other_l.shape() {
        return Err(Error::ShapeMismatchBinaryOp { lhs: mask_l.shape().clone(), rhs: other_l.shape().clone(), op: "pick_scalar" });
    }

    let dtype = other.wgpu_dtype();
    let device = other.wgpu_device();
    let key = ShaderKind::PickScalar(dtype.into());
    let elem_bytes = other.wgpu_elem_bytes();
    let n = mask_l.shape().element_count();
    let out = device.alloc_storage_output(n * elem_bytes)?;
    if n == 0 {
        return Ok(S::wrap_wgpu(out, dtype, device));
    }

    let (mask_start, mask_offset) = buffer_start(mask_l, 4, device);
    let (other_start, other_offset) = buffer_start(other_l, elem_bytes, device);
    let rank = mask_l.shape().rank();
    let params = device.upload_uniform(&PickScalarParams {
        n: n as u32,
        rank: rank as u32,
        mask_start,
        other_start,
        dims_base: 0,
        mask_strides_base: rank as u32,
        other_strides_base: (2 * rank) as u32,
        mask_contiguous: u32::from(mask_l.is_contiguous()),
        other_contiguous: u32::from(other_l.is_contiguous()),
        scalar_bits,
        scalar_is_true: u32::from(scalar_is_true),
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
        _pad3: 0,
        _pad4: 0,
    })?;
    let meta = device.upload_storage_u32(&pick_scalar_meta(mask_l, other_l))?;

    let shader = device.shader(key)?;
    let bind_group = BindGroupBuilder::new()
        .storage_read_at(0, &mask.buffer, mask_offset)
        .storage_read_at(1, other.wgpu_buffer(), other_offset)
        .storage_rw(2, &out)
        .uniform(3, &params)
        .storage_read(4, &meta)
        .build(device, shader.layout());
    shader.run(device, &bind_group, shader.workgroups_1d(n));

    Ok(S::wrap_wgpu(out, dtype, device))
}
