use crate::WgpuBoolStorage;
use crate::launch::{binary_index_meta, buffer_start, layout_index_meta};
use crate::ops::{BindGroupBuilder, cmp_op_code};
use crate::shaders::ShaderKind;
use crate::storage::WgpuStorage;
use luma_tensor::{CmpOp, DTypeKindT, Error, Layout, Result};

/// Uniform payload for binary comparisons.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CmpParams {
    n: u32,
    op: u32,
    rank: u32,
    lhs_start: u32,
    rhs_start: u32,
    dims_base: u32,
    lhs_strides_base: u32,
    rhs_strides_base: u32,
    lhs_contiguous: u32,
    rhs_contiguous: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Uniform payload for scalar comparisons.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CmpScalarParams {
    n: u32,
    op: u32,
    scalar_bits: u32,
    rank: u32,
    start: u32,
    dims_base: u32,
    strides_base: u32,
    contiguous: u32,
}

pub(crate) fn launch_cmp<S: WgpuStorage>(lhs: &S, lhs_l: &Layout, rhs: &S, rhs_l: &Layout, op: CmpOp) -> Result<WgpuBoolStorage> {
    if lhs_l.shape() != rhs_l.shape() {
        return Err(Error::ShapeMismatchBinaryOp { lhs: lhs_l.shape().clone(), rhs: rhs_l.shape().clone(), op: "wgpu f_cmp" });
    }

    let dtype = lhs.wgpu_dtype();
    if dtype != rhs.wgpu_dtype() {
        return Err(Error::DTypeMismatch { lhs: dtype.into(), rhs: rhs.wgpu_dtype().into(), op: "wgpu f_cmp" });
    }

    let key = ShaderKind::Cmp(dtype.into());
    let elem_bytes = dtype.size_of();

    let device = lhs.wgpu_device();
    let n = lhs_l.shape().element_count();
    let out = device.alloc_storage_output(n * 4)?;
    if n == 0 {
        return Ok(WgpuBoolStorage::new(out, device));
    }

    let (lhs_start, lhs_offset) = buffer_start(lhs_l, elem_bytes, device);
    let (rhs_start, rhs_offset) = buffer_start(rhs_l, elem_bytes, device);
    let rank = lhs_l.shape().rank();
    let params = device.upload_uniform(&CmpParams {
        n: n as u32,
        op: cmp_op_code(op),
        rank: rank as u32,
        lhs_start,
        rhs_start,
        dims_base: 0,
        lhs_strides_base: rank as u32,
        rhs_strides_base: (2 * rank) as u32,
        lhs_contiguous: u32::from(lhs_l.is_contiguous()),
        rhs_contiguous: u32::from(rhs_l.is_contiguous()),
        _pad0: 0,
        _pad1: 0,
    })?;
    let meta = device.upload_storage_u32(&binary_index_meta(lhs_l, rhs_l))?;

    let shader = device.shader(key)?;
    let bind_group = BindGroupBuilder::new()
        .storage_read_at(0, lhs.wgpu_buffer(), lhs_offset)
        .storage_read_at(1, rhs.wgpu_buffer(), rhs_offset)
        .storage_rw(2, &out)
        .uniform(3, &params)
        .storage_read(4, &meta)
        .build(device, shader.layout());
    shader.run(device, &bind_group, shader.workgroups_1d(n));

    Ok(WgpuBoolStorage::new(out, device))
}

pub(crate) fn launch_cmp_scalar<S: WgpuStorage>(input: &S, layout: &Layout, scalar_bits: u32, op: CmpOp) -> Result<WgpuBoolStorage> {
    let dtype = input.wgpu_dtype();
    let key = ShaderKind::CmpScalar(dtype.into());
    let elem_bytes = dtype.size_of();

    let device = input.wgpu_device();
    let n = layout.shape().element_count();
    let out = device.alloc_storage_output(n * 4)?;
    if n == 0 {
        return Ok(WgpuBoolStorage::new(out, device));
    }

    let (start, offset) = buffer_start(layout, elem_bytes, device);
    let rank = layout.shape().rank();
    let params = device.upload_uniform(&CmpScalarParams {
        n: n as u32,
        op: cmp_op_code(op),
        scalar_bits,
        rank: rank as u32,
        start,
        dims_base: 0,
        strides_base: rank as u32,
        contiguous: u32::from(layout.is_contiguous()),
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

    Ok(WgpuBoolStorage::new(out, device))
}
