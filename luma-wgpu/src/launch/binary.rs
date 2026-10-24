use crate::launch::{binary_index_meta, buffer_start, layout_index_meta};
use crate::ops::{BindGroupBuilder, binary_op_code};
use crate::shaders::ShaderKind;
use crate::storage::WgpuStorage;
use luma_tensor::{BinaryOp, DTypeKindT, Error, Layout, Result};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BinaryParams {
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BinaryScalarParams {
    n: u32,
    op: u32,
    rank: u32,
    start: u32,
    dims_base: u32,
    strides_base: u32,
    contiguous: u32,
    scalar_is_lhs: u32,
    scalar_bits: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BinaryInplaceParams {
    n: u32,
    op: u32,
    rank: u32,
    dst_start: u32,
    src_start: u32,
    dims_base: u32,
    dst_strides_base: u32,
    src_strides_base: u32,
    dst_contiguous: u32,
    src_contiguous: u32,
    _pad0: u32,
    _pad1: u32,
}

pub(crate) fn launch_binary<S: WgpuStorage>(lhs: &S, lhs_l: &Layout, rhs: &S, rhs_l: &Layout, op: BinaryOp) -> Result<S> {
    let dtype = lhs.wgpu_dtype();
    if dtype != rhs.wgpu_dtype() {
        return Err(Error::DTypeMismatch { lhs: dtype.into(), rhs: rhs.wgpu_dtype().into(), op: "wgpu f_binary" });
    }

    let key = ShaderKind::Binary(dtype.into());
    let elem_bytes = dtype.size_of();

    let device = lhs.wgpu_device();
    let n = lhs_l.shape().element_count();
    let out = device.alloc_storage_output(n * elem_bytes)?;
    if n == 0 {
        return Ok(S::wrap_wgpu(out, dtype, device));
    }

    let (lhs_start, lhs_offset) = buffer_start(lhs_l, elem_bytes, device);
    let (rhs_start, rhs_offset) = buffer_start(rhs_l, elem_bytes, device);
    let rank = lhs_l.shape().rank();
    let params = device.upload_uniform(&BinaryParams {
        n: n as u32,
        op: binary_op_code(op),
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

    Ok(S::wrap_wgpu(out, dtype, device))
}

pub(crate) fn launch_binary_scalar<S: WgpuStorage>(
    input: &S,
    layout: &Layout,
    scalar_bits: u32,
    scalar_is_lhs: bool,
    op: BinaryOp,
) -> Result<S> {
    let dtype = input.wgpu_dtype();
    let key = ShaderKind::BinaryScalar(dtype.into());
    let elem_bytes = dtype.size_of();

    let device = input.wgpu_device();
    let n = layout.shape().element_count();
    let out = device.alloc_storage_output(n * elem_bytes)?;
    if n == 0 {
        return Ok(S::wrap_wgpu(out, dtype, device));
    }

    let (start, offset) = buffer_start(layout, elem_bytes, device);
    let rank = layout.shape().rank();
    let params = device.upload_uniform(&BinaryScalarParams {
        n: n as u32,
        op: binary_op_code(op),
        rank: rank as u32,
        start,
        dims_base: 0,
        strides_base: rank as u32,
        contiguous: u32::from(layout.is_contiguous()),
        scalar_is_lhs: u32::from(scalar_is_lhs),
        scalar_bits,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    })?;
    let meta = device.upload_storage_u32(&layout_index_meta(layout))?;

    let shader = device.shader(key)?;
    let bind_group = BindGroupBuilder::new()
        .storage_read_at(0, input.wgpu_buffer(), offset)
        .storage_rw(1, &out)
        .uniform(2, &params)
        .storage_read(3, &meta)
        .build(device, shader.layout());
    shader.run(device, &bind_group, shader.workgroups_1d(n));

    Ok(S::wrap_wgpu(out, dtype, device))
}

pub(crate) fn launch_binary_inplace<S: WgpuStorage>(dst: &S, dst_l: &Layout, src: &S, src_l: &Layout, op: BinaryOp) -> Result<()> {
    if dst_l.shape() != src_l.shape() {
        return Err(Error::ShapeMismatchBinaryOp { lhs: dst_l.shape().clone(), rhs: src_l.shape().clone(), op: "wgpu f_binary_" });
    }

    let dtype = dst.wgpu_dtype();
    if dtype != src.wgpu_dtype() {
        return Err(Error::DTypeMismatch { lhs: dtype.into(), rhs: src.wgpu_dtype().into(), op: "wgpu f_binary_" });
    }

    let key = ShaderKind::BinaryInplace(dtype.into());
    let elem_bytes = dtype.size_of();

    let device = dst.wgpu_device();
    let n = dst_l.shape().element_count();
    if n == 0 {
        return Ok(());
    }

    let (dst_start, dst_offset) = buffer_start(dst_l, elem_bytes, device);
    let (src_start, src_offset) = buffer_start(src_l, elem_bytes, device);
    let rank = dst_l.shape().rank();
    let params = device.upload_uniform(&BinaryInplaceParams {
        n: n as u32,
        op: binary_op_code(op),
        rank: rank as u32,
        dst_start,
        src_start,
        dims_base: 0,
        dst_strides_base: rank as u32,
        src_strides_base: (2 * rank) as u32,
        dst_contiguous: u32::from(dst_l.is_contiguous()),
        src_contiguous: u32::from(src_l.is_contiguous()),
        _pad0: 0,
        _pad1: 0,
    })?;
    let meta = device.upload_storage_u32(&binary_index_meta(dst_l, src_l))?;

    let shader = device.shader(key)?;
    let bind_group = BindGroupBuilder::new()
        .storage_rw_at(0, dst.wgpu_buffer(), dst_offset)
        .storage_read_at(1, src.wgpu_buffer(), src_offset)
        .uniform(2, &params)
        .storage_read(3, &meta)
        .build(device, shader.layout());
    shader.run(device, &bind_group, shader.workgroups_1d(n));

    Ok(())
}

pub(crate) fn launch_binary_scalar_inplace<S: WgpuStorage>(dst: &S, dst_l: &Layout, scalar_bits: u32, op: BinaryOp) -> Result<()> {
    let dtype = dst.wgpu_dtype();
    let key = ShaderKind::BinaryScalarInplace(dtype.into());
    let elem_bytes = dtype.size_of();

    let device = dst.wgpu_device();
    let n = dst_l.shape().element_count();
    if n == 0 {
        return Ok(());
    }

    let (start, offset) = buffer_start(dst_l, elem_bytes, device);
    let rank = dst_l.shape().rank();
    let params = device.upload_uniform(&BinaryScalarParams {
        n: n as u32,
        op: binary_op_code(op),
        rank: rank as u32,
        start,
        dims_base: 0,
        strides_base: rank as u32,
        contiguous: u32::from(dst_l.is_contiguous()),
        scalar_is_lhs: 0,
        scalar_bits,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    })?;
    let meta = device.upload_storage_u32(&layout_index_meta(dst_l))?;

    let shader = device.shader(key)?;
    let bind_group = BindGroupBuilder::new()
        .storage_rw_at(0, dst.wgpu_buffer(), offset)
        .uniform(1, &params)
        .storage_read(2, &meta)
        .build(device, shader.layout());
    shader.run(device, &bind_group, shader.workgroups_1d(n));

    Ok(())
}
