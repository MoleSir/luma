use crate::WgpuIntStorage;
use crate::launch::buffer_start;
use crate::ops::BindGroupBuilder;
use crate::shaders::ShaderKind;
use crate::storage::WgpuStorage;
use luma_tensor::{DType, Error, Layout, Result, Shape};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct IndexSelectParams {
    n: u32,
    rank: u32,
    src_start: u32,
    src_dims_base: u32,
    src_strides_base: u32,
    src_contiguous: u32,
    ids_start: u32,
    ids_stride: u32,
    src_dim: u32,
    ids_count: u32,
    right: u32,
    pad: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GatherParams {
    n: u32,
    rank: u32,
    src_start: u32,
    src_dims_base: u32,
    src_strides_base: u32,
    src_contiguous: u32,
    idx_start: u32,
    idx_dims_base: u32,
    idx_strides_base: u32,
    idx_contiguous: u32,
    src_dim: u32,
    ids_dim: u32,
    right: u32,
    pad: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct IndexAddParams {
    n: u32,
    rank: u32,
    src_start: u32,
    src_dims_base: u32,
    src_strides_base: u32,
    src_contiguous: u32,
    ids_start: u32,
    ids_stride: u32,
    ids_count: u32,
    src_dim: u32,
    dst_dim: u32,
    right: u32,
    pad: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScatterAddParams {
    n: u32,
    rank: u32,
    src_start: u32,
    src_dims_base: u32,
    src_strides_base: u32,
    src_contiguous: u32,
    idx_start: u32,
    idx_dims_base: u32,
    idx_strides_base: u32,
    idx_contiguous: u32,
    src_dim: u32,
    dst_dim: u32,
    right: u32,
    pad: u32,
    _pad0: u32,
    _pad1: u32,
}

fn index_pad(dtype: DType, op: &'static str) -> Result<u32> {
    match dtype {
        DType::I32 => Ok(i32::MAX as u32),
        DType::U32 => Ok(u32::MAX),
        _ => Err(Error::Msg(format!("{op}: expected i32/u32 indices, got {dtype:?}"))),
    }
}

fn validate_dim(layout: &Layout, dim: usize, op: &'static str) -> Result<()> {
    if dim >= layout.shape().rank() {
        return Err(Error::DimOutOfRange { shape: layout.shape().clone(), dim: dim as i32, op });
    }
    Ok(())
}

pub(crate) fn launch_index_select<S: WgpuStorage>(
    x: &S,
    x_l: &Layout,
    idx: &WgpuIntStorage,
    idx_l: &Layout,
    dim: usize,
    out_shape: &Shape,
) -> Result<S> {
    validate_dim(x_l, dim, "index_select")?;
    if idx_l.shape().rank() != 1 {
        return Err(Error::Msg("wgpu index_select: indices must be 1-D".into()));
    }

    let dtype = x.wgpu_dtype();
    let device = x.wgpu_device();
    let elem_bytes = x.wgpu_elem_bytes();
    let pad = index_pad(idx.dtype(), "index_select")?;
    let n = out_shape.element_count();
    let out = device.alloc_storage_output(n * elem_bytes)?;
    let wrapped = S::wrap_wgpu(out, dtype, device);
    if n == 0 {
        return Ok(wrapped);
    }

    let ids_count = idx_l.shape().element_count();
    if ids_count == 0 {
        return Ok(wrapped);
    }

    let rank = x_l.shape().rank();
    let src_dim = x_l.dims()[dim];
    let right: usize = x_l.dims()[dim + 1..].iter().product();
    let (src_start, src_offset) = buffer_start(x_l, elem_bytes, device);
    let (ids_start, ids_offset) = buffer_start(idx_l, 4, device);

    let params = device.upload_uniform(&IndexSelectParams {
        n: n as u32,
        rank: rank as u32,
        src_start,
        src_dims_base: 0,
        src_strides_base: rank as u32,
        src_contiguous: u32::from(x_l.is_contiguous()),
        ids_start,
        ids_stride: idx_l.stride()[0] as u32,
        src_dim: src_dim as u32,
        ids_count: ids_count as u32,
        right: right as u32,
        pad,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
        _pad3: 0,
    })?;
    let meta = device.upload_storage_u32(&crate::launch::layout_index_meta(x_l))?;

    let shader = device.shader(ShaderKind::IndexSelect(dtype.into()))?;
    let bind_group = BindGroupBuilder::new()
        .storage_read_at(0, x.wgpu_buffer(), src_offset)
        .storage_read_at(1, idx.wgpu_buffer(), ids_offset)
        .storage_rw(2, wrapped.wgpu_buffer())
        .uniform(3, &params)
        .storage_read(4, &meta)
        .build(device, shader.layout());
    shader.run(device, &bind_group, shader.workgroups_1d(n));

    Ok(wrapped)
}

pub(crate) fn launch_gather<S: WgpuStorage>(
    x: &S,
    x_l: &Layout,
    idx: &WgpuIntStorage,
    idx_l: &Layout,
    dim: usize,
    out_shape: &Shape,
) -> Result<S> {
    validate_dim(x_l, dim, "gather")?;
    if x_l.shape().rank() != idx_l.shape().rank() {
        return Err(Error::ShapeMismatchBinaryOp { lhs: x_l.shape().clone(), rhs: idx_l.shape().clone(), op: "gather" });
    }
    debug_assert_eq!(out_shape.dims(), idx_l.shape().dims(), "wgpu gather shape must match the layer");

    let dtype = x.wgpu_dtype();
    let device = x.wgpu_device();
    let elem_bytes = x.wgpu_elem_bytes();
    let pad = index_pad(idx.dtype(), "gather")?;
    let n = out_shape.element_count();
    let out = device.alloc_storage_output(n * elem_bytes)?;
    let wrapped = S::wrap_wgpu(out, dtype, device);
    if n == 0 {
        return Ok(wrapped);
    }

    let rank = x_l.shape().rank();
    let src_dim = x_l.dims()[dim];
    let ids_dim = idx_l.dims()[dim];
    let right: usize = x_l.dims()[dim + 1..].iter().product();
    let (src_start, src_offset) = buffer_start(x_l, elem_bytes, device);
    let (idx_start, idx_offset) = buffer_start(idx_l, 4, device);

    let mut meta = Vec::with_capacity(4 * rank);
    meta.extend(x_l.dims().iter().map(|&v| v as u32));
    meta.extend(x_l.stride().iter().map(|&v| v as u32));
    meta.extend(idx_l.dims().iter().map(|&v| v as u32));
    meta.extend(idx_l.stride().iter().map(|&v| v as u32));

    let params = device.upload_uniform(&GatherParams {
        n: n as u32,
        rank: rank as u32,
        src_start,
        src_dims_base: 0,
        src_strides_base: rank as u32,
        src_contiguous: u32::from(x_l.is_contiguous()),
        idx_start,
        idx_dims_base: (2 * rank) as u32,
        idx_strides_base: (3 * rank) as u32,
        idx_contiguous: u32::from(idx_l.is_contiguous()),
        src_dim: src_dim as u32,
        ids_dim: ids_dim as u32,
        right: right as u32,
        pad,
        _pad0: 0,
        _pad1: 0,
    })?;
    let meta = device.upload_storage_u32(&meta)?;

    let shader = device.shader(ShaderKind::Gather(dtype.into()))?;
    let bind_group = BindGroupBuilder::new()
        .storage_read_at(0, x.wgpu_buffer(), src_offset)
        .storage_read_at(1, idx.wgpu_buffer(), idx_offset)
        .storage_rw(2, wrapped.wgpu_buffer())
        .uniform(3, &params)
        .storage_read(4, &meta)
        .build(device, shader.layout());
    shader.run(device, &bind_group, shader.workgroups_1d(n));

    Ok(wrapped)
}

pub(crate) fn launch_index_add<S: WgpuStorage>(
    init: &S,
    init_l: &Layout,
    idx: &WgpuIntStorage,
    idx_l: &Layout,
    src: &S,
    src_l: &Layout,
    dim: usize,
) -> Result<S> {
    validate_dim(init_l, dim, "index_add")?;
    validate_dim(src_l, dim, "index_add")?;
    if idx_l.shape().rank() != 1 {
        return Err(Error::Msg("wgpu index_add: indices must be 1-D".into()));
    }
    if init.wgpu_dtype() != src.wgpu_dtype() {
        return Err(Error::DTypeMismatch { lhs: init.wgpu_dtype().into(), rhs: src.wgpu_dtype().into(), op: "index_add" });
    }

    let idx_count = idx_l.shape().element_count();
    if src_l.dims()[dim] != idx_count {
        return Err(Error::ShapeMismatchBinaryOp { lhs: src_l.shape().clone(), rhs: idx_l.shape().clone(), op: "index_add" });
    }

    let device = init.wgpu_device();
    let out = crate::launch::launch_copy(init, init_l)?;
    let n: usize = src_l.dims()[..dim].iter().product::<usize>() * src_l.dims()[dim + 1..].iter().product::<usize>();
    if n == 0 || idx_count == 0 {
        return Ok(out);
    }

    let dtype = init.wgpu_dtype();
    let elem_bytes = init.wgpu_elem_bytes();
    let pad = index_pad(idx.dtype(), "index_add")?;
    let rank = src_l.shape().rank();
    let src_dim = src_l.dims()[dim];
    let dst_dim = init_l.dims()[dim];
    let right: usize = src_l.dims()[dim + 1..].iter().product();
    let (src_start, src_offset) = buffer_start(src_l, elem_bytes, device);
    let (ids_start, ids_offset) = buffer_start(idx_l, 4, device);

    let params = device.upload_uniform(&IndexAddParams {
        n: n as u32,
        rank: rank as u32,
        src_start,
        src_dims_base: 0,
        src_strides_base: rank as u32,
        src_contiguous: u32::from(src_l.is_contiguous()),
        ids_start,
        ids_stride: idx_l.stride()[0] as u32,
        ids_count: idx_count as u32,
        src_dim: src_dim as u32,
        dst_dim: dst_dim as u32,
        right: right as u32,
        pad,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    })?;
    let meta = device.upload_storage_u32(&crate::launch::layout_index_meta(src_l))?;

    let shader = device.shader(ShaderKind::IndexAdd(dtype.into()))?;
    let bind_group = BindGroupBuilder::new()
        .storage_read_at(0, src.wgpu_buffer(), src_offset)
        .storage_read_at(1, idx.wgpu_buffer(), ids_offset)
        .storage_rw(2, out.wgpu_buffer())
        .uniform(3, &params)
        .storage_read(4, &meta)
        .build(device, shader.layout());
    shader.run(device, &bind_group, shader.workgroups_1d(n));

    Ok(out)
}

pub(crate) fn launch_scatter_add<S: WgpuStorage>(
    init: &S,
    init_l: &Layout,
    idx: &WgpuIntStorage,
    idx_l: &Layout,
    src: &S,
    src_l: &Layout,
    dim: usize,
) -> Result<S> {
    validate_dim(init_l, dim, "scatter_add")?;
    validate_dim(src_l, dim, "scatter_add")?;
    if idx_l.shape() != src_l.shape() {
        return Err(Error::ShapeMismatchBinaryOp { lhs: src_l.shape().clone(), rhs: idx_l.shape().clone(), op: "scatter_add" });
    }
    if init.wgpu_dtype() != src.wgpu_dtype() {
        return Err(Error::DTypeMismatch { lhs: init.wgpu_dtype().into(), rhs: src.wgpu_dtype().into(), op: "scatter_add" });
    }

    let device = init.wgpu_device();
    let out = crate::launch::launch_copy(init, init_l)?;
    let n: usize = src_l.dims()[..dim].iter().product::<usize>() * src_l.dims()[dim + 1..].iter().product::<usize>();
    if n == 0 || src_l.dims()[dim] == 0 {
        return Ok(out);
    }

    let dtype = init.wgpu_dtype();
    let elem_bytes = init.wgpu_elem_bytes();
    let pad = index_pad(idx.dtype(), "scatter_add")?;
    let rank = src_l.shape().rank();
    let src_dim = src_l.dims()[dim];
    let dst_dim = init_l.dims()[dim];
    let right: usize = src_l.dims()[dim + 1..].iter().product();
    let (src_start, src_offset) = buffer_start(src_l, elem_bytes, device);
    let (idx_start, idx_offset) = buffer_start(idx_l, 4, device);

    let mut meta = Vec::with_capacity(4 * rank);
    meta.extend(src_l.dims().iter().map(|&v| v as u32));
    meta.extend(src_l.stride().iter().map(|&v| v as u32));
    meta.extend(idx_l.dims().iter().map(|&v| v as u32));
    meta.extend(idx_l.stride().iter().map(|&v| v as u32));

    let params = device.upload_uniform(&ScatterAddParams {
        n: n as u32,
        rank: rank as u32,
        src_start,
        src_dims_base: 0,
        src_strides_base: rank as u32,
        src_contiguous: u32::from(src_l.is_contiguous()),
        idx_start,
        idx_dims_base: (2 * rank) as u32,
        idx_strides_base: (3 * rank) as u32,
        idx_contiguous: u32::from(idx_l.is_contiguous()),
        src_dim: src_dim as u32,
        dst_dim: dst_dim as u32,
        right: right as u32,
        pad,
        _pad0: 0,
        _pad1: 0,
    })?;
    let meta = device.upload_storage_u32(&meta)?;

    let shader = device.shader(ShaderKind::ScatterAdd(dtype.into()))?;
    let bind_group = BindGroupBuilder::new()
        .storage_read_at(0, src.wgpu_buffer(), src_offset)
        .storage_read_at(1, idx.wgpu_buffer(), idx_offset)
        .storage_rw(2, out.wgpu_buffer())
        .uniform(3, &params)
        .storage_read(4, &meta)
        .build(device, shader.layout());
    shader.run(device, &bind_group, shader.workgroups_1d(n));

    Ok(out)
}
