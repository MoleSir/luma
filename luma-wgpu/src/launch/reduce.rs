use crate::ops::{BindGroupBuilder, reduce_op_code};
use crate::shaders::ShaderKind;
use crate::{WgpuIntBuffer, WgpuIntStorage, WgpuStorage};
use luma_tensor::{Error, Layout, ReduceOp, Result, Shape};

/// Uniform payload for reduce.
///
/// Metadata layout (all `u32`):
/// `dims[rank] | strides[rank] | reduced_mask[rank] |
///  reduced_dims[red_count] | reduced_strides[red_count]`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceParams {
    out_count: u32,
    rank: u32,
    start: u32,
    red_count: u32,
    red_total: u32,
    op: u32,
    dims_base: u32,
    strides_base: u32,
    mask_base: u32,
    red_dims_base: u32,
    red_strides_base: u32,
    _pad: u32,
}

/// Uniform payload for argmin/argmax over a single dim.
///
/// Metadata layout is the same `dims | strides | mask` prefix as
/// [`ReduceParams`]; the reduced dim is marked in `mask`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ArgReduceParams {
    out_count: u32,
    rank: u32,
    start: u32,
    dim_size: u32,
    dim_stride: u32,
    dim: u32,
    take_max: u32,
    dims_base: u32,
    strides_base: u32,
    mask_base: u32,
    _pad0: u32,
    _pad1: u32,
}

/// `dims | strides | reduced_mask | reduced_dims | reduced_strides`, all `u32`.
fn reduce_meta(layout: &Layout, reduced: &[usize]) -> Vec<u32> {
    let rank = layout.shape().rank();
    let mut meta = Vec::with_capacity(3 * rank + 2 * reduced.len());
    meta.extend(layout.dims().iter().map(|&v| v as u32));
    meta.extend(layout.stride().iter().map(|&v| v as u32));
    meta.extend((0..rank).map(|d| u32::from(reduced.contains(&d))));
    meta.extend(reduced.iter().map(|&d| layout.dims()[d] as u32));
    meta.extend(reduced.iter().map(|&d| layout.stride()[d] as u32));
    meta
}

fn validate_dims(layout: &Layout, dims: &mut Vec<usize>, op: &'static str) -> Result<()> {
    dims.sort_unstable();
    dims.dedup();
    let rank = layout.shape().rank();
    if let Some(&dim) = dims.iter().find(|&&d| d >= rank) {
        return Err(Error::DimOutOfRange { shape: layout.shape().clone(), dim: dim as i32, op });
    }
    Ok(())
}

pub(crate) fn launch_reduce<S: WgpuStorage>(
    x: &S,
    layout: &Layout,
    dims: &[usize],
    _keepdim: bool,
    op: ReduceOp,
    out_shape: &Shape,
) -> Result<S> {
    let device = x.wgpu_device();
    let in_buf = x.wgpu_buffer();
    let dtype = x.wgpu_dtype();
    let key = ShaderKind::Reduce(dtype.into());
    let elem_bytes = x.wgpu_elem_bytes();

    let mut reduced = dims.to_vec();
    validate_dims(layout, &mut reduced, "reduce")?;

    let rank = layout.shape().rank();
    let out_count = out_shape.element_count();
    let out = device.alloc_storage_output(out_count * elem_bytes)?;
    if out_count == 0 {
        return Ok(S::wrap_wgpu(out, dtype, device));
    }

    let red_dims: Vec<u32> = reduced.iter().map(|&d| layout.dims()[d] as u32).collect();
    let red_total: usize = red_dims.iter().map(|&d| d as usize).product();

    if red_total == 0 {
        return Ok(S::wrap_wgpu(out, dtype, device));
    }

    let meta = reduce_meta(layout, &reduced);
    let params = device.upload_uniform(&ReduceParams {
        out_count: out_count as u32,
        rank: rank as u32,
        start: layout.start_offset() as u32,
        red_count: reduced.len() as u32,
        red_total: red_total as u32,
        op: reduce_op_code(op),
        dims_base: 0,
        strides_base: rank as u32,
        mask_base: (2 * rank) as u32,
        red_dims_base: (3 * rank) as u32,
        red_strides_base: (3 * rank + reduced.len()) as u32,
        _pad: 0,
    })?;
    let meta = device.upload_storage_u32(&meta)?;

    let shader = device.shader(key)?;
    let bind_group = BindGroupBuilder::new()
        .storage_read(0, in_buf)
        .storage_rw(1, &out)
        .uniform(2, &params)
        .storage_read(3, &meta)
        .build(device, shader.layout());
    // One workgroup per output; every workgroup reduces its slice cooperatively.
    shader.run(device, &bind_group, [out_count as u32, 1, 1]);

    Ok(S::wrap_wgpu(out, dtype, device))
}

pub(crate) fn launch_arg_reduce<S: WgpuStorage>(
    x: &S,
    layout: &Layout,
    dim: usize,
    _keepdim: bool,
    take_max: bool,
    out_shape: &Shape,
) -> Result<WgpuIntStorage> {
    let device = x.wgpu_device();
    let dtype = x.wgpu_dtype();
    let in_buf = x.wgpu_buffer();
    let key = ShaderKind::ArgReduce(dtype.into());

    let rank = layout.shape().rank();
    if dim >= rank {
        return Err(Error::DimOutOfRange { shape: layout.shape().clone(), dim: dim as i32, op: "arg_reduce" });
    }

    let out_count = out_shape.element_count();
    let out = device.alloc_storage_output(out_count * std::mem::size_of::<u32>())?;
    if out_count == 0 {
        return Ok(WgpuIntStorage { buffer: WgpuIntBuffer::U32(out), device: device.clone() });
    }

    let meta = reduce_meta(layout, &[dim]);
    let params = device.upload_uniform(&ArgReduceParams {
        out_count: out_count as u32,
        rank: rank as u32,
        start: layout.start_offset() as u32,
        dim_size: layout.dims()[dim] as u32,
        dim_stride: layout.stride()[dim] as u32,
        dim: dim as u32,
        take_max: u32::from(take_max),
        dims_base: 0,
        strides_base: rank as u32,
        mask_base: (2 * rank) as u32,
        _pad0: 0,
        _pad1: 0,
    })?;
    let meta = device.upload_storage_u32(&meta)?;

    let shader = device.shader(key)?;
    let bind_group = BindGroupBuilder::new()
        .storage_read(0, in_buf)
        .storage_rw(1, &out)
        .uniform(2, &params)
        .storage_read(3, &meta)
        .build(device, shader.layout());
    // One workgroup per output; arg reduce uses the same block layout as reduce.
    shader.run(device, &bind_group, [out_count as u32, 1, 1]);

    Ok(WgpuIntStorage { buffer: WgpuIntBuffer::U32(out), device: device.clone() })
}
