use crate::launch::{buffer_start, layout_index_meta};
use crate::ops::BindGroupBuilder;
use crate::shaders::ShaderKind;
use crate::storage::WgpuStorage;
use luma_tensor::{Layout, Result};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CopyParams {
    n: u32,
    rank: u32,
    start: u32,
    dims_base: u32,
    strides_base: u32,
    contiguous: u32,
    _pad0: u32,
    _pad1: u32,
}

pub(crate) fn launch_copy<S: WgpuStorage>(x: &S, layout: &Layout) -> Result<S> {
    let dtype = x.wgpu_dtype();
    let elem_bytes = x.wgpu_elem_bytes();
    let device = x.wgpu_device();
    let n = layout.shape().element_count();
    let out = device.alloc_storage_output(n * elem_bytes)?;
    if n == 0 {
        return Ok(S::wrap_wgpu(out, dtype, device));
    }

    let (start, offset) = buffer_start(layout, elem_bytes, device);
    let rank = layout.shape().rank();
    let params = device.upload_uniform(&CopyParams {
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

    let shader = device.shader(ShaderKind::Copy(dtype.into()))?;
    let bind_group = BindGroupBuilder::new()
        .storage_read_at(0, x.wgpu_buffer(), offset)
        .storage_rw(1, &out)
        .uniform(2, &params)
        .storage_read(3, &meta)
        .build(device, shader.layout());
    shader.run(device, &bind_group, shader.workgroups_1d(n));

    Ok(S::wrap_wgpu(out, dtype, device))
}
