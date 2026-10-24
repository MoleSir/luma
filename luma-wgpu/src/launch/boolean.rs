use crate::launch::{binary_index_meta, buffer_start, layout_index_meta};
use crate::ops::BindGroupBuilder;
use crate::shaders::ShaderKind;
use crate::{Wgpu, WgpuBoolStorage};
use luma_tensor::{Error, Layout, Result};

#[derive(Clone, Copy)]
pub(crate) enum BoolOp {
    And,
    Or,
    Xor,
}

fn bool_op_code(op: BoolOp) -> u32 {
    match op {
        BoolOp::And => 0,
        BoolOp::Or => 1,
        BoolOp::Xor => 2,
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BoolCopyParams {
    n: u32,
    rank: u32,
    start: u32,
    dims_base: u32,
    strides_base: u32,
    contiguous: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BoolBinaryParams {
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

fn bool_copy_params(device: &Wgpu, layout: &Layout, n: usize) -> Result<wgpu::Buffer> {
    let (start, _) = buffer_start(layout, 4, device);
    let rank = layout.shape().rank();
    Ok(device.upload_uniform(&BoolCopyParams {
        n: n as u32,
        rank: rank as u32,
        start,
        dims_base: 0,
        strides_base: rank as u32,
        contiguous: u32::from(layout.is_contiguous()),
        _pad0: 0,
        _pad1: 0,
    })?)
}

pub(crate) fn launch_bool_not(x: &WgpuBoolStorage, layout: &Layout) -> Result<WgpuBoolStorage> {
    let n = layout.shape().element_count();
    let out = x.device.alloc_storage_output(n * 4)?;
    if n != 0 {
        let (_, offset) = buffer_start(layout, 4, &x.device);
        let params = bool_copy_params(&x.device, layout, n)?;
        let meta = x.device.upload_storage_u32(&layout_index_meta(layout))?;
        let shader = x.device.shader(ShaderKind::BoolNot)?;
        let bind_group = BindGroupBuilder::new()
            .storage_read_at(0, &x.buffer, offset)
            .storage_rw(1, &out)
            .uniform(2, &params)
            .storage_read(3, &meta)
            .build(&x.device, shader.layout());
        shader.run(&x.device, &bind_group, shader.workgroups_1d(n));
    }
    Ok(WgpuBoolStorage::new(out, &x.device))
}

pub(crate) fn launch_bool_binary(
    lhs: &WgpuBoolStorage,
    lhs_l: &Layout,
    rhs: &WgpuBoolStorage,
    rhs_l: &Layout,
    op: BoolOp,
) -> Result<WgpuBoolStorage> {
    if lhs_l.shape() != rhs_l.shape() {
        return Err(Error::ShapeMismatchBinaryOp { lhs: lhs_l.shape().clone(), rhs: rhs_l.shape().clone(), op: "wgpu b_binary" });
    }

    let device = &lhs.device;
    let n = lhs_l.shape().element_count();
    let out = device.alloc_storage_output(n * 4)?;
    if n == 0 {
        return Ok(WgpuBoolStorage::new(out, device));
    }

    let (lhs_start, lhs_offset) = buffer_start(lhs_l, 4, device);
    let (rhs_start, rhs_offset) = buffer_start(rhs_l, 4, device);
    let rank = lhs_l.shape().rank();
    let params = device.upload_uniform(&BoolBinaryParams {
        n: n as u32,
        op: bool_op_code(op),
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

    let shader = device.shader(ShaderKind::BoolBinary)?;
    let bind_group = BindGroupBuilder::new()
        .storage_read_at(0, &lhs.buffer, lhs_offset)
        .storage_read_at(1, &rhs.buffer, rhs_offset)
        .storage_rw(2, &out)
        .uniform(3, &params)
        .storage_read(4, &meta)
        .build(device, shader.layout());
    shader.run(device, &bind_group, shader.workgroups_1d(n));

    Ok(WgpuBoolStorage::new(out, device))
}
