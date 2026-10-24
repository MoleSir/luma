use crate::ops::BindGroupBuilder;
use crate::shaders::ShaderKind;
use crate::{Wgpu, WgpuError, WgpuFloatStorage};
use luma_tensor::{Error, FloatDType, Result, Shape};

const OP_UNIFORM: u32 = 0;
const OP_NORMAL: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RandomParams {
    n: u32,
    op: u32,
    seed: u32,
    p0: f32,
    p1: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

fn launch_random(
    device: &Wgpu,
    shape: &Shape,
    dtype: FloatDType,
    op: u32,
    p0: f32,
    p1: f32,
    op_name: &'static str,
) -> Result<WgpuFloatStorage> {
    let (key, elem_bytes) = match dtype {
        FloatDType::F32 => (ShaderKind::Random(FloatDType::F32), 4usize),
        FloatDType::F16 => (ShaderKind::Random(FloatDType::F16), 2usize),
        FloatDType::F64 | FloatDType::BF16 => {
            return Err(WgpuError::UnsupportedDType(dtype.into(), op_name).into());
        }
    };

    let n = shape.element_count();
    let out = device.alloc_storage_output(n * elem_bytes)?;
    if n == 0 {
        return Ok(WgpuFloatStorage::new(out, dtype, device));
    }

    let params =
        device.upload_uniform(&RandomParams { n: n as u32, op, seed: device.next_random_seed(), p0, p1, _pad0: 0, _pad1: 0, _pad2: 0 })?;

    let shader = device.shader(key)?;
    let bind_group = BindGroupBuilder::new().storage_rw(0, &out).uniform(1, &params).build(device, shader.layout());
    shader.run(device, &bind_group, shader.workgroups_1d(n));

    Ok(WgpuFloatStorage::new(out, dtype, device))
}

pub(crate) fn launch_random_uniform(device: &Wgpu, shape: &Shape, lo: f64, hi: f64, dtype: FloatDType) -> Result<WgpuFloatStorage> {
    let lo32 = lo as f32;
    let span = hi as f32 - lo32;
    if !lo.is_finite() || !hi.is_finite() || lo >= hi || !lo32.is_finite() || !span.is_finite() || span <= 0.0 {
        return Err(Error::Rand(format!("invalid uniform range [{lo}, {hi})")));
    }
    launch_random(device, shape, dtype, OP_UNIFORM, lo32, span, "wgpu f_rand_uniform")
}

pub(crate) fn launch_random_normal(device: &Wgpu, shape: &Shape, mean: f64, std: f64, dtype: FloatDType) -> Result<WgpuFloatStorage> {
    let mean32 = mean as f32;
    let std32 = std as f32;
    if !mean.is_finite() || !std.is_finite() || std < 0.0 || !mean32.is_finite() || !std32.is_finite() || std32 < 0.0 {
        return Err(Error::Rand(format!("invalid normal parameters mean={mean}, std={std}")));
    }
    launch_random(device, shape, dtype, OP_NORMAL, mean32, std32, "wgpu f_rand_normal")
}
