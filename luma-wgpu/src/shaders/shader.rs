use crate::{Wgpu, WgpuResult};
use luma_tensor::{DType, FloatDType, IntDType};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderKind {
    Binary(DType),
    BinaryScalar(DType),
    BinaryInplace(DType),
    BinaryScalarInplace(DType),
    Unary(DType),
    UnaryInplace(DType),
    FloatUnary(FloatDType),
    FloatUnaryInplace(FloatDType),
    Cmp(DType),
    CmpScalar(DType),
    Matmul(DType),
    Reduce(DType),
    ArgReduce(DType),
    Random(FloatDType),
    BoolBinary,
    BoolNot,
    Copy(DType),
    CastFloat(DType, FloatDType),
    CastInt(DType, IntDType),
    Pick(DType),
    PickScalar(DType),
    Cat(DType),
    IndexSelect(DType),
    Gather(DType),
    IndexAdd(DType),
    ScatterAdd(DType),
}

impl ShaderKind {
    /// Whether this kernel needs a device created with `shader-f16`.
    pub fn requires_f16(self) -> bool {
        match self {
            ShaderKind::Binary(d)
            | ShaderKind::BinaryScalar(d)
            | ShaderKind::BinaryInplace(d)
            | ShaderKind::BinaryScalarInplace(d)
            | ShaderKind::Unary(d)
            | ShaderKind::UnaryInplace(d)
            | ShaderKind::Cmp(d)
            | ShaderKind::CmpScalar(d)
            | ShaderKind::Matmul(d)
            | ShaderKind::Reduce(d)
            | ShaderKind::ArgReduce(d)
            | ShaderKind::Copy(d)
            | ShaderKind::Pick(d)
            | ShaderKind::PickScalar(d)
            | ShaderKind::Cat(d)
            | ShaderKind::IndexSelect(d)
            | ShaderKind::Gather(d)
            | ShaderKind::IndexAdd(d)
            | ShaderKind::ScatterAdd(d) => d == DType::F16,

            ShaderKind::FloatUnary(d) | ShaderKind::FloatUnaryInplace(d) | ShaderKind::Random(d) => d == FloatDType::F16,

            ShaderKind::CastFloat(src, dst) => src == DType::F16 || dst == FloatDType::F16,
            ShaderKind::CastInt(src, _) => src == DType::F16,

            ShaderKind::BoolBinary | ShaderKind::BoolNot => false,
        }
    }
}

/// A compiled compute pipeline plus the resources needed to launch it.
pub struct Shader {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    workgroup_size: [u32; 3],
}

impl Shader {
    /// Compile a shader entry point into a [`Shader`].
    ///
    /// `layout: None` lets wgpu infer the bind-group layout from the WGSL
    /// declarations; the resulting layout is then retrieved from the pipeline.
    pub fn new(
        device: &Wgpu,
        label: &'static str,
        shader: &'static str,
        entry: &'static str,
        workgroup_size: [u32; 3],
    ) -> WgpuResult<Self> {
        let module = device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(shader.into()) });

        let pipeline = device.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });

        let layout = pipeline.get_bind_group_layout(0);

        Ok(Self { pipeline, layout, workgroup_size })
    }

    /// The bind-group layout inferred from the shader.
    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    /// Build a bind group for this kernel's layout.
    pub fn bind(&self, device: &Wgpu, entries: &[wgpu::BindGroupEntry<'_>]) -> wgpu::BindGroup {
        device
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor { label: Some("luma-wgpu bind group"), layout: &self.layout, entries })
    }

    /// Number of workgroups needed for `n` 1-D invocations.
    pub fn workgroups_1d(&self, n: usize) -> [u32; 3] {
        let wg = self.workgroup_size[0].max(1);
        [(n as u32).div_ceil(wg), 1, 1]
    }

    /// Record and submit one dispatch.
    pub fn run(&self, device: &Wgpu, bind_group: &wgpu::BindGroup, workgroups: [u32; 3]) {
        let mut encoder =
            device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("luma-wgpu kernel encoder") });

        {
            let mut pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("luma-wgpu kernel pass"), timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
        }

        device.queue().submit(Some(encoder.finish()));
    }
}
