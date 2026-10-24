//! The wgpu device handle.
//!
//! Mirrors `luma-cuda::Cuda`: a cheap-to-clone handle (`Arc<WgpuInner>`) that
//! owns the adapter, logical device, queue, allocator, and pipeline cache.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use luma_tensor::Device;

use crate::allocator::WgpuAllocator;
use crate::shaders::{PipelineCache, Shader, ShaderKind};
use crate::{WgpuBoolStorage, WgpuError, WgpuFloatStorage, WgpuIntStorage, WgpuResult};

#[derive(Clone)]
pub struct Wgpu(pub(crate) Arc<WgpuInner>);

pub struct WgpuInner {
    pub(crate) ordinal: usize,
    pub(crate) instance: wgpu::Instance,
    pub(crate) adapter: wgpu::Adapter,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) allocator: WgpuAllocator,
    pub(crate) pipelines: PipelineCache,
    pub(crate) rng_seed: AtomicU32,
}

impl Wgpu {
    pub fn new(ordinal: usize) -> WgpuResult<Self> {
        let instance: wgpu::Instance = wgpu::Instance::default();
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        let Some(adapter) = adapters.into_iter().nth(ordinal) else {
            return Err(WgpuError::Init(format!("no wgpu adapter at ordinal {ordinal}")));
        };

        // Only request shader-f16 when the adapter actually offers it; the
        // f16 shaders are unavailable otherwise.
        let mut required_features = wgpu::Features::empty();
        if adapter.features().contains(wgpu::Features::SHADER_F16) {
            required_features |= wgpu::Features::SHADER_F16;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("luma-wgpu"),
            required_features,
            required_limits: wgpu::Limits::defaults(),
            ..Default::default()
        }))
        .map_err(|e| WgpuError::Init(e.to_string()))?;

        Ok(Wgpu(Arc::new(WgpuInner {
            ordinal,
            instance,
            adapter,
            device,
            queue,
            allocator: WgpuAllocator::new(),
            pipelines: PipelineCache::default(),
            // Same base seed as the CUDA backend. The counter makes every
            // random construction use a fresh seed while keeping runs
            // reproducible.
            rng_seed: AtomicU32::new(299_792_458),
        })))
    }

    pub fn instance(&self) -> &wgpu::Instance {
        &self.0.instance
    }

    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.0.adapter
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.0.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.0.queue
    }

    pub fn ordinal(&self) -> usize {
        self.0.ordinal
    }

    /// Whether this device can run `enable f16;` shaders.
    pub fn supports_f16(&self) -> bool {
        self.device().features().contains(wgpu::Features::SHADER_F16)
    }

    pub fn allocator(&self) -> &WgpuAllocator {
        &self.0.allocator
    }

    pub fn pipelines(&self) -> &PipelineCache {
        &self.0.pipelines
    }

    pub(crate) fn next_random_seed(&self) -> u32 {
        self.0.rng_seed.fetch_add(1, Ordering::Relaxed)
    }

    /// Block until all previously submitted GPU work has completed.
    pub fn synchronize(&self) -> WgpuResult<()> {
        self.device().poll(wgpu::PollType::wait_indefinitely()).map_err(|e| WgpuError::Map(e.to_string()))?;
        Ok(())
    }

    #[inline]
    pub fn alloc(&self, size: u64, usage: wgpu::BufferUsages) -> WgpuResult<wgpu::Buffer> {
        self.allocator().alloc(self.device(), size, usage)
    }

    #[inline]
    pub fn alloc_storage_output(&self, size: usize) -> WgpuResult<wgpu::Buffer> {
        self.alloc(size as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST)
    }

    #[inline]
    pub fn alloc_uniform(&self, size: usize) -> WgpuResult<wgpu::Buffer> {
        self.alloc(size as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST)
    }

    /// Upload one `Pod` value into a fresh uniform buffer.
    /// 1. alloc uniform buffer / 2. write buffer with &[u8] of value
    pub fn upload_uniform<T: bytemuck::Pod>(&self, value: &T) -> WgpuResult<wgpu::Buffer> {
        let buffer = self.alloc_uniform(std::mem::size_of::<T>())?;
        self.queue().write_buffer(&buffer, 0, bytemuck::bytes_of(value));
        Ok(buffer)
    }

    /// Upload a `u32` metadata array into a read-only storage buffer.
    ///
    /// Shape/stride metadata is kept out of uniform buffers because WGSL
    /// uniform arrays have a 16-byte element stride; a storage buffer has the
    /// natural 4-byte stride and no fixed maximum rank.
    pub fn upload_storage_u32(&self, data: &[u32]) -> WgpuResult<wgpu::Buffer> {
        let bytes = data.len().max(1) * std::mem::size_of::<u32>();
        let buffer = self.alloc(bytes as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST)?;
        if !data.is_empty() {
            self.queue().write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        }
        Ok(buffer)
    }

    /// Get a cached shader, compiling it on first use.
    pub fn shader(&self, key: ShaderKind) -> WgpuResult<Arc<Shader>> {
        if key.requires_f16() && !self.supports_f16() {
            return Err(WgpuError::UnsupportedFeature("shader-f16"));
        }
        self.pipelines().get(self, key)
    }
}

impl fmt::Debug for Wgpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Wgpu").field("ordinal", &self.0.ordinal).finish_non_exhaustive()
    }
}

impl Default for Wgpu {
    fn default() -> Self {
        Wgpu::new(0).expect("failed to create default wgpu device (adapter 0)")
    }
}

impl Device for Wgpu {
    type FloatStorage = WgpuFloatStorage;
    type IntStorage = WgpuIntStorage;
    type BoolStorage = WgpuBoolStorage;

    fn name(&self) -> String {
        format!("wgpu:{}", self.0.ordinal)
    }

    fn same_device(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
