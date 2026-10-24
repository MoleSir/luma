use crate::allocator::align4;
use crate::{Wgpu, WgpuError};
use luma_tensor::dtype::DTypeKindT;
use luma_tensor::{Bool, BoolDType, DType, Float, FloatDType, Int, IntDType, Storage};
use std::sync::mpsc;

/// Common buffer access for the generic wgpu launch code.
pub trait WgpuStorage: Sized {
    type DType: DTypeKindT;

    fn wgpu_dtype(&self) -> Self::DType;
    fn wgpu_device(&self) -> &Wgpu;
    fn wgpu_buffer(&self) -> &wgpu::Buffer;
    fn wrap_wgpu(buffer: wgpu::Buffer, dtype: Self::DType, device: &Wgpu) -> Self;

    /// Bytes per logical element in the wgpu buffer.
    fn wgpu_elem_bytes(&self) -> usize {
        self.wgpu_dtype().size_of()
    }
}

// =============================================================================================
//                      Float Storage
// =============================================================================================

pub struct WgpuFloatStorage {
    pub buffer: WgpuFloatBuffer,
    pub device: Wgpu,
}

pub enum WgpuFloatBuffer {
    /// Native WGSL `f32` storage.
    F32(wgpu::Buffer),
    /// Optional via WebGPU `shader-f16` (`enable f16;`).
    F16(wgpu::Buffer),
}

impl WgpuFloatStorage {
    pub(crate) fn new(buffer: wgpu::Buffer, dtype: FloatDType, device: &Wgpu) -> Self {
        let buffer = match dtype {
            FloatDType::F32 => WgpuFloatBuffer::F32(buffer),
            FloatDType::F16 => WgpuFloatBuffer::F16(buffer),
            FloatDType::F64 | FloatDType::BF16 => unreachable!("unsupported wgpu float dtype"),
        };
        WgpuFloatStorage { buffer, device: device.clone() }
    }

    pub(crate) fn upload_bytes(device: &Wgpu, dtype: FloatDType, data: &[u8]) -> luma_tensor::Result<Self> {
        if dtype != FloatDType::F16 && dtype != FloatDType::F32 {
            return Err(WgpuError::UnsupportedDType(dtype.into(), "wgpu float storage").into());
        }
        if dtype == FloatDType::F16 && !device.supports_f16() {
            return Err(WgpuError::UnsupportedFeature("shader-f16").into());
        }

        // `Queue::write_buffer` requires a 4-byte-multiple data length, so
        // f16/u8 payloads may need zero padding. The buffer itself is also
        // padded by the allocator.
        let padded = align4(data.len());
        let mut bytes = data.to_vec();
        bytes.resize(padded, 0);
        let buffer = device.alloc_storage_output(padded)?;
        device.queue().write_buffer(&buffer, 0, &bytes);
        Ok(WgpuFloatStorage::new(buffer, dtype, device))
    }

    pub fn dtype(&self) -> DType {
        match &self.buffer {
            WgpuFloatBuffer::F32(_) => DType::F32,
            WgpuFloatBuffer::F16(_) => DType::F16,
        }
    }

    pub fn len(&self) -> usize {
        (match &self.buffer {
            WgpuFloatBuffer::F32(buffer) | WgpuFloatBuffer::F16(buffer) => buffer.size(),
        } / self.dtype().size_in_bytes() as u64) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Storage<Wgpu, Float> for WgpuFloatStorage {
    fn dtype(&self) -> FloatDType {
        match &self.buffer {
            WgpuFloatBuffer::F32(_) => FloatDType::F32,
            WgpuFloatBuffer::F16(_) => FloatDType::F16,
        }
    }

    fn device(&self) -> &Wgpu {
        &self.device
    }
}

impl WgpuStorage for WgpuFloatStorage {
    type DType = FloatDType;

    fn wgpu_dtype(&self) -> Self::DType {
        Storage::<Wgpu, Float>::dtype(self)
    }

    fn wgpu_device(&self) -> &Wgpu {
        &self.device
    }

    fn wgpu_buffer(&self) -> &wgpu::Buffer {
        match &self.buffer {
            WgpuFloatBuffer::F32(buffer) | WgpuFloatBuffer::F16(buffer) => buffer,
        }
    }

    fn wrap_wgpu(buffer: wgpu::Buffer, dtype: Self::DType, device: &Wgpu) -> Self {
        WgpuFloatStorage::new(buffer, dtype, device)
    }
}

// =============================================================================================
//                      Int Storage
// =============================================================================================

pub struct WgpuIntStorage {
    pub buffer: WgpuIntBuffer,
    pub device: Wgpu,
}

pub enum WgpuIntBuffer {
    I32(wgpu::Buffer),
    U32(wgpu::Buffer),
}

impl WgpuIntStorage {
    pub(crate) fn new(buffer: wgpu::Buffer, dtype: IntDType, device: &Wgpu) -> Self {
        let buffer = match dtype {
            IntDType::I32 => WgpuIntBuffer::I32(buffer),
            IntDType::U32 => WgpuIntBuffer::U32(buffer),
            IntDType::U8 => unreachable!("wgpu does not support u8 storage"),
        };
        WgpuIntStorage { buffer, device: device.clone() }
    }

    pub(crate) fn upload_bytes(device: &Wgpu, dtype: IntDType, data: &[u8]) -> luma_tensor::Result<Self> {
        if dtype != IntDType::I32 && dtype != IntDType::U32 {
            return Err(WgpuError::UnsupportedDType(dtype.into(), "wgpu int storage").into());
        }

        let padded = align4(data.len());
        let mut bytes = data.to_vec();
        bytes.resize(padded, 0);
        let buffer = device.alloc_storage_output(padded)?;
        device.queue().write_buffer(&buffer, 0, &bytes);
        Ok(WgpuIntStorage::new(buffer, dtype, device))
    }

    pub fn dtype(&self) -> DType {
        match &self.buffer {
            WgpuIntBuffer::I32(_) => DType::I32,
            WgpuIntBuffer::U32(_) => DType::U32,
        }
    }

    pub fn len(&self) -> usize {
        (match &self.buffer {
            WgpuIntBuffer::I32(buffer) | WgpuIntBuffer::U32(buffer) => buffer.size(),
        } / 4) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Storage<Wgpu, Int> for WgpuIntStorage {
    fn dtype(&self) -> IntDType {
        match &self.buffer {
            WgpuIntBuffer::I32(_) => IntDType::I32,
            WgpuIntBuffer::U32(_) => IntDType::U32,
        }
    }

    fn device(&self) -> &Wgpu {
        &self.device
    }
}

impl WgpuStorage for WgpuIntStorage {
    type DType = IntDType;

    fn wgpu_dtype(&self) -> Self::DType {
        Storage::<Wgpu, Int>::dtype(self)
    }

    fn wgpu_device(&self) -> &Wgpu {
        &self.device
    }

    fn wgpu_buffer(&self) -> &wgpu::Buffer {
        match &self.buffer {
            WgpuIntBuffer::I32(buffer) | WgpuIntBuffer::U32(buffer) => buffer,
        }
    }

    fn wrap_wgpu(buffer: wgpu::Buffer, dtype: Self::DType, device: &Wgpu) -> Self {
        WgpuIntStorage::new(buffer, dtype, device)
    }
}

// =============================================================================================
//                      Bool Storage
// =============================================================================================

pub struct WgpuBoolStorage {
    pub buffer: wgpu::Buffer,
    pub device: Wgpu,
}

impl WgpuBoolStorage {
    pub(crate) fn new(buffer: wgpu::Buffer, device: &Wgpu) -> Self {
        Self { buffer, device: device.clone() }
    }

    /// Logical bools are stored as one `u32` per element.
    pub fn len(&self) -> usize {
        (self.buffer.size() / 4) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Storage<Wgpu, Bool> for WgpuBoolStorage {
    fn dtype(&self) -> BoolDType {
        BoolDType::Bool
    }

    fn device(&self) -> &Wgpu {
        &self.device
    }
}

impl WgpuStorage for WgpuBoolStorage {
    type DType = BoolDType;

    fn wgpu_dtype(&self) -> Self::DType {
        BoolDType::Bool
    }

    fn wgpu_device(&self) -> &Wgpu {
        &self.device
    }

    fn wgpu_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    fn wrap_wgpu(buffer: wgpu::Buffer, _dtype: Self::DType, device: &Wgpu) -> Self {
        WgpuBoolStorage::new(buffer, device)
    }

    fn wgpu_elem_bytes(&self) -> usize {
        4
    }
}

/// Copy a storage buffer back to the host, blocking until the map completes.
///
/// `byte_offset` is the layout start offset in bytes; `logical_bytes` is the
/// tensor's real byte size. The source offset is rounded down to wgpu's 4-byte
/// copy alignment and the extra prefix is skipped after mapping.
pub(crate) fn readback_bytes(device: &Wgpu, src: &wgpu::Buffer, byte_offset: usize, logical_bytes: usize) -> luma_tensor::Result<Vec<u8>> {
    if logical_bytes == 0 {
        return Ok(Vec::new());
    }

    let skip = byte_offset % 4;
    let src_offset = (byte_offset - skip) as u64;
    let copy_bytes = align4(logical_bytes + skip);
    let staging = device.alloc(copy_bytes as u64, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)?;

    let mut encoder = device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("luma-wgpu readback") });
    encoder.copy_buffer_to_buffer(src, src_offset, &staging, 0, copy_bytes as u64);
    device.queue().submit(Some(encoder.finish()));

    let (tx, rx) = mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    device.device().poll(wgpu::PollType::wait_indefinitely()).map_err(|e| WgpuError::Map(e.to_string()))?;

    rx.recv().map_err(|e| WgpuError::Map(e.to_string()))?.map_err(|e| WgpuError::Map(e.to_string()))?;

    let view = staging.slice(..).get_mapped_range().map_err(|e| WgpuError::Map(e.to_string()))?;
    let bytes = view.to_vec();
    drop(view);
    staging.unmap();

    Ok(bytes[skip..skip + logical_bytes].to_vec())
}
