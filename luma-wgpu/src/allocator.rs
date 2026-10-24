use crate::WgpuResult;

/// Round a byte count up to wgpu's 4-byte copy alignment
/// (`COPY_BUFFER_ALIGNMENT`).
#[inline]
pub(crate) fn align4(bytes: usize) -> usize {
    bytes.div_ceil(4) * 4
}

pub struct WgpuAllocator {}

impl Default for WgpuAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl WgpuAllocator {
    pub fn new() -> Self {
        Self {}
    }

    pub fn alloc(&self, device: &wgpu::Device, size: u64, usage: wgpu::BufferUsages) -> WgpuResult<wgpu::Buffer> {
        // Buffers are always allocated at least 4 bytes and 4-byte aligned;
        // callers pass logical byte sizes.
        let size = align4(size.max(1) as usize) as u64;

        let buffer =
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("luma-wgpu buffer"), size, usage, mapped_at_creation: false });
        Ok(buffer)
    }
}
