mod allocator;
mod kernels;
mod ops;
mod storage;
mod dispatch;

pub use allocator::{CpuAllocator, PoolAllocator, SystemAllocator};
pub use storage::*;
use std::fmt;
use std::sync::{Arc, RwLock};
use luma_tensor::Device;

/// The CPU device.
///
/// Carries a pluggable [`CpuAllocator`] shared across clones (the `Arc`), so
/// tensors created through any clone of the same device see the same
/// allocator. The default is [`SystemAllocator`] — plain allocation, no
/// pooling — so behaviour matches the pre-allocator device exactly.
#[derive(Clone)]
pub struct Cpu {
    allocator: Arc<RwLock<dyn CpuAllocator>>,
}

impl Default for Cpu {
    fn default() -> Self {
        Self { allocator: Arc::new(RwLock::new(SystemAllocator)) }
    }
}

impl fmt::Debug for Cpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cpu").finish_non_exhaustive()
    }
}

impl Cpu {
    /// Create a CPU device with a custom allocator (e.g. a pooling allocator
    /// for inference workloads).
    pub fn with_allocator(allocator: impl CpuAllocator) -> Self {
        Self { allocator: Arc::new(RwLock::new(allocator)) }
    }

    /// 与调用方共享 allocator 句柄（测试需要读计数/配置时用）。
    pub fn with_allocator_shared(allocator: Arc<RwLock<dyn CpuAllocator>>) -> Self {
        Self { allocator }
    }

    /// The allocator backing storage allocation on this device.
    pub fn allocator(&self) -> &Arc<RwLock<dyn CpuAllocator>> {
        &self.allocator
    }

    /// 从 allocator 拿一块并用迭代器填满（`.collect()` 的路由版）。内核层唯一
    /// 的分配入口——池化 allocator 从这里拦截所有计算类分配的复用。
    pub(crate) fn collect_alloc<U: allocator::AllocVec>(&self, iter: impl IntoIterator<Item = U>) -> Vec<U> {
        let guard = self.allocator.read().expect("allocator poisoned");
        allocator::collect_alloc(&*guard, iter)
    }

    /// `vec![value; n]` 的路由版。
    pub(crate) fn fill_alloc<U: allocator::AllocVec + Copy>(&self, n: usize, value: U) -> Vec<U> {
        let guard = self.allocator.read().expect("allocator poisoned");
        allocator::fill_alloc(&*guard, n, value)
    }

    /// 裸分配（不填内容）：push 循环等手动填写的内核用。
    pub(crate) fn alloc_vec<U: allocator::AllocVec>(&self, n: usize) -> Vec<U> {
        let guard = self.allocator.read().expect("allocator poisoned");
        U::alloc_vec(&*guard, n)
    }
}

impl Device for Cpu {
    type FloatStorage = CpuFloatStorage;
    type IntStorage = CpuIntStorage;
    type BoolStorage = CpuBoolStorage;

    fn name(&self) -> String {
        "cpu".into()
    }
}
