pub mod allocator;
pub mod kernels;
mod ops;
mod storage;
pub use allocator::{CpuAllocator, SystemAllocator};
pub use storage::*;

use std::fmt;
use std::sync::{Arc, RwLock};

use crate::{DType, Device};

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

// ============================================================================
// Dispatch macros: map a storage enum to its concrete element type, run the
// generic kernel, and re-wrap the result.
// ============================================================================

/// Run `$body` with `$data` bound to the inner `&Vec<t>` of a `CpuFloatStorage`,
/// then wrap the returned `Vec<t>` back into a `CpuFloatStorage`.
///
/// The rebuild preserves the storage's device instance (result tensors inherit
/// it via `from_storage`).
#[macro_export]
macro_rules! dispatch_float {
    ($storage:expr, |$data:ident| $body:expr) => {
        match $storage {
            $crate::CpuFloatStorage::F32($data, _) => $crate::CpuFloatStorage::F32($body, $storage.device().clone()),
            $crate::CpuFloatStorage::F64($data, _) => $crate::CpuFloatStorage::F64($body, $storage.device().clone()),
        }
    };
}

/// Like [`dispatch_float`] but `$body` yields a non-storage value (e.g. Vec<bool>).
#[macro_export]
macro_rules! dispatch_float_raw {
    ($storage:expr, |$data:ident| $body:expr) => {
        match $storage {
            $crate::CpuFloatStorage::F32($data, _) => $body,
            $crate::CpuFloatStorage::F64($data, _) => $body,
        }
    };
}

/// Dispatch two float storages of the SAME variant; errors on mismatch.
///
/// The result inherits the LHS's device (the `&self` storage in op code).
#[macro_export]
macro_rules! dispatch_float2 {
    ($lhs:expr, $rhs:expr, $op:literal, |$a:ident, $b:ident| $body:expr) => {
        match ($lhs, $rhs) {
            ($crate::CpuFloatStorage::F32($a, _), $crate::CpuFloatStorage::F32($b, _)) => {
                Ok($crate::CpuFloatStorage::F32($body, $lhs.device().clone()))
            }
            ($crate::CpuFloatStorage::F64($a, _), $crate::CpuFloatStorage::F64($b, _)) => {
                Ok($crate::CpuFloatStorage::F64($body, $lhs.device().clone()))
            }
            (l, r) => Err($crate::Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: $op }),
        }
    };
}

/// Dispatch two float storages of the SAME variant, `$body` yields a raw value.
#[macro_export]
macro_rules! dispatch_float2_raw {
    ($lhs:expr, $rhs:expr, $op:literal, |$a:ident, $b:ident| $body:expr) => {
        match ($lhs, $rhs) {
            ($crate::CpuFloatStorage::F32($a, _), $crate::CpuFloatStorage::F32($b, _)) => Ok($body),
            ($crate::CpuFloatStorage::F64($a, _), $crate::CpuFloatStorage::F64($b, _)) => Ok($body),
            (l, r) => Err($crate::Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: $op }),
        }
    };
}

#[macro_export]
macro_rules! dispatch_int {
    ($storage:expr, |$data:ident| $body:expr) => {
        match $storage {
            $crate::CpuIntStorage::I32($data, _) => $crate::CpuIntStorage::I32($body, $storage.device().clone()),
            $crate::CpuIntStorage::U32($data, _) => $crate::CpuIntStorage::U32($body, $storage.device().clone()),
            $crate::CpuIntStorage::U8($data, _) => $crate::CpuIntStorage::U8($body, $storage.device().clone()),
        }
    };
}

#[macro_export]
macro_rules! dispatch_int_raw {
    ($storage:expr, |$data:ident| $body:expr) => {
        match $storage {
            $crate::CpuIntStorage::I32($data, _) => $body,
            $crate::CpuIntStorage::U32($data, _) => $body,
            $crate::CpuIntStorage::U8($data, _) => $body,
        }
    };
}

#[macro_export]
macro_rules! dispatch_int2 {
    ($lhs:expr, $rhs:expr, $op:literal, |$a:ident, $b:ident| $body:expr) => {
        match ($lhs, $rhs) {
            ($crate::CpuIntStorage::I32($a, _), $crate::CpuIntStorage::I32($b, _)) => {
                Ok($crate::CpuIntStorage::I32($body, $lhs.device().clone()))
            }
            ($crate::CpuIntStorage::U32($a, _), $crate::CpuIntStorage::U32($b, _)) => {
                Ok($crate::CpuIntStorage::U32($body, $lhs.device().clone()))
            }
            ($crate::CpuIntStorage::U8($a, _), $crate::CpuIntStorage::U8($b, _)) => {
                Ok($crate::CpuIntStorage::U8($body, $lhs.device().clone()))
            }
            (l, r) => Err($crate::Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: $op }),
        }
    };
}

#[macro_export]
macro_rules! dispatch_int2_raw {
    ($lhs:expr, $rhs:expr, $op:literal, |$a:ident, $b:ident| $body:expr) => {
        match ($lhs, $rhs) {
            ($crate::CpuIntStorage::I32($a, _), $crate::CpuIntStorage::I32($b, _)) => Ok($body),
            ($crate::CpuIntStorage::U32($a, _), $crate::CpuIntStorage::U32($b, _)) => Ok($body),
            ($crate::CpuIntStorage::U8($a, _), $crate::CpuIntStorage::U8($b, _)) => Ok($body),
            (l, r) => Err($crate::Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: $op }),
        }
    };
}

/// Read an int storage's elements (in `layout` order) as `usize`, mapping the
/// dtype's MAX sentinel to `kernels::indexing::PAD`. Used by indexing kernels.
pub(crate) fn int_ids_as_usize(storage: &CpuIntStorage, layout: &crate::Layout) -> Vec<usize> {
    use kernels::element::{CpuInt, CpuNum};
    use kernels::indexing::PAD;
    macro_rules! collect {
        ($data:expr) => {
            layout
                .storage_indices()
                .map(|i| {
                    let v = $data[i];
                    if v == CpuInt::MAX { PAD } else { v.to_usize() }
                })
                .collect()
        };
    }
    match storage {
        CpuIntStorage::I32(d, _) => collect!(d),
        CpuIntStorage::U32(d, _) => collect!(d),
        CpuIntStorage::U8(d, _) => collect!(d),
    }
}

/// Build an int storage of the given dtype from `usize` indices.
pub(crate) fn usize_to_int_storage(data: &[usize], dtype: DType, device: &Cpu) -> CpuIntStorage {
    match dtype {
        DType::I32 => CpuIntStorage::I32(data.iter().map(|&v| v as i32).collect(), device.clone()),
        DType::U32 => CpuIntStorage::U32(data.iter().map(|&v| v as u32).collect(), device.clone()),
        DType::U8 => CpuIntStorage::U8(data.iter().map(|&v| v as u8).collect(), device.clone()),
        _ => CpuIntStorage::U32(data.iter().map(|&v| v as u32).collect(), device.clone()),
    }
}
