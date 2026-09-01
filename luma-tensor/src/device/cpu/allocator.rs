//! CPU memory allocators: pluggable backing-buffer allocation for CPU storage.
//!
//! The contract is `alloc_*` hands out a fresh `Vec`, `dealloc_*` takes it
//! back — what "back" means is the implementation's business: [`SystemAllocator`]
//! drops it, a future pool allocator recycles it. The trait is deliberately
//! concrete per element type (dyn-safe, no generics) and device-agnostic
//! (`Vec` is a CPU concept; a CUDA device would define its own allocator).

use std::vec::Vec;

/// 元素类型 → allocator 具体方法的静态分派：泛型 kernel 里用 `U::alloc_vec`
/// 把"给 T 分配"落到 `alloc_f32`/`alloc_f64`/... 上。
pub trait AllocVec: Sized {
    fn alloc_vec(alloc: &dyn CpuAllocator, n: usize) -> Vec<Self>;
}

impl AllocVec for f32 {
    fn alloc_vec(alloc: &dyn CpuAllocator, n: usize) -> Vec<Self> {
        alloc.alloc_f32(n)
    }
}
impl AllocVec for f64 {
    fn alloc_vec(alloc: &dyn CpuAllocator, n: usize) -> Vec<Self> {
        alloc.alloc_f64(n)
    }
}
impl AllocVec for i32 {
    fn alloc_vec(alloc: &dyn CpuAllocator, n: usize) -> Vec<Self> {
        alloc.alloc_i32(n)
    }
}
impl AllocVec for u32 {
    fn alloc_vec(alloc: &dyn CpuAllocator, n: usize) -> Vec<Self> {
        alloc.alloc_u32(n)
    }
}
impl AllocVec for u8 {
    fn alloc_vec(alloc: &dyn CpuAllocator, n: usize) -> Vec<Self> {
        alloc.alloc_u8(n)
    }
}
impl AllocVec for bool {
    fn alloc_vec(alloc: &dyn CpuAllocator, n: usize) -> Vec<Self> {
        alloc.alloc_bool(n)
    }
}
impl AllocVec for i64 {
    fn alloc_vec(alloc: &dyn CpuAllocator, n: usize) -> Vec<Self> {
        alloc.alloc_i64(n)
    }
}
impl AllocVec for usize {
    fn alloc_vec(alloc: &dyn CpuAllocator, n: usize) -> Vec<Self> {
        alloc.alloc_usize(n)
    }
}

/// `.collect()` 的 allocator 路由版：先按 size_hint 上界从 allocator 拿块
/// （这些索引迭代器的 size_hint 是精确的，保证池分桶命中），再 extend 填满。
pub(crate) fn collect_alloc<U: AllocVec>(alloc: &dyn CpuAllocator, iter: impl IntoIterator<Item = U>) -> Vec<U> {
    let iter = iter.into_iter();
    let n = iter.size_hint().1.unwrap_or_else(|| iter.size_hint().0);
    let mut v = U::alloc_vec(alloc, n);
    v.extend(iter);
    v
}

/// `vec![value; n]` 的 allocator 路由版。
pub(crate) fn fill_alloc<U: AllocVec + Copy>(alloc: &dyn CpuAllocator, n: usize, value: U) -> Vec<U> {
    let mut v = U::alloc_vec(alloc, n);
    v.resize(n, value);
    v
}

/// Allocator for CPU tensor storage.
///
/// Six element types × alloc/dealloc. `dealloc_*` takes ownership of the
/// `Vec` and decides its fate.
pub trait CpuAllocator: Send + Sync + 'static {
    fn alloc_f32(&self, n: usize) -> Vec<f32>;
    fn alloc_f64(&self, n: usize) -> Vec<f64>;
    fn alloc_i32(&self, n: usize) -> Vec<i32>;
    fn alloc_u32(&self, n: usize) -> Vec<u32>;
    fn alloc_u8(&self, n: usize) -> Vec<u8>;
    fn alloc_bool(&self, n: usize) -> Vec<bool>;
    fn alloc_i64(&self, n: usize) -> Vec<i64>;
    fn alloc_usize(&self, n: usize) -> Vec<usize>;

    fn dealloc_f32(&self, v: Vec<f32>);
    fn dealloc_f64(&self, v: Vec<f64>);
    fn dealloc_i32(&self, v: Vec<i32>);
    fn dealloc_u32(&self, v: Vec<u32>);
    fn dealloc_u8(&self, v: Vec<u8>);
    fn dealloc_bool(&self, v: Vec<bool>);
    fn dealloc_i64(&self, v: Vec<i64>);
    fn dealloc_usize(&self, v: Vec<usize>);
}

/// Default allocator: plain allocation, `dealloc` drops.
///
/// Semantically identical to the pre-allocator behaviour — the whole point of
/// this phase is that swapping the allocator changes nothing observable.
#[derive(Debug, Default)]
pub struct SystemAllocator;

impl CpuAllocator for SystemAllocator {
    fn alloc_f32(&self, n: usize) -> Vec<f32> {
        Vec::with_capacity(n)
    }
    fn alloc_f64(&self, n: usize) -> Vec<f64> {
        Vec::with_capacity(n)
    }
    fn alloc_i32(&self, n: usize) -> Vec<i32> {
        Vec::with_capacity(n)
    }
    fn alloc_u32(&self, n: usize) -> Vec<u32> {
        Vec::with_capacity(n)
    }
    fn alloc_u8(&self, n: usize) -> Vec<u8> {
        Vec::with_capacity(n)
    }
    fn alloc_bool(&self, n: usize) -> Vec<bool> {
        Vec::with_capacity(n)
    }
    fn alloc_i64(&self, n: usize) -> Vec<i64> {
        Vec::with_capacity(n)
    }
    fn alloc_usize(&self, n: usize) -> Vec<usize> {
        Vec::with_capacity(n)
    }

    fn dealloc_f32(&self, _v: Vec<f32>) {}
    fn dealloc_f64(&self, _v: Vec<f64>) {}
    fn dealloc_i32(&self, _v: Vec<i32>) {}
    fn dealloc_u32(&self, _v: Vec<u32>) {}
    fn dealloc_u8(&self, _v: Vec<u8>) {}
    fn dealloc_bool(&self, _v: Vec<bool>) {}
    fn dealloc_i64(&self, _v: Vec<i64>) {}
    fn dealloc_usize(&self, _v: Vec<usize>) {}
}
