//! CPU memory allocators: pluggable backing-buffer allocation for CPU storage.
//!
//! The contract is `alloc_*` hands out a fresh `Vec`, `dealloc_*` takes it
//! back — what "back" means is the implementation's business: [`SystemAllocator`]
//! drops it, a future pool allocator recycles it. The trait is deliberately
//! concrete per element type (dyn-safe, no generics) and device-agnostic
//! (`Vec` is a CPU concept; a CUDA device would define its own allocator).

use std::collections::HashMap;
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
    /// 测试/配置探针：downcast 到具体实现。
    fn as_any(&self) -> &dyn std::any::Any;

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
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

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

/// 分桶内存池：同 capacity 的空闲块复用。
///
/// alloc：桶空才系统分配（计数 +1）；dealloc：clear() 后入桶，桶超上限则
/// drop（峰值保护——池保留的内存不归还系统，上限防止无限涨）。
///
/// `system_allocs` 是测试探针：稳态推理（死槽清理后）第二次 run 应为 0。
#[derive(Debug, Default)]
pub struct PoolAllocator {
    f32: std::sync::Mutex<HashMap<usize, Vec<Vec<f32>>>>,
    f64: std::sync::Mutex<HashMap<usize, Vec<Vec<f64>>>>,
    i32: std::sync::Mutex<HashMap<usize, Vec<Vec<i32>>>>,
    u32: std::sync::Mutex<HashMap<usize, Vec<Vec<u32>>>>,
    u8: std::sync::Mutex<HashMap<usize, Vec<Vec<u8>>>>,
    bool: std::sync::Mutex<HashMap<usize, Vec<Vec<bool>>>>,
    i64: std::sync::Mutex<HashMap<usize, Vec<Vec<i64>>>>,
    usize_: std::sync::Mutex<HashMap<usize, Vec<Vec<usize>>>>,
    /// 池 miss、真正走系统分配的次数（测试探针）。
    system_allocs: std::sync::atomic::AtomicUsize,
    /// 每桶保留上限；0 = 无上限。
    max_per_bucket: usize,
}

impl PoolAllocator {
    pub fn new() -> Self {
        Self::default()
    }

    /// 每桶保留上限（0 = 无上限）。
    pub fn with_max_per_bucket(max: usize) -> Self {
        let mut p = Self::default();
        p.max_per_bucket = max;
        p
    }

    pub fn system_allocs(&self) -> usize {
        self.system_allocs.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn take<T>(bucket: &mut HashMap<usize, Vec<Vec<T>>>, n: usize, counter: &std::sync::atomic::AtomicUsize) -> Vec<T> {
        // 桶键 = 实际 capacity；`Vec::with_capacity(n)` 的容量会被分配器字节
        // 对齐圆整（4 元素 f32 请求可能得 6 容量），所以从 n 向上小范围探测——
        // 任何 capacity >= n 的块都满足请求。向上探测成本：稳态通常 1-2 次
        // HashMap miss，可忽略。
        for c in n..=n.saturating_add(64) {
            if let Some(v) = bucket.get_mut(&c).and_then(|b| b.pop()) {
                return v;
            }
        }
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Vec::with_capacity(n)
    }

    fn give<T>(bucket: &mut HashMap<usize, Vec<Vec<T>>>, mut v: Vec<T>, max: usize) {
        // 关键：入池前 clear——池里的 Vec 必须 len == 0。否则下次弹出时 kernel
        // 的 extend 会"追加"到旧数据后面（错误内容 + realloc），或 resize 原样
        // 保留旧值（fill 路径）。clear 保留 capacity，零成本。
        v.clear();
        let cap = v.capacity();
        let free = bucket.entry(cap).or_default();
        if max > 0 && free.len() >= max {
            return; // 超上限：drop（v 在此结束）
        }
        free.push(v);
    }
}

macro_rules! pool_alloc {
    ($($field:ident: $t:ty => $alloc:ident / $dealloc:ident),* $(,)?) => {
        impl CpuAllocator for PoolAllocator {
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
            $(fn $alloc(&self, n: usize) -> Vec<$t> {
                Self::take(&mut self.$field.lock().expect("pool poisoned"), n, &self.system_allocs)
            }
            fn $dealloc(&self, v: Vec<$t>) {
                Self::give(&mut self.$field.lock().expect("pool poisoned"), v, self.max_per_bucket);
            })*
        }
    };
}

pool_alloc! {
    f32: f32 => alloc_f32 / dealloc_f32,
    f64: f64 => alloc_f64 / dealloc_f64,
    i32: i32 => alloc_i32 / dealloc_i32,
    u32: u32 => alloc_u32 / dealloc_u32,
    u8: u8 => alloc_u8 / dealloc_u8,
    bool: bool => alloc_bool / dealloc_bool,
    i64: i64 => alloc_i64 / dealloc_i64,
    usize_: usize => alloc_usize / dealloc_usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 同尺寸 alloc → dealloc → alloc：第二次从池拿，系统分配只发生一次。
    #[test]
    fn reuses_freed_block() {
        let p = PoolAllocator::new();
        let v = p.alloc_f32(64);
        assert_eq!(v.capacity(), 64);
        p.dealloc_f32(v); // 池只接收显式 dealloc（Storage Drop 的路由入口）
        let v2 = p.alloc_f32(64);
        assert_eq!(v2.capacity(), 64);
        assert_eq!(p.system_allocs(), 1, "第二次分配应命中池");
        p.dealloc_f32(v2);
    }

    /// 不同尺寸不共享桶，各自系统分配。
    #[test]
    fn different_sizes_separate_buckets() {
        let p = PoolAllocator::new();
        drop(p.alloc_f32(64));
        drop(p.alloc_f32(128));
        assert_eq!(p.system_allocs(), 2);
    }

    /// 每桶上限：超过后 dealloc 直接释放（不保留）。
    #[test]
    fn bucket_cap_limits_retention() {
        let p = PoolAllocator::with_max_per_bucket(2);
        // 先攒 5 块（池空 → 5 次系统分配），再全部归还 → 桶上限 2 只留 2 块
        let mut held = Vec::new();
        for _ in 0..5 {
            held.push(p.alloc_f32(32));
        }
        assert_eq!(p.system_allocs(), 5, "池空时每次都系统分配");
        for v in held {
            p.dealloc_f32(v);
        }
        // 同时持有 5 块（不归还）：桶里只有 2 → 2 命中 + 3 系统分配
        let mut held2 = Vec::new();
        for _ in 0..5 {
            held2.push(p.alloc_f32(32));
        }
        assert_eq!(p.system_allocs(), 5 + 3, "桶上限 2：5 块里 2 命中 3 系统");
        for v in held2 {
            p.dealloc_f32(v);
        }
    }

    /// dealloc 后容量保持（clear 不清 capacity），桶按 capacity 命中。
    #[test]
    fn capacity_preserved_after_dealloc() {
        let p = PoolAllocator::new();
        p.dealloc_f32(p.alloc_f32(64));
        let v = p.alloc_f32(64);
        assert_eq!(v.capacity(), 64);
        assert_eq!(v.len(), 0, "入池前 clear，len 应为 0");
        p.dealloc_f32(v);
    }
}
