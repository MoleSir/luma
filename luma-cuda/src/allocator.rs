use std::{
    cmp::max, collections::BTreeMap, error::Error, fmt, marker::PhantomData, mem::size_of,
    sync::{Arc, Mutex},
};
use cudarc::driver::{
    sys::CUdeviceptr, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceSlice, DriverError, SyncOnDrop,
};

pub type SegId = usize;

/// 段最小粒度：一次新开段至少 1MB，并按 1MB 网格取整（reserved 数字好看、可预测）。
const MIN_SEG: usize = 1 << 20;

/// 分配错误。
#[derive(Debug)]
pub enum AllocError {
    /// CUDA 底层分配失败（如显存不足）。
    Cuda(DriverError),
    /// 非法请求（0 字节等）。
    InvalidRequest { bytes: usize },
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocError::Cuda(e) => write!(f, "cuda alloc failed: {e}"),
            AllocError::InvalidRequest { bytes } => write!(f, "invalid alloc request of {bytes} bytes"),
        }
    }
}

impl Error for AllocError {}

impl From<DriverError> for AllocError {
    fn from(e: DriverError) -> Self {
        AllocError::Cuda(e)
    }
}

/// 给上层用的共享分配器：状态在 `Arc<Mutex<Pool>>` 里，`CudaBuffer` Drop 时回来还。
#[derive(Clone)]
pub struct CachingAllocator {
    pool: Arc<Mutex<Pool>>,
    stream: Arc<CudaStream>,
}

/// 有类型标签的显存句柄：Arc 共享段的底层字节，记住自己在段里的位置。
/// `T` 只在 `CachingAllocator::alloc::<T>` 那一刻贴上，底层永远是字节。
pub struct CudaBuffer<T> {
    storage: Arc<CudaSlice<u8>>, // 与池里那个 Segment 共享同一块字节
    offset: usize,               // 本 buffer 在段内的字节起点
    len: usize,                  // T 的元素个数（DeviceSlice::len）
    bytes: usize,                // 实际占用（256 对齐；Drop 归还用）
    seg: SegId,                  // 归还定位
    pool: Arc<Mutex<Pool>>,      // Drop 时回去还
    _t: PhantomData<T>,
}

/// 多段内存池。段用 `Vec<Option<Segment>>` + 下标作稳定 `SegId`：
/// empty_cache 把空段槽位置 None（不缩数组），索引永不失效。
pub struct Pool {
    stream: Arc<CudaStream>,
    segments: Vec<Option<Segment>>,
    allocated: usize,
    reserved: usize,
    active: usize,
    pool_miss: usize,
    peak: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    pub allocated: usize, // Σ 各段 live：正借出去的字节
    pub reserved: usize,  // Σ 各段 total：从驱动拿到的总字节
    pub active: usize,    // 活跃段数（Some 的槽）
    pub pool_miss: usize, // 真正新开段（stream.alloc）的次数 —— 探针
    pub peak: usize,      // allocated 的历史峰值
}

pub struct Segment {
    storage: Arc<CudaSlice<u8>>,
    total: usize,
    live: usize,
    free: BTreeMap<usize, usize>,
}

// ============================================================================
// CachingAllocator + CudaBuffer
// ============================================================================

impl CachingAllocator {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        CachingAllocator {
            pool: Arc::new(Mutex::new(Pool::new(stream))),
            stream: stream.clone(),
        }
    }

    /// 新段从哪条流分配（也是测试里做 memcpy 用的默认流）。
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// 借出一块能装下 `len` 个 `T` 的显存。`T` 只在这里被贴到字节上。
    pub fn alloc<T>(&self, len: usize) -> Result<CudaBuffer<T>, AllocError> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or(AllocError::InvalidRequest { bytes: 0 })?;
        if bytes == 0 {
            return Err(AllocError::InvalidRequest { bytes: 0 });
        }

        let (storage, seg, offset) = self.pool.lock().expect("pool poisoned").alloc_raw(bytes)?;
        Ok(CudaBuffer {
            storage,
            offset,
            len,
            bytes: align_up(bytes, 256),
            seg,
            pool: self.pool.clone(),
            _t: PhantomData,
        })
    }

    pub fn empty_cache(&self) {
        self.pool.lock().expect("pool poisoned").empty_cache();
    }

    pub fn stats(&self) -> Stats {
        self.pool.lock().expect("pool poisoned").stats()
    }

    pub fn assert_consistency(&self) {
        self.pool.lock().expect("pool poisoned").assert_consistency();
    }
}

impl<T> CudaBuffer<T> {
    /// 段内字节起点（测试里验证"是否复用了同一块"）。
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// 实际占用的段内字节数（对齐后）。
    pub fn byte_len(&self) -> usize {
        self.bytes
    }
}

impl<T> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        // v1 顺序安全：先等流跑完，避免"还了又被借走时还有旧 kernel 在写"。
        let _ = self.storage.stream().synchronize();
        if let Ok(mut g) = self.pool.lock() {
            g.free_raw(self.seg, self.offset, self.bytes);
        }
    }
}

// --- 让 cudarc 的泛型 memcpy（memcpy_htod / clone_dtoh ...）接受 CudaBuffer ---

impl<T> DeviceSlice<T> for CudaBuffer<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        self.storage.stream()
    }
}

impl<T> DevicePtr<T> for CudaBuffer<T> {
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (CUdeviceptr, SyncOnDrop<'a>) {
        // 复用底层 CudaSlice 的读同步守卫，基址 + 自己的字节偏移
        let (base, guard) = <CudaSlice<u8> as DevicePtr<u8>>::device_ptr(&self.storage, stream);
        (base + self.offset as u64, guard)
    }
}

impl<T> DevicePtrMut<T> for CudaBuffer<T> {
    fn device_ptr_mut<'a>(&'a mut self, stream: &'a CudaStream) -> (CUdeviceptr, SyncOnDrop<'a>) {
        // v1 简化：单流 + Drop 同步兜底，写路径先与读路径共用守卫。
        DevicePtr::device_ptr(self, stream)
    }
}

// ============================================================================
// Pool
// ============================================================================

impl Pool {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Pool {
            stream: stream.clone(),
            segments: Vec::new(),
            allocated: 0,
            reserved: 0,
            active: 0,
            pool_miss: 0,
            peak: 0,
        }
    }

    pub fn stats(&self) -> Stats {
        Stats {
            allocated: self.allocated,
            reserved: self.reserved,
            active: self.active,
            pool_miss: self.pool_miss,
            peak: self.peak,
        }
    }

    /// 借出 `bytes` 字节。先在现有各段做全局 best-fit；全都不够就新开一段。
    /// 返回 `(共享的底层字节, 段, 段内偏移)` —— 调用方包成 `CudaBuffer`。
    pub fn alloc_raw(&mut self, bytes: usize) -> Result<(Arc<CudaSlice<u8>>, SegId, usize), AllocError> {
        let bytes = align_up(bytes, 256);
        if bytes == 0 {
            return Err(AllocError::InvalidRequest { bytes: 0 });
        }

        // 全局 best-fit：跨段扫，选 run_len 最小者。元组 = (段id, offset, run_len)
        let mut best_sel: Option<(SegId, usize, usize)> = None;
        for (i, slot) in self.segments.iter().enumerate() {
            if let Some(seg) = slot {
                if let Some((off, run_len)) = seg.find_best(bytes) {
                    best_sel = match best_sel {
                        // 已有更小的就保留；否则换成当前的（同大小时不换 → 偏向更早的段）
                        Some((_, _, last_len)) if last_len < run_len => best_sel,
                        _ => Some((i, off, run_len)),
                    };
                }
            }
        }

        let (seg_id, offset) = match best_sel {
            Some((id, off, _)) => (id, off),
            None => (self.new_segment(bytes)?, 0),
        };
        let offset = self.segments[seg_id]
            .as_mut()
            .unwrap()
            .alloc_raw_in(bytes, offset)?;
        let slice = self.segments[seg_id].as_ref().unwrap().storage().clone();

        self.allocated += bytes;
        self.peak = self.peak.max(self.allocated);
        Ok((slice, seg_id, offset))
    }

    /// 新建一段能装下 `bytes` 的段，返回其 SegId。
    fn new_segment(&mut self, bytes: usize) -> Result<SegId, AllocError> {
        let seg_bytes = align_up(max(bytes, MIN_SEG), MIN_SEG);
        let seg = Segment::new(seg_bytes, &self.stream)?;
        self.pool_miss += 1;
        self.reserved += seg.total;
        self.active += 1;
        self.segments.push(Some(seg));
        Ok(self.segments.len() - 1)
    }

    pub fn free_raw(&mut self, id: SegId, offset: usize, bytes: usize) {
        assert!(self.segments[id].is_some(), "free on dead segment {id}");
        let bytes = align_up(bytes, 256);
        assert!(self.allocated >= bytes, "allocated underflow on free");
        self.segments[id].as_mut().unwrap().free_raw(offset, bytes);
        self.allocated -= bytes;
    }

    /// 释放所有整段空闲的段（存储随 Arc 归零自动还给驱动），并重算计数。
    pub fn empty_cache(&mut self) {
        for slot in self.segments.iter_mut() {
            if let Some(seg) = slot {
                if seg.live == 0 {
                    *slot = None;
                }
            }
        }
        self.recount();
    }

    /// 从段状态重算计数（empty_cache 后最可靠，避免增量漂移）。peak 保留历史。
    fn recount(&mut self) {
        self.allocated = self.segments.iter().flatten().map(|s| s.live).sum();
        self.reserved = self.segments.iter().flatten().map(|s| s.total).sum();
        self.active = self.segments.iter().filter(|s| s.is_some()).count();
    }

    pub fn assert_consistency(&self) {
        for seg in self.segments.iter().flatten() {
            seg.assert_consistency();
        }
        // 计数必须等于从段状态重算出来的值
        let sum_live: usize = self.segments.iter().flatten().map(|s| s.live).sum();
        let sum_total: usize = self.segments.iter().flatten().map(|s| s.total).sum();
        let n_active = self.segments.iter().filter(|s| s.is_some()).count();
        assert_eq!(self.allocated, sum_live, "pool allocated != Σlive");
        assert_eq!(self.reserved, sum_total, "pool reserved != Σtotal");
        assert_eq!(self.active, n_active, "pool active != count(Some)");
    }
}

// ============================================================================
// Segment
// ============================================================================

impl Segment {
    /// 新建一段 `total` 字节的段：整段空闲 `free = {0: total}`。向上取整到 256。
    pub fn new(total: usize, stream: &Arc<CudaStream>) -> Result<Self, AllocError> {
        let total = align_up(total.max(1), 256);
        let storage = Arc::new(unsafe { stream.alloc::<u8>(total)? });
        let mut free = BTreeMap::new();
        free.insert(0, total);
        Ok(Segment { storage, total, live: 0, free })
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn live(&self) -> usize {
        self.live
    }

    /// 底层整段字节（`CudaBuffer` 会 Arc 共享它）。
    pub fn storage(&self) -> &Arc<CudaSlice<u8>> {
        &self.storage
    }

    /// best-fit 查找：返回 `(offset, run_len)`（够用里最小的空闲区间），只查不改。
    /// 返回 `None` 是"没有合适的空闲区间"这一正常分支（触发新段），不是错误。
    pub fn find_best(&self, bytes: usize) -> Option<(usize, usize)> {
        assert!(bytes % 256 == 0);
        let mut best: Option<(usize, usize)> = None;
        for (&offset, &size) in self.free.iter() {
            if size >= bytes {
                best = match best {
                    Some((_, last)) if last < size => best, // 已有更小
                    _ => Some((offset, size)),
                };
            }
        }
        best
    }

    /// 从 `offset`（必须是一个空闲区间起点）切走 `bytes`，返回 offset。
    pub fn alloc_raw_in(&mut self, bytes: usize, offset: usize) -> Result<usize, AllocError> {
        assert!(bytes % 256 == 0);
        if bytes == 0 || bytes > self.total {
            return Err(AllocError::InvalidRequest { bytes });
        }
        let origin = self
            .free
            .get(&offset)
            .copied()
            .filter(|&size| size >= bytes)
            .ok_or(AllocError::InvalidRequest { bytes })?;

        self.free.remove(&offset);
        let leftover = origin - bytes;
        if leftover > 0 {
            self.free.insert(offset + bytes, leftover);
        }
        self.live += bytes;
        Ok(offset)
    }

    /// 归还 `[offset, offset+bytes)`：标为空闲，并与前/后相邻空闲区间链式合并。
    pub fn free_raw(&mut self, offset: usize, bytes: usize) {
        assert!(offset % 256 == 0, "offset must be 256-aligned");
        let bytes = align_up(bytes, 256);
        assert!(bytes > 0, "cannot free 0 bytes");
        assert!(offset + bytes <= self.total, "free range out of segment");
        assert!(self.live >= bytes, "live underflow / double free");

        // 前驱：最后一块起点 < offset 的空闲区间。若它正好结束在 offset，并入它。
        let mut start = offset;
        if let Some((&k, &v)) = self.free.range(..offset).next_back() {
            assert!(k + v <= offset, "double free: overlaps left free run");
            if k + v == offset {
                start = k;
            }
        }

        // 后继：第一块起点 >= offset 的空闲区间。若它正好从 offset+bytes 开始，并入它。
        let mut end = offset + bytes;
        if let Some((&k, &v)) = self.free.range(offset..).next() {
            assert!(k >= offset + bytes, "double free: overlaps right free run");
            if k == offset + bytes {
                end = k + v;
            }
        }

        self.free.retain(|&k, _| k < start || k >= end);
        self.free.insert(start, end - start);
        self.live -= bytes;
    }

    /// 不变量自检：错了就 panic。
    pub fn assert_consistency(&self) {
        let mut sum_free = 0usize;
        let mut prev_key: Option<usize> = None;
        for (&k, &v) in self.free.iter() {
            assert!(v > 0, "zero-length free run at {k}");
            assert!(k + v <= self.total, "free run [{k}, {}) exceeds total {}", k + v, self.total);
            if let Some(pk) = prev_key {
                assert!(
                    pk + self.free[&pk] < k,
                    "free runs at {pk} and {k} are touching (coalesce missed)"
                );
            }
            sum_free += v;
            prev_key = Some(k);
        }
        assert_eq!(self.live + sum_free, self.total, "conservation broken: live+free != total");
    }
}

fn align_up(v: usize, a: usize) -> usize {
    (v + a - 1) & !(a - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaContext;

    fn allocator() -> CachingAllocator {
        CachingAllocator::new(&CudaContext::new(0).unwrap().default_stream())
    }

    /// Test A — 同段两个 buffer 各写各的互不干扰（trait 实现可被 memcpy 消费）。
    #[test]
    fn two_buffers_same_segment() {
        let alloc = allocator();
        let stream = alloc.stream().clone();
        let n = 64 * 1024; // 256KB f32

        let mut a = alloc.alloc::<f32>(n).unwrap();
        let mut b = alloc.alloc::<f32>(n).unwrap();
        // 同一个 1MB 新段被切两次 → 偏移不同
        assert_eq!(a.offset(), 0);
        assert_eq!(b.offset(), align_up(n * 4, 256));
        assert_ne!(a.offset(), b.offset());

        let ha: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let hb: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();
        stream.memcpy_htod(&ha, &mut a).unwrap();
        stream.memcpy_htod(&hb, &mut b).unwrap();
        stream.synchronize().unwrap();

        assert_eq!(stream.clone_dtoh(&a).unwrap(), ha);
        assert_eq!(stream.clone_dtoh(&b).unwrap(), hb);
        alloc.assert_consistency();

        drop(a);
        drop(b);
        alloc.empty_cache();
        assert_eq!((alloc.stats().reserved, alloc.stats().active), (0, 0));
    }

    /// Test B — Drop 回池：释放后同尺寸分配必须复用到同一 offset。
    #[test]
    fn buffer_reuse_after_drop() {
        let alloc = allocator();
        let n = 256 * 1024; // 1MB f32 → 正好一段

        let off0 = alloc.alloc::<f32>(n).unwrap().offset();
        drop(alloc.alloc::<f32>(n).unwrap());
        let off1 = alloc.alloc::<f32>(n).unwrap().offset();
        assert_eq!(off0, off1, "freed block must be reused");
        assert_eq!(alloc.stats().pool_miss, 1, "single segment serves everything");
        alloc.assert_consistency();
    }

    /// Test C — 跨类型复用：f32 -> u8 -> f32 同一块字节，无脏数据。
    #[test]
    fn cross_type_reuse() {
        let alloc = allocator();
        let stream = alloc.stream().clone();
        let nf = 64 * 1024; // f32 个数
        let nb = nf * 4; // 同字节数的 u8

        let mut f1 = alloc.alloc::<f32>(nf).unwrap();
        let off_f = f1.offset();
        let host1 = vec![1.5f32; nf];
        stream.memcpy_htod(&host1, &mut f1).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(stream.clone_dtoh(&f1).unwrap(), host1);
        drop(f1);

        let mut b = alloc.alloc::<u8>(nb).unwrap();
        assert_eq!(b.offset(), off_f, "u8 must reuse the freed f32 bytes");
        let hostb: Vec<u8> = (0..nb).map(|i| (i % 251) as u8).collect();
        stream.memcpy_htod(&hostb, &mut b).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(stream.clone_dtoh(&b).unwrap(), hostb);
        drop(b);

        let mut f2 = alloc.alloc::<f32>(nf).unwrap();
        assert_eq!(f2.offset(), off_f, "f32 must reuse the same bytes again");
        let host2 = vec![42.0f32; nf];
        stream.memcpy_htod(&host2, &mut f2).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(stream.clone_dtoh(&f2).unwrap(), host2, "no stale data from u8 phase");
        drop(f2);

        alloc.assert_consistency();
    }

    /// Result：0 长度请求必须是 Err，而不是静默 None。
    #[test]
    fn zero_request_is_error() {
        let alloc = allocator();
        assert!(matches!(alloc.alloc::<f32>(0), Err(AllocError::InvalidRequest { .. })));
    }
}
