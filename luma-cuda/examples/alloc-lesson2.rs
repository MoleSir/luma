use std::{cmp::max, collections::BTreeMap, sync::Arc};
use cudarc::driver::{CudaSlice, CudaStream, DriverError};

pub type SegId = usize;

fn main() {
    
}

/// 段最小粒度：一次新开段至少 1MB，并按 1MB 网格取整（reserved 数字好看、可预测）。
const MIN_SEG: usize = 1 << 20;

#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    pub allocated: usize, // Σ 各段 live：正借出去的字节
    pub reserved: usize,  // Σ 各段 total：从驱动拿到的总字节
    pub active: usize,    // 活跃段数（Some 的槽）
    pub pool_miss: usize, // 真正新开段（stream.alloc）的次数 —— 探针
    pub peak: usize,      // allocated 的历史峰值
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

pub struct Segment {
    storage: Arc<CudaSlice<u8>>,
    total: usize,
    live: usize,
    free: BTreeMap<usize, usize>,
}

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

    /// 借出 `bytes` 字节。先在现有各段做全局 best-fit（最小够用区间、平局取段下标小者，
    /// 保证布局确定性）；全都不够就新开一段。返回 `(段, 段内偏移)`。
    pub fn alloc_raw(&mut self, bytes: usize) -> Option<(SegId, usize)> {
        let bytes = align_up(bytes, 256);
        if bytes == 0 {
            return None;
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
            None => {
                let id = self.new_segment(bytes);
                (id, 0)
            }
        };
        let offset = self.segments[seg_id].as_mut().unwrap().alloc_raw_in(bytes, offset)?;

        self.allocated += bytes;
        self.peak = self.peak.max(self.allocated);
        Some((seg_id, offset))
    }

    /// 新建一段能装下 `bytes` 的段，返回其 SegId。
    fn new_segment(&mut self, bytes: usize) -> SegId {
        let seg_bytes = align_up(max(bytes, MIN_SEG), MIN_SEG);
        let seg = Segment::new(seg_bytes, &self.stream).expect("cudaMalloc failed");
        self.pool_miss += 1;
        self.reserved += seg.total;
        self.active += 1;
        self.segments.push(Some(seg));
        self.segments.len() - 1
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

impl Segment {
    /// 新建一段 `total` 字节的段：整段空闲 `free = {0: total}`。向上取整到 256。
    pub fn new(total: usize, stream: &Arc<CudaStream>) -> Result<Self, DriverError> {
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

    /// 底层整段字节（Lesson 3 的 `CudaBuffer` 会 Arc 共享它）。
    pub fn storage(&self) -> &Arc<CudaSlice<u8>> {
        &self.storage
    }

    /// best-fit 查找：返回 `(offset, run_len)`（够用里最小的空闲区间），只查不改。
    /// 调用方 `alloc_raw_in` 用它拿 offset，两处逻辑保持一致。
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

    /// 已经确定从 `offset`（必须是一个空闲区间起点）切走 `bytes`。
    pub fn alloc_raw_in(&mut self, bytes: usize, offset: usize) -> Option<usize> {
        assert!(bytes % 256 == 0);
        if bytes == 0 || bytes > self.total {
            return None;
        }
        let origin = self.free[&offset]; // 由 find_best 保证存在且 >= bytes
        self.free.remove(&offset);
        let leftover = origin - bytes;
        if leftover > 0 {
            self.free.insert(offset + bytes, leftover);
        }
        self.live += bytes;
        Some(offset)
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

    fn stream() -> Arc<CudaStream> {
        CudaContext::new(0).unwrap().default_stream()
    }

    /// 确定性 LCG（可复现的随机序列）。
    fn next_rng(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *state >> 33
    }

    /// Test A — 稳态复用：warmup 后 200 轮同负载，pool_miss 不再增长、位置完全确定。
    #[test]
    fn steady_state_reuse() {
        let mut pool = Pool::new(&stream());
        let sizes = [1usize << 20, 2 << 20, 4 << 20];

        // warmup：空池各触发一次新段 → 3 次 pool_miss
        let mut warm = Vec::new();
        for &s in &sizes {
            warm.push(pool.alloc_raw(s).unwrap());
        }
        for ((id, off), s) in warm.iter().zip(&sizes) {
            pool.free_raw(*id, *off, *s);
        }
        let miss0 = pool.stats().pool_miss;
        assert_eq!(miss0, 3, "warmup must create exactly 3 segments");

        // 记录第一轮的位置，之后每轮都必须一模一样
        let mut ref_pos = Vec::new();
        for round in 0..200 {
            let mut pos = Vec::new();
            for &s in &sizes {
                pos.push(pool.alloc_raw(s).unwrap());
            }
            if round == 0 {
                ref_pos = pos.clone();
            } else {
                assert_eq!(pos, ref_pos, "round {round}: positions drifted");
            }
            for ((id, off), s) in pos.into_iter().zip(sizes) {
                pool.free_raw(id, off, s);
            }
            pool.assert_consistency();
        }
        assert_eq!(pool.stats().pool_miss, miss0, "pool_miss must not grow after warmup");
    }

    /// Test B — 碎片化 vs 合并 + empty_cache（把 C++ 剧本搬进 Pool）。
    #[test]
    fn fragmentation_coalesce_empty_cache() {
        let mut pool = Pool::new(&stream());
        let mb = 1 << 20;
        let mb256 = 256 * mb;

        // 段 0：256MB 整段（seed，随即释放成整块空闲）
        let (s0, _) = pool.alloc_raw(mb256).unwrap();
        pool.free_raw(s0, 0, mb256);

        // 依次切四个 48MB：全部应落在段 0，offset 0/48/96/144
        let a = pool.alloc_raw(48 * mb).unwrap();
        let b = pool.alloc_raw(48 * mb).unwrap();
        let c = pool.alloc_raw(48 * mb).unwrap();
        let d = pool.alloc_raw(48 * mb).unwrap();
        assert_eq!((a.0, a.1), (s0, 0));
        assert_eq!((b.0, b.1), (s0, 48 * mb));
        assert_eq!((c.0, c.1), (s0, 96 * mb));
        assert_eq!((d.0, d.1), (s0, 144 * mb));
        pool.assert_consistency();

        // 释放 B、D → 两个洞
        pool.free_raw(b.0, b.1, 48 * mb);
        pool.free_raw(d.0, d.1, 48 * mb);
        let st = pool.stats();
        println!("after free B,D: reserved={}MB allocated={}MB miss={}",
                 st.reserved >> 20, st.allocated >> 20, st.pool_miss);

        // 要 128MB：洞最大 112MB（D洞+尾64 已合并）→ 必须新段（碎片化！）
        let e = pool.alloc_raw(128 * mb).unwrap();
        assert_eq!(e.0, s0 + 1, "128MB can't fit in fragmented seg0 -> new segment");
        assert_eq!(pool.stats().pool_miss, 2, "fragmentation forced a 2nd segment");

        // 释放 A、C、E → 段 0 合并回整块 256
        pool.free_raw(a.0, a.1, 48 * mb);
        pool.free_raw(c.0, c.1, 48 * mb);
        pool.free_raw(e.0, e.1, 128 * mb);

        // 192MB 直接复用段 0 → pool_miss 不变
        let f = pool.alloc_raw(192 * mb).unwrap();
        assert_eq!((f.0, f.1), (s0, 0), "192MB should reuse coalesced seg0");
        assert_eq!(pool.stats().pool_miss, 2, "coalescing must avoid a 3rd segment");
        pool.assert_consistency();

        // 全释放 + empty_cache → reserved / active 归零，但仍可用
        pool.free_raw(f.0, f.1, 192 * mb);
        pool.empty_cache();
        let st = pool.stats();
        println!("after empty_cache: reserved={}MB active={} miss={}", st.reserved >> 20, st.active, st.pool_miss);
        assert_eq!((st.reserved, st.active), (0, 0), "empty_cache must release everything");
        let again = pool.alloc_raw(1 * mb).unwrap();
        assert_eq!(again.1, 0);
        pool.free_raw(again.0, again.1, 1 * mb);
        pool.assert_consistency();
    }

    /// Test C — 随机 alloc/free 压力：影子账本 + 每步一致性。
    #[test]
    fn random_pool_stress() {
        let mut pool = Pool::new(&stream());
        let mut ledger: Vec<(SegId, usize, usize)> = Vec::new(); // (id, offset, len)
        let mut state = 4242u64;

        for _ in 0..3000 {
            if ledger.is_empty() || next_rng(&mut state) % 4 != 0 {
                let want = 256 * (1 + (next_rng(&mut state) as usize % 512)); // 256B..=128KB
                if let Some((id, off)) = pool.alloc_raw(want) {
                    ledger.push((id, off, want));
                }
            } else {
                let idx = (next_rng(&mut state) as usize) % ledger.len();
                let (id, off, len) = ledger.remove(idx);
                pool.free_raw(id, off, len);
            }
            pool.assert_consistency();
        }
        for (id, off, len) in ledger.drain(..) {
            pool.free_raw(id, off, len);
        }
        pool.assert_consistency();
        pool.empty_cache();
        let st = pool.stats();
        assert_eq!((st.allocated, st.reserved, st.active), (0, 0, 0));
    }
}
