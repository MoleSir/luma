use std::{collections::BTreeMap, sync::Arc};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DriverError};

fn next_rng(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state >> 33
}

fn main() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();

    let mut seg = Segment::new(1 << 20, &stream).unwrap(); // 1 MB
    let mut ledger: Vec<(usize, usize)> = Vec::new(); // 影子账本：借走了什么
    let mut state = 12345u64;

    for _ in 0..5000 {
        // 空账本时只借不还；否则 3/4 概率借、1/4 概率还
        if ledger.is_empty() || next_rng(&mut state) % 4 != 0 {
            let want = 256 * (1 + next_rng(&mut state) as usize % 512); // 256B..=128KB
            if let Some(off) = seg.alloc_raw(want) {
                ledger.push((off, want));
            }
        } else {
            let idx = (next_rng(&mut state) as usize) % ledger.len();
            let (off, len) = ledger.remove(idx);
            seg.free_raw(off, len);
        }
        seg.assert_consistency();
    }

    // 全部归还
    for (off, len) in ledger.drain(..) {
        seg.free_raw(off, len);
    }
    seg.assert_consistency();

    assert_eq!(seg.live(), 0);
    assert_eq!(seg.free_map().len(), 1, "all frees must coalesce into one run");
    assert_eq!(seg.free_map().get(&0), Some(&seg.total()), "segment must be whole again");
}


/// 一段从驱动批发来的连续显存，内部按"空闲区间树"记账。
///
/// `free: BTreeMap<usize, usize>` 的 key = 空闲区间起点，value = 长度，
/// 只记录"空着没人用"的区间；借出去的字节不在这张表里，只体现在 `live`。
pub struct Segment {
    storage: Arc<CudaSlice<u8>>,
    total: usize,
    live: usize,
    free: BTreeMap<usize, usize>,
}

impl Segment {
    /// 新建一段 `total` 字节的段：整段空闲（`free = {0: total}`）。
    /// `total` 会被向上取整到 256 的倍数，保证偏移对齐与 cudaMalloc 一致。
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

    pub fn storage(&self) -> &CudaSlice<u8> {
        &self.storage
    }

    pub fn free_map(&self) -> &BTreeMap<usize, usize> {
        &self.free
    }

    /// 借出 `bytes` 字节（内部向上取整到 256）。
    /// best-fit：挑"够用里最小"的空闲区间，从它头部切走，剩尾巴留作新空闲区间。
    /// 返回区间起点 offset；任何单块都装不下时返回 None（碎片化）。
    pub fn alloc_raw(&mut self, bytes: usize) -> Option<usize> {
        let bytes = align_up(bytes, 256);
        if bytes == 0 || bytes > self.total {
            return None;
        }

        // best-fit：size >= bytes 里最小的那块
        let mut best: Option<(usize, usize)> = None;
        for (&offset, &size) in self.free.iter() {
            if size >= bytes {
                match best {
                    None => best = Some((offset, size)),
                    Some((_, bs)) if size < bs => best = Some((offset, size)),
                    _ => {}
                }
            }
        }
        let (offset, origin) = best?;

        // 从头部切走 bytes，若剩 >0 才把尾巴插回树（避免 0 长度区间）
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
                start = k; // 向左扩展
            }
        }

        // 后继：第一块起点 >= offset 的空闲区间。若它正好从 offset+bytes 开始，并入它。
        let mut end = offset + bytes;
        if let Some((&k, &v)) = self.free.range(offset..).next() {
            assert!(k >= offset + bytes, "double free: overlaps right free run");
            if k == offset + bytes {
                end = k + v; // 向右扩展
            }
        }

        // 吃掉 [start, end) 内所有旧空闲区间，再整体插回一个合并后的区间
        self.free.retain(|&k, _| k < start || k >= end);
        self.free.insert(start, end - start);
        self.live -= bytes;
    }

    /// 不变量自检：每次 alloc/free 后调用，错了就 panic。
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
