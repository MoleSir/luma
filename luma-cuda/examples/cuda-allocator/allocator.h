// allocator.h — 一个仿 PyTorch CUDACachingAllocator 的最小"缓存分配器"
//
// 设计要点（与 torch native 后端一一对应）：
//   * 向驱动要内存的单位是 Segment —— 一次 cudaMalloc 的一整块连续显存。
//   * Segment 内部切成 Block —— 真正可分配的最小单位。
//   * 分配 = 在空闲块里 best-fit；没有就新开 Segment 并 split。
//   * 释放 = 标 free 后与段内相邻空闲块 coalesce 合并成大块。
//   * 元数据（Segment/Block）全在主机侧 —— 显存里只有"裸字节"。
//
// 学习的核心：allocator 从头到尾不关心类型。拿到/交出的都是 void*（等价
// torch 的 DataPtr void*）。"这是 f32 还是 f64"只在你 reinterpret_cast 那一刻
// 才出现 —— 见 main.cu。

#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <unordered_map>

// ---------------------------------------------------------------------------
// 对齐与粒度参数（torch 里这些是可配置项）
// ---------------------------------------------------------------------------
inline constexpr size_t kAlignBytes  = 256;            // cudaMalloc 保证 256B 对齐，我们保持每个 payload 也 256B 对齐
inline constexpr size_t kMinSegment  = size_t(1) << 20; // 一次 cudaMalloc 的最小段：1 MB
inline constexpr size_t kMinSplit    = 256;            // 分裂后剩余 < 256B 就不分裂（宁可多给一点）

inline size_t align_up(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

// ---------------------------------------------------------------------------
// Stats —— 分配器可观测状态（torch 的 memory_stats 就是这些字段的放大版）
// ---------------------------------------------------------------------------
struct Stats {
    size_t allocated_bytes   = 0;  // 正在被使用的显存字节
    size_t reserved_bytes    = 0;  // 从驱动拿到的总字节（所有 Segment 之和）
    size_t peak_bytes        = 0;  // allocated_bytes 的历史峰值
    size_t live_blocks       = 0;  // 当前在用的 Block 数
    size_t free_blocks       = 0;  // 空闲 Block 数
    size_t segments          = 0;  // 当前 Segment 数
    size_t cuda_malloc_calls = 0;  // 探针：真正调用 cudaMalloc 的次数
    size_t cuda_free_calls   = 0;  // 探针：真正调用 cudaFree 的次数（empty_cache 触发）
};

// ---------------------------------------------------------------------------
// Block / Segment
//
// 每个 Segment 里有一串"按地址升序"的 Block，串成双向链表；Block 元数据是
// 主机侧堆对象（new 出来的），链表指针即 prev/next。这样 coalesce 时只需改
// 指针，与经典 malloc lab / torch 的做法一致。
// ---------------------------------------------------------------------------
struct Segment;  // fwd

struct Block {
    Segment* seg;        // 属于哪个段
    size_t   offset;     // 段内字节偏移；payload = (char*)seg->ptr + offset
    size_t   size;       // 块大小（已 256B 对齐取整）
    bool     free;       // 是否空闲
    size_t   id;         // 单调分配序号（仅观察用；空闲块为 0）
    Block*   prev;       // 段内前一块（地址更低）
    Block*   next;       // 段内后一块（地址更高）
};

struct Segment {
    void*   ptr;         // cudaMalloc 基址 —— 学习重点：这就是"裸 void*"
    size_t  size;        // 段大小（已按 kMinSegment 取整）
    size_t  live_bytes;  // 本段内在用的字节（empty_cache 判断能否整段归还）
    Block*  first;       // 段内地址最低的 Block
    Segment* next;       // 全局 Segment 单向链
};

// ---------------------------------------------------------------------------
// CachingAllocator
// ---------------------------------------------------------------------------
class CachingAllocator {
public:
    CachingAllocator() = default;
    ~CachingAllocator();

    CachingAllocator(const CachingAllocator&) = delete;
    CachingAllocator& operator=(const CachingAllocator&) = delete;

    void* alloc(size_t bytes);   // 返回 256B 对齐的无类型裸指针（失败返回 nullptr）
    void  free(void* p);         // 交还一块 —— 回池 + coalesce，绝不立刻 cudaFree
    void  empty_cache();         // 释放所有"整段空闲"的 Segment（reserved 回落）
    Stats stats() const;

    // 一致性校验（学习辅助：验证统计量守恒）
    void assert_consistency() const;

private:
    Block* best_fit(size_t bytes);
    void   split_block(Block* b, size_t take);     // 从 b 前部切出 take，剩余做新空闲块
    void   insert_free(Block* b);
    void   remove_free(Block* b);
    Block* add_segment(size_t bytes);              // cudaMalloc 新段并返回第一个 Block
    void   coalesce(Block* b);                     // b 刚被标 free，尝试与左右合并

    std::list<Block*> free_;                       // 空闲块表（unsorted，best-fit 线性扫）
    std::unordered_map<void*, Block*> by_ptr_;     // live payload 指针 -> Block（O(1) 释放定位）
    Segment* segments_ = nullptr;
    Segment* tail_     = nullptr;

    size_t id_counter_   = 0;
    size_t allocated_    = 0;
    size_t reserved_     = 0;
    size_t peak_         = 0;
    size_t live_blocks_  = 0;
    size_t free_blocks_  = 0;
    size_t segments_n_   = 0;
    size_t n_malloc_     = 0;
    size_t n_free_       = 0;
};
