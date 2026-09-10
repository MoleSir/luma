// allocator.cu — 缓存分配器实现（纯主机侧代码，.cu 只是为了统一用 nvcc 编译）
//
// 心智模型（对照 torch native 后端）：
//   reserved  = 从驱动 cudaMalloc 拿到的总内存（由若干 Segment 组成）
//   allocated = 借给上层 tensor 的内存
//   free_     = reserved 里暂时没人用、可被再次分配的部分
//
// 释放不立刻还驱动：块先进 free_ 等待复用；只有整段全空 + empty_cache()
// 才 cudaFree 归还。这就是"缓存分配器"名字的由来。

#include "allocator.h"

#include <cstdio>
#include <cstdlib>

// cudaMalloc 失败时打印并终止（教学版不做优雅恢复）
[[noreturn]] static void oom(size_t bytes) {
    fprintf(stderr, "[allocator] cudaMalloc(%zu bytes) failed\n", bytes);
    exit(1);
}

CachingAllocator::~CachingAllocator() {
    empty_cache();
}

// ---------------------------------------------------------------------------
// 内部工具
// ---------------------------------------------------------------------------

// 把一个 Block 插进空闲表（free_ 无序，best_fit 时线性扫）
void CachingAllocator::insert_free(Block* b) {
    free_.push_back(b);
    ++free_blocks_;
}

void CachingAllocator::remove_free(Block* b) {
    for (auto it = free_.begin(); it != free_.end(); ++it) {
        if (*it == b) {
            free_.erase(it);
            --free_blocks_;
            return;
        }
    }
    fprintf(stderr, "[allocator] internal: block %p not in free list\n", (void*)b);
    abort();
}

// best-fit：找 size 最小且 >= bytes 的空闲块。碎片少、但可能留下很窄的洞，
// 这正是后面"碎片化"测试想展示的现象。
Block* CachingAllocator::best_fit(size_t bytes) {
    Block* best = nullptr;
    for (Block* b : free_) {
        if (b->size >= bytes && (best == nullptr || b->size < best->size)) {
            best = b;
        }
    }
    return best;
}

// 从空闲块 b 的头部切出 take 字节给"即将被分配"的部分，剩下的做成新空闲块。
// 前提：b->size - take >= kMinSplit，否则不该分裂。
void CachingAllocator::split_block(Block* b, size_t take) {
    Block* r = new Block;
    r->seg    = b->seg;
    r->offset = b->offset + take;
    r->size   = b->size - take;
    r->free   = true;
    r->id     = 0;
    // 把 r 插到 b 后面（保持段内地址升序）
    r->prev = b;
    r->next = b->next;
    if (b->next) b->next->prev = r;
    b->next = r;

    b->size = take;
    insert_free(r);
}

// cudaMalloc 一个新段，把整段做成一个空闲 Block 返回（调用方随后 split）。
Block* CachingAllocator::add_segment(size_t bytes) {
    size_t seg_bytes = align_up(bytes > kMinSegment ? bytes : kMinSegment, kMinSegment);

    void* ptr = nullptr;
    if (cudaMalloc(&ptr, seg_bytes) != cudaSuccess) oom(seg_bytes);
    ++n_malloc_;

    Segment* s = new Segment;
    s->ptr = ptr;
    s->size = seg_bytes;
    s->live_bytes = 0;
    s->next = nullptr;
    if (tail_) tail_->next = s;
    else segments_ = s;
    tail_ = s;
    ++segments_n_;
    reserved_ += seg_bytes;

    Block* b = new Block;
    b->seg    = s;
    b->offset = 0;
    b->size   = seg_bytes;
    b->free   = true;
    b->id     = 0;
    b->prev   = nullptr;
    b->next   = nullptr;
    s->first  = b;
    insert_free(b);
    return b;
}

// coalesce：b 刚刚被标 free（已在 free_ 里）。先把右侧相邻 free 并进 b，
// 再把 b 并进左侧相邻 free（保留低地址那块做幸存者），消灭碎片洞。
void CachingAllocator::coalesce(Block* b) {
    // 右合并：把 r = b->next 收编进 b
    while (b->next && b->next->free) {
        Block* r = b->next;
        b->size += r->size;
        b->next = r->next;
        if (r->next) r->next->prev = b;
        remove_free(r);
        delete r;  // 元数据节点消失；显存字节仍在，只是并进了 b
    }
    // 左合并：把 b 并进 l = b->prev（l 成为幸存者）
    if (b->prev && b->prev->free) {
        Block* l = b->prev;
        l->size += b->size;
        l->next = b->next;
        if (b->next) b->next->prev = l;
        if (b->seg->first == b) b->seg->first = l;
        remove_free(b);
        delete b;
    }
}

// ---------------------------------------------------------------------------
// 公共 API
// ---------------------------------------------------------------------------

void* CachingAllocator::alloc(size_t bytes) {
    if (bytes == 0) return nullptr;

    size_t req = align_up(bytes, kAlignBytes);
    Block* b = best_fit(req);
    if (b == nullptr) {
        // 空闲块不够 → 向驱动要新段，取整段再切
        b = add_segment(req);
    }

    // 切出 req：若剩余 >= kMinSplit 就分裂，否则整块给出（宁多勿碎）
    if (b->size - req >= kMinSplit) {
        split_block(b, req);
    }
    remove_free(b);

    // 标记为在用
    b->free = false;
    b->id   = ++id_counter_;
    b->seg->live_bytes += b->size;
    allocated_ += b->size;
    ++live_blocks_;
    if (allocated_ > peak_) peak_ = allocated_;

    void* payload = (char*)b->seg->ptr + b->offset;
    by_ptr_[payload] = b;
    return payload;
}

void CachingAllocator::free(void* p) {
    if (p == nullptr) return;

    auto it = by_ptr_.find(p);
    if (it == by_ptr_.end()) {
        fprintf(stderr, "[allocator] free() of unknown pointer %p\n", p);
        abort();
    }
    Block* b = it->second;
    by_ptr_.erase(it);

    // 标空闲并放回池（不是 cudaFree！）
    b->free = true;
    b->seg->live_bytes -= b->size;
    allocated_ -= b->size;
    --live_blocks_;
    insert_free(b);
    coalesce(b);
}

void CachingAllocator::empty_cache() {
    Segment* s = segments_;
    Segment* prev = nullptr;
    while (s) {
        Segment* next = s->next;
        if (s->live_bytes == 0) {
            // 整段全空 → 还给驱动
            if (cudaFree(s->ptr) != cudaSuccess) {
                fprintf(stderr, "[allocator] cudaFree(%p) failed\n", s->ptr);
                abort();
            }
            ++n_free_;
            reserved_ -= s->size;
            --segments_n_;

            // 释放段内所有 Block 元数据
            Block* b = s->first;
            while (b) {
                Block* bn = b->next;
                remove_free(b);
                delete b;
                b = bn;
            }

            if (prev) prev->next = next;
            else segments_ = next;
            if (tail_ == s) tail_ = prev;
            delete s;
        } else {
            prev = s;
        }
        s = next;
    }
}

Stats CachingAllocator::stats() const {
    Stats st;
    st.allocated_bytes    = allocated_;
    st.reserved_bytes     = reserved_;
    st.peak_bytes         = peak_;
    st.live_blocks        = live_blocks_;
    st.free_blocks        = free_blocks_;
    st.segments           = segments_n_;
    st.cuda_malloc_calls  = n_malloc_;
    st.cuda_free_calls    = n_free_;
    return st;
}

// 一致性守恒：段内空闲+在用 == reserved；空闲表计数与 free_blocks_ 一致。
// 每轮大测试结尾调一次，当作"分配器没算错账"的自检。
void CachingAllocator::assert_consistency() const {
    size_t reserved = 0, live = 0;
    for (Segment* s = segments_; s; s = s->next) {
        reserved += s->size;
        for (Block* b = s->first; b; b = b->next) {
            live += b->free ? 0 : b->size;
        }
    }
    if (reserved != reserved_ || live != allocated_) {
        fprintf(stderr, "[allocator] consistency broken: reserved %zu/%zu live %zu/%zu\n",
                reserved, reserved_, live, allocated_);
        abort();
    }
    if (free_.size() != free_blocks_) {
        fprintf(stderr, "[allocator] free list size mismatch %zu/%zu\n", free_.size(), free_blocks_);
        abort();
    }
}
