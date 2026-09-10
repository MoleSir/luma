// main.cu — 用 CUDA kernel 给缓存分配器做"压力测试 + 现象演示"
//
// 教学主线：
//   1. naive baseline —— 直连 cudaMalloc/cudaFree，看驱动调用有多频繁。
//   2. CachingAllocator 复用 —— 第二次同样负载 0 次新 cudaMalloc。
//   3. 碎片化与 coalesce —— 交错分配释放后出现的"洞"，以及合并如何救回。
//   4. kernel 正确性 + split/merge —— 类型只在你 reinterpret_cast 那一刻出现。
//
// 全程可见："C++ 拿到 void*" —— 分配器给的是无类型裸指针，我们在用的时候
// 才把它当成 float*/unsigned char*。

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "allocator.h"

// ---------------------------------------------------------------------------
// CUDA 工具与 kernel
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t e_ = (call);                                                  \
        if (e_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_),   \
                    __FILE__, __LINE__);                                          \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

// 最简单的写 kernel：一维 grid-stride 填常量。学习 grid/block 的基本切分。
template <typename T>
__global__ void fill_kernel(T* x, T v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

// 读两个数组写第三个：z = a*x + y。学习多缓冲读写。
__global__ void saxpy_kernel(const float* x, const float* y, float a, float* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = a * x[i] + y[i];
}

template <typename T>
static void launch_fill(T* d, T v, int n) {
    fill_kernel<T><<<(n + 255) / 256, 256>>>(d, v, n);
    CUDA_CHECK(cudaGetLastError());  // 启动即查错，避免 kernel 静默失败
}

// 把 device 数组拷回 host 并逐元素比对（教学用最直白的方式验证正确性）。
template <typename T>
static bool verify(const char* name, T* d, int n, const T* expect) {
    std::vector<T> host(n);
    CUDA_CHECK(cudaMemcpy(host.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        if (host[i] != expect[i]) {
            printf("  [FAIL] %s mismatch at %d: got %f want %f\n", name, i,
                   (double)host[i], (double)expect[i]);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// 打印辅助
// ---------------------------------------------------------------------------

static void print_stats(const char* tag, const Stats& st) {
    printf("%-24s allocated %7zuMB reserved %7zuMB peak %7zuMB | live %2zu free %2zu "
           "segs %2zu | cudaMalloc %zu cudaFree %zu\n",
           tag, st.allocated_bytes >> 20, st.reserved_bytes >> 20, st.peak_bytes >> 20,
           st.live_blocks, st.free_blocks, st.segments,
           st.cuda_malloc_calls, st.cuda_free_calls);
}

constexpr size_t MB = size_t(1) << 20;
static int g_fail = 0;

// ---------------------------------------------------------------------------
// Test 1 —— naive baseline：直连驱动 vs 缓存分配器，比谁 cudaMalloc 得少
// ---------------------------------------------------------------------------
static void test1_naive_baseline() {
    printf("\n=== Test 1: naive cudaMalloc vs CachingAllocator ===\n");
    const int ROUNDS = 100;
    const size_t SZ = 1 * MB;

    // ---- naive：每次分配都是一次驱动调用
    {
        int calls = 0;
        std::vector<float*> ptrs;
        for (int i = 0; i < ROUNDS; ++i) {
            float* p = nullptr;
            CUDA_CHECK(cudaMalloc(&p, SZ));
            ++calls;
            launch_fill(p, (float)i, 1 << 18);
            ptrs.push_back(p);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        for (float* p : ptrs) CUDA_CHECK(cudaFree(p));
        printf("  naive    : %d alloc/free of %zuMB  -> cudaMalloc called %d times\n",
               ROUNDS, SZ >> 20, calls);
    }

    // ---- caching allocator：释放的块回池，第二次分配命中同一个块
    {
        CachingAllocator a;
        for (int i = 0; i < ROUNDS; ++i) {
            void* p = a.alloc(SZ);
            launch_fill((float*)p, (float)i, 1 << 18);  // ← 类型在这一行才出现
            CUDA_CHECK(cudaDeviceSynchronize());
            a.free(p);
        }
        Stats st = a.stats();
        print_stats("caching allocator", st);
        printf("  caching  : %d alloc/free of %zuMB -> cudaMalloc called %zu times\n",
               ROUNDS, SZ >> 20, st.cuda_malloc_calls);
        if (st.cuda_malloc_calls == 1) printf("  [PASS] reuse works: 1 driver call for %d allocations\n", ROUNDS);
        else { printf("  [FAIL] expected 1 cudaMalloc, got %zu\n", st.cuda_malloc_calls); ++g_fail; }
    }
}

// ---------------------------------------------------------------------------
// Test 2 —— 稳态复用：warmup 后同样负载不再触发任何新 cudaMalloc
// ---------------------------------------------------------------------------
static void test2_reuse_steady_state() {
    printf("\n=== Test 2: steady-state reuse (0 extra driver calls) ===\n");
    CachingAllocator a;

    const size_t sizes[3] = {1 * MB, 2 * MB, 4 * MB};
    const int FLOATS[3] = {1 << 18, 1 << 19, 1 << 20};

    // warmup：从空设备分配一遍再全释放，让三个整段落进 free 池
    void* warm[3];
    for (int i = 0; i < 3; ++i) warm[i] = a.alloc(sizes[i]);
    for (int i = 0; i < 3; ++i) a.free(warm[i]);

    size_t calls_after_warmup = a.stats().cuda_malloc_calls;
    printf("  warmup done: cudaMalloc called %zu times (3 segments)\n", calls_after_warmup);

    // 记录第一次复用的指针，之后每一轮都应拿到相同指针（best-fit 确定性）
    void* ref[3];
    bool ok_ptr = true, ok_calls = true;
    for (int round = 0; round < 200; ++round) {
        void* p[3];
        for (int i = 0; i < 3; ++i) {
            p[i] = a.alloc(sizes[i]);
            launch_fill((float*)p[i], (float)(round + i), FLOATS[i]);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        if (round == 0)
            for (int i = 0; i < 3; ++i) ref[i] = p[i];
        else
            for (int i = 0; i < 3; ++i)
                if (p[i] != ref[i]) ok_ptr = false;
        for (int i = 0; i < 3; ++i) a.free(p[i]);
    }

    Stats st = a.stats();
    if (st.cuda_malloc_calls == calls_after_warmup) {
        printf("  [PASS] 200 rounds of alloc/free -> 0 extra cudaMalloc (%zu total)\n", st.cuda_malloc_calls);
    } else {
        printf("  [FAIL] driver calls grew: %zu -> %zu\n", calls_after_warmup, st.cuda_malloc_calls);
        ok_calls = false;
    }
    printf("  %s\n", ok_ptr ? "  [PASS] deterministic pointer reuse across rounds"
                            : "  [FAIL] pointer reuse not stable");
    if (!ok_ptr) ++g_fail;
    if (!ok_calls) ++g_fail;
    a.assert_consistency();
}

// ---------------------------------------------------------------------------
// Test 3 —— 碎片化 vs coalesce，以及 empty_cache
//
// 剧本：
//   用一个大段(256MB)切成 A,B,C,D(各48MB)+尾64MB；
//   释放 B,D 形成"洞" → 想分配 128MB 时洞不够连续 → 被迫新开一段（碎片化！）
//   释放剩余后 coalesce 把它们合并回一个 256MB 大块 → 128MB(此剧本用 192MB)
//   分配直接复用，不再新开段。
// ---------------------------------------------------------------------------
static void test3_fragmentation_coalesce_empty_cache() {
    printf("\n=== Test 3: fragmentation, coalescing, empty_cache ===\n");
    CachingAllocator a;

    // 先造一个大空闲块当"容器段"
    void* seed = a.alloc(256 * MB);
    a.free(seed);

    void* A = a.alloc(48 * MB);
    void* B = a.alloc(48 * MB);
    void* C = a.alloc(48 * MB);
    void* D = a.alloc(48 * MB);
    print_stats("A,B,C,D 48MB carved", a.stats());

    // 制造两个洞：free B 和 D。此时段内空闲是分散的：洞(48)+洞(48)+尾(64)
    a.free(B);
    a.free(D);
    print_stats("free B,D (two holes)", a.stats());
    {
        // 空闲字节总量其实够 128MB，但被两个洞 + 尾部分散 —— 下面用 alloc 探测碎片
        Stats st = a.stats();
        printf("   total free bytes = %zuMB (but fragmented into several holes)\n",
               (st.reserved_bytes - st.allocated_bytes) >> 20);
    }

    // 分配 128MB：分散的洞拼不出 → 必须新开一段（碎片化代价）
    void* E = a.alloc(128 * MB);
    size_t before = a.stats().cuda_malloc_calls;
    printf("  alloc E=128MB with fragmented holes -> cudaMalloc called %zu times\n", before);
    if (before == 2) printf("  [PASS] fragmentation forced a new segment (holes not contiguous)\n");
    else { printf("  [FAIL] expected 1 new segment, total %zu\n", before); ++g_fail; }
    print_stats("E=128MB allocated", a.stats());

    // 释放 A、C、E：A 与洞B合并、再与 C 合并、再与洞D+尾合并 → 一个 256MB 大块
    a.free(A);
    a.free(C);
    a.free(E);
    print_stats("free A,C,E (coalesced)", a.stats());

    // 分配 192MB：现在有一个 256MB 连续块可复用 → 不应再 cudaMalloc
    void* F = a.alloc(192 * MB);
    Stats after = a.stats();
    print_stats("F=192MB reused", after);
    if (after.cuda_malloc_calls == before)
        printf("  [PASS] coalescing reunited holes: 192MB served without new segment\n");
    else { printf("  [FAIL] still needed new segment: %zu\n", after.cuda_malloc_calls); ++g_fail; }

    // 全部释放 + empty_cache：reserved 应归零（整段都还给了驱动）
    a.free(F);
    a.empty_cache();
    Stats st2 = a.stats();
    printf("  after empty_cache: reserved=%zuMB segments=%zu cudaFree=%zu\n",
           st2.reserved_bytes >> 20, st2.segments, st2.cuda_free_calls);
    if (st2.reserved_bytes == 0 && st2.segments == 0)
        printf("  [PASS] empty_cache returned all memory to the driver\n");
    else { printf("  [FAIL] reserved=%zu segments=%zu\n", st2.reserved_bytes, st2.segments); ++g_fail; }

    // empty_cache 之后 allocator 仍可用
    void* again = a.alloc(1 * MB);
    if (again) printf("  [PASS] allocator still works after empty_cache\n");
    else { printf("  [FAIL] alloc failed after empty_cache\n"); ++g_fail; }
    a.free(again);
    a.assert_consistency();
}

// ---------------------------------------------------------------------------
// Test 4 —— kernel 正确性 + 跨"类型"复用 + split/merge 在真实读写下验证
// ---------------------------------------------------------------------------
static void test4_kernel_correctness() {
    printf("\n=== Test 4: kernel correctness across reuse ===\n");
    CachingAllocator a;

    // ---- Phase A：基本 kernel 正确性
    const int N = 1 << 20;  // 4MB floats
    float* x = (float*)a.alloc(N * sizeof(float));
    float* y = (float*)a.alloc(N * sizeof(float));
    float* z = (float*)a.alloc(N * sizeof(float));
    launch_fill(x, 1.5f, N);
    launch_fill(y, 2.0f, N);
    saxpy_kernel<<<(N + 255) / 256, 256>>>(x, y, 3.0f, z, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> expz(N, 6.5f);
    std::vector<float> expxy(N, 1.5f);
    if (verify("z=3x+y", z, N, expz.data()) && verify("x", x, N, expxy.data())) {
        printf("  [PASS] Phase A: saxpy correctness (z = 3*1.5 + 2.0 = 6.5)\n");
    } else ++g_fail;

    // ---- Phase B：把刚释放的 4MB 当"另一种类型"复用（char 数组）
    a.free(x);
    a.free(z);

    const int NB = 4 << 20;  // 4MB 字节 —— 和上面两个 4MB float 段同大小
    unsigned char* buf = (unsigned char*)a.alloc(NB);
    launch_fill(buf, (unsigned char)0xAB, NB);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<unsigned char> expb(NB, 0xAB);
    if (verify("buf as u8", buf, NB, expb.data()))
        printf("  [PASS] Phase B: freed float block reused as unsigned char (no stale data)\n");
    else ++g_fail;

    // 再换回 float：同一块内存，先被 0xAB 覆盖，现在应该装的是 42.0
    a.free(buf);
    float* g = (float*)a.alloc((NB / 4) * sizeof(float));
    launch_fill(g, 42.0f, NB / 4);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> expg(NB / 4, 42.0f);
    if (verify("g as f32", g, NB / 4, expg.data()))
        printf("  [PASS] Phase B: same bytes reinterpreted float->u8->float correctly\n");
    else ++g_fail;
    a.free(g);
    a.free(y);

    // ---- Phase C：split / merge（一个段内 10MB -> 切 3+2 -> 释放 3 合并回 8 -> 复用 7）
    void* c10 = a.alloc(10 * MB);
    a.free(c10);                          // 腾出一个整 10MB 空闲段
    size_t m0 = a.stats().cuda_malloc_calls;

    void* a3 = a.alloc(3 * MB);           // 切: 10 -> 3(用) + 7(空)
    void* a2 = a.alloc(2 * MB);           // 切:  7 -> 2(用) + 5(空)
    a.free(a3);                           // 3 与右邻 5 合并 -> 8(空)
    void* a7 = a.alloc(7 * MB);           // 复用 8（>=7 再切） -> 不该新开段
    Stats cst = a.stats();
    print_stats("split/merge workout", cst);
    if (cst.cuda_malloc_calls == m0)
        printf("  [PASS] Phase C: split then coalesce served 7MB without new segment\n");
    else { printf("  [FAIL] split/merge needed extra cudaMalloc: %zu\n", cst.cuda_malloc_calls); ++g_fail; }

    // 在这些进进出出的块上跑真实 kernel 确认没算错账
    launch_fill((unsigned char*)a7, (unsigned char)7, 7 << 20);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<unsigned char> e7(7 << 20, 7);
    if (verify("a7 after churn", (unsigned char*)a7, 7 << 20, e7.data()))
        printf("  [PASS] Phase C: kernel write survives allocator churn\n");
    else ++g_fail;

    a.free(a2);
    a.free(a7);
    a.assert_consistency();
}

int main() {
    // 选卡：默认 device0（这台机器上是 4090），可用 CUDA_DEVICE=4 切到 H200。
    int dev = 0;
    if (const char* e = getenv("CUDA_DEVICE")) dev = atoi(e);
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("device %d: %s  (CC %d.%d, %zu MB)\n\n",
           dev, prop.name, prop.major, prop.minor, (size_t)prop.totalGlobalMem >> 20);
    printf("mental model: the allocator trades in raw void* (like torch DataPtr).\n"
           "A type is only applied at reinterpret_cast time inside each test.\n");

    test1_naive_baseline();
    test2_reuse_steady_state();
    test3_fragmentation_coalesce_empty_cache();
    test4_kernel_correctness();

    printf("\n==========================================\n");
    if (g_fail == 0) {
        printf("ALL TESTS PASSED\n");
        return 0;
    }
    printf("%d CHECK(S) FAILED\n", g_fail);
    return 1;
}
