// Shared (header-only) helpers for the flash-attention kernels.
//
// Every `kernels/*.cu` includes this file. All state is `inline` so the three
// translation units share a single thread-local error buffer and the linker
// keeps exactly one copy of each definition.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstddef>
#include <cstdio>

// Return codes shared by all kernels (mirrored in `src/ffi.rs`).
enum flash_attn_status {
    FLASH_ATTN_OK = 0,
    FLASH_ATTN_ERR_NULL_PTR = 1,
    FLASH_ATTN_ERR_INVALID_SHAPE = 2,
    FLASH_ATTN_ERR_INVALID_HEAD_SIZE = 3,
    FLASH_ATTN_ERR_INVALID_START_POS = 4,
    FLASH_ATTN_ERR_CUDA = 5,
};

namespace flash_attn_detail {

inline thread_local char g_last_error[256] = {0};

inline void clear_last_error() { g_last_error[0] = '\0'; }

inline void set_last_error(const char* msg) {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s", msg);
}

// Sets MaxDynamicSharedMemorySize when needed; returns false on failure.
template <typename Kernel>
inline bool ensure_smem(Kernel kernel, size_t smem) {
    if (smem <= 48 * 1024) {
        return true;
    }
    cudaError_t err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)
    );
    if (err != cudaSuccess) {
        set_last_error(cudaGetErrorString(err));
        return false;
    }
    return true;
}

constexpr int kBlockN = 64;  // Bc: KV tile along the sequence
constexpr int kBlockM = 64;  // Br: query tile along the sequence

// Storage type <-> f32. Compute always happens in f32; half/bfloat16 only
// affect storage, matching the framework's "native storage, f32 compute" policy.
template <typename T>
__device__ __forceinline__ float f_to_float(T v) {
    return static_cast<float>(v);
}
template <>
__device__ __forceinline__ float f_to_float<__half>(__half v) {
    return __half2float(v);
}
template <>
__device__ __forceinline__ float f_to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T f_from_float(float v) {
    return static_cast<T>(v);
}
template <>
__device__ __forceinline__ __half f_from_float<__half>(float v) {
    return __float2half_rn(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 f_from_float<__nv_bfloat16>(float v) {
    return __float2bfloat16_rn(v);
}

// Logical KV position `j` of sequence `s` -> physical slot in the pooled cache.
// 1. block_tables (num_seqs, max_blocks)，对每个 seq 有 max_blocks 个值表示每个逻辑块在实际块的索引
// 2. s 表示 seq 索引
// 3. max_blocks
// 4. j 是 token 在 seq 中的逻辑索引，我们要的就是这个逻辑索引的真实物理索引
// 5. block_size
__device__ __forceinline__ int block_slot(
    const int* __restrict__ block_tables,
    int s,
    int max_blocks,
    int j,
    int block_size
) {
    // j 是在整个 seq 的逻辑 index，按照 block 划分，
    // logical_block 是 j 的 逻辑 block 索引
    // offset 是 j 在 block 内部的偏移（对物理/逻辑都是一致的）
    const int logical_block = j / block_size;
    const int offset = j % block_size;
    // s * max_blocks 得到当前 seq 的起始位置
    // logitcal_block 表示用逻辑块索引查表，
    // 所以 block_tables[s * max_blocks + logical_block] 得到的是 j 这个索引所在物理块的 block index
    // 大 cache 的 shape 是  (max_blocks, block_size, kv_heads, head_size)
    // 在 token 的角度，这个 shape 是 (max_blocks, block_size)
    // 最后返回的就是 token 相对这个的索引（token slot）
    return block_tables[s * max_blocks + logical_block] * block_size + offset;
}

// Warp-wide sum reduction (32 lanes).
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

}  // namespace flash_attn_detail
