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
__device__ __forceinline__ int block_slot(
    const int* __restrict__ block_tables,
    int s,
    int max_blocks,
    int j,
    int block_size
) {
    const int logical_block = j / block_size;
    const int offset = j % block_size;
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
