#include <cmath>
#include <cstddef>
#include <cstdint>
#include "./utils.cuh"

#define WARP_SIZE 32
const int BLOCK_SIZE = 1024;

template <typename T> __device__ void sum(
    const size_t src_numel, 
    const size_t el_to_sum_per_block,
    const size_t num_dims, 
    const size_t* dims,
    const size_t* strides,
    const T* src, T* dst 
) {
    __shared__ T shr[BLOCK_SIZE];
    size_t tid = threadIdx.x;
    size_t dst_id = blockIdx.x;

    shr[tid] = 0;
    size_t start_idx = dst_id * el_to_sum_per_block;
    size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
    size_t idx = start_idx + tid;

    while (idx < stop_idx) {
        size_t phy_idx = logical_index_to_physical_index(idx, num_dims, dims, strides);
        shr[tid] += src[phy_idx];
        idx += blockDim.x;
    }

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) shr[tid] += shr[tid + s];
    }

    if (tid == 0) {
        dst[dst_id] = shr[0];
    }
}