#include "common.cuh"

#include <cstdint>

// ---------------------------------------------------------------------------
// flash_attn_with_kvcache: single-query-token decode reading a paged KV cache.
//
// Layouts (all contiguous):
//   q       (num_seqs, q_heads,  head_size)
//   o       (num_seqs, q_heads,  head_size)
//   k_cache (num_blocks, block_size, kv_heads, head_size)
//   v_cache (num_blocks, block_size, kv_heads, head_size)
//   block_tables: (num_seqs, max_blocks)                            i32
//   context_lens: (num_seqs,)                                       i32
// ---------------------------------------------------------------------------

namespace {

using flash_attn_detail::block_slot;
using flash_attn_detail::f_from_float;
using flash_attn_detail::f_to_float;
using flash_attn_detail::set_last_error;
using flash_attn_detail::warp_reduce_sum;

template <typename T, const int H>
__global__ void flash_attn_decode_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k_cache,
    const T* __restrict__ v_cache,
    T* __restrict__ o,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int num_seqs,
    int q_heads,
    int kv_heads,
    int block_size,
    int max_blocks,
    float scale
) {
    constexpr int ELEMS = H / 32;  // head-dim elements handled per lane

    const int s = blockIdx.x;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (warp >= q_heads) return;
    const int q_head = warp;

    const int group_size = q_heads / kv_heads;
    const int kv_head = q_head / group_size;
    const int D = kv_heads * H;
    const int L = context_lens[s];

    const long q_off = (long)s * q_heads * H + (long)q_head * H;

    float qi[ELEMS];
    float acc[ELEMS];
#pragma unroll
    for (int e = 0; e < ELEMS; ++e) {
        qi[e] = f_to_float(q[q_off + lane + e * 32]);
        acc[e] = 0.0f;
    }

    float m = -CUDART_INF_F;
    float l = 0.0f;

    for (int j = 0; j < L; ++j) {
        const int slot = block_slot(block_tables, s, max_blocks, j, block_size);
        const T* kbase = k_cache + (long)slot * D + (long)kv_head * H;
        const T* vbase = v_cache + (long)slot * D + (long)kv_head * H;

        float partial = 0.0f;
#pragma unroll
        for (int e = 0; e < ELEMS; ++e) partial += qi[e] * f_to_float(kbase[lane + e * 32]);
        const float dot = warp_reduce_sum(partial);
        const float score = __shfl_sync(0xffffffffu, dot, 0) * scale;

        const float m_new = fmaxf(m, score);
        const float alpha = __expf(m - m_new);
        const float p = __expf(score - m_new);
        l = l * alpha + p;
#pragma unroll
        for (int e = 0; e < ELEMS; ++e) acc[e] = acc[e] * alpha + p * f_to_float(vbase[lane + e * 32]);
        m = m_new;
    }

    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
#pragma unroll
    for (int e = 0; e < ELEMS; ++e) o[q_off + lane + e * 32] = f_from_float<T>(acc[e] * inv);

    (void)num_seqs;
}

template <typename T, const int H>
int launch_decode(
    const T* q, const T* k_cache, const T* v_cache, T* o,
    const int* block_tables, const int* context_lens,
    int num_seqs, int q_heads, int kv_heads, int block_size, int max_blocks,
    float scale, cudaStream_t stream
) {
    dim3 grid(num_seqs);
    dim3 block(q_heads * 32);
    flash_attn_decode_kernel<T, H><<<grid, block, 0, stream>>>(
        q, k_cache, v_cache, o, block_tables, context_lens,
        num_seqs, q_heads, kv_heads, block_size, max_blocks, scale
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_last_error(cudaGetErrorString(err));
        return FLASH_ATTN_ERR_CUDA;
    }
    return FLASH_ATTN_OK;
}

bool check_shape_common(int num_seqs, int q_heads, int kv_heads, int head_size) {
    if (num_seqs <= 0 || q_heads <= 0 || kv_heads <= 0) {
        set_last_error("flash_attn_with_kvcache: num_seqs/q_heads/kv_heads must be > 0");
        return false;
    }
    if (q_heads % kv_heads != 0) {
        set_last_error("flash_attn_with_kvcache: q_heads must be a multiple of kv_heads (GQA)");
        return false;
    }
    if (head_size != 32 && head_size != 64 && head_size != 128) {
        set_last_error("flash_attn_with_kvcache: head_size must be 32/64/128");
        return false;
    }
    return true;
}

template <typename T>
int flash_attn_with_kvcache_impl(
    const T* q, const T* k_cache, const T* v_cache, T* o,
    const int* block_tables, const int* context_lens,
    int num_seqs, int q_heads, int kv_heads, int head_size,
    int block_size, int max_blocks,
    float scale, cudaStream_t stream
) {
    flash_attn_detail::clear_last_error();
    cudaGetLastError();

    if (q == nullptr || k_cache == nullptr || v_cache == nullptr || o == nullptr ||
        block_tables == nullptr || context_lens == nullptr) {
        set_last_error("flash_attn_with_kvcache: null pointer argument");
        return FLASH_ATTN_ERR_NULL_PTR;
    }
    if (!check_shape_common(num_seqs, q_heads, kv_heads, head_size)) {
        return FLASH_ATTN_ERR_INVALID_SHAPE;
    }
    if (block_size <= 0) {
        set_last_error("flash_attn_with_kvcache: block_size must be > 0");
        return FLASH_ATTN_ERR_INVALID_SHAPE;
    }

    switch (head_size) {
        case 32:
            return launch_decode<T, 32>(q, k_cache, v_cache, o, block_tables, context_lens,
                                        num_seqs, q_heads, kv_heads, block_size, max_blocks, scale, stream);
        case 64:
            return launch_decode<T, 64>(q, k_cache, v_cache, o, block_tables, context_lens,
                                        num_seqs, q_heads, kv_heads, block_size, max_blocks, scale, stream);
        case 128:
            return launch_decode<T, 128>(q, k_cache, v_cache, o, block_tables, context_lens,
                                         num_seqs, q_heads, kv_heads, block_size, max_blocks, scale, stream);
        default:
            return FLASH_ATTN_ERR_INVALID_HEAD_SIZE;
    }
}

}  // namespace

#define DEFINE_FLASH_ATTN_KVCACHE_ENTRY(NAME, TYPE)                                                      \
    extern "C" int NAME(                                                                                 \
        const TYPE* q, const TYPE* k_cache, const TYPE* v_cache, TYPE* o,                               \
        const int* block_tables, const int* context_lens,                                               \
        int num_seqs, int q_heads, int kv_heads, int head_size,                                         \
        int block_size, int max_blocks,                                                                 \
        float scale, void* stream                                                                       \
    ) noexcept {                                                                                         \
        return flash_attn_with_kvcache_impl<TYPE>(                                                      \
            q, k_cache, v_cache, o, block_tables, context_lens,                                         \
            num_seqs, q_heads, kv_heads, head_size, block_size, max_blocks,                             \
            scale, reinterpret_cast<cudaStream_t>(stream)                                               \
        );                                                                                              \
    }

DEFINE_FLASH_ATTN_KVCACHE_ENTRY(flash_attn_with_kvcache_f32, float)
DEFINE_FLASH_ATTN_KVCACHE_ENTRY(flash_attn_with_kvcache_f16, __half)
DEFINE_FLASH_ATTN_KVCACHE_ENTRY(flash_attn_with_kvcache_bf16, __nv_bfloat16)

#undef DEFINE_FLASH_ATTN_KVCACHE_ENTRY
