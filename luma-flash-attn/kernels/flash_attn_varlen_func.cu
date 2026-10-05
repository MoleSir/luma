#include "common.cuh"

#include <cstdint>

// ---------------------------------------------------------------------------
// flash_attn_varlen_func: packed variable-length prefill attention.
//
// Layouts (all contiguous):
//   q   (total_q,   q_heads,  head_size)
//   o   (total_q,   q_heads,  head_size)
//   k/v contiguous: (total_kv, kv_heads, head_size)
//   k/v paged:      (num_blocks, block_size, kv_heads, head_size)   (optional prefix cache)
//   cu_seqlens_q/k: (num_seqs + 1,)            i32
//   block_tables:   (num_seqs, max_blocks)     i32   (only when k_paged)
//
// Causal is bottom-right aligned: sequence `s` has q_len and k_len, and the
// first query token attends to KV index (k_len - q_len).
// ---------------------------------------------------------------------------

namespace {

using flash_attn_detail::block_slot;
using flash_attn_detail::ensure_smem;
using flash_attn_detail::f_from_float;
using flash_attn_detail::f_to_float;
using flash_attn_detail::set_last_error;

template <typename T, const int H>
__global__ void flash_attn_varlen_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    T* __restrict__ o,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ cu_seqlens_k,
    const int* __restrict__ block_tables,  // may be nullptr (no prefix cache)
    int num_seqs,
    int max_seqlen_q,
    int max_seqlen_k,
    int q_heads,
    int kv_heads,
    int block_size,
    int max_blocks,
    int k_paged,
    float scale
) {
    constexpr int Bc = flash_attn_detail::kBlockN;
    constexpr int Br = flash_attn_detail::kBlockM;

    const int tid = threadIdx.x;
    const int s = blockIdx.x;
    const int q_head = blockIdx.y;
    const int q_tile = blockIdx.z;

    const int q_start = cu_seqlens_q[s];
    const int q_len = cu_seqlens_q[s + 1] - q_start;
    const int kv_start = cu_seqlens_k[s];
    const int k_len = cu_seqlens_k[s + 1] - kv_start;
    const int start_pos = k_len - q_len;  // right-aligned causal

    const int row = q_tile * Br + tid;
    const int q_global = start_pos + row;

    const int group_size = q_heads / kv_heads;
    const int kv_head = q_head / group_size;
    const int D = kv_heads * H;

    const bool row_valid = (row < q_len);
    const long q_off = (long)(q_start + row) * q_heads * H + (long)q_head * H;

    extern __shared__ float sram[];
    float* k_block = sram;              // (Bc, H)
    float* v_block = sram + (Bc * H);   // (Bc, H)

    float qi[H];
    float oi[H];
    if (row_valid) {
#pragma unroll
        for (int x = 0; x < H; ++x) qi[x] = f_to_float(q[q_off + x]);
    }
#pragma unroll
    for (int x = 0; x < H; ++x) oi[x] = 0.0f;

    float m_prev = -CUDART_INF_F;
    float l_prev = 0.0f;

    const int Tc = (k_len + Bc - 1) / Bc;
    const int q_global_max = start_pos + q_tile * Br + (Br - 1);

    for (int jt = 0; jt < Tc; ++jt) {
        if (jt * Bc > q_global_max) break;

        for (int i = tid; i < Bc * H; i += Br) {
            const int r = i / H;
            const int col = i % H;
            const int row_g = jt * Bc + r;
            if (row_g < k_len) {
                const T* kbase;
                const T* vbase;
                if (k_paged) {
                    const int slot = block_slot(block_tables, s, max_blocks, row_g, block_size);
                    kbase = k + (long)slot * D + (long)kv_head * H;
                    vbase = v + (long)slot * D + (long)kv_head * H;
                } else {
                    const long off = (long)(kv_start + row_g) * D + (long)kv_head * H;
                    kbase = k + off;
                    vbase = v + off;
                }
                k_block[i] = f_to_float(kbase[col]);
                v_block[i] = f_to_float(vbase[col]);
            } else {
                k_block[i] = 0.0f;
                v_block[i] = 0.0f;
            }
        }
        __syncthreads();

        if (row_valid) {
            float m_curr = -CUDART_INF_F;
            float sc[Bc];

            for (int y = 0; y < Bc; ++y) {
                const int kv_idx = jt * Bc + y;
                const bool visible = (kv_idx < k_len) && (q_global >= kv_idx);
                float score = -CUDART_INF_F;
                if (visible) {
                    float sum = 0.0f;
#pragma unroll
                    for (int x = 0; x < H; ++x) sum += qi[x] * k_block[y * H + x];
                    score = sum * scale;
                }
                sc[y] = score;
                m_curr = fmaxf(m_curr, score);
            }

            float l_curr = 0.0f;
            for (int y = 0; y < Bc; ++y) {
                const float p = (sc[y] == -CUDART_INF_F) ? 0.0f : __expf(sc[y] - m_curr);
                sc[y] = p;
                l_curr += p;
            }

            if (m_curr != -CUDART_INF_F) {
                const float m_new = fmaxf(m_prev, m_curr);
                const float alpha = __expf(m_prev - m_new);
                const float beta = __expf(m_curr - m_new);
                const float l_new = alpha * l_prev + beta * l_curr;
                for (int x = 0; x < H; ++x) {
                    float pv = 0.0f;
                    for (int y = 0; y < Bc; ++y) pv += sc[y] * v_block[y * H + x];
                    oi[x] = (alpha * l_prev * oi[x] + beta * pv) / l_new;
                }
                m_prev = m_new;
                l_prev = l_new;
            }
        }

        __syncthreads();
    }

    if (row_valid) {
        const bool write = (l_prev > 0.0f);
#pragma unroll
        for (int x = 0; x < H; ++x) {
            o[q_off + x] = write ? f_from_float<T>(oi[x]) : f_from_float<T>(0.0f);
        }
    }
    (void)num_seqs; (void)max_seqlen_q; (void)max_seqlen_k;
}

template <typename T, const int H>
int launch_varlen(
    const T* q, const T* k, const T* v, T* o,
    const int* cu_seqlens_q, const int* cu_seqlens_k, const int* block_tables,
    int num_seqs, int max_seqlen_q, int max_seqlen_k,
    int q_heads, int kv_heads, int block_size, int max_blocks, int k_paged,
    float scale, cudaStream_t stream
) {
    constexpr int Bc = flash_attn_detail::kBlockN;
    constexpr int Br = flash_attn_detail::kBlockM;
    const size_t smem = 2 * Bc * H * sizeof(float);

    if (!ensure_smem(flash_attn_varlen_kernel<T, H>, smem)) {
        return FLASH_ATTN_ERR_CUDA;
    }

    dim3 grid(num_seqs, q_heads, (max_seqlen_q + Br - 1) / Br);
    dim3 block(Br);
    flash_attn_varlen_kernel<T, H><<<grid, block, smem, stream>>>(
        q, k, v, o, cu_seqlens_q, cu_seqlens_k, block_tables,
        num_seqs, max_seqlen_q, max_seqlen_k, q_heads, kv_heads,
        block_size, max_blocks, k_paged, scale
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
        set_last_error("flash_attn_varlen: num_seqs/q_heads/kv_heads must be > 0");
        return false;
    }
    if (q_heads % kv_heads != 0) {
        set_last_error("flash_attn_varlen: q_heads must be a multiple of kv_heads (GQA)");
        return false;
    }
    if (head_size != 32 && head_size != 64 && head_size != 128) {
        set_last_error("flash_attn_varlen: head_size must be 32/64/128");
        return false;
    }
    return true;
}

template <typename T>
int flash_attn_varlen_impl(
    const T* q, const T* k, const T* v, T* o,
    const int* cu_seqlens_q, const int* cu_seqlens_k, const int* block_tables,
    int num_seqs, int max_seqlen_q, int max_seqlen_k,
    int q_heads, int kv_heads, int head_size,
    int block_size, int max_blocks, int k_paged,
    float scale, cudaStream_t stream
) {
    flash_attn_detail::clear_last_error();
    cudaGetLastError();

    if (q == nullptr || o == nullptr || cu_seqlens_q == nullptr || cu_seqlens_k == nullptr) {
        set_last_error("flash_attn_varlen: q/o/cu_seqlens must not be null");
        return FLASH_ATTN_ERR_NULL_PTR;
    }
    if ((k_paged && (k == nullptr || v == nullptr || block_tables == nullptr)) ||
        (!k_paged && (k == nullptr || v == nullptr))) {
        set_last_error("flash_attn_varlen: k/v (and block_tables when paged) must not be null");
        return FLASH_ATTN_ERR_NULL_PTR;
    }
    if (!check_shape_common(num_seqs, q_heads, kv_heads, head_size)) {
        return FLASH_ATTN_ERR_INVALID_SHAPE;
    }
    if (max_seqlen_q <= 0 || max_seqlen_k <= 0 || block_size <= 0) {
        set_last_error("flash_attn_varlen: max_seqlen_q/k and block_size must be > 0");
        return FLASH_ATTN_ERR_INVALID_SHAPE;
    }

    switch (head_size) {
        case 32:
            return launch_varlen<T, 32>(q, k, v, o, cu_seqlens_q, cu_seqlens_k, block_tables,
                                        num_seqs, max_seqlen_q, max_seqlen_k, q_heads, kv_heads,
                                        block_size, max_blocks, k_paged, scale, stream);
        case 64:
            return launch_varlen<T, 64>(q, k, v, o, cu_seqlens_q, cu_seqlens_k, block_tables,
                                        num_seqs, max_seqlen_q, max_seqlen_k, q_heads, kv_heads,
                                        block_size, max_blocks, k_paged, scale, stream);
        case 128:
            return launch_varlen<T, 128>(q, k, v, o, cu_seqlens_q, cu_seqlens_k, block_tables,
                                         num_seqs, max_seqlen_q, max_seqlen_k, q_heads, kv_heads,
                                         block_size, max_blocks, k_paged, scale, stream);
        default:
            return FLASH_ATTN_ERR_INVALID_HEAD_SIZE;
    }
}

}  // namespace

#define DEFINE_FLASH_ATTN_VARLEN_ENTRY(NAME, TYPE)                                                       \
    extern "C" int NAME(                                                                                 \
        const TYPE* q, const TYPE* k, const TYPE* v, TYPE* o,                                           \
        const int* cu_seqlens_q, const int* cu_seqlens_k, const int* block_tables,                      \
        int num_seqs, int max_seqlen_q, int max_seqlen_k,                                               \
        int q_heads, int kv_heads, int head_size,                                                       \
        int block_size, int max_blocks, int k_paged,                                                    \
        float scale, void* stream                                                                       \
    ) noexcept {                                                                                         \
        return flash_attn_varlen_impl<TYPE>(                                                            \
            q, k, v, o, cu_seqlens_q, cu_seqlens_k, block_tables,                                       \
            num_seqs, max_seqlen_q, max_seqlen_k, q_heads, kv_heads, head_size,                         \
            block_size, max_blocks, k_paged, scale, reinterpret_cast<cudaStream_t>(stream)              \
        );                                                                                              \
    }

DEFINE_FLASH_ATTN_VARLEN_ENTRY(flash_attn_varlen_f32, float)
DEFINE_FLASH_ATTN_VARLEN_ENTRY(flash_attn_varlen_f16, __half)
DEFINE_FLASH_ATTN_VARLEN_ENTRY(flash_attn_varlen_bf16, __nv_bfloat16)

#undef DEFINE_FLASH_ATTN_VARLEN_ENTRY
