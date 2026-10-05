#include "common.cuh"
#include <cstdint>

namespace {

using flash_attn_detail::block_slot;
using flash_attn_detail::f_from_float;
using flash_attn_detail::f_to_float;
using flash_attn_detail::set_last_error;
using flash_attn_detail::warp_reduce_sum;

//   q       (num_seqs, q_heads,  head_size)          -- seqlen_q == 1 (decode)
//   o       (num_seqs, q_heads,  head_size)
//   k_cache (num_blocks, block_size, kv_heads, head_size)
//   v_cache (num_blocks, block_size, kv_heads, head_size)
//   block_tables: (num_seqs, max_blocks)                            i32
//   context_lens: (num_seqs,)                                       i32
//
// `seqlen_q > 1` (prefill) is handled by `flash_attn_prefill_kernel` below;
// the C entry point dispatches on `q_seq_len`.
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

    // seq 索引
    const int s = blockIdx.x;
    // threadDim.x = q_heads * 32
    // threadIdx.x / 32 就是 thread 负责的 q_head 索引
    // threadIdx.x % 32 表示用 32 个 thread 处理一个 q_head 
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (warp >= q_heads) return;
    const int q_head = warp;

    const int group_size = q_heads / kv_heads;
    const int kv_head = q_head / group_size;
    const int D = kv_heads * H;
    // context_lens[s] 获取当前处理的 seq 的 kv 长度
    const int L = context_lens[s];

    // 计算 q 的偏移，(num_seqs, q_heads, head_size)
    // q[s, q_head, :] -> 得到这个 thread 负责的 token
    const long q_off = (long)s * q_heads * H + (long)q_head * H;

    // 但一个 token 由 32 个 thread 一起完成，每个 thread 只需要 ELEMS 个 thread
    float qi[ELEMS];
    float acc[ELEMS];
#pragma unroll
    for (int e = 0; e < ELEMS; ++e) {
        qi[e] = f_to_float(q[q_off + lane + e * 32]);
        acc[e] = 0.0f;
    }

    float m = -CUDART_INF_F;
    float l = 0.0f;

    // 循环 L 次，每次处理一个 kv token，进行 online softmax
    for (int j = 0; j < L; ++j) {
        // 传入 j 表示当前 kv token 的逻辑 token index，得到物理 slot
        const int slot = block_slot(block_tables, s, max_blocks, j, block_size);
        // 计算 k/v 的位置
        const T* kbase = k_cache + (long)slot * D + (long)kv_head * H;
        const T* vbase = v_cache + (long)slot * D + (long)kv_head * H;

        // 计算 q 和 k 的 dot，再次说明，这是多个 thread 一起完成的，每个 thread 算 ELEMS 得到部分结果
        float partial = 0.0f;
#pragma unroll
        for (int e = 0; e < ELEMS; ++e) {
            partial += qi[e] * f_to_float(kbase[lane + e * 32]);
        }
        // 正好连续 32 个 thread 合作处理一个 q token，利用 warp 求和得到 q @ k，只有 thread lane == 0 才是正确的
        const float dot = warp_reduce_sum(partial);
        // 每个 thread 都去取 lane 0 的正确结果然后 * scale
        const float score = __shfl_sync(0xffffffffu, dot, 0) * scale;

        // 计算新最大值
        const float m_new = fmaxf(m, score);
        // 校准
        const float alpha = __expf(m - m_new);
        // 计算 exp
        const float p = __expf(score - m_new);
        // 更新 l
        l = l * alpha + p;

        // 计算和 v 的矩阵乘
#pragma unroll
        for (int e = 0; e < ELEMS; ++e) {
            acc[e] = acc[e] * alpha + p * f_to_float(vbase[lane + e * 32]);
        }

        // 更新 m
        m = m_new;
    }

    // 之前计算的少一个 分母
    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
#pragma unroll
    for (int e = 0; e < ELEMS; ++e) {
        o[q_off + lane + e * 32] = f_from_float<T>(acc[e] * inv);
    }

    (void)num_seqs;
}

/// - `q`: `(batch, 1, q_num_heads, head_size)`
/// - `k_cache` / `v_cache`: `(num_blocks, block_size, kv_num_heads, head_size)`
/// - `cache_seqlens`: `(batch,)` `i32` (valid cached length per sequence)
/// - `block_table`: `(batch, max_blocks)` `i32` (logical block -> physical block)
/// - returns: `(batch, 1, q_num_heads, head_size)`
template <typename T, const int H>
int launch_decode(
    const T* q, const T* k_cache, const T* v_cache, T* o,
    const int* block_tables, const int* context_lens,
    int num_seqs, int q_heads, int kv_heads, int block_size, int max_blocks,
    float scale, cudaStream_t stream
) {
    dim3 grid(num_seqs);
    dim3 block(q_heads * 32);
    // 每个 block 负责计算一条 seq，每个 seq 只有一个 q，每个 block 使用 q_heads * 32 个 thread 计算
    // 那么分配给 head_size 的是 32 个 thread 完成一个 q 的 head_size
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

/// - `q`: `(batch, q_seq_len, q_heads, head_size)` (`q_seq_len > 1`)
/// - `k_cache` / `v_cache`: `(num_blocks, block_size, kv_heads, head_size)`
/// - `context_lens`: `(batch,)` `i32` — TOTAL cached length, INCLUDING the q
///   tokens (their K/V must already be present in the cache)
/// - `block_tables`: `(batch, max_blocks)` `i32` (logical block -> physical)
/// - returns: `(batch, q_seq_len, q_heads, head_size)`
///
/// Causal is bottom-right aligned: local query row `i` sits at global position
/// `context_lens[s] - q_seq_len + i` and attends to keys `j <= that`.
template <typename T, const int Bc, const int Br, const int H>
__global__ void flash_attn_prefill_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k_cache,
    const T* __restrict__ v_cache,
    T* __restrict__ o,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int q_seq_len,
    int q_heads,
    int kv_heads,
    int block_size,
    int max_blocks,
    float scale
) {
    const int tid = threadIdx.x;
    const int s = blockIdx.x;
    const int q_head = blockIdx.y;
    const int q_tile = blockIdx.z;

    const int group_size = q_heads / kv_heads;
    const int kv_head = q_head / group_size;
    const int D = kv_heads * H;
    const int L = context_lens[s];

    // 每个 thread 负责一行 q token（全局行号）
    const int row = q_tile * Br + tid;
    const bool row_valid = (row < q_seq_len);
    // 右下对齐：本行在整条序列中的全局位置
    const int g = L - q_seq_len + row;

    const long q_off = (long)(s * q_seq_len + row) * q_heads * H + (long)q_head * H;

    extern __shared__ float sram[];
    float* k_block = sram;             // (Bc, H)
    float* v_block = sram + (Bc * H);  // (Bc, H)

    float qi[H];
    float oi[H];
    if (row_valid && g >= 0) {
#pragma unroll
        for (int x = 0; x < H; ++x) qi[x] = f_to_float(q[q_off + x]);
    }
#pragma unroll
    for (int x = 0; x < H; ++x) oi[x] = 0.0f;

    float m_prev = -CUDART_INF_F;
    float l_prev = 0.0f;

    const int Tc = (L + Bc - 1) / Bc;
    const int g_max = L - q_seq_len + q_tile * Br + (Br - 1);

    for (int jt = 0; jt < Tc; ++jt) {
        if (jt * Bc > g_max) break;

        // 加载共享 k/v tile（全部来自 paged cache）
        for (int i = tid; i < Bc * H; i += Br) {
            const int r = i / H;
            const int col = i % H;
            const int row_g = jt * Bc + r;
            if (row_g < L) {
                const int slot = block_slot(block_tables, s, max_blocks, row_g, block_size);
                k_block[i] = f_to_float(k_cache[(long)slot * D + (long)kv_head * H + col]);
                v_block[i] = f_to_float(v_cache[(long)slot * D + (long)kv_head * H + col]);
            } else {
                k_block[i] = 0.0f;
                v_block[i] = 0.0f;
            }
        }
        __syncthreads();

        if (row_valid && g >= 0) {
            float m_curr = -CUDART_INF_F;
            float s_blk[Bc];

            for (int y = 0; y < Bc; ++y) {
                const int kv_idx = jt * Bc + y;
                const bool visible = (kv_idx < L) && (kv_idx <= g);
                float score = -CUDART_INF_F;
                if (visible) {
                    float sum = 0.0f;
#pragma unroll
                    for (int x = 0; x < H; ++x) sum += qi[x] * k_block[y * H + x];
                    score = sum * scale;
                }
                s_blk[y] = score;
                m_curr = fmaxf(m_curr, score);
            }

            float l_curr = 0.0f;
            for (int y = 0; y < Bc; ++y) {
                const float p = (s_blk[y] == -CUDART_INF_F) ? 0.0f : __expf(s_blk[y] - m_curr);
                s_blk[y] = p;
                l_curr += p;
            }

            if (m_curr != -CUDART_INF_F) {
                const float m_new = fmaxf(m_prev, m_curr);
                const float alpha = __expf(m_prev - m_new);
                const float beta = __expf(m_curr - m_new);
                const float l_new = alpha * l_prev + beta * l_curr;
                for (int x = 0; x < H; ++x) {
                    float pv = 0.0f;
                    for (int y = 0; y < Bc; ++y) pv += s_blk[y] * v_block[y * H + x];
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
}

/// - `q`: `(batch, q_seq_len, q_heads, head_size)`
/// - `k_cache` / `v_cache`: `(num_blocks, block_size, kv_heads, head_size)`
/// - `context_lens`: `(batch,)` i32 (total cached length, incl. q)
/// - `block_tables`: `(batch, max_blocks)` i32
/// - returns: `(batch, q_seq_len, q_heads, head_size)`
template <typename T, const int H>
int launch_prefill(
    const T* q, const T* k_cache, const T* v_cache, T* o,
    const int* block_tables, const int* context_lens,
    int num_seqs, int q_seq_len, int q_heads, int kv_heads,
    int block_size, int max_blocks, float scale, cudaStream_t stream
) {
    constexpr int Bc = flash_attn_detail::kBlockN;
    constexpr int Br = flash_attn_detail::kBlockM;
    const size_t smem = 2 * Bc * H * sizeof(float);
    if (!flash_attn_detail::ensure_smem(flash_attn_prefill_kernel<T, Bc, Br, H>, smem)) {
        return FLASH_ATTN_ERR_CUDA;
    }

    dim3 grid(num_seqs, q_heads, (q_seq_len + Br - 1) / Br);
    dim3 block(Br);
    flash_attn_prefill_kernel<T, Bc, Br, H><<<grid, block, smem, stream>>>(
        q, k_cache, v_cache, o, block_tables, context_lens,
        q_seq_len, q_heads, kv_heads, block_size, max_blocks, scale
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

/// Dispatch on `q_seq_len`: single token -> decode kernel, multi token -> prefill.
template <typename T, const int H>
int launch_kvcache(
    const T* q, const T* k_cache, const T* v_cache, T* o,
    const int* block_tables, const int* context_lens,
    int num_seqs, int q_seq_len, int q_heads, int kv_heads,
    int block_size, int max_blocks, float scale, cudaStream_t stream
) {
    if (q_seq_len == 1) {
        return launch_decode<T, H>(q, k_cache, v_cache, o, block_tables, context_lens,
                                   num_seqs, q_heads, kv_heads, block_size, max_blocks, scale, stream);
    }
    return launch_prefill<T, H>(q, k_cache, v_cache, o, block_tables, context_lens,
                                num_seqs, q_seq_len, q_heads, kv_heads, block_size, max_blocks, scale, stream);
}

template <typename T>
int flash_attn_with_kvcache_impl(
    const T* q, const T* k_cache, const T* v_cache, T* o,
    const int* block_tables, const int* context_lens,
    int num_seqs, int q_seq_len, int q_heads, int kv_heads, int head_size,
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
    if (q_seq_len <= 0) {
        set_last_error("flash_attn_with_kvcache: q_seq_len must be > 0");
        return FLASH_ATTN_ERR_INVALID_SHAPE;
    }
    if (block_size <= 0) {
        set_last_error("flash_attn_with_kvcache: block_size must be > 0");
        return FLASH_ATTN_ERR_INVALID_SHAPE;
    }

    switch (head_size) {
        case 32:
            return launch_kvcache<T, 32>(q, k_cache, v_cache, o, block_tables, context_lens,
                                         num_seqs, q_seq_len, q_heads, kv_heads, block_size, max_blocks, scale, stream);
        case 64:
            return launch_kvcache<T, 64>(q, k_cache, v_cache, o, block_tables, context_lens,
                                         num_seqs, q_seq_len, q_heads, kv_heads, block_size, max_blocks, scale, stream);
        case 128:
            return launch_kvcache<T, 128>(q, k_cache, v_cache, o, block_tables, context_lens,
                                          num_seqs, q_seq_len, q_heads, kv_heads, block_size, max_blocks, scale, stream);
        default:
            return FLASH_ATTN_ERR_INVALID_HEAD_SIZE;
    }
}

}  // namespace

#define DEFINE_FLASH_ATTN_KVCACHE_ENTRY(NAME, TYPE)                                                      \
    extern "C" int NAME(                                                                                 \
        const TYPE* q, const TYPE* k_cache, const TYPE* v_cache, TYPE* o,                               \
        const int* block_tables, const int* context_lens,                                               \
        int num_seqs, int q_seq_len, int q_heads, int kv_heads, int head_size,                          \
        int block_size, int max_blocks,                                                                 \
        float scale, void* stream                                                                       \
    ) noexcept {                                                                                         \
        return flash_attn_with_kvcache_impl<TYPE>(                                                      \
            q, k_cache, v_cache, o, block_tables, context_lens,                                         \
            num_seqs, q_seq_len, q_heads, kv_heads, head_size, block_size, max_blocks,                  \
            scale, reinterpret_cast<cudaStream_t>(stream)                                               \
        );                                                                                              \
    }

DEFINE_FLASH_ATTN_KVCACHE_ENTRY(flash_attn_with_kvcache_f32, float)
DEFINE_FLASH_ATTN_KVCACHE_ENTRY(flash_attn_with_kvcache_f16, __half)
DEFINE_FLASH_ATTN_KVCACHE_ENTRY(flash_attn_with_kvcache_bf16, __nv_bfloat16)

#undef DEFINE_FLASH_ATTN_KVCACHE_ENTRY
