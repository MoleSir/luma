#include "common.cuh"

#include <cstdint>

// ---------------------------------------------------------------------------
// flash_attn_func: batched Flash Attention forward (causal + optional mask, GQA).
//
// Layouts (all contiguous):
//   q, o : (batch, q_seq_len,  q_num_heads,  head_size)   f32
//   k, v : (batch, kv_seq_len, kv_num_heads, head_size)   f32
//   attn_mask: (batch, kv_seq_len), 1=valid / 0=padding, may be nullptr
//   head_size only 32 / 64 / 128; q_num_heads must be a multiple of kv_num_heads.
//
// causal: Q token `i` (global position start_pos + i) sees only K index j <= that.
// scale:  softmax scale (typically 1/sqrt(head_size)).
// ---------------------------------------------------------------------------

namespace {

using flash_attn_detail::ensure_smem;
using flash_attn_detail::f_from_float;
using flash_attn_detail::f_to_float;
using flash_attn_detail::set_last_error;

/// - `q`: `(batch, q_seq_len,  q_num_heads,  H)`
/// - `k`: `(batch, kv_seq_len, kv_num_heads, H)`
/// - `v`: same shape as `k`
/// - returns: `(batch, q_seq_len, q_num_heads, H)`
template <typename T, const int Bc, const int Br, const int H>
__global__ void flash_attn_kernel(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v, T* __restrict__ o,
    const int q_seq_len, const int kv_seq_len, const int q_num_heads, const int kv_num_heads,
    const float scale, const int Tc, const uint8_t* __restrict__ attn_mask, const int start_pos
) {
    const int group_size = q_num_heads / kv_num_heads;

    // dim3 grid(batch, q_num_heads, Tr);
    // 三个维度分别是：batch、q 的 head 数量、以及 q seq 分块索引
    const int tid         = threadIdx.x;
    const int batch_idx   = blockIdx.x;
    const int q_head_idx  = blockIdx.y;
    const int q_tile_idx  = blockIdx.z;
    const int kv_head_idx = q_head_idx / group_size;

    // 加载 k/v 块，每个大小为 Bc x H
    extern __shared__ float sram[];
    float* k_block = sram;              // (Bc * H)
    float* v_block = sram + (Bc * H);   // (Bc * H)

    // 计算各维度的 stride 方便计算偏移
    const int q_batch_stride  = q_seq_len  * q_num_heads  * H;
    const int kv_batch_stride = kv_seq_len * kv_num_heads * H;
    const int q_head_stride   = H;
    const int kv_head_stride  = H;

    // 获取当前这个 Block 处理的起始位置，将每个 ptr 看成：
    // q_ptr (q_seq_len, H) and k/v_ptr (kv_seq_len, H)
    const T* q_ptr = q + batch_idx * q_batch_stride + q_head_idx * q_head_stride;
    const T* k_ptr = k + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    const T* v_ptr = v + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    T* o_ptr       = o + batch_idx * q_batch_stride + q_head_idx * q_head_stride;

    // 将 ptr 看成二维矩阵，剩下的就是一个 stride，即 seq len 维度的 stride
    const int q_stride  = q_num_heads * H;
    const int kv_stride = kv_num_heads * H;

    // 计算当前 Block 处理的这一组 Br 个 token 的起始位置
    const int q_row_idx = q_tile_idx * Br + tid;
    // q_row_idx 是相对 Tensor 的索引，加上 start_pos 得到相对全局序列的索引
    const int q_global_idx = start_pos + q_row_idx;

    // attn_mask 原来的 shape 为 (batch, kv_seq_len)
    // 由于我们处理的 block 是 batch_idx 索引，选择当前 batch 的起始位置，表示当前这个 batch 共享的 mask，长度为 kv_seq_len
    const uint8_t* mask_row = (attn_mask != nullptr) ? (attn_mask + batch_idx * kv_seq_len) : nullptr;

    // q_seq_len 可能无法整除 Br，导致有些 thread 处理的是无效 row index
    bool q_valid = (q_row_idx < q_seq_len);
    if (q_valid && mask_row != nullptr) {
        /*
            - q_global_idx < kv_seq_len: q 的位置不可能超过 kv_seq_len
            - mask_row[q_global_idx] != 0: q_global_idx 表示这个 thread 处理的 token 在整个序列的索引，取出这个位置是否被 mask 
        */
        q_valid = (q_global_idx < kv_seq_len) && (mask_row[q_global_idx] != 0);
    }

    float qi[H];
    float oi[H];
    
    // 一个 Block 负责处理 Br 个 Q 和对应一系列 K/V 的处理，而每个用 Br 个 thread，正好每个 thread 处理一行
    // 如果这个 thread q 不超范围，自己加载一行 q 
//     if (q_row_idx < q_seq_len) {
// #pragma unroll
//         for (int x = 0; x < H; ++x) {
//             // 再次提醒，q_ptr 现在逻辑 shape 是 (Br, H)，stride 是 q_stride
//             qi[x] = f_to_float(q_ptr[q_row_idx * q_stride + x]);
//         }
//     }

    // 这样写没有分支，避免 warp diverages
    for (int x = 0; x < H; ++x) {
        // 再次提醒，q_ptr 现在逻辑 shape 是 (Br, H)，stride 是 q_stride
        qi[x] = q_row_idx < q_seq_len ? f_to_float(q_ptr[q_row_idx * q_stride + x]) : 0.0f;
    }
    
#pragma unroll
    for (int x = 0; x < H; ++x) {
        oi[x] = 0.0f;
    }

    // online softmax，每个 thread 负责一行
    float m_prev = -CUDART_INF_F;
    float l_prev = 0.0f;

    // k/v 的循环
    for (int j = 0; j < Tc; ++j) {
        const int q_global_max = start_pos + q_tile_idx * Br + (Br - 1);
        if (j * Bc > q_global_max) {
            break;
        }

        // 加载共享 k/v，大小均为 (Bc, H)，由 Br 个 thread 搬运，使用老套路
        for (int i = tid; i < Bc * H; i += Br) {
            const int row = i / H;
            const int col = i % H;
            const int row_g = j * Bc + row;
            k_block[i] = row_g < kv_seq_len ? f_to_float(k_ptr[row_g * kv_stride + col]) : 0.0f;
            v_block[i] = row_g < kv_seq_len ? f_to_float(v_ptr[row_g * kv_stride + col]) : 0.0f;
        }
        __syncthreads();

        // 计算任务：一个 thread 独立使用一个 token（一个 q_seq 位置），和一个 Block 中的 k/v 计算
        // 计算 matmul 时，一个 thread 需要完成自己的 token 和 Bc 个 kv 向量做点积，得到 Bc 个结果
        if (q_row_idx < q_seq_len) {
            float m_curr = -CUDART_INF_F;
            // s[Bc] 保存 Bc 个点积结果
            float s[Bc];

            // 一个 thread 需要完成自己的 token 和 Bc 个 kv 向量做点积
            // for y 的循环，每次处理一个 kv 向量 dot 结果
            for (int y = 0; y < Bc; ++y) {
                // j * Bc 本次循环的起始 kv seq 索引 + 本次内部循环处理的 y
                // kv_idx 是当前处理的 kv seq len 的一个索引
                const int kv_idx = j * Bc + y;
                // 避免分块超过索引范围
                // 三个条件同时满足：
                // 1. kv_idx < kv_seq_len：必须分块 out of range
                // 2. mask_row[kv_idx] != 0 mask 有效
                // 3. q_global_idx >= kv_idx causal mask，q 是输入，kv 是历史，q 只能查询到比自己小于等于的 kv 历史
                bool visible = (kv_idx < kv_seq_len);
                if (visible && mask_row != nullptr) {
                    visible = (mask_row[kv_idx] != 0);
                }
                if (visible) {
                    visible = (q_global_idx >= kv_idx);
                }

                float score = -CUDART_INF_F;
                // 如果有效
                if (visible) {
                    // 计算一个 q 和 k 的向量点积
                    float sum = 0.0f;
#pragma unroll
                    for (int x = 0; x < H; ++x) {
                        sum += qi[x] * k_block[y * H + x];
                    }
                    score = sum * scale;
                }
                // 计算当前 token 对 Bc 个 k 的点积同时，正好计算这个 block QK^T 的最大值
                s[y] = score;
                m_curr = fmaxf(m_curr, score);
            }
            // 到这里：每个 thread 完成了自己的 token 和 Bc 个 k 的点积，得到 Bc 个值，同时记录其中的最大值。

            // online softmax
            float l_curr = 0.0f;
            // 对每个位置计算 exp(x - x_max)，这里的 x_max 是本块的最大值
            // 同时计算当前块计算的 exp 求和
            for (int y = 0; y < Bc; ++y) {
                float p = (s[y] == -CUDART_INF_F) ? 0.0f : __expf(s[y] - m_curr);
                s[y] = p;
                l_curr += p;
            }
            // 到这里，每个 thread 完成了自己的 token 和 Bc 个 k 的点积分 - x_max 的 exp 值 以及 他们的求和
            // 但有问题：我们 - x_max 这个值是块内的最大值，不是全局

            if (m_curr != -CUDART_INF_F) {
                // 找到当前所有 block 的最大值
                const float m_new = fmaxf(m_prev, m_curr);
                // alpha = exp(之前最大 - 总最大)
                const float alpha = __expf(m_prev - m_new);
                // beta = exp(块最大 - 总最大)
                const float beta  = __expf(m_curr - m_new);
                // 更新 l: 
                // 1. l_prev 用的 x_max 是 m_prev，所以用更新因子 alpha
                // 2. l_curr 用的 x_max 是块内部的最大 m_curr，所以更新因子用 beta
                const float l_new = alpha * l_prev + beta * l_curr;

                // (Bc,) @ (Bc, H)
                // 计算 softmax 后和 v 的计算，每个 thread 得到了 Bc 个结果，表示这个 token 和 H 个 v 向量的缩放因子
                for (int x = 0; x < H; ++x) {
                    // 计算一个点积，但需要用 s 控制缩放
                    float pv = 0.0f;
                    for (int y = 0; y < Bc; ++y) {
                        pv += s[y] * v_block[y * H + x];
                    }
                    // 更新一个 oi
                    oi[x] = (alpha * l_prev * oi[x] + beta * pv) / l_new;
                }

                // 准备下一个块，准备下一次的 m / l
                m_prev = m_new;
                l_prev = l_new;
            }
        }

        __syncthreads();
    }

    if (q_row_idx < q_seq_len) {
        const bool write_valid = q_valid && (l_prev > 0.0f);
#pragma unroll
        for (int x = 0; x < H; ++x) {
            o_ptr[q_row_idx * q_stride + x] = write_valid ? f_from_float<T>(oi[x]) : f_from_float<T>(0.0f);
        }
    }
}

/// - `q`: `(batch, q_seq_len,  q_num_heads,  H)`
/// - `k`: `(batch, kv_seq_len, kv_num_heads, H)`
/// - `v`: same shape as `k`
/// - returns: `(batch, q_seq_len, q_num_heads, H)`
template <typename T, const int Bc, const int Br, const int H>
int launch_flash_attn(
    const T* q, const T* k, const T* v, T* o,
    int batch, int q_seq_len, int kv_seq_len, int q_num_heads, int kv_num_heads,
    int start_pos, float scale, const uint8_t* attn_mask, cudaStream_t stream
) {
    // q seq 长度每 Br 分块，kv seq 长度每 Bc 分块
    const int Tr = (q_seq_len  + Br - 1) / Br;
    const int Tc = (kv_seq_len + Bc - 1) / Bc;

    // 共享内存大小，用于 kv tile 的加载： 的 Bc 表示 kv 每个块的 seq 长度，H 是维度
    const size_t smem = 2 * Bc * H * sizeof(float);
    if (!ensure_smem(flash_attn_kernel<T, Bc, Br, H>, smem)) {
        return FLASH_ATTN_ERR_CUDA;
    }

    // block 划分：每个 Block 负责计算一块 Q 和一系列 K/V 的分块计算结果
    dim3 grid(batch, q_num_heads, Tr);
    // Q 一块大小为 (Br, H)，正好用 Br 个 thread 来处理
    dim3 block(Br);

    flash_attn_kernel<T, Bc, Br, H><<<grid, block, smem, stream>>>(
        q, k, v, o,
        q_seq_len, kv_seq_len, q_num_heads, kv_num_heads,
        scale, Tc, attn_mask, start_pos
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_last_error(cudaGetErrorString(err));
        return FLASH_ATTN_ERR_CUDA;
    }
    return FLASH_ATTN_OK;
}

template <typename T>
int flash_attn_impl(
    const T* q, const T* k, const T* v, T* o,
    int batch, int q_seq_len, int kv_seq_len, int q_num_heads, int kv_num_heads, int head_size,
    int start_pos, float scale, const uint8_t* attn_mask, cudaStream_t stream
) {
    if (q == nullptr || k == nullptr || v == nullptr || o == nullptr) {
        set_last_error("flash_attn: q/k/v/o must not be null");
        return FLASH_ATTN_ERR_NULL_PTR;
    }
    if (batch <= 0 || q_seq_len <= 0 || kv_seq_len <= 0 || q_num_heads <= 0 || kv_num_heads <= 0) {
        set_last_error("flash_attn: shape parameters must be positive");
        return FLASH_ATTN_ERR_INVALID_SHAPE;
    }
    if (q_num_heads % kv_num_heads != 0) {
        set_last_error("flash_attn: q_num_heads must be a multiple of kv_num_heads (GQA)");
        return FLASH_ATTN_ERR_INVALID_SHAPE;
    }
    if (start_pos < 0) {
        set_last_error("flash_attn: start_pos must be >= 0");
        return FLASH_ATTN_ERR_INVALID_START_POS;
    }

    switch (head_size) {
        case 32:
            return launch_flash_attn<T, flash_attn_detail::kBlockN, flash_attn_detail::kBlockM, 32>(
                q, k, v, o, batch, q_seq_len, kv_seq_len, q_num_heads, kv_num_heads, start_pos, scale, attn_mask, stream);
        case 64:
            return launch_flash_attn<T, flash_attn_detail::kBlockN, flash_attn_detail::kBlockM, 64>(
                q, k, v, o, batch, q_seq_len, kv_seq_len, q_num_heads, kv_num_heads, start_pos, scale, attn_mask, stream);
        case 128:
            return launch_flash_attn<T, flash_attn_detail::kBlockN, flash_attn_detail::kBlockM, 128>(
                q, k, v, o, batch, q_seq_len, kv_seq_len, q_num_heads, kv_num_heads, start_pos, scale, attn_mask, stream);
        default:
            set_last_error("flash_attn: unsupported head_size (only 32/64/128)");
            return FLASH_ATTN_ERR_INVALID_HEAD_SIZE;
    }
}

}  // namespace

extern "C" const char* flash_attn_last_error(void) noexcept {
    return flash_attn_detail::g_last_error;
}

#define DEFINE_FLASH_ATTN_FUNC_ENTRY(NAME, TYPE)                                                        \
    extern "C" int NAME(                                                                                \
        const TYPE* q, const TYPE* k, const TYPE* v, TYPE* o,                                          \
        int batch, int q_seq_len, int kv_seq_len, int q_num_heads, int kv_num_heads, int head_size,     \
        int start_pos, float scale, const uint8_t* attn_mask, void* stream                             \
    ) noexcept {                                                                                        \
        flash_attn_detail::clear_last_error();                                                          \
        cudaGetLastError();                                                                             \
        return flash_attn_impl<TYPE>(                                                                   \
            q, k, v, o, batch, q_seq_len, kv_seq_len, q_num_heads, kv_num_heads,                        \
            head_size, start_pos, scale, attn_mask, reinterpret_cast<cudaStream_t>(stream)              \
        );                                                                                              \
    }

DEFINE_FLASH_ATTN_FUNC_ENTRY(flash_attn_f32, float)
DEFINE_FLASH_ATTN_FUNC_ENTRY(flash_attn_f16, __half)
DEFINE_FLASH_ATTN_FUNC_ENTRY(flash_attn_bf16, __nv_bfloat16)

#undef DEFINE_FLASH_ATTN_FUNC_ENTRY
