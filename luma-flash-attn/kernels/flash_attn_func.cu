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

template <typename T, const int Bc, const int Br, const int H>
__global__ void flash_attn_kernel(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v, T* __restrict__ o,
    const int q_seq_len, const int kv_seq_len, const int q_num_heads, const int kv_num_heads,
    const float scale, const int Tc, const uint8_t* __restrict__ attn_mask, const int start_pos
) {
    const int group_size = q_num_heads / kv_num_heads;

    const int tid         = threadIdx.x;
    const int batch_idx   = blockIdx.x;
    const int q_head_idx  = blockIdx.y;
    const int q_tile_idx  = blockIdx.z;
    const int kv_head_idx = q_head_idx / group_size;

    extern __shared__ float sram[];
    float* k_block = sram;              // (Bc * H)
    float* v_block = sram + (Bc * H);   // (Bc * H)

    const int q_batch_stride  = q_seq_len  * q_num_heads  * H;
    const int kv_batch_stride = kv_seq_len * kv_num_heads * H;
    const int q_head_stride   = H;
    const int kv_head_stride  = H;

    const T* q_ptr = q + batch_idx * q_batch_stride + q_head_idx * q_head_stride;
    const T* k_ptr = k + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    const T* v_ptr = v + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    T* o_ptr       = o + batch_idx * q_batch_stride + q_head_idx * q_head_stride;

    const int q_stride  = q_num_heads * H;
    const int kv_stride = kv_num_heads * H;

    const int q_row_idx = q_tile_idx * Br + tid;
    const int q_global_idx = start_pos + q_row_idx;

    const uint8_t* mask_row = (attn_mask != nullptr) ? (attn_mask + batch_idx * kv_seq_len) : nullptr;

    bool q_valid = (q_row_idx < q_seq_len);
    if (q_valid && mask_row != nullptr) {
        q_valid = (q_global_idx < kv_seq_len) && (mask_row[q_global_idx] != 0);
    }

    float qi[H];
    float oi[H];

    if (q_row_idx < q_seq_len) {
#pragma unroll
        for (int x = 0; x < H; ++x) {
            qi[x] = f_to_float(q_ptr[q_row_idx * q_stride + x]);
        }
    }
#pragma unroll
    for (int x = 0; x < H; ++x) {
        oi[x] = 0.0f;
    }

    float m_prev = -CUDART_INF_F;
    float l_prev = 0.0f;

    for (int j = 0; j < Tc; ++j) {
        const int q_global_max = start_pos + q_tile_idx * Br + (Br - 1);
        if (j * Bc > q_global_max) {
            break;
        }

        for (int i = tid; i < Bc * H; i += Br) {
            const int row = i / H;
            const int col = i % H;
            const int row_g = j * Bc + row;
            if (row_g < kv_seq_len) {
                k_block[i] = f_to_float(k_ptr[row_g * kv_stride + col]);
                v_block[i] = f_to_float(v_ptr[row_g * kv_stride + col]);
            } else {
                k_block[i] = 0.0f;
                v_block[i] = 0.0f;
            }
        }
        __syncthreads();

        if (q_row_idx < q_seq_len) {
            float m_curr = -CUDART_INF_F;
            float s[Bc];

            for (int y = 0; y < Bc; ++y) {
                const int kv_idx = j * Bc + y;

                bool visible = (kv_idx < kv_seq_len);
                if (visible && mask_row != nullptr) {
                    visible = (mask_row[kv_idx] != 0);
                }
                if (visible) {
                    visible = (q_global_idx >= kv_idx);
                }

                float score = -CUDART_INF_F;
                if (visible) {
                    float sum = 0.0f;
#pragma unroll
                    for (int x = 0; x < H; ++x) {
                        sum += qi[x] * k_block[y * H + x];
                    }
                    score = sum * scale;
                }
                s[y] = score;
                m_curr = fmaxf(m_curr, score);
            }

            float l_curr = 0.0f;
            for (int y = 0; y < Bc; ++y) {
                float p = (s[y] == -CUDART_INF_F) ? 0.0f : __expf(s[y] - m_curr);
                s[y] = p;
                l_curr += p;
            }

            if (m_curr != -CUDART_INF_F) {
                const float m_new = fmaxf(m_prev, m_curr);
                const float alpha = __expf(m_prev - m_new);
                const float beta  = __expf(m_curr - m_new);
                const float l_new = alpha * l_prev + beta * l_curr;

                for (int x = 0; x < H; ++x) {
                    float pv = 0.0f;
                    for (int y = 0; y < Bc; ++y) {
                        pv += s[y] * v_block[y * H + x];
                    }
                    oi[x] = (alpha * l_prev * oi[x] + beta * pv) / l_new;
                }

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

template <typename T, const int Bc, const int Br, const int H>
int launch_flash_attn(
    const T* q, const T* k, const T* v, T* o,
    int batch, int q_seq_len, int kv_seq_len, int q_num_heads, int kv_num_heads,
    int start_pos, float scale, const uint8_t* attn_mask, cudaStream_t stream
) {
    const int Tr = (q_seq_len  + Br - 1) / Br;
    const int Tc = (kv_seq_len + Bc - 1) / Bc;

    const size_t smem = 2 * Bc * H * sizeof(float);
    if (!ensure_smem(flash_attn_kernel<T, Bc, Br, H>, smem)) {
        return FLASH_ATTN_ERR_CUDA;
    }

    dim3 grid(batch, q_num_heads, Tr);
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
