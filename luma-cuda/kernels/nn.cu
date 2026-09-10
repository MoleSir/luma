#include <stdint.h>

// ============================================================
//  nn kernels: softmax + rms_norm (one block per row, last dim)
//
//  grid:   num_rows  blocks
//  block:  min(row_size, 1024) = bd
//  shared: next_pow2(bd) * sizeof(TYPE)   ← needed for binary-tree reduce
// ============================================================

__device__ inline int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// ============================================================
//  softmax
// ============================================================
#define SOFTMAX_KERNEL(TYPE, FN_NAME, EXP_FN, FMAX_FN, MAX_NEG) \
extern "C" __global__ void FN_NAME( \
    const int num_rows,\
    const int row_size,\
    const TYPE* input,\
    TYPE* output\
) {\
    extern __shared__ char s[];\
    TYPE* shr = (TYPE*)s;\
    \
    const int tid = threadIdx.x;\
    const int row = blockIdx.x;\
    if (row >= num_rows) return;\
    \
    const TYPE* in_row = input  + (size_t)row * row_size;\
    TYPE*       out_row = output + (size_t)row * row_size;\
    \
    int blk_p2 = next_pow2(blockDim.x);\
    \
    /* ---- step 1: find max ---- */\
    TYPE thread_max = MAX_NEG;\
    for (int i = tid; i < row_size; i += blockDim.x)\
        thread_max = FMAX_FN(thread_max, in_row[i]);\
    shr[tid] = thread_max;\
    __syncthreads();\
    if (tid == 0) for (int i = blockDim.x; i < blk_p2; i++) shr[i] = MAX_NEG;\
    __syncthreads();\
    for (int s = blk_p2 / 2; s > 0; s >>= 1) {\
        if (tid < s) shr[tid] = FMAX_FN(shr[tid], shr[tid + s]);\
        __syncthreads();\
    }\
    TYPE max_val = shr[0];\
    \
    /* ---- step 2: sum of exp(x - max) ---- */\
    TYPE thread_sum = (TYPE)0;\
    for (int i = tid; i < row_size; i += blockDim.x)\
        thread_sum += EXP_FN(in_row[i] - max_val);\
    shr[tid] = thread_sum;\
    __syncthreads();\
    if (tid == 0) for (int i = blockDim.x; i < blk_p2; i++) shr[i] = (TYPE)0;\
    __syncthreads();\
    for (int s = blk_p2 / 2; s > 0; s >>= 1) {\
        if (tid < s) shr[tid] += shr[tid + s];\
        __syncthreads();\
    }\
    TYPE sum_val = shr[0];\
    \
    /* ---- step 3: normalize ---- */\
    for (int i = tid; i < row_size; i += blockDim.x)\
        out_row[i] = EXP_FN(in_row[i] - max_val) / sum_val;\
}

SOFTMAX_KERNEL(float,  softmax_f32, expf, fmaxf, -1e30f)
SOFTMAX_KERNEL(double, softmax_f64, exp,  fmax,  -1e300)


// ============================================================
//  rms_norm
// ============================================================
#define RMS_NORM_KERNEL(TYPE, FN_NAME, RSQRT_FN) \
extern "C" __global__ void FN_NAME( \
    const int num_rows,\
    const int row_size,\
    const TYPE* input,\
    const TYPE* weight,\
    const TYPE eps,\
    TYPE* output\
) {\
    extern __shared__ char s[];\
    TYPE* shr = (TYPE*)s;\
    \
    const int tid = threadIdx.x;\
    const int row = blockIdx.x;\
    if (row >= num_rows) return;\
    \
    const TYPE* in_row = input  + (size_t)row * row_size;\
    TYPE*       out_row = output + (size_t)row * row_size;\
    \
    int blk_p2 = next_pow2(blockDim.x);\
    \
    /* ---- mean(x^2) ---- */\
    TYPE thread_sum = (TYPE)0;\
    for (int i = tid; i < row_size; i += blockDim.x)\
        thread_sum += in_row[i] * in_row[i];\
    shr[tid] = thread_sum;\
    __syncthreads();\
    if (tid == 0) for (int i = blockDim.x; i < blk_p2; i++) shr[i] = (TYPE)0;\
    __syncthreads();\
    for (int s = blk_p2 / 2; s > 0; s >>= 1) {\
        if (tid < s) shr[tid] += shr[tid + s];\
        __syncthreads();\
    }\
    TYPE inv_rms = RSQRT_FN(shr[0] / (TYPE)row_size + eps);\
    \
    /* ---- normalize * weight ---- */\
    for (int i = tid; i < row_size; i += blockDim.x)\
        out_row[i] = in_row[i] * inv_rms * weight[i];\
}

RMS_NORM_KERNEL(float,  rms_norm_f32, rsqrtf)
RMS_NORM_KERNEL(double, rms_norm_f64, rsqrt)
