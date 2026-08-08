#include <cstdint>
#include <cfloat>
#include <climits>
#include "./utils.cuh"

#define BLOCK_SIZE 256

#define ACCUM_SUM(a, b) ((a) + (b))
#define ACCUM_PROD(a, b) ((a) * (b))
#define ACCUM_MIN(a, b) ming((a), (b))
#define ACCUM_MAX(a, b) maxg((a), (b))
#define ACCUM_ALL(a, b) ((a) & (b))
#define ACCUM_ANY(a, b) ((a) | (b))

#define REDUCE_KERNEL(NAME, TYPE, DType, INIT, ACCUM_FN) \
extern "C" __global__ void s##NAME##_##DType( \
    const size_t reduce_size, \
    const size_t reduce_stride, \
    const size_t num_dims, \
    const size_t* dims, \
    const size_t* strides, \
    const size_t reduce_dim, \
    const TYPE* src, \
    TYPE* dst \
) { \
    __shared__ TYPE shr[BLOCK_SIZE]; \
    size_t tid = threadIdx.x; \
    size_t bid = blockIdx.x; \
    \
    size_t base = 0; \
    size_t rem = bid; \
    for (int d = (int)num_dims - 1; d >= 0; d--) { \
        if (d == (int)reduce_dim) continue; \
        base += (rem % dims[d]) * strides[d]; \
        rem /= dims[d]; \
    } \
    \
    TYPE acc = (TYPE)(INIT); \
    for (size_t i = tid; i < reduce_size; i += blockDim.x) { \
        TYPE val = src[base + i * reduce_stride]; \
        acc = ACCUM_FN(acc, val); \
    } \
    shr[tid] = acc; \
    __syncthreads(); \
    \
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) { \
        if (tid < s) { shr[tid] = ACCUM_FN(shr[tid], shr[tid + s]); } \
        __syncthreads(); \
    } \
    \
    if (tid == 0) dst[bid] = shr[0]; \
}

// ---- f32 ----
REDUCE_KERNEL(sum, float, f32, 0.0f, ACCUM_SUM)
REDUCE_KERNEL(min, float, f32, INFINITY, ACCUM_MIN)
REDUCE_KERNEL(max, float, f32, -INFINITY, ACCUM_MAX)
REDUCE_KERNEL(prod, float, f32, 1.0f, ACCUM_PROD)

// ---- f64 ----
REDUCE_KERNEL(sum, double, f64, 0.0, ACCUM_SUM)
REDUCE_KERNEL(min, double, f64, INFINITY, ACCUM_MIN)
REDUCE_KERNEL(max, double, f64, -INFINITY, ACCUM_MAX)
REDUCE_KERNEL(prod, double, f64, 1.0, ACCUM_PROD)

// ---- i32 ----
REDUCE_KERNEL(sum, int32_t, i32, 0, ACCUM_SUM)
REDUCE_KERNEL(min, int32_t, i32, INT32_MAX, ACCUM_MIN)
REDUCE_KERNEL(max, int32_t, i32, INT32_MIN, ACCUM_MAX)
REDUCE_KERNEL(prod, int32_t, i32, 1, ACCUM_PROD)

// ---- u32 ----
REDUCE_KERNEL(sum, uint32_t, u32, 0u, ACCUM_SUM)
REDUCE_KERNEL(min, uint32_t, u32, UINT32_MAX, ACCUM_MIN)
REDUCE_KERNEL(max, uint32_t, u32, 0u, ACCUM_MAX)
REDUCE_KERNEL(prod, uint32_t, u32, 1u, ACCUM_PROD)

// ---- u8 ----
REDUCE_KERNEL(sum, uint8_t, u8, (uint8_t)0, ACCUM_SUM)
REDUCE_KERNEL(min, uint8_t, u8, UINT8_MAX, ACCUM_MIN)
REDUCE_KERNEL(max, uint8_t, u8, (uint8_t)0, ACCUM_MAX)
REDUCE_KERNEL(prod, uint8_t, u8, (uint8_t)1, ACCUM_PROD)

// ---- bool all / any ----
REDUCE_KERNEL(all, uint8_t, u8, (uint8_t)0xFF, ACCUM_ALL)
REDUCE_KERNEL(any, uint8_t, u8, (uint8_t)0, ACCUM_ANY)

#define ARG_REDUCE_KERNEL(NAME, TYPE, DType, INIT, CMP_) \
extern "C" __global__ void s##NAME##_##DType( \
    const size_t reduce_size, \
    const size_t reduce_stride, \
    const size_t num_dims, \
    const size_t* dims, \
    const size_t* strides, \
    const size_t reduce_dim, \
    const TYPE* src, \
    int32_t* dst \
) { \
    __shared__ TYPE     shr_val[BLOCK_SIZE]; \
    __shared__ int32_t  shr_idx[BLOCK_SIZE]; \
    size_t tid = threadIdx.x; \
    size_t bid = blockIdx.x; \
    \
    size_t base = 0; \
    size_t rem = bid; \
    for (int d = (int)num_dims - 1; d >= 0; d--) { \
        if (d == (int)reduce_dim) continue; \
        base += (rem % dims[d]) * strides[d]; \
        rem /= dims[d]; \
    } \
    \
    TYPE    acc_val = (TYPE)(INIT); \
    int32_t acc_idx = 0; \
    for (size_t i = tid; i < reduce_size; i += blockDim.x) { \
        TYPE val = src[base + i * reduce_stride]; \
        if (CMP_(acc_val, val)) { \
            acc_val = val; \
            acc_idx = (int32_t)i; \
        } \
    } \
    shr_val[tid] = acc_val; \
    shr_idx[tid] = acc_idx; \
    __syncthreads(); \
    \
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) { \
        if (tid < s) { \
            TYPE other_val = shr_val[tid + s]; \
            if (CMP_(shr_val[tid], other_val)) { \
                shr_val[tid] = other_val; \
                shr_idx[tid] = shr_idx[tid + s]; \
            } \
        } \
        __syncthreads(); \
    } \
    \
    if (tid == 0) dst[bid] = shr_idx[0]; \
}

#define CMP_ARGMIN(a, b) ((b) < (a))
#define CMP_ARGMAX(a, b) ((b) > (a))

// ---- f32 argmax / argmin ----
ARG_REDUCE_KERNEL(argmax, float, f32, -INFINITY, CMP_ARGMAX)
ARG_REDUCE_KERNEL(argmin, float, f32,  INFINITY, CMP_ARGMIN)

// ---- f64 argmax / argmin ----
ARG_REDUCE_KERNEL(argmax, double, f64, -INFINITY, CMP_ARGMAX)
ARG_REDUCE_KERNEL(argmin, double, f64,  INFINITY, CMP_ARGMIN)

// ---- i32 argmax / argmin ----
ARG_REDUCE_KERNEL(argmax, int32_t, i32, INT32_MIN, CMP_ARGMAX)
ARG_REDUCE_KERNEL(argmin, int32_t, i32, INT32_MAX, CMP_ARGMIN)
