#include "./utils.cuh"
#include <cstdint>

#define PICK_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t numel,  \
    const size_t num_dims, \
    const size_t *dims, \
    const size_t *strides, \
    const size_t *strides_t, \
    const size_t *strides_f, \
    const uint8_t *mask, \
    const TYPENAME *t, \
    const TYPENAME *f, \
    TYPENAME *out \
) {  \
    if (is_contiguous(num_dims, dims, strides) \
        && is_contiguous(num_dims, dims, strides_f) \
        && is_contiguous(num_dims, dims, strides_t)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            out[i] = mask[i] ? t[i] : f[i]; \
        } \
    } \
    else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned strided_i = logical_index_to_physical_index(i, num_dims, dims, strides); \
            unsigned strided_i_t = logical_index_to_physical_index(i, num_dims, dims, strides_t); \
            unsigned strided_i_f = logical_index_to_physical_index(i, num_dims, dims, strides_f); \
            out[i] = mask[strided_i] ? t[strided_i_t] : f[strided_i_f]; \
        } \
    } \
} \

#define PICK_TRUE_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t numel,  \
    const size_t num_dims, \
    const size_t *dims, \
    const size_t *strides, \
    const size_t *strides_f, \
    const uint8_t *mask, \
    const TYPENAME t, \
    const TYPENAME *f, \
    TYPENAME *out \
) {  \
    if (is_contiguous(num_dims, dims, strides) && is_contiguous(num_dims, dims, strides_f)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            out[i] = mask[i] ? t : f[i]; \
        } \
    } \
    else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned strided_i = logical_index_to_physical_index(i, num_dims, dims, strides); \
            unsigned strided_i_f = logical_index_to_physical_index(i, num_dims, dims, strides_f); \
            out[i] = mask[strided_i] ? t : f[strided_i_f]; \
        } \
    } \
} \

#define PICK_FALSE_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t numel,  \
    const size_t num_dims, \
    const size_t *dims, \
    const size_t *strides, \
    const size_t *strides_t, \
    const uint8_t *mask, \
    const TYPENAME *t, \
    const TYPENAME f, \
    TYPENAME *out \
) {  \
    if (is_contiguous(num_dims, dims, strides) && is_contiguous(num_dims, dims, strides_t)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            out[i] = mask[i] ? t[i] : f; \
        } \
    } \
    else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned strided_i = logical_index_to_physical_index(i, num_dims, dims, strides); \
            unsigned strided_i_t = logical_index_to_physical_index(i, num_dims, dims, strides_t); \
            out[i] = mask[strided_i] ? t[strided_i_t] : f; \
        } \
    } \
} \

PICK_OP(float, pick_f32)
PICK_OP(double, pick_f64)
PICK_OP(uint8_t, pick_u8)
PICK_OP(uint32_t, pick_u32)
PICK_OP(int32_t, pick_i32)

PICK_TRUE_OP(float, pick_true_f32)
PICK_TRUE_OP(double, pick_true_f64)
PICK_TRUE_OP(uint8_t, pick_true_u8)
PICK_TRUE_OP(uint32_t, pick_true_u32)
PICK_TRUE_OP(int32_t, pick_true_i32)

PICK_FALSE_OP(float, pick_false_f32)
PICK_FALSE_OP(double, pick_false_f64)
PICK_FALSE_OP(uint8_t, pick_false_u8)
PICK_FALSE_OP(uint32_t, pick_false_u32)
PICK_FALSE_OP(int32_t, pick_false_i32)
