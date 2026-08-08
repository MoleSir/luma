#include <cstdint>
#include "./utils.cuh"

#define COPY_STRIDED_OFFSET(TYPE, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t* dims, \
    const size_t* strides, \
    const size_t dst_offset, \
    const TYPE* src, \
    TYPE* dst \
) { \
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x; \
    const size_t tstride = gridDim.x * blockDim.x; \
    if (is_contiguous(num_dims, dims, strides)) { \
        for (size_t i = tid; i < numel; i += tstride) { \
            dst[dst_offset + i] = src[i]; \
        } \
    } else { \
        for (size_t i = tid; i < numel; i += tstride) { \
            dst[dst_offset + i] = src[logical_index_to_physical_index(i, num_dims, dims, strides)]; \
        } \
    } \
}

COPY_STRIDED_OFFSET(float,    ucopy_f32)
COPY_STRIDED_OFFSET(double,   ucopy_f64)
COPY_STRIDED_OFFSET(int32_t,  ucopy_i32)
COPY_STRIDED_OFFSET(uint32_t, ucopy_u32)
COPY_STRIDED_OFFSET(uint8_t,  ucopy_u8)

#define COPY2D(TYPE, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t d1, \
    const size_t d2, \
    const size_t src_s, \
    const size_t dst_s, \
    const TYPE* src, \
    TYPE* dst \
) { \
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x; \
    const size_t tstride = gridDim.x * blockDim.x; \
    const size_t total = d1 * d2; \
    for (size_t i = tid; i < total; i += tstride) { \
        size_t row = i / d2; \
        size_t col = i % d2; \
        dst[row * dst_s + col] = src[row * src_s + col]; \
    } \
}

COPY2D(float,    ucopy2d_f32)
COPY2D(double,   ucopy2d_f64)
COPY2D(int32_t,  ucopy2d_i32)
COPY2D(uint32_t, ucopy2d_u32)
COPY2D(uint8_t,  ucopy2d_u8)
