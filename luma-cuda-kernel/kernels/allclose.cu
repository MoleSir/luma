#include <cstdint>
#include "./utils.cuh"

#define ALLCLOSE_FLOAT_KERNEL(TYPENAME, FN_NAME, ABS_FN) \
extern "C" __global__ void FN_NAME( \
    const size_t numel,\
    const size_t num_dims,\
    const size_t *dims,\
    const size_t *a_strides,\
    const size_t *b_strides,\
    const TYPENAME* a,\
    const TYPENAME* b,\
    const TYPENAME rtol,\
    const TYPENAME atol,\
    int* result\
) {\
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\
    const size_t tstride = blockDim.x * gridDim.x;\
    for (size_t i = tid; i < numel; i += tstride) {\
        size_t a_i = 0;\
        size_t b_i = 0;\
        logical_index_to_physical_index_2(i, num_dims, dims, a_strides, b_strides, &a_i, &b_i);\
        TYPENAME x = a[a_i];\
        TYPENAME y = b[b_i];\
        TYPENAME diff = x - y;\
        if (diff < (TYPENAME)0) diff = -diff;\
        if (diff > atol + rtol * (y < (TYPENAME)0 ? -y : y)) {\
            atomicOr((unsigned int*)result, 1u);\
        }\
    }\
}

#define ALLCLOSE_INT_KERNEL(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel,\
    const size_t num_dims,\
    const size_t *dims,\
    const size_t *a_strides,\
    const size_t *b_strides,\
    const TYPENAME* a,\
    const TYPENAME* b,\
    int* result\
) {\
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\
    const size_t tstride = blockDim.x * gridDim.x;\
    for (size_t i = tid; i < numel; i += tstride) {\
        size_t a_i = 0;\
        size_t b_i = 0;\
        logical_index_to_physical_index_2(i, num_dims, dims, a_strides, b_strides, &a_i, &b_i);\
        if (a[a_i] != b[b_i]) {\
            atomicOr((unsigned int*)result, 1u);\
        }\
    }\
}

ALLCLOSE_FLOAT_KERNEL(float,  allclose_f32, (void))
ALLCLOSE_FLOAT_KERNEL(double, allclose_f64, (void))
ALLCLOSE_INT_KERNEL(int32_t,  allclose_i32)
ALLCLOSE_INT_KERNEL(uint32_t, allclose_u32)
ALLCLOSE_INT_KERNEL(uint8_t,  allclose_u8)
