#include <cmath>
#include <cstddef>
#include <cstdint>
#include "./utils.cuh"

#define UNARY_OP(TYPENAME, FN_NAME, FUNC)\
extern "C" __global__ void FN_NAME(\
    const size_t numel,\
    const size_t num_dims,\
    const size_t* dims,\
    const size_t* strides,\
    const TYPENAME* input,\
    TYPENAME* output\
) {\
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\
    const size_t tstride = gridDim.x * blockDim.x;\
    if (is_contiguous(num_dims, dims, strides)) {\
        for (size_t i = tid; i < numel; i += tstride) {\
            TYPENAME x = input[i];\
            output[i] = FUNC;\
        }\
    } else {\
        for (size_t i = tid; i < numel; i += tstride) {\
            size_t input_i = logical_index_to_physical_index(i, num_dims, dims, strides);\
            TYPENAME x = input[input_i];\
            output[i] = FUNC;\
        }\
    }\
}\
