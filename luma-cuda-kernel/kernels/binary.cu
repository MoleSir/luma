#include <cstdint>
#include "./utils.cuh"

#define BINARY_OP_OUT(TYPENAME, OUT_TYPENAME, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel,\
    const size_t num_dims,\
    const size_t *dims_and_strides,\
    const TYPENAME* lhs,\
    const TYPENAME* rhs,\
    OUT_TYPENAME* out\
) {\
    const size_t* dims = dims_and_strides;\
    const size_t* lhs_strides = dims_and_strides + 1 * num_dims;\
    const size_t* rhs_strides = dims_and_strides + 2 * num_dims;\
    bool lhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, lhs_strides);\
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides);\
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\
    const size_t tstride = blockDim.x * gridDim.x;\
    if (lhs_cont && rhs_cont) {\
        for (size_t i = tid; i < numel; i += tstride) {\
            TYPENAME x = lhs[i];\
            TYPENAME y = rhs[i];\
            out[i] = FUNC;\
        }\
    } else if (lhs_cont) {\
        for (size_t i = tid; i < numel; i += tstride) {\
            TYPENAME x = lhs[i];\
            TYPENAME y = rhs[logical_index_to_physical_index(i, num_dims, dims, rhs_strides)];\
            out[i] = FUNC;\
        }\
    } else if (rhs_cont) {\
        for (size_t i = tid; i < numel; i += tstride) {\
            TYPENAME x = lhs[logical_index_to_physical_index(i, num_dims, dims, lhs_strides)];\
            TYPENAME y = rhs[i];\
            out[i] = FUNC;\
        }\
    } else {\
        for (size_t i = tid; i < numel; i += tstride) {\
            size_t lhs_i = 0;\
            size_t rhs_i = 0;\
            logical_index_to_physical_index_2(i, num_dims, dims, lhs_strides, rhs_strides, &lhs_i, &rhs_i);\
            TYPENAME x = lhs[lhs_i];\
            TYPENAME y = rhs[rhs_i];\
            out[i] = FUNC;\
        }\
    }\
}\

#define BINARY_OP(TYPENAME, FN_NAME, FUNC) BINARY_OP_OUT(TYPENAME, TYPENAME, FN_NAME, FUNC)

BINARY_OP(float, badd_f32, x + y)
BINARY_OP(double, badd_f64, x + y);
BINARY_OP(uint8_t, badd_u8, x + y);
BINARY_OP(uint32_t, badd_u32, x + y);
BINARY_OP(int64_t, badd_i64, x + y);
BINARY_OP(float, bdiv_f32, x / y)
BINARY_OP(double, bdiv_f64, x / y);
BINARY_OP(uint8_t, bdiv_u8, x / y);
BINARY_OP(uint32_t, bdiv_u32, x / y);
BINARY_OP(int64_t, bdiv_i64, x / y);
BINARY_OP(float, bmul_f32, x * y)
BINARY_OP(double, bmul_f64, x * y);
BINARY_OP(uint8_t, bmul_u8, x * y);
BINARY_OP(uint32_t, bmul_u32, x * y);
BINARY_OP(int64_t, bmul_i64, x * y);
BINARY_OP(float, bsub_f32, x - y)
BINARY_OP(double, bsub_f64, x - y);
BINARY_OP(uint8_t, bsub_u8, x - y);
BINARY_OP(uint32_t, bsub_u32, x - y);
BINARY_OP(int64_t, bsub_i64, x - y);
BINARY_OP(float, bminimum_f32, ming(x, y));
BINARY_OP(double, bminimum_f64, ming(x, y));
BINARY_OP(uint8_t, bminimum_u8, ming(x, y));
BINARY_OP(uint32_t, bminimum_u32, ming(x, y));
BINARY_OP(int64_t, bminimum_i64, ming(x, y));
BINARY_OP(float, bmaximum_f32, maxg(x, y));
BINARY_OP(double, bmaximum_f64, maxg(x, y));
BINARY_OP(uint8_t, bmaximum_u8, maxg(x, y));
BINARY_OP(uint32_t, bmaximum_u32, maxg(x, y));
BINARY_OP(int64_t, bmaximum_i64, maxg(x, y));

BINARY_OP_OUT(float, uint8_t, eq_f32, x == y)
BINARY_OP_OUT(double, uint8_t, eq_f64, x == y)
BINARY_OP_OUT(uint8_t, uint8_t, eq_u8, x == y)
BINARY_OP_OUT(uint32_t, uint8_t, eq_u32, x == y)
BINARY_OP_OUT(int64_t, uint8_t, eq_i64, x == y)

BINARY_OP_OUT(float, uint8_t, ne_f32, x != y)
BINARY_OP_OUT(double, uint8_t, ne_f64, x != y)
BINARY_OP_OUT(uint8_t, uint8_t, ne_u8, x != y)
BINARY_OP_OUT(uint32_t, uint8_t, ne_u32, x != y)
BINARY_OP_OUT(int64_t, uint8_t, ne_i64, x != y)

BINARY_OP_OUT(float, uint8_t, lt_f32, x < y)
BINARY_OP_OUT(double, uint8_t, lt_f64, x < y)
BINARY_OP_OUT(uint8_t, uint8_t, lt_u8, x < y)
BINARY_OP_OUT(uint32_t, uint8_t, lt_u32, x < y)
BINARY_OP_OUT(int64_t, uint8_t, lt_i64, x < y)

BINARY_OP_OUT(float, uint8_t, le_f32, x <= y)
BINARY_OP_OUT(double, uint8_t, le_f64, x <= y)
BINARY_OP_OUT(uint8_t, uint8_t, le_u8, x <= y)
BINARY_OP_OUT(uint32_t, uint8_t, le_u32, x <= y)
BINARY_OP_OUT(int64_t, uint8_t, le_i64, x <= y)

BINARY_OP_OUT(float, uint8_t, gt_f32, x > y)
BINARY_OP_OUT(double, uint8_t, gt_f64, x > y)
BINARY_OP_OUT(uint8_t, uint8_t, gt_u8, x > y)
BINARY_OP_OUT(uint32_t, uint8_t, gt_u32, x > y)
BINARY_OP_OUT(int64_t, uint8_t, gt_i64, x > y)

BINARY_OP_OUT(float, uint8_t, ge_f32, x >= y)
BINARY_OP_OUT(double, uint8_t, ge_f64, x >= y)
BINARY_OP_OUT(uint8_t, uint8_t, ge_u8, x >= y)
BINARY_OP_OUT(uint32_t, uint8_t, ge_u32, x >= y)
BINARY_OP_OUT(int64_t, uint8_t, ge_i64, x >= y)