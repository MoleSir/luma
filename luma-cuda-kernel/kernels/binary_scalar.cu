#include <cstdint>
#include "./utils.cuh"

#define BINARY_SCALAR_OP_OUT(TYPENAME, OUT_TYPENAME, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel,\
    const size_t num_dims,\
    const size_t *dims,\
    const size_t *lhs_strides,\
    const TYPENAME* lhs,\
    TYPENAME rhs,\
    OUT_TYPENAME* out\
) {\
    bool lhs_cont = is_contiguous(num_dims, dims, lhs_strides);\
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\
    const size_t tstride = blockDim.x * gridDim.x;\
    if (lhs_cont) {\
        for (size_t i = tid; i < numel; i += tstride) {\
            TYPENAME x = lhs[i];\
            TYPENAME y = rhs;\
            out[i] = FUNC;\
        }\
    } else {\
        for (size_t i = tid; i < numel; i += tstride) {\
            TYPENAME x = lhs[logical_index_to_physical_index(i, num_dims, dims, lhs_strides)];\
            TYPENAME y = rhs;\
            out[i] = FUNC;\
        }\
    }\
}

// ---- add ----
BINARY_SCALAR_OP_OUT(float,    float,    bsadd_f32, x + y)
BINARY_SCALAR_OP_OUT(double,   double,   bsadd_f64, x + y)
BINARY_SCALAR_OP_OUT(int32_t,  int32_t,  bsadd_i32, x + y)
BINARY_SCALAR_OP_OUT(uint32_t, uint32_t, bsadd_u32, x + y)
BINARY_SCALAR_OP_OUT(uint8_t,  uint8_t,  bsadd_u8,  x + y)

// ---- sub ----
BINARY_SCALAR_OP_OUT(float,    float,    bssub_f32, x - y)
BINARY_SCALAR_OP_OUT(double,   double,   bssub_f64, x - y)
BINARY_SCALAR_OP_OUT(int32_t,  int32_t,  bssub_i32, x - y)
BINARY_SCALAR_OP_OUT(uint32_t, uint32_t, bssub_u32, x - y)
BINARY_SCALAR_OP_OUT(uint8_t,  uint8_t,  bssub_u8,  x - y)

// ---- mul ----
BINARY_SCALAR_OP_OUT(float,    float,    bsmul_f32, x * y)
BINARY_SCALAR_OP_OUT(double,   double,   bsmul_f64, x * y)
BINARY_SCALAR_OP_OUT(int32_t,  int32_t,  bsmul_i32, x * y)
BINARY_SCALAR_OP_OUT(uint32_t, uint32_t, bsmul_u32, x * y)
BINARY_SCALAR_OP_OUT(uint8_t,  uint8_t,  bsmul_u8,  x * y)

// ---- div ----
BINARY_SCALAR_OP_OUT(float,    float,    bsdiv_f32, x / y)
BINARY_SCALAR_OP_OUT(double,   double,   bsdiv_f64, x / y)
BINARY_SCALAR_OP_OUT(int32_t,  int32_t,  bsdiv_i32, x / y)
BINARY_SCALAR_OP_OUT(uint32_t, uint32_t, bsdiv_u32, x / y)
BINARY_SCALAR_OP_OUT(uint8_t,  uint8_t,  bsdiv_u8,  x / y)

// ---- min ----
BINARY_SCALAR_OP_OUT(float,    float,    bsmin_f32, ming(x, y))
BINARY_SCALAR_OP_OUT(double,   double,   bsmin_f64, ming(x, y))
BINARY_SCALAR_OP_OUT(int32_t,  int32_t,  bsmin_i32, ming(x, y))
BINARY_SCALAR_OP_OUT(uint32_t, uint32_t, bsmin_u32, ming(x, y))
BINARY_SCALAR_OP_OUT(uint8_t,  uint8_t,  bsmin_u8,  ming(x, y))

// ---- max ----
BINARY_SCALAR_OP_OUT(float,    float,    bsmax_f32, maxg(x, y))
BINARY_SCALAR_OP_OUT(double,   double,   bsmax_f64, maxg(x, y))
BINARY_SCALAR_OP_OUT(int32_t,  int32_t,  bsmax_i32, maxg(x, y))
BINARY_SCALAR_OP_OUT(uint32_t, uint32_t, bsmax_u32, maxg(x, y))
BINARY_SCALAR_OP_OUT(uint8_t,  uint8_t,  bsmax_u8,  maxg(x, y))
\
