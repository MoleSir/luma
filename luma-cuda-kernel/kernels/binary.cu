#include <cstdint>
#include "./utils.cuh"

#define BINARY_OP_OUT(TYPENAME, OUT_TYPENAME, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel,\
    const size_t num_dims,\
    const size_t *dims,\
    const size_t *lhs_strides,\
    const size_t *rhs_strides,\
    const TYPENAME* lhs,\
    const TYPENAME* rhs,\
    OUT_TYPENAME* out\
) {\
    bool lhs_cont = is_contiguous(num_dims, dims, lhs_strides);\
    bool rhs_cont = is_contiguous(num_dims, dims, rhs_strides);\
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

// ---- arithmetic ----
BINARY_OP(float,       badd_f32, x + y)
BINARY_OP(double,      badd_f64, x + y)
BINARY_OP(uint8_t,     badd_u8, x + y)
BINARY_OP(uint32_t,    badd_u32, x + y)
BINARY_OP(int32_t,     badd_i32, x + y)

BINARY_OP(float,       bsub_f32, x - y)
BINARY_OP(double,      bsub_f64, x - y)
BINARY_OP(uint8_t,     bsub_u8, x - y)
BINARY_OP(uint32_t,    bsub_u32, x - y)
BINARY_OP(int32_t,     bsub_i32, x - y)

BINARY_OP(float,       bmul_f32, x * y)
BINARY_OP(double,      bmul_f64, x * y)
BINARY_OP(uint8_t,     bmul_u8, x * y)
BINARY_OP(uint32_t,    bmul_u32, x * y)
BINARY_OP(int32_t,     bmul_i32, x * y)

BINARY_OP(float,       bdiv_f32, x / y)
BINARY_OP(double,      bdiv_f64, x / y)
BINARY_OP(uint8_t,     bdiv_u8, x / y)
BINARY_OP(uint32_t,    bdiv_u32, x / y)
BINARY_OP(int32_t,     bdiv_i32, x / y)

BINARY_OP(float,       bmin_f32, ming(x, y))
BINARY_OP(double,      bmin_f64, ming(x, y))
BINARY_OP(uint8_t,     bmin_u8, ming(x, y))
BINARY_OP(uint32_t,    bmin_u32, ming(x, y))
BINARY_OP(int32_t,     bmin_i32, ming(x, y))

BINARY_OP(float,       bmax_f32, maxg(x, y))
BINARY_OP(double,      bmax_f64, maxg(x, y))
BINARY_OP(uint8_t,     bmax_u8, maxg(x, y))
BINARY_OP(uint32_t,    bmax_u32, maxg(x, y))
BINARY_OP(int32_t,     bmax_i32, maxg(x, y))

// ---- comparison (output uint8_t) ----
BINARY_OP_OUT(float,    uint8_t, beq_f32,  x == y)
BINARY_OP_OUT(double,   uint8_t, beq_f64,  x == y)
BINARY_OP_OUT(uint8_t,  uint8_t, beq_u8,   x == y)
BINARY_OP_OUT(uint32_t, uint8_t, beq_u32,  x == y)
BINARY_OP_OUT(int32_t,  uint8_t, beq_i32,  x == y)

BINARY_OP_OUT(float,    uint8_t, bne_f32,  x != y)
BINARY_OP_OUT(double,   uint8_t, bne_f64,  x != y)
BINARY_OP_OUT(uint8_t,  uint8_t, bne_u8,   x != y)
BINARY_OP_OUT(uint32_t, uint8_t, bne_u32,  x != y)
BINARY_OP_OUT(int32_t,  uint8_t, bne_i32,  x != y)

BINARY_OP_OUT(float,    uint8_t, blt_f32,  x < y)
BINARY_OP_OUT(double,   uint8_t, blt_f64,  x < y)
BINARY_OP_OUT(uint8_t,  uint8_t, blt_u8,   x < y)
BINARY_OP_OUT(uint32_t, uint8_t, blt_u32,  x < y)
BINARY_OP_OUT(int32_t,  uint8_t, blt_i32,  x < y)

BINARY_OP_OUT(float,    uint8_t, ble_f32,  x <= y)
BINARY_OP_OUT(double,   uint8_t, ble_f64,  x <= y)
BINARY_OP_OUT(uint8_t,  uint8_t, ble_u8,   x <= y)
BINARY_OP_OUT(uint32_t, uint8_t, ble_u32,  x <= y)
BINARY_OP_OUT(int32_t,  uint8_t, ble_i32,  x <= y)

BINARY_OP_OUT(float,    uint8_t, bgt_f32,  x > y)
BINARY_OP_OUT(double,   uint8_t, bgt_f64,  x > y)
BINARY_OP_OUT(uint8_t,  uint8_t, bgt_u8,   x > y)
BINARY_OP_OUT(uint32_t, uint8_t, bgt_u32,  x > y)
BINARY_OP_OUT(int32_t,  uint8_t, bgt_i32,  x > y)

BINARY_OP_OUT(float,    uint8_t, bge_f32,  x >= y)
BINARY_OP_OUT(double,   uint8_t, bge_f64,  x >= y)
BINARY_OP_OUT(uint8_t,  uint8_t, bge_u8,   x >= y)
BINARY_OP_OUT(uint32_t, uint8_t, bge_u32,  x >= y)
BINARY_OP_OUT(int32_t,  uint8_t, bge_i32,  x >= y)

// ---- bool logical ----
BINARY_OP(uint8_t, band_u8, x & y)
BINARY_OP(uint8_t, bor_u8,  x | y)
BINARY_OP(uint8_t, bxor_u8, x ^ y)

extern "C" __global__ void pick_f32(
    const size_t numel, const size_t num_dims,
    const size_t *dims, const size_t *mask_strides, const size_t *val_strides,
    const uint8_t *ids, const float *t, const float *f, float *out
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numel; i += blockDim.x * gridDim.x) {
        out[i] = ids[i] ? t[i] : f[i];
    }
}

extern "C" __global__ void pick_f64(
    const size_t numel, const size_t num_dims,
    const size_t *dims, const size_t *mask_strides, const size_t *val_strides,
    const uint8_t *ids, const double *t, const double *f, double *out
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numel; i += blockDim.x * gridDim.x) {
        out[i] = ids[i] ? t[i] : f[i];
    }
}
