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

// ---- sub (scalar - tensor) ----
BINARY_SCALAR_OP_OUT(float,    float,    bslsub_f32, y - x)
BINARY_SCALAR_OP_OUT(double,   double,   bslsub_f64, y - x)
BINARY_SCALAR_OP_OUT(int32_t,  int32_t,  bslsub_i32, y - x)
BINARY_SCALAR_OP_OUT(uint32_t, uint32_t, bslsub_u32, y - x)
BINARY_SCALAR_OP_OUT(uint8_t,  uint8_t,  bslsub_u8,  y - x)

// ---- div (scalar / tensor) ----
BINARY_SCALAR_OP_OUT(float,    float,    bsldiv_f32, y / x)
BINARY_SCALAR_OP_OUT(double,   double,   bsldiv_f64, y / x)
BINARY_SCALAR_OP_OUT(int32_t,  int32_t,  bsldiv_i32, y / x)
BINARY_SCALAR_OP_OUT(uint32_t, uint32_t, bsldiv_u32, y / x)
BINARY_SCALAR_OP_OUT(uint8_t,  uint8_t,  bsldiv_u8,  y / x)

// ---- half-precision ----
BINARY_SCALAR_OP_OUT(__half,        __half,        bsadd_f16,  x + y)
BINARY_SCALAR_OP_OUT(__nv_bfloat16, __nv_bfloat16, bsadd_bf16, x + y)
BINARY_SCALAR_OP_OUT(__half,        __half,        bssub_f16,  x - y)
BINARY_SCALAR_OP_OUT(__nv_bfloat16, __nv_bfloat16, bssub_bf16, x - y)
BINARY_SCALAR_OP_OUT(__half,        __half,        bsmul_f16,  x * y)
BINARY_SCALAR_OP_OUT(__nv_bfloat16, __nv_bfloat16, bsmul_bf16, x * y)
BINARY_SCALAR_OP_OUT(__half,        __half,        bsdiv_f16,  x / y)
BINARY_SCALAR_OP_OUT(__nv_bfloat16, __nv_bfloat16, bsdiv_bf16, x / y)
BINARY_SCALAR_OP_OUT(__half,        __half,        bsmin_f16,  ming(x, y))
BINARY_SCALAR_OP_OUT(__nv_bfloat16, __nv_bfloat16, bsmin_bf16, ming(x, y))
BINARY_SCALAR_OP_OUT(__half,        __half,        bsmax_f16,  maxg(x, y))
BINARY_SCALAR_OP_OUT(__nv_bfloat16, __nv_bfloat16, bsmax_bf16, maxg(x, y))
BINARY_SCALAR_OP_OUT(__half,        __half,        bslsub_f16, y - x)
BINARY_SCALAR_OP_OUT(__nv_bfloat16, __nv_bfloat16, bslsub_bf16, y - x)
BINARY_SCALAR_OP_OUT(__half,        __half,        bsldiv_f16, y / x)
BINARY_SCALAR_OP_OUT(__nv_bfloat16, __nv_bfloat16, bsldiv_bf16, y / x)

// ---- in-place (dst is both the lhs and the output) ----
#define BINARY_SCALAR_OP_INPLACE(TYPENAME, FN_NAME, OP) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims, \
    const size_t *dst_strides, \
    TYPENAME *dst, \
    TYPENAME rhs \
) { \
    bool dst_cont = is_contiguous(num_dims, dims, dst_strides); \
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x; \
    const size_t tstride = blockDim.x * gridDim.x; \
    if (dst_cont) { \
        for (size_t i = tid; i < numel; i += tstride) { \
            dst[i] = dst[i] OP rhs; \
        } \
    } else { \
        for (size_t i = tid; i < numel; i += tstride) { \
            size_t di = logical_index_to_physical_index(i, num_dims, dims, dst_strides); \
            dst[di] = dst[di] OP rhs; \
        } \
    } \
}

#define BINARY_SCALAR_OP_INPLACE_FN(TYPENAME, FN_NAME, FN) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims, \
    const size_t *dst_strides, \
    TYPENAME *dst, \
    TYPENAME rhs \
) { \
    bool dst_cont = is_contiguous(num_dims, dims, dst_strides); \
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x; \
    const size_t tstride = blockDim.x * gridDim.x; \
    if (dst_cont) { \
        for (size_t i = tid; i < numel; i += tstride) { \
            dst[i] = FN(dst[i], rhs); \
        } \
    } else { \
        for (size_t i = tid; i < numel; i += tstride) { \
            size_t di = logical_index_to_physical_index(i, num_dims, dims, dst_strides); \
            dst[di] = FN(dst[di], rhs); \
        } \
    } \
}

BINARY_SCALAR_OP_INPLACE(float,    bsiadd_f32, +)
BINARY_SCALAR_OP_INPLACE(double,   bsiadd_f64, +)
BINARY_SCALAR_OP_INPLACE(int32_t,  bsiadd_i32, +)
BINARY_SCALAR_OP_INPLACE(uint32_t, bsiadd_u32, +)
BINARY_SCALAR_OP_INPLACE(uint8_t,  bsiadd_u8,  +)

BINARY_SCALAR_OP_INPLACE(float,    bsisub_f32, -)
BINARY_SCALAR_OP_INPLACE(double,   bsisub_f64, -)
BINARY_SCALAR_OP_INPLACE(int32_t,  bsisub_i32, -)
BINARY_SCALAR_OP_INPLACE(uint32_t, bsisub_u32, -)
BINARY_SCALAR_OP_INPLACE(uint8_t,  bsisub_u8,  -)

BINARY_SCALAR_OP_INPLACE(float,    bsimul_f32, *)
BINARY_SCALAR_OP_INPLACE(double,   bsimul_f64, *)
BINARY_SCALAR_OP_INPLACE(int32_t,  bsimul_i32, *)
BINARY_SCALAR_OP_INPLACE(uint32_t, bsimul_u32, *)
BINARY_SCALAR_OP_INPLACE(uint8_t,  bsimul_u8,  *)

BINARY_SCALAR_OP_INPLACE(float,    bsidiv_f32, /)
BINARY_SCALAR_OP_INPLACE(double,   bsidiv_f64, /)
BINARY_SCALAR_OP_INPLACE(int32_t,  bsidiv_i32, /)
BINARY_SCALAR_OP_INPLACE(uint32_t, bsidiv_u32, /)
BINARY_SCALAR_OP_INPLACE(uint8_t,  bsidiv_u8,  /)

BINARY_SCALAR_OP_INPLACE_FN(float,    bsimin_f32, ming)
BINARY_SCALAR_OP_INPLACE_FN(double,   bsimin_f64, ming)
BINARY_SCALAR_OP_INPLACE_FN(int32_t,  bsimin_i32, ming)
BINARY_SCALAR_OP_INPLACE_FN(uint32_t, bsimin_u32, ming)
BINARY_SCALAR_OP_INPLACE_FN(uint8_t,  bsimin_u8,  ming)

BINARY_SCALAR_OP_INPLACE_FN(float,    bsimax_f32, maxg)
BINARY_SCALAR_OP_INPLACE_FN(double,   bsimax_f64, maxg)
BINARY_SCALAR_OP_INPLACE_FN(int32_t,  bsimax_i32, maxg)
BINARY_SCALAR_OP_INPLACE_FN(uint32_t, bsimax_u32, maxg)
BINARY_SCALAR_OP_INPLACE_FN(uint8_t,  bsimax_u8,  maxg)

BINARY_SCALAR_OP_INPLACE(__half,        bsiadd_f16,  +)
BINARY_SCALAR_OP_INPLACE(__nv_bfloat16, bsiadd_bf16, +)
BINARY_SCALAR_OP_INPLACE(__half,        bsisub_f16,  -)
BINARY_SCALAR_OP_INPLACE(__nv_bfloat16, bsisub_bf16, -)
BINARY_SCALAR_OP_INPLACE(__half,        bsimul_f16,  *)
BINARY_SCALAR_OP_INPLACE(__nv_bfloat16, bsimul_bf16, *)
BINARY_SCALAR_OP_INPLACE(__half,        bsidiv_f16,  /)
BINARY_SCALAR_OP_INPLACE(__nv_bfloat16, bsidiv_bf16, /)
BINARY_SCALAR_OP_INPLACE_FN(__half,        bsimin_f16,  ming)
BINARY_SCALAR_OP_INPLACE_FN(__nv_bfloat16, bsimin_bf16, ming)
BINARY_SCALAR_OP_INPLACE_FN(__half,        bsimax_f16,  maxg)
BINARY_SCALAR_OP_INPLACE_FN(__nv_bfloat16, bsimax_bf16, maxg)
