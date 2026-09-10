#include <cmath>
#include <cstddef>
#include <cstdint>
#include "./utils.cuh"

// Simple unary: out[i] = FUNC(x)
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

// Unary with 1 extra parameter: out[i] = FUNC(x, param)
#define UNARY_OP_PARAM1(TYPENAME, FN_NAME, PARAM_TYPE, PARAM_NAME, FUNC)\
extern "C" __global__ void FN_NAME(\
    const size_t numel,\
    const size_t num_dims,\
    const size_t* dims,\
    const size_t* strides,\
    const PARAM_TYPE PARAM_NAME,\
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

// Unary with 2 extra parameters: out[i] = FUNC(x, param1, param2)
#define UNARY_OP_PARAM2(TYPENAME, FN_NAME, PARAM1_TYPE, PARAM1_NAME, PARAM2_TYPE, PARAM2_NAME, FUNC)\
extern "C" __global__ void FN_NAME(\
    const size_t numel,\
    const size_t num_dims,\
    const size_t* dims,\
    const size_t* strides,\
    const PARAM1_TYPE PARAM1_NAME,\
    const PARAM2_TYPE PARAM2_NAME,\
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

// ---- simple unary (float) ----
UNARY_OP(float,  uexp_f32,     expg(x))
UNARY_OP(double, uexp_f64,     expg(x))
UNARY_OP(float,  uln_f32,      logg(x))
UNARY_OP(double, uln_f64,      logg(x))
UNARY_OP(float,  usin_f32,     sing(x))
UNARY_OP(double, usin_f64,     sing(x))
UNARY_OP(float,  ucos_f32,     cosg(x))
UNARY_OP(double, ucos_f64,     cosg(x))
UNARY_OP(float,  utanh_f32,    tanhg(x))
UNARY_OP(double, utanh_f64,    tanhg(x))
UNARY_OP(float,  usqrt_f32,    sqrtg(x))
UNARY_OP(double, usqrt_f64,    sqrtg(x))
UNARY_OP(float,  urecip_f32,   recipg(x))
UNARY_OP(double, urecip_f64,   recipg(x))
UNARY_OP(float,  uerf_f32,     erfg(x))
UNARY_OP(double, uerf_f64,     erfg(x))
UNARY_OP(float,  ufloor_f32,   floorg(x))
UNARY_OP(double, ufloor_f64,   floorg(x))
UNARY_OP(float,  uceil_f32,    ceilg(x))
UNARY_OP(double, uceil_f64,    ceilg(x))
UNARY_OP(float,  uround_f32,   roundg(x))
UNARY_OP(double, uround_f64,   roundg(x))

// Neg
UNARY_OP(float,  uneg_f32,  -x)
UNARY_OP(double, uneg_f64,  -x)
UNARY_OP(int32_t,  uneg_i32,  -x)
UNARY_OP(uint32_t, uneg_u32,  0u - x)
UNARY_OP(uint8_t,  uneg_u8,   (uint8_t)(0u - x))

// Abs
UNARY_OP(float,    uabs_f32,  fabsf(x))
UNARY_OP(double,   uabs_f64,  fabs(x))
UNARY_OP(int32_t,  uabs_i32,  abs(x))
UNARY_OP(uint32_t, uabs_u32,  x)
UNARY_OP(uint8_t,  uabs_u8,   x)

// Sqr
UNARY_OP(float,  usqr_f32,  x * x)
UNARY_OP(double, usqr_f64,  x * x)

// Relu
UNARY_OP(float,  urelu_f32,  maxg(x, 0.0f))
UNARY_OP(double, urelu_f64,  maxg(x, 0.0))

// Sigmoid
UNARY_OP(float,  usigmoid_f32,  1.0f / (1.0f + expg(-x)))
UNARY_OP(double, usigmoid_f64,  1.0 / (1.0 + expg(-x)))

// Silu
UNARY_OP(float,  usilu_f32,  x / (1.0f + expg(-x)))
UNARY_OP(double, usilu_f64,  x / (1.0 + expg(-x)))

// Gelu
UNARY_OP(float,  ugelu_f32, \
    (0.5f * x) * (1.0f + tanhg(0.7978845608028654f * (x + 0.044715f * x * x * x))))
UNARY_OP(double, ugelu_f64, \
    (0.5 * x) * (1.0 + tanhg(0.7978845608028654 * (x + 0.044715 * x * x * x))))

// GeluErf
UNARY_OP(float,  ugelu_erf_f32,  0.5f * x * (1.0f + erfg(x * 0.7071067811865475f)))
UNARY_OP(double, ugelu_erf_f64,  0.5 * x * (1.0 + erfg(x * 0.7071067811865475)))

// Sign
UNARY_OP(float,  usign_f32,  copysigng(1.0f, x))
UNARY_OP(double, usign_f64,  copysigng(1.0, x))
UNARY_OP(int32_t,  usign_i32,  (0 < x) - (x < 0))
UNARY_OP(uint32_t, usign_u32,  (uint32_t)(x != 0u))
UNARY_OP(uint8_t,  usign_u8,   (uint8_t)(x != (uint8_t)0))

// ---- parameterized unary (float) ----
UNARY_OP_PARAM1(float,  uleaky_relu_f32,  float, alpha,  (x > 0.0f) ? x : (alpha * x))
UNARY_OP_PARAM1(double, uleaky_relu_f64,  double, alpha, (x > 0.0) ? x : (alpha * x))
UNARY_OP_PARAM1(float,  upow_f32,         float, exp,    powg(x, exp))
UNARY_OP_PARAM1(double, upow_f64,         double, exp,   powg(x, exp))
UNARY_OP_PARAM1(int32_t,  upow_i32,       int32_t, exp,  (int32_t)powg((float)x, (float)exp))
UNARY_OP_PARAM1(uint32_t, upow_u32,       uint32_t, exp, (uint32_t)powg((float)x, (float)exp))
UNARY_OP_PARAM1(uint8_t,  upow_u8,        uint8_t, exp,  (uint8_t)powg((float)x, (float)exp))
UNARY_OP_PARAM2(float,  uaffine_f32,      float, mul, float, add,  mul * x + add)
UNARY_OP_PARAM2(double, uaffine_f64,      double, mul, double, add, mul * x + add)
UNARY_OP_PARAM2(int32_t,  uaffine_i32,    int32_t, mul, int32_t, add,  mul * x + add)
UNARY_OP_PARAM2(uint32_t, uaffine_u32,    uint32_t, mul, uint32_t, add, mul * x + add)
UNARY_OP_PARAM2(uint8_t,  uaffine_u8,     uint8_t, mul, uint8_t, add,  mul * x + add)

// ---- clamp ----
#define CLAMP_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t* dims, \
    const size_t* strides, \
    const bool has_min, \
    const TYPENAME min_val, \
    const bool has_max, \
    const TYPENAME max_val, \
    const TYPENAME* input, \
    TYPENAME* output \
) { \
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x; \
    const size_t tstride = gridDim.x * blockDim.x; \
    if (is_contiguous(num_dims, dims, strides)) { \
        for (size_t i = tid; i < numel; i += tstride) { \
            TYPENAME v = input[i]; \
            if (has_min && v < min_val) v = min_val; \
            if (has_max && v > max_val) v = max_val; \
            output[i] = v; \
        } \
    } else { \
        for (size_t i = tid; i < numel; i += tstride) { \
            size_t input_i = logical_index_to_physical_index(i, num_dims, dims, strides); \
            TYPENAME v = input[input_i]; \
            if (has_min && v < min_val) v = min_val; \
            if (has_max && v > max_val) v = max_val; \
            output[i] = v; \
        } \
    } \
}

CLAMP_OP(float,    uclamp_f32)
CLAMP_OP(double,   uclamp_f64)
CLAMP_OP(int32_t,  uclamp_i32)
CLAMP_OP(uint32_t, uclamp_u32)
CLAMP_OP(uint8_t,  uclamp_u8)

// ---- bool not ----
UNARY_OP(uint8_t, unot_u8, !x)
