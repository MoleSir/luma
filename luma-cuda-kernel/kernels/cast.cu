#include <cstdint>
#include "./utils.cuh"

#define CAST_OP(IN_TYPE, OUT_TYPE, FN_NAME, EXPR) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t* dims, \
    const size_t* strides, \
    const IN_TYPE* input, \
    OUT_TYPE* output \
) { \
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x; \
    const size_t tstride = gridDim.x * blockDim.x; \
    if (is_contiguous(num_dims, dims, strides)) { \
        for (size_t i = tid; i < numel; i += tstride) { \
            IN_TYPE x = input[i]; \
            output[i] = EXPR; \
        } \
    } else { \
        for (size_t i = tid; i < numel; i += tstride) { \
            size_t input_i = logical_index_to_physical_index(i, num_dims, dims, strides); \
            IN_TYPE x = input[input_i]; \
            output[i] = EXPR; \
        } \
    } \
}

// ---- identity (contiguous copy) ----
CAST_OP(float,    float,    ucast_f32_to_f32, x)
CAST_OP(double,   double,   ucast_f64_to_f64, x)
CAST_OP(int32_t,  int32_t,  ucast_i32_to_i32, x)
CAST_OP(uint32_t, uint32_t, ucast_u32_to_u32, x)
CAST_OP(uint8_t,  uint8_t,  ucast_u8_to_u8,  x)

// ---- float <-> float ----
CAST_OP(float,  double, ucast_f32_to_f64, (double)(x))
CAST_OP(double, float,  ucast_f64_to_f32, (float)(x))

// ---- float -> int ----
CAST_OP(float, int32_t,  ucast_f32_to_i32, (int32_t)(x))
CAST_OP(float, uint32_t, ucast_f32_to_u32, (uint32_t)(x))
CAST_OP(float, uint8_t,  ucast_f32_to_u8,  (uint8_t)(x))
CAST_OP(double, int32_t,  ucast_f64_to_i32, (int32_t)(x))
CAST_OP(double, uint32_t, ucast_f64_to_u32, (uint32_t)(x))
CAST_OP(double, uint8_t,  ucast_f64_to_u8,  (uint8_t)(x))

// ---- int -> float ----
CAST_OP(int32_t,  float,  ucast_i32_to_f32, (float)(x))
CAST_OP(int32_t,  double, ucast_i32_to_f64, (double)(x))
CAST_OP(uint32_t, float,  ucast_u32_to_f32, (float)(x))
CAST_OP(uint32_t, double, ucast_u32_to_f64, (double)(x))
CAST_OP(uint8_t,  float,  ucast_u8_to_f32,  (float)(x))
CAST_OP(uint8_t,  double, ucast_u8_to_f64,  (double)(x))

// ---- int -> int ----
CAST_OP(int32_t,  uint32_t, ucast_i32_to_u32, (uint32_t)(x))
CAST_OP(int32_t,  uint8_t,  ucast_i32_to_u8,  (uint8_t)(x))
CAST_OP(uint32_t, int32_t,  ucast_u32_to_i32, (int32_t)(x))
CAST_OP(uint32_t, uint8_t,  ucast_u32_to_u8,  (uint8_t)(x))
CAST_OP(uint8_t,  int32_t,  ucast_u8_to_i32,  (int32_t)(x))
CAST_OP(uint8_t,  uint32_t, ucast_u8_to_u32,  (uint32_t)(x))

// ---- to bool ----
CAST_OP(float,    uint8_t, ucast_f32_to_bool, (uint8_t)(x != 0.0f))
CAST_OP(double,   uint8_t, ucast_f64_to_bool, (uint8_t)(x != 0.0))
CAST_OP(int32_t,  uint8_t, ucast_i32_to_bool, (uint8_t)(x != 0))
CAST_OP(uint32_t, uint8_t, ucast_u32_to_bool, (uint8_t)(x != 0u))
