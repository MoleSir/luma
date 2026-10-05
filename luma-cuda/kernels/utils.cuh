#include <stdint.h>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ auto is_contiguous(const size_t num_dims, const size_t* dims, const size_t* strides) -> bool {
    size_t acc = 1;
    // dims: (2, 3, 4)
    // strides: (12, 4, 1)
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        if (dims[dim_idx] > 1 && acc != strides[dim_idx]) {
            return false;
        }
        acc *= dims[dim_idx];
    } 
    return true;
}

__device__ inline auto logical_index_to_physical_index(size_t index, const size_t num_dims, const size_t* dims, const size_t* strides) -> size_t {
    size_t physical_index = 0;

    // from last dim -> 0 dim, get each dim's index
    for (size_t i = 0; i < num_dims; i++) {
        size_t d = num_dims - 1 - i;
        size_t dim_index = index % dims[d];
        physical_index += dim_index * strides[d];
        index /= dims[d];
    }

    return physical_index;
}

__device__ inline void logical_index_to_physical_index_2(
    size_t index,
    const size_t num_dims, 
    const size_t* dims, const size_t* lhs_strides, const size_t* rhs_strides,
    size_t* lhs_physical_index, size_t* rhs_physical_index
) {
    *lhs_physical_index = 0;
    *rhs_physical_index = 0;

    // from last dim -> 0 dim, get each dim's index
    for (size_t i = 0; i < num_dims; i++) {
        size_t d = num_dims - 1 - i;

        size_t dim_index = index % dims[d];
        *lhs_physical_index += dim_index * lhs_strides[d];
        *rhs_physical_index += dim_index * rhs_strides[d];

        index /= dims[d];
    }
}

__device__ __forceinline__ bool isnang(float a) { return isnan(a); }
__device__ __forceinline__ bool isnang(double a) { return isnan(a); }
__device__ __forceinline__ float recipg(float a) { return 1.0 / a; }
__device__ __forceinline__ double recipg(double a) { return 1.0 / a; }
__device__ __forceinline__ float cosg(float a) { return cosf(a); }
__device__ __forceinline__ double cosg(double a) { return cos(a); }
__device__ __forceinline__ float sing(float a) { return sinf(a); }
__device__ __forceinline__ double sing(double a) { return sin(a); }
__device__ __forceinline__ float sqrtg(float a) { return sqrtf(a); }
__device__ __forceinline__ double sqrtg(double a) { return sqrt(a); }
__device__ __forceinline__ float powg(float a, float b) { return powf(a, b); }
__device__ __forceinline__ double powg(double a, double b) { return pow(a, b); }
__device__ __forceinline__ float tanhg(float a) { return tanhf(a); }
__device__ __forceinline__ double tanhg(double a) { return tanh(a); }
__device__ __forceinline__ float erfg(float a) { return erff(a); }
__device__ __forceinline__ double erfg(double a) { return erf(a); }
__device__ __forceinline__ float ceilg(float a) { return ceilf(a); }
__device__ __forceinline__ double ceilg(double a) { return ceil(a); }
__device__ __forceinline__ float floorg(float a) { return floorf(a); }
__device__ __forceinline__ double floorg(double a) { return floor(a); }
__device__ __forceinline__ float roundg(float a) { return roundf(a); }
__device__ __forceinline__ double roundg(double a) { return round(a); }
__device__ __forceinline__ float normcdfg(float a) { return normcdff(a); }
__device__ __forceinline__ double normcdfg(double a) { return normcdf(a); }
__device__ __forceinline__ float maxg(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ double maxg(double a, double b) { return fmax(a, b); }
__device__ __forceinline__ float ming(float a, float b) { return fminf(a, b); }
__device__ __forceinline__ double ming(double a, double b) { return fmin(a, b); }
__device__ __forceinline__ float logg(float a) { return logf(a); }
__device__ __forceinline__ double logg(double a) { return log(a); }
__device__ __forceinline__ float expg(float a) { return expf(a); }
__device__ __forceinline__ double expg(double a) { return exp(a); }
__device__ __forceinline__ float absg(float a) { return fabsf(a); }
__device__ __forceinline__ double absg(double a) { return fabs(a); }
__device__ __forceinline__ float copysigng(float a, float b) { return copysignf(a, b); }
__device__ __forceinline__ double copysigng(double a, double b) { return copysign(a, b); }

__device__ __forceinline__ int64_t ming(int64_t a, int64_t b) { return min(a, b); }
__device__ __forceinline__ int64_t maxg(int64_t a, int64_t b) { return max(a, b); }
__device__ __forceinline__ int32_t ming(int32_t a, int32_t b) { return min(a, b); }
__device__ __forceinline__ int32_t maxg(int32_t a, int32_t b) { return max(a, b); }
__device__ __forceinline__ uint32_t ming(uint32_t a, uint32_t b) { return min(a, b); }
__device__ __forceinline__ uint32_t maxg(uint32_t a, uint32_t b) { return max(a, b); }
__device__ __forceinline__ uint8_t ming(uint8_t a, uint8_t b) { return min(a, b); }
__device__ __forceinline__ uint8_t maxg(uint8_t a, uint8_t b) { return max(a, b); }

// ---- half / bfloat16 ----
// All half-precision math is computed by widening to f32, applying the f32
// kernel, then narrowing back. This keeps a single set of implementations and
// matches the CPU backend's "native storage, f32 for transcendentals" policy.
#define DEFINE_HALF_MATH(T, TOF, FROMF) \
__device__ __forceinline__ T expg(T a) { return FROMF(expf(TOF(a))); } \
__device__ __forceinline__ T logg(T a) { return FROMF(logf(TOF(a))); } \
__device__ __forceinline__ T sing(T a) { return FROMF(sinf(TOF(a))); } \
__device__ __forceinline__ T cosg(T a) { return FROMF(cosf(TOF(a))); } \
__device__ __forceinline__ T tanhg(T a) { return FROMF(tanhf(TOF(a))); } \
__device__ __forceinline__ T sqrtg(T a) { return FROMF(sqrtf(TOF(a))); } \
__device__ __forceinline__ T rsqrtg(T a) { return FROMF(rsqrtf(TOF(a))); } \
__device__ __forceinline__ T recipg(T a) { return FROMF(1.0f / TOF(a)); } \
__device__ __forceinline__ T erfg(T a) { return FROMF(erff(TOF(a))); } \
__device__ __forceinline__ T ceilg(T a) { return FROMF(ceilf(TOF(a))); } \
__device__ __forceinline__ T floorg(T a) { return FROMF(floorf(TOF(a))); } \
__device__ __forceinline__ T roundg(T a) { return FROMF(roundf(TOF(a))); } \
__device__ __forceinline__ T powg(T a, T b) { return FROMF(powf(TOF(a), TOF(b))); } \
__device__ __forceinline__ T absg(T a) { return FROMF(fabsf(TOF(a))); } \
__device__ __forceinline__ T maxg(T a, T b) { return FROMF(fmaxf(TOF(a), TOF(b))); } \
__device__ __forceinline__ T ming(T a, T b) { return FROMF(fminf(TOF(a), TOF(b))); } \
__device__ __forceinline__ T copysigng(T a, T b) { return FROMF(copysignf(TOF(a), TOF(b))); } \
__device__ __forceinline__ T maxg(T a, float b) { return FROMF(fmaxf(TOF(a), b)); } \
__device__ __forceinline__ T relug(T a) { return FROMF(fmaxf(TOF(a), 0.0f)); } \
__device__ __forceinline__ T sigmoidg(T a) { float v = TOF(a); return FROMF(1.0f / (1.0f + expf(-v))); } \
__device__ __forceinline__ T silug(T a) { float v = TOF(a); return FROMF(v / (1.0f + expf(-v))); } \
__device__ __forceinline__ T gelug(T a) { \
    float v = TOF(a); \
    float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v); \
    return FROMF(0.5f * v * (1.0f + tanhf(inner))); \
} \
__device__ __forceinline__ T gelu_erfg(T a) { \
    float v = TOF(a); \
    return FROMF(0.5f * v * (1.0f + erff(v * 0.7071067811865475f))); \
} \
__device__ __forceinline__ T signg(T a) { return FROMF(copysignf(1.0f, TOF(a))); } \
__device__ __forceinline__ T leaky_relug(T a, T alpha) { return TOF(a) > 0.0f ? a : FROMF(TOF(alpha) * TOF(a)); } \
__device__ __forceinline__ T uaffineg(T x, T mul, T add) { return FROMF(TOF(mul) * TOF(x) + TOF(add)); }

DEFINE_HALF_MATH(__half, __half2float, __float2half_rn)
DEFINE_HALF_MATH(__nv_bfloat16, __bfloat162float, __float2bfloat16_rn)
