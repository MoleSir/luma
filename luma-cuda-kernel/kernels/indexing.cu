#include "./utils.cuh"
#include <cstdint>

template <typename T>
__host__ __device__
constexpr T max_value();

template <>
__host__ __device__
constexpr int64_t max_value<int64_t>() {
    return 0x7FFFFFFFFFFFFFFFLL;
}

template <>
__host__ __device__
constexpr uint32_t max_value<uint32_t>() {
    return 0xFFFFFFFFu;
}

template <>
__host__ __device__
constexpr uint8_t max_value<uint8_t>() {
    return 0xFFu;
}

template <>
__host__ __device__
constexpr int32_t max_value<int32_t>() {
    return 0x7FFFFFFF;
}

template <>
__host__ __device__
constexpr int16_t max_value<int16_t>() {
    return 0x7FFF;
}

// =============================================================================== //
//                index [select]|[add]
// =============================================================================== //

template<typename T, typename I>
__device__ void index_select(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const I *ids,
    const T *inp,
    T *out,
    const size_t left_size,
    const size_t src_dim_size,
    const size_t ids_dim_size,
    const size_t right_size
) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    bool b = is_contiguous(num_dims, dims, strides);
    for (unsigned int dst_i = blockIdx.x * blockDim.x + threadIdx.x; dst_i < numel; dst_i += blockDim.x * gridDim.x) {
          unsigned int left_i = dst_i / (ids_dim_size * right_size);
          unsigned int id_i = dst_i / right_size % ids_dim_size;
          unsigned int right_i = dst_i % right_size;
          if (ids[id_i] == max_value<I>()) {
            out[dst_i] = static_cast<T>(0);
          } else {
            assert(ids[id_i] < src_dim_size);
            unsigned int src_i = left_i * (src_dim_size * right_size) + ids[id_i] * right_size + right_i;
            unsigned strided_i = b ? src_i : logical_index_to_physical_index(src_i, num_dims, dims, strides);
            out[dst_i] = inp[strided_i];
          }
    }
}

#define INDEX_SELECT_OP(TYPENAME, INDEX_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t numel,  \
    const size_t num_dims, \
    const size_t *info, \
    const INDEX_TYPENAME *ids, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const size_t left_size, \
    const size_t src_dim_size, \
    const size_t ids_dim_size, \
    const size_t right_size \
) { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size); } \

template<typename T, typename I>
__device__ void index_add(
    const I *ids,
    const size_t ids_dim_size,
    const T *inp,
    T *out,
    const size_t left_size,
    const size_t src_dim_size,
    const size_t dst_dim_size,
    const size_t right_size
) {
      const size_t numel = left_size * right_size;
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
          const size_t pre = i / right_size;
          const size_t post = i % right_size;
          for (unsigned int j = 0; j < ids_dim_size; ++j) {
              const I idx = ids[j];
              const size_t src_i = (pre * ids_dim_size + j) * right_size + post;
              if (idx < max_value<I>()) {
                assert(idx < dst_dim_size);
                const size_t dst_i = (pre * dst_dim_size + idx) * right_size + post;
                out[dst_i] += inp[src_i];
              }
          }
      }
}

#define INDEX_ADD_OP(TYPENAME, INDEX_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const INDEX_TYPENAME *ids, \
    const size_t ids_dim_size, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const size_t left_size, \
    const size_t src_dim_size, \
    const size_t dst_dim_size, \
    const size_t right_size \
) { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size); } \

// =============================================================================== //
//                gather
// =============================================================================== //

template<typename T, typename I>
__device__ void gather(
    const size_t numel,
    const I *ids,
    const T *inp,
    T *out,
    const size_t left_size,
    const size_t src_dim_size,
    const size_t ids_dim_size,
    const size_t right_size
) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        size_t post = i % right_size;
        const I idx = ids[i];
        if (ids[i] == max_value<I>()) {
          out[i] = static_cast<T>(0);
        } else {
          assert(idx < src_dim_size);
          size_t pre = i / (right_size * ids_dim_size);
          size_t src_i = (pre * src_dim_size + idx) * right_size + post;
          out[i] = inp[src_i];
        }
    }
}

#define GATHER_OP(TYPENAME, INDEX_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t numel,  \
    const INDEX_TYPENAME *ids, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const size_t left_size, \
    const size_t src_dim_size, \
    const size_t ids_dim_size, \
    const size_t right_size \
) { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size); } \

// =============================================================================== //
//                scatter[add]
// =============================================================================== //

template<typename T, typename I>
__device__ void scatter(
    const I *ids,
    const T *inp,
    T *out,
    const size_t left_size,
    const size_t src_dim_size,
    const size_t dst_dim_size,
    const size_t right_size
) {
      const size_t numel = left_size * right_size;
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
          const size_t pre = i / right_size;
          const size_t post = i % right_size;
          for (unsigned int j = 0; j < src_dim_size; ++j) {
              const size_t src_i = (pre * src_dim_size + j) * right_size + post;
              const I idx = ids[src_i];
              if (idx < max_value<I>()) {
                assert(idx < dst_dim_size);
                const size_t dst_i = (pre * dst_dim_size + idx) * right_size + post;
                out[dst_i] = inp[src_i];
              }
          }
      }
}

template<typename T, typename I>
__device__ void scatter_add(
    const I *ids,
    const T *inp,
    T *out,
    const size_t left_size,
    const size_t src_dim_size,
    const size_t dst_dim_size,
    const size_t right_size
) {
      const size_t numel = left_size * right_size;
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
          const size_t pre = i / right_size;
          const size_t post = i % right_size;
          for (unsigned int j = 0; j < src_dim_size; ++j) {
              const size_t src_i = (pre * src_dim_size + j) * right_size + post;
              const I idx = ids[src_i];
              if (idx < max_value<I>()) {
                assert(idx < dst_dim_size);
                const size_t dst_i = (pre * dst_dim_size + idx) * right_size + post;
                out[dst_i] += inp[src_i];
              }
          }
      }
}

#define SCATTER_OP(TYPENAME, INDEX_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const INDEX_TYPENAME *ids, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const size_t left_size, \
    const size_t src_dim_size, \
    const size_t dst_dim_size, \
    const size_t right_size \
) { scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); } \

#define SCATTER_ADD_OP(TYPENAME, INDEX_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const INDEX_TYPENAME *ids, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const size_t left_size, \
    const size_t src_dim_size, \
    const size_t dst_dim_size, \
    const size_t right_size \
) { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); } \

// =============================================================================== //
//                impl
// =============================================================================== //

INDEX_SELECT_OP(float, int32_t, is_i32_f32)
INDEX_SELECT_OP(double, int32_t, is_i32_f64)
INDEX_SELECT_OP(uint8_t, int32_t, is_i32_u8)
INDEX_SELECT_OP(uint32_t, int32_t, is_i32_u32)
INDEX_SELECT_OP(int32_t, int32_t, is_i32_i32)

INDEX_SELECT_OP(float, uint32_t, is_u32_f32)
INDEX_SELECT_OP(double, uint32_t, is_u32_f64)
INDEX_SELECT_OP(uint8_t, uint32_t, is_u32_u8)
INDEX_SELECT_OP(uint32_t, uint32_t, is_u32_u32)
INDEX_SELECT_OP(int32_t, uint32_t, is_u32_i32)

GATHER_OP(float, int32_t, gather_i32_f32)
GATHER_OP(double, int32_t, gather_i32_f64)
GATHER_OP(uint8_t, int32_t, gather_i32_u8)
GATHER_OP(uint32_t, int32_t, gather_i32_u32)
GATHER_OP(int32_t, int32_t, gather_i32_i32)

GATHER_OP(float, uint32_t, gather_u32_f32)
GATHER_OP(double, uint32_t, gather_u32_f64)
GATHER_OP(uint8_t, uint32_t, gather_u32_u8)
GATHER_OP(uint32_t, uint32_t, gather_u32_u32)
GATHER_OP(int32_t, uint32_t, gather_u32_i32)

INDEX_ADD_OP(float, int32_t, ia_i32_f32)
INDEX_ADD_OP(double, int32_t, ia_i32_f64)
INDEX_ADD_OP(uint8_t, int32_t, ia_i32_u8)
INDEX_ADD_OP(uint32_t, int32_t, ia_i32_u32)
INDEX_ADD_OP(int32_t, int32_t, ia_i32_i32)

INDEX_ADD_OP(float, uint32_t, ia_u32_f32)
INDEX_ADD_OP(double, uint32_t, ia_u32_f64)
INDEX_ADD_OP(uint8_t, uint32_t, ia_u32_u8)
INDEX_ADD_OP(uint32_t, uint32_t, ia_u32_u32)
INDEX_ADD_OP(int32_t, uint32_t, ia_u32_i32)

SCATTER_ADD_OP(float, int32_t, sa_i32_f32)
SCATTER_ADD_OP(double, int32_t, sa_i32_f64)
SCATTER_ADD_OP(uint8_t, int32_t, sa_i32_u8)
SCATTER_ADD_OP(uint32_t, int32_t, sa_i32_u32)
SCATTER_ADD_OP(int32_t, int32_t, sa_i32_i32)

SCATTER_ADD_OP(float, uint32_t, sa_u32_f32)
SCATTER_ADD_OP(double, uint32_t, sa_u32_f64)
SCATTER_ADD_OP(uint8_t, uint32_t, sa_u32_u8)
SCATTER_ADD_OP(uint32_t, uint32_t, sa_u32_u32)
SCATTER_ADD_OP(int32_t, uint32_t, sa_u32_i32)

SCATTER_OP(float, int32_t, s_i32_f32)
SCATTER_OP(double, int32_t, s_i32_f64)
SCATTER_OP(uint8_t, int32_t, s_i32_u8)
SCATTER_OP(int32_t, int32_t, s_i32_i32)
SCATTER_OP(uint32_t, int32_t, s_i32_u32)

SCATTER_OP(float, uint32_t, s_u32_f32)
SCATTER_OP(double, uint32_t, s_u32_f64)
SCATTER_OP(uint8_t, uint32_t, s_u32_u8)
SCATTER_OP(int32_t, uint32_t, s_u32_i32)
SCATTER_OP(uint32_t, uint32_t, s_u32_u32)
