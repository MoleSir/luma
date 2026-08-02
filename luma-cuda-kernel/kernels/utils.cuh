#include <stdint.h>
#include <cmath>

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