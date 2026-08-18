use crate::Layout;
use crate::device::cuda::{Cuda, CudaError, CudaResult};
use cudarc::{
    cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig, sys::cublasOperation_t},
    driver::{CudaSlice, DeviceRepr},
};

fn matmul_config<T: Copy>(
    alpha: T,
    beta: T,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> CudaResult<StridedBatchedConfig<T>> {
    let lhs_stride = lhs_l.stride();
    let rhs_stride = rhs_l.stride();
    let lhs_dims = lhs_l.dims();
    let rhs_dims = rhs_l.dims();

    if lhs_dims.len() < 2 || rhs_dims.len() < 2 {
        return Err(CudaError::MatMulNonContiguous { msg: "matmul requires at least 2D".into() });
    }

    let (lhs_m2, lhs_m1) = (lhs_stride[lhs_stride.len() - 2], lhs_stride[lhs_stride.len() - 1]);
    let (rhs_m2, rhs_m1) = (rhs_stride[rhs_stride.len() - 2], rhs_stride[rhs_stride.len() - 1]);

    let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        return Err(CudaError::MatMulNonContiguous {
            msg: format!("LHS stride {:?} invalid for shape {:?} (m={}, k={})", lhs_stride, lhs_dims, m, k),
        });
    };

    let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        return Err(CudaError::MatMulNonContiguous {
            msg: format!("RHS stride {:?} invalid for shape {:?} (k={}, n={})", rhs_stride, rhs_dims, k, n),
        });
    };

    let stride_a: i64 = if rhs_stride.len() >= 3 { rhs_stride[rhs_stride.len() - 3] as i64 } else { (n * k) as i64 };
    let stride_b: i64 = if lhs_stride.len() >= 3 { lhs_stride[lhs_stride.len() - 3] as i64 } else { (m * k) as i64 };
    let stride_c: i64 = (m * n) as i64;

    Ok(StridedBatchedConfig {
        batch_size: b as i32,
        gemm: GemmConfig { alpha, beta, m: n as i32, n: m as i32, k: k as i32, lda, ldb, ldc: n as i32, transa, transb },
        stride_a,
        stride_b,
        stride_c,
    })
}

pub(crate) fn launch_matmul<T: Copy + DeviceRepr>(
    device: &Cuda,
    alpha: T,
    beta: T,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &CudaSlice<T>,
    lhs_l: &Layout,
    rhs: &CudaSlice<T>,
    rhs_l: &Layout,
) -> CudaResult<CudaSlice<T>>
where
    CudaBlas: Gemm<T>,
{
    let cfg = matmul_config(alpha, beta, (b, m, n, k), lhs_l, rhs_l)?;
    let mut out = device.alloc::<T>(b * m * n)?;
    let blas = device.0.blas.lock().unwrap();
    unsafe {
        blas.gemm_strided_batched(cfg, rhs, lhs, &mut out).map_err(CudaError::Cublas)?;
    }
    Ok(out)
}

pub(crate) fn launch_add_matmul_<T: Copy + DeviceRepr>(
    device: &Cuda,
    alpha: T,
    beta: T,
    dst: &mut CudaSlice<T>,
    _dst_l: &Layout,
    lhs: &CudaSlice<T>,
    lhs_l: &Layout,
    rhs: &CudaSlice<T>,
    rhs_l: &Layout,
    (b, m, n, k): (usize, usize, usize, usize),
) -> CudaResult<()>
where
    CudaBlas: Gemm<T>,
{
    let cfg = matmul_config(alpha, beta, (b, m, n, k), lhs_l, rhs_l)?;
    let blas = device.0.blas.lock().unwrap();
    unsafe {
        blas.gemm_strided_batched(cfg, rhs, lhs, dst).map_err(CudaError::Cublas)?;
    }
    Ok(())
}
