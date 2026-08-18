use cudarc::{cublas::result::CublasError, curand::result::CurandError, driver::DriverError};

#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error(transparent)]
    CudaDriver(#[from] DriverError),

    #[error(transparent)]
    Curand(#[from] CurandError),

    #[error(transparent)]
    Cublas(#[from] CublasError),

    #[error("op {2} with diff cuda, {0} and {1}")]
    DiffCuda(String, String, String),

    #[error("matmul not contiguous: {msg}")]
    MatMulNonContiguous { msg: String },

    #[error("unsupport int matmul")]
    UnsupportIntMatmul,
}

pub type CudaResult<T> = std::result::Result<T, CudaError>;
