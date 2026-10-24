use cudarc::nccl;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum NcclError {
    /// An error bubbled up from the tensor layer (shape/dtype/kernel).
    #[error(transparent)]
    Tensor(#[from] luma_tensor::Error),

    #[error(transparent)]
    Nccl(#[from] WrapNcclError),

    #[error(transparent)]
    Cuda(#[from] luma_cuda::CudaError),
}

#[derive(Debug)]
pub struct WrapNcclError(pub nccl::result::NcclError);

impl std::fmt::Display for WrapNcclError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "nccl: {:?}", self.0.0)
    }
}

impl std::error::Error for WrapNcclError {}

pub(super) fn map_nccl_err(e: nccl::result::NcclError) -> NcclError {
    NcclError::Nccl(WrapNcclError(e))
}
