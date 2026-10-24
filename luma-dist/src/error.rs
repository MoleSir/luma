use thiserror::Error;

#[derive(Debug, Error)]
pub enum DistError {
    /// An error bubbled up from the tensor layer (shape/dtype/kernel).
    #[error(transparent)]
    Tensor(#[from] luma_tensor::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[cfg(feature = "nccl")]
    #[error(transparent)]
    Nccl(#[from] crate::backends::nccl::NcclError),
    #[cfg(feature = "nccl")]
    #[error(transparent)]
    Cuda(#[from] luma_cuda::CudaError),

    #[cfg(feature = "tcp")]
    #[error(transparent)]
    Tcp(#[from] crate::backends::tcp::TcpError),

    /// A backend-specific failure, rendered as a string.
    #[error("dist backend error: {0}")]
    Backend(String),

    /// The requested operation is not supported by this backend.
    #[error("dist unsupported op: {0}")]
    Unsupported(String),

    /// A rank argument was outside `0..world_size`.
    #[error("invalid rank {rank} for world_size {world_size}")]
    InvalidRank { rank: usize, world_size: usize },

    #[error("unmatch world size in one group {0} != {1}")]
    UnmatchWorldSize(usize, usize),

    #[error("there are duplicate ranks!")]
    DuplicateRank(usize),

    /// A collective received an operand with an incompatible shape.
    #[error("dist shape mismatch: {0}")]
    Shape(String),

    /// The `Comm`s handed to a local group constructor are inconsistent
    /// (empty, mismatched world sizes, or duplicated/out-of-range ranks).
    #[error("invalid local group: {0}")]
    InvalidGroup(String),

    /// Multi-process bootstrap failure (env vars, rendezvous, …).
    #[error("bootstrap: {0}")]
    Bootstrap(String),

    /// Rendezvous transport failure (bind, connect, timeout, frame, …).
    #[error("rendezvous: {0}")]
    Rendezvous(String),

    #[error("multi broadcast roots '{0:?}' > 1")]
    MultiBroadcastRoot(Vec<usize>),

    #[error("no broadcast root")]
    NoBroadcastRoot,
}

/// Convenience alias used across the distributed API.
pub type DistResult<T> = std::result::Result<T, DistError>;
