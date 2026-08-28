#[thiserrorctx::context_error]
pub enum MlError {
    #[error(transparent)]
    Tensor(#[from] luma_tensor::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("knn error: {0}")]
    Knn(String),

    #[error("Mismatched number of samples: x has {x_samples}, y has {y_samples}")]
    SampleSizeMismatch {
        x_samples: usize,
        y_samples: usize,
    },

}