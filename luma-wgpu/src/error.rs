//! wgpu backend errors.
//!
//! Mirrors `luma-cuda`'s error module: a backend-specific error enum converted
//! into the shared [`luma_tensor::Error::Device`] variant at the trait seam.

use luma_tensor::DType;

#[derive(Debug, thiserror::Error)]
pub enum WgpuError {
    #[error("wgpu init failed: {0}")]
    Init(String),

    #[error("wgpu unsupported dtype {0:?} for op {1}")]
    UnsupportedDType(DType, &'static str),

    #[error("wgpu device does not support feature '{0}'")]
    UnsupportedFeature(&'static str),

    #[error("wgpu validation failed: {0}")]
    Validation(String),

    #[error("wgpu buffer map failed: {0}")]
    Map(String),

    #[error("wgpu: {0}")]
    Msg(String),
}

pub type WgpuResult<T> = std::result::Result<T, WgpuError>;

impl From<WgpuError> for luma_tensor::Error {
    fn from(error: WgpuError) -> Self {
        luma_tensor::Error::Device { device: "wgpu".into(), error: Box::new(error) }
    }
}
