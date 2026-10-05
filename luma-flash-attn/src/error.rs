use core::ffi::{c_char, CStr};

use thiserror::Error;

use crate::api::ffi;

#[derive(Debug, Error)]
pub enum FlashAttnError {
    #[error("flash_attn: null pointer parameter ({0})")]
    NullPtr(String),

    #[error("flash_attn: invalid shape parameter ({0})")]
    InvalidShape(String),

    #[error("flash_attn: unsupported head_size ({0})")]
    InvalidHeadSize(String),

    #[error("flash_attn: invalid start_pos ({0})")]
    InvalidStartPos(String),

    #[error("flash_attn: buffer too small ({0})")]
    BufferTooSmall(String),

    #[error("flash_attn: CUDA error ({0})")]
    Cuda(String),

    #[error("flash_attn: unsupported ({0})")]
    Unsupported(String),

    #[error("flash_attn: unknown error code {0}")]
    Unknown(i32),

    #[error(transparent)]
    Tensor(#[from] luma_tensor::Error),
}

impl FlashAttnError {
    pub fn from_status(status: i32) -> Result<(), Self> {
        if status == ffi::FLASH_ATTN_OK {
            return Ok(());
        }
        let msg = last_error_string();
        Err(match status {
            ffi::FLASH_ATTN_ERR_NULL_PTR => Self::NullPtr(msg),
            ffi::FLASH_ATTN_ERR_INVALID_SHAPE => Self::InvalidShape(msg),
            ffi::FLASH_ATTN_ERR_INVALID_HEAD_SIZE => Self::InvalidHeadSize(msg),
            ffi::FLASH_ATTN_ERR_INVALID_START_POS => Self::InvalidStartPos(msg),
            ffi::FLASH_ATTN_ERR_CUDA => Self::Cuda(msg),
            other => Self::Unknown(other),
        })
    }
}

impl From<FlashAttnError> for luma_tensor::CustomOpError {
    fn from(e: FlashAttnError) -> Self {
        luma_tensor::CustomOpError::msg(e.to_string())
    }
}

impl From<FlashAttnError> for luma_tensor::Error {
    fn from(e: FlashAttnError) -> Self {
        luma_tensor::Error::CustomOp(e.into())
    }
}

fn last_error_string() -> String {
    unsafe {
        let p: *const c_char = ffi::flash_attn_last_error();
        if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}
