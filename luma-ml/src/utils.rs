use luma_tensor::{DTypeKind, Device, Float, Tensor};
use thiserrorctx::Context;
use crate::error::{MlError, MlResult};

/// 检查输入 x y 是否满足样本/标签对的格式：
/// - `x`: (n_samples, n_features)
/// - `y`: (n_samples)
pub fn validate_xy_shapes<Dev, K1, K2>(x: &Tensor<Dev, K1>, y: &Tensor<Dev, K2>, sample_weight: Option<&Tensor<Dev, Float>>) -> MlResult<(usize, usize)> 
where 
    Dev: Device, 
    K1: DTypeKind<Dev>, 
    K2: DTypeKind<Dev>,
{
    if !x.same_device(y) {
        Err(MlError::Tensor(luma_tensor::Error::DeviceMismatch { lhs: x.device().name(), rhs: y.device().name() }))?;
    }
    let (n_samples, n_features) = x.dims2().map_err(MlError::Tensor).context("expect x as 2-dims")?;
    let n_samples_y = y.dims1().map_err(MlError::Tensor).context("expect y as 1-dim")?;
    if n_samples != n_samples_y {
        thiserrorctx::bail!(MlError::SampleSizeMismatch { x_samples: n_samples, y_samples: n_samples_y });
    }

    if let Some(sample_weight) = sample_weight {
        let n_samples_sw = sample_weight.dims1().map_err(MlError::Tensor).context("expect sample_weight as 1-dim")?;
        if n_samples != n_samples_sw {
            thiserrorctx::bail!(MlError::SampleSizeMismatch { x_samples: n_samples, y_samples: n_samples_sw });
        }
    }

    Ok((n_samples, n_features))
}