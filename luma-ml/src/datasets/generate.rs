use derive_builder::Builder;
use luma_tensor::{Device, FloatDType, IndexOp, Tensor};
use crate::error::MlResult;

/// Make a regression dataset
/// 
/// ## Args
/// - `n_samples`
/// - `n_features`
/// - `option`: RegressionOption
/// 
/// ## Returns
/// - `x_train`: (n_samples, n_features)
/// - `y_train`: (n_samples,)
/// - `coef`: (n_features + 1,)
pub fn make_regression<Dev: Device>(n_samples: usize, n_features: usize, device: &Dev, option: RegressionOption) -> MlResult<RegressionData<Dev>> {
    option.generate(n_samples, n_features, device)
}

#[derive(Builder)]
#[builder(pattern = "owned")] // 允许 builder().noise(0.1).generate(...) 链式调用
pub struct RegressionOption {
    #[builder(default)]
    pub mean: f64,
    #[builder(default = "1.0")]
    pub std: f64,
    #[builder(default = "0.1")]
    pub noise: f64,
    #[builder(default = "None")]
    pub seed: Option<u32>,
    #[builder(default = "FloatDType::F32")]
    pub dtype: FloatDType,
}

pub struct RegressionData<Dev: Device> {
    pub x: Tensor<Dev>,
    pub y: Tensor<Dev>,
    pub coef: Tensor<Dev>,
}

impl RegressionOption {
    pub fn generate<Dev: Device>(self, n_samples: usize, n_features: usize, device: &Dev) -> MlResult<RegressionData<Dev>> {
        let mean = self.mean;
        let std = self.std;
        let noise = self.noise;
        let dtype = self.dtype;

        let weight_bias = Tensor::randn(mean, std, (n_features + 1,), (device, dtype))?;
        let weight = weight_bias.i(..n_features)?;
        let bias = weight_bias.i(n_features)?.to_scalar()?;

        let x = Tensor::randn(mean, std, (n_samples, n_features), (device, dtype))?;
        let y = x.matmul(&weight.unsqueeze(1)?)?.squeeze(1)?;
        
        y.add_(bias)?;
        if self.noise > 0.0 {
            y.add_(noise * Tensor::randn(0.0, 1.0, (n_samples,), (device, dtype))?)?;
        }

        Ok(RegressionData { x, y, coef: weight_bias })
    }
}

impl Default for RegressionOption {
    fn default() -> Self {
        RegressionOptionBuilder::default().build().expect("build error")
    }
}
