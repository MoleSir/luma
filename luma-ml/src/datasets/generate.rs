use crate::error::MlResult;
use derive_builder::Builder;
use luma_tensor::{Device, FloatDType, IndexOp, Tensor};

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
pub fn make_regression<Dev: Device>(
    n_samples: usize,
    n_features: usize,
    device: &Dev,
    option: RegressionOption,
) -> MlResult<RegressionData<Dev>> {
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

#[cfg(test)]
mod tests {
    use luma_tensor::{Cpu, IndexOp};

    use crate::datasets::{RegressionOption, RegressionOptionBuilder, make_regression};

    #[test]
    fn test_make_regression_shapes_and_linearity() {
        let data = make_regression(50, 3, &Cpu, RegressionOptionBuilder::default().noise(0.0).build().unwrap()).unwrap();

        assert_eq!(data.x.dims(), [50, 3]);
        assert_eq!(data.y.dims(), [50]);
        assert_eq!(data.coef.dims(), [4]);

        // 无噪声时 y == x @ w + b 应精确成立
        let w = data.coef.i(..3).unwrap();
        let b = data.coef.i(3).unwrap().to_scalar().unwrap();
        let y_expected = data.x.matmul(&w.unsqueeze(1).unwrap()).unwrap().squeeze(1).unwrap();
        y_expected.add_(b).unwrap();

        let max_diff =
            data.y.to_vec().unwrap().into_iter().zip(y_expected.to_vec().unwrap()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(max_diff < 1e-9, "max diff = {max_diff}");
    }

    #[test]
    fn test_make_regression_dtype_option() {
        // dtype 选项应生效（默认 F32）
        let data_f32 = make_regression(10, 2, &Cpu, RegressionOption::default()).unwrap();
        assert_eq!(data_f32.x.dtype(), luma_tensor::FloatDType::F32);

        let data_f64 =
            make_regression(10, 2, &Cpu, RegressionOptionBuilder::default().dtype(luma_tensor::FloatDType::F64).build().unwrap()).unwrap();
        assert_eq!(data_f64.x.dtype(), luma_tensor::FloatDType::F64);
        assert_eq!(data_f64.y.dtype(), luma_tensor::FloatDType::F64);
    }
}
