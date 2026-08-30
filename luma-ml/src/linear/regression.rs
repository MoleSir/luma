use luma_tensor::{no_grad, Device, Tensor};

use crate::{core::{PredictFit, PredictModel}, error::MlResult, utils, PredictFitWithWeight};

pub enum LinearRegression {
    GradientDescent { n_iter: usize, learning_rate: f64, },
    NormalEquations,
}

pub struct LinearRegressionModel<Dev: Device> {
    pub weights: Tensor<Dev>,
    pub bias: f64,
}

impl Default for LinearRegression {
    fn default() -> Self {
        LinearRegression::GradientDescent { n_iter: 1000, learning_rate: 0.01 }
    }
}

impl<Dev: Device> PredictFit<Tensor<Dev>> for LinearRegression {
    type Output = Tensor<Dev>;
    type Model = LinearRegressionModel<Dev>;

    /// fit a linear regression model
    /// $$
    /// y = w x + b
    /// $$
    /// 
    /// ## Args
    /// - `x`: (n_samples, n_features) 
    /// - `y`: (n_samples,)
    /// 
    /// ## Return
    /// - linear gression model
    fn fit(&self, x: &Tensor<Dev>, y: &Tensor<Dev>) -> MlResult<Self::Model> {
        match self {
            Self::GradientDescent { n_iter, learning_rate } => Self::fit_gd(x, y, None, *n_iter, *learning_rate),
            Self::NormalEquations => Self::fit_ne(x, y),
        }
    }
}

impl<Dev: Device> PredictFitWithWeight<Tensor<Dev>> for LinearRegression {
    type Weight = Tensor<Dev>;

    fn fit_with_weight(&self, x: &Tensor<Dev>, y: &Self::Output, weight: &Self::Weight) -> MlResult<Self::Model> {
        match self {
            Self::GradientDescent { n_iter, learning_rate } => Self::fit_gd(x, y, Some(weight), *n_iter, *learning_rate),
            Self::NormalEquations => Self::fit_ne(x, y),
        }
    }
}

impl<Dev: Device> PredictModel for LinearRegressionModel<Dev> {
    type Input = Tensor<Dev>;
    type Output = Tensor<Dev>;

    /// ## Args:
    /// - `x`: (n_samples, n_features)
    /// 
    /// ## Return
    /// - `y`: (n_samples, )
    fn predict(&self, x: &Tensor<Dev>) -> MlResult<Tensor<Dev>> {
        no_grad!();
        let y = x.matmul(&self.weights)?;
        y.add_(self.bias)?;
        Ok(y.squeeze(1)?)
    }
}

impl LinearRegression {
    #[allow(unused)]
    fn fit_ne<Dev: Device>(x: &Tensor<Dev>, y: &Tensor<Dev>) -> MlResult<LinearRegressionModel<Dev>> {
        unimplemented!("fit with Normal Equations")
    }

    /// - `x`: (n_samples, n_features)
    /// - `y`: (n_samples),
    /// - `sample_weight`: Option<(n_samples)>,    
    fn fit_gd<Dev: Device>(x: &Tensor<Dev>, y: &Tensor<Dev>, sample_weight: Option<&Tensor<Dev>>, n_iter: usize, learning_rate: f64) -> MlResult<LinearRegressionModel<Dev>> {
        let device = x.device();
        let dtype = x.dtype();

        let (n_samples, n_features) = utils::validate_xy_shapes(x, y, sample_weight)?;
        
        let weights = Tensor::zeros((n_features, 1), (device, dtype))?;
        let mut bias = 0.0;

        let y = y.unsqueeze(1)?; // (n_samples, 1)
        let x_t = x.transpose_last()?;

        // (n_samples, 1)
        let sample_weight = match sample_weight {
            Some(w) => w.unsqueeze(1)?,
            None => Tensor::ones((n_samples, 1), (device, dtype))?,
        };
        let weight_sum = sample_weight.sum_all()?.to_scalar()?;

        for _ in 0..n_iter {
            // (n_samples, n_features) @ (n_features, 1) => (n_samples, 1)
            let y_pred = x.matmul(&weights)?.add_(bias)?;

            // (n_samples, 1)
            let dloss = (&y_pred - &y).mul_(&sample_weight)?.mul_(2.0)?;

            // (n_features, n_samples) @ (n_samples, 1) => (n_features, 1)
            let w_grad = x_t.matmul(&dloss)?.div_(weight_sum)?;
            let b_grad = dloss.sum_all()?.div_(weight_sum)?.to_scalar()?;

            w_grad.mul_(learning_rate)?;
            let b_grad = learning_rate * b_grad;
            weights.sub_(w_grad)?;
            bias -= b_grad;
        }

        Ok(LinearRegressionModel { weights, bias })
    }
}

#[cfg(test)]
mod tests {
    use luma_tensor::{Cpu, Tensor};

    use crate::{datasets::{make_regression, RegressionOption}, linear::LinearRegression, core::PredictFit};

    #[test]
    fn test_gd_1d() {
        const N_SAMPLES: usize = 50;
        const W: f64 = 3.0;
        const B: f64 = 2.5;
        let x_train = Tensor::rand(-1.0, 1.0, (N_SAMPLES,), &Cpu).unwrap();
        let y_train = W * &x_train + B;
        let x_train = x_train.unsqueeze(1).unwrap();
        let y_train = y_train + 0.1 * Tensor::randn(0.0, 1.0, (N_SAMPLES,), &Cpu).unwrap();

        let trainer = LinearRegression::default();
        let model = trainer.fit(&x_train, &y_train).unwrap();
        println!("{}", model.weights);
        println!("{}", model.bias);

        // 噪声 0.1 + 50 样本，GD 1000 轮应收敛到真实参数附近
        let weights = model.weights.to_vec().unwrap();
        assert!((weights[0] - W).abs() < 0.3, "w = {}", weights[0]);
        assert!((model.bias - B).abs() < 0.5, "b = {}", model.bias);
    }

    #[test]
    fn test_gd_nd() {
        const N_SAMPLES: usize = 50;
        const N_FEATURES: usize = 5;
        let train_data = make_regression(N_SAMPLES, N_FEATURES, &Cpu, RegressionOption::default()).unwrap();
        let x_train = train_data.x;
        let y_train = train_data.y;
        let weight_bias = train_data.coef;

        let trainer = LinearRegression::default();
        let model = trainer.fit(&x_train, &y_train).unwrap();
        println!("{}", weight_bias);
        println!("{}", model.weights);
        println!("{}", model.bias);
    }
}