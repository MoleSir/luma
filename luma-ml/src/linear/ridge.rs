use luma_tensor::{no_grad, Device, Tensor};

use crate::{core::{PredictFit, PredictModel}, error::MlResult, utils, PredictFitWithWeight};

pub struct RidgeRegression {
    pub n_iter: usize, 
    pub learning_rate: f64,
    pub alpha: f64,
}

pub struct RidgeRegressionModel<Dev: Device> {
    pub weights: Tensor<Dev>,
    pub bias: f64,
    pub alpha: f64,
}

impl Default for RidgeRegression {
    fn default() -> Self {
        RidgeRegression { n_iter: 1000, learning_rate: 0.01, alpha: 0.1 }
    }
}

impl<Dev: Device> PredictFit<Tensor<Dev>> for RidgeRegression {
    type Output = Tensor<Dev>;
    type Model = RidgeRegressionModel<Dev>;

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
        self.fit_with(x, y, None)
    }
}

impl<Dev: Device> PredictFitWithWeight<Tensor<Dev>> for RidgeRegression {
    type Weight = Tensor<Dev>;

    fn fit_with_weight(&self, x: &Tensor<Dev>, y: &Self::Output, weight: &Self::Weight) -> MlResult<Self::Model> {
        self.fit_with(x, y, Some(weight))
    }
}

impl<Dev: Device> PredictModel for RidgeRegressionModel<Dev> {
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

impl RidgeRegression {
    /// - `x`: (n_samples, n_features)
    /// - `y`: (n_samples),
    /// - `sample_weight`: Option<(n_samples)>,    
    fn fit_with<Dev: Device>(&self, x: &Tensor<Dev>, y: &Tensor<Dev>, sample_weight: Option<&Tensor<Dev>>) -> MlResult<RidgeRegressionModel<Dev>> {
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

        /*
            y_pred = w0 * x0 + w1 * x1 + ... + b
            loss = (y_pred - y)^2 + \alpha \sum |w|

            dloss/dy_pred = 2 * (y_pred - y) + \alpha \sum |w|

            dloss/dwi =  2 * (y_pred - y) * xi + \alpha * w.sign()
            dloss/db = 2 * (y_pred - y)
        */
        for _ in 0..self.n_iter {
            // (n_samples, n_features) @ (n_features, 1) => (n_samples, 1)
            let y_pred = x.matmul(&weights)?.add_(bias)?;

            // (n_samples, 1)
            let dloss = (&y_pred - &y).mul_(&sample_weight)?.mul_(2.0)?;

            // (n_features, n_samples) @ (n_samples, 1) => (n_features, 1)
            let w_grad = x_t.matmul(&dloss)?.div_(weight_sum)?;
            let b_grad = dloss.sum_all()?.div_(weight_sum)?.to_scalar()?;

            let l2_grad = &weights * (2.0 * self.alpha); 
            w_grad.add_(l2_grad)?;

            w_grad.mul_(self.learning_rate)?;
            let b_grad = self.learning_rate * b_grad;
            weights.sub_(w_grad)?;
            bias -= b_grad;
        }

        Ok(RidgeRegressionModel { weights, bias, alpha: self.alpha  })
    }
}

#[cfg(test)]
mod tests {
    use luma_tensor::{Cpu, Tensor};

    use crate::{datasets::{make_regression, RegressionOption}, linear::RidgeRegression, core::PredictFit};

    #[test]
    fn test_gd_1d() {
        const N_SAMPLES: usize = 50;
        const W: f64 = 3.0;
        const B: f64 = 2.5;
        let x_train = Tensor::rand(-1.0, 1.0, (N_SAMPLES,), &Cpu).unwrap();
        let y_train = W * &x_train + B;
        let x_train = x_train.unsqueeze(1).unwrap();
        let y_train = y_train + 0.1 * Tensor::randn(0.0, 1.0, (N_SAMPLES,), &Cpu).unwrap();

        let trainer = RidgeRegression::default();
        let model = trainer.fit(&x_train, &y_train).unwrap();
        println!("{}", model.weights);
        println!("{}", model.bias);
    }

    #[test]
    fn test_gd_nd() {
        const N_SAMPLES: usize = 50;
        const N_FEATURES: usize = 5;
        let train_data = make_regression(N_SAMPLES, N_FEATURES, &Cpu, RegressionOption::default()).unwrap();
        let x_train = train_data.x;
        let y_train = train_data.y;
        let weight_bias = train_data.coef;

        let trainer = RidgeRegression::default();
        let model = trainer.fit(&x_train, &y_train).unwrap();
        println!("{}", weight_bias);
        println!("{}", model.weights);
        println!("{}", model.bias);
    }
}