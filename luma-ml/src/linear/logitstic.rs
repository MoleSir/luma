use luma_tensor::{Bool, Device, Tensor};

use crate::{
    core::{PredictFit, PredictModel},
    error::MlResult,
    utils,
};

pub struct LogisticRegression {
    pub n_iter: usize,
    pub learning_rate: f64,
    pub threshold: f64,
}

pub struct LogisticRegressionModel<Dev: Device> {
    pub weights: Tensor<Dev>,
    pub bias: f64,
    pub threshold: f64,
}

impl Default for LogisticRegression {
    fn default() -> Self {
        LogisticRegression { n_iter: 1000, learning_rate: 0.1, threshold: 0.5 }
    }
}

impl<Dev: Device> PredictFit<Tensor<Dev>> for LogisticRegression {
    type Output = Tensor<Dev, Bool>;
    type Model = LogisticRegressionModel<Dev>;

    /// fit a logistic regression model (Binary Classification)
    /// $$
    /// z = w x + b
    /// y\_pred = \frac{1}{1 + e^{-z}}
    /// $$
    ///
    /// ## Args
    /// - `x`: (n_samples, n_features)
    /// - `y`: (n_samples,) , boolean values for binary classes
    ///
    /// ## Return
    /// - logistic regression model
    fn fit(&self, x: &Tensor<Dev>, y: &Tensor<Dev, Bool>) -> MlResult<Self::Model> {
        let (n_samples, n_features) = utils::validate_xy_shapes(x, y, None)?;
        let device = x.device();
        let dtype = x.dtype();

        let y_float = y.cast(dtype)?.unsqueeze(1)?;
        let weights = Tensor::zeros((n_features, 1), (device, dtype))?;
        let mut bias = 0.0;

        let lr = self.learning_rate;
        let n_samples_t = n_samples as f64;
        let x_t = x.transpose_last()?; // (n_features, n_samples)

        // train model
        /*
            z = XW + b
            y_pred = sigmoid(z) = 1 / (1 + exp(-z))

            Loss (BCE) = - y*log(y_pred) - (1-y)*log(1-y_pred)

            dloss/dz = y_pred - y
            dloss/dw = X^T @ (y_pred - y) / N
            dloss/db = mean(y_pred - y)
        */
        for _ in 0..self.n_iter {
            // forward pass
            let z = x.matmul(&weights)? + bias; // (n_samples, 1)            
            let y_pred = z.sigmoid()?;

            // backward
            let y_pred_grad = y_pred - &y_float; // (n_samples, 1)

            // W_grad: (n_features, n_samples) @ (n_samples, 1) => (n_features, 1)
            let w_grad = x_t.matmul(&y_pred_grad)? / n_samples_t;
            let b_grad = y_pred_grad.mean_all()?.to_scalar()?;

            // Update
            w_grad.mul_(lr)?;
            let b_grad = lr * b_grad;
            weights.sub_(w_grad)?;
            bias -= b_grad;
        }

        Ok(LogisticRegressionModel { weights, bias, threshold: self.threshold })
    }
}

impl<Dev: Device> PredictModel for LogisticRegressionModel<Dev> {
    type Input = Tensor<Dev>;
    type Output = Tensor<Dev, Bool>;

    /// ## Args
    /// - `x`: (n_samples, n_features)
    ///
    /// ## Return
    /// - y: (n_samples,)
    fn predict(&self, x: &Tensor<Dev>) -> MlResult<Tensor<Dev, Bool>> {
        self.predict_threshold(x, self.threshold)
    }
}

impl<Dev: Device> LogisticRegressionModel<Dev> {
    pub fn predict_proba(&self, x: &Tensor<Dev>) -> MlResult<Tensor<Dev>> {
        let z = x.matmul(&self.weights)?;
        z.add_(self.bias)?;
        z.sigmoid_()?;
        Ok(z.squeeze(1)?)
    }

    pub fn predict_threshold(&self, x: &Tensor<Dev>, threshold: f64) -> MlResult<Tensor<Dev, Bool>> {
        let probs = self.predict_proba(x)?;
        let preds = probs.ge(threshold)?;
        Ok(preds)
    }
}

#[cfg(test)]
mod tests {
    use luma_tensor::{Cpu, IndexOp};

    use crate::{
        PredictFit, PredictModel,
        datasets::{load_iris, train_test_split},
    };

    use super::LogisticRegression;

    #[test]
    fn test_iris() {
        let iris = load_iris(&Cpu).unwrap();
        let x = iris.data;
        let y = iris.target;
        let x = x.i((.., ..2)).unwrap().contiguous().unwrap();
        let y = y.eq(0).unwrap();
        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.3).unwrap();
        // println!("{}", x_train.shape());
        // println!("{}", x_test.shape());
        // println!("{}", y_train.shape());
        // println!("{}", y_test.shape());

        let trainer = LogisticRegression::default();
        let model = trainer.fit(&x_train, &y_train).unwrap();

        let y_pred = model.predict(&x_test).unwrap();

        let n_correct = y_pred.xor(&y_test).unwrap().false_count().unwrap();
        println!("Correct count: {}", n_correct);
        println!("Accurry: {}%", (n_correct as f64 / x_test.dims()[0] as f64) * 100.0);
    }
}
