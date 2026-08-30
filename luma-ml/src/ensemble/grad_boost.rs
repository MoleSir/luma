use std::marker::PhantomData;

use luma_tensor::{Device, Tensor};

use crate::{MlResult, PredictFit, PredictModel};

pub struct GradientBoostRegressor<F> {
    pub fitter: F,
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub loss: Loss,
}

#[derive(Clone, Copy, Debug)]
pub enum Loss {
    Mse,
    Mae,
}

pub struct GradientBoostRegressorModel<Dev: Device, F> {
    pub models: Vec<F>,
    pub learning_rate: f64,
    marker: PhantomData<fn() -> Dev>,
}

impl<Dev: Device, F> PredictFit<Tensor<Dev>> for GradientBoostRegressor<F>
where
    F: PredictFit<Tensor<Dev>, Output = Tensor<Dev>>,
{
    type Model = GradientBoostRegressorModel<Dev, F::Model>;
    type Output = Tensor<Dev>;

    fn fit(&self, x: &Tensor<Dev>, y: &Tensor<Dev>) -> crate::MlResult<Self::Model> {
        let y_pred = y.zeros_like()?;
        let mut models = vec![];

        for _ in 0..self.n_estimators {
            // MSE => Loss = (pred - y)
            // 负梯度 => y - pred
            let grad = self.loss.gradient(y, &y_pred)?;
            grad.mul_(-1.0)?;

            let model = self.fitter.fit(x, &grad)?;

            let update = model.predict(x)?;
            update.mul_(self.learning_rate)?;

            y_pred.add_(&update)?;

            models.push(model);
        }

        Ok(GradientBoostRegressorModel { models, learning_rate: self.learning_rate, marker: PhantomData })
    }
}

impl<Dev: Device, M> PredictModel for GradientBoostRegressorModel<Dev, M>
where
    M: PredictModel<Input = Tensor<Dev>, Output = Tensor<Dev>>,
{
    type Input = M::Input;
    type Output = M::Output;

    fn predict(&self, x: &Self::Input) -> crate::MlResult<Self::Output> {
        let y = self.models[0].predict(x)?;
        y.mul_(self.learning_rate)?;
        for model in self.models.iter().skip(1) {
            let model_y = model.predict(x)?;
            model_y.mul_(self.learning_rate)?;
            y.add_(model_y)?;
        }
        Ok(y)
    }
}

impl Loss {
    pub fn gradient<Dev: Device>(&self, y: &Tensor<Dev>, y_pred: &Tensor<Dev>) -> MlResult<Tensor<Dev>> {
        match self {
            Loss::Mse => {
                // Loss = (y_pred - y)^2
                // dLoss = 2 * (y_pred - y)
                let grad = y_pred - y; // y_pred - y
                grad.mul_(2.0)?;
                Ok(grad)
            }
            Loss::Mae => {
                // Loss = |y_pred - y|
                let grad = y_pred - y;
                grad.sign_()?;
                Ok(grad)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use luma_tensor::{Cpu, Tensor};

    use crate::{
        core::{PredictFit, PredictModel},
        ensemble::{GradientBoostRegressor, Loss},
        tree::DecisionTreeRegressor,
    };

    /// y = 3x + 2 + 0.1 * noise, x ∈ [0, 10)
    /// 输入用默认 F32，验证决策树 predict 的 to_dtype 对齐（y.zeros_like 是 F32）
    fn make_linear_data() -> (Tensor<Cpu>, Tensor<Cpu>) {
        const N_SAMPLES: usize = 100;
        let x = Tensor::<Cpu>::rand(0.0, 10.0, (N_SAMPLES,), &Cpu).unwrap();
        let y = 3.0 * &x + 2.0 + 0.1 * Tensor::<Cpu>::randn(0.0, 1.0, (N_SAMPLES,), &Cpu).unwrap();
        let x = x.unsqueeze(1).unwrap();
        (x, y)
    }

    #[test]
    fn test_loss_mse_gradient() {
        // Loss = (y_pred - y)^2, 梯度 = 2 * (y_pred - y)
        let y = Tensor::<Cpu>::new(vec![1.0, 2.0, 3.0], &Cpu).unwrap();
        let y_pred = Tensor::<Cpu>::new(vec![2.0, 2.0, 1.0], &Cpu).unwrap();

        let grad = Loss::Mse.gradient(&y, &y_pred).unwrap();
        assert_eq!(grad.to_vec().unwrap(), vec![2.0, 0.0, -4.0]);
    }

    #[test]
    fn test_gradient_boost_reduces_train_error() {
        let (x, y) = make_linear_data();

        // 训练误差应随 n_estimators 增加而下降
        let train_mse = |n_estimators: usize| {
            let trainer =
                GradientBoostRegressor { fitter: DecisionTreeRegressor::new(3), n_estimators, learning_rate: 0.1, loss: Loss::Mse };
            let model = trainer.fit(&x, &y).unwrap();
            let y_pred = model.predict(&x).unwrap();
            (y_pred - &y).sqr().unwrap().mean_all().unwrap().to_scalar().unwrap()
        };

        let mse_5 = train_mse(5);
        let mse_20 = train_mse(20);
        assert!(mse_20 < mse_5, "mse_5 = {mse_5}, mse_20 = {mse_20}");
        // 噪声标准差 0.1，20 轮后训练误差应接近噪声下限，远小于 0.5
        assert!(mse_20 < 0.5, "mse_20 = {mse_20}");
    }
}
