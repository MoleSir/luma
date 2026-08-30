use luma_tensor::{Device, Tensor};

use crate::{MlResult, PredictFit, PredictFitWithWeight, PredictModel, utils};

pub struct AdaBoostRegressor<F> {
    pub fiter: F,
    pub n_estimators: usize,
    pub learning_rate: f64,
}

pub struct AdaBoostModel<M> {
    pub estimators: Vec<(M, f64)>,
}

impl<Dev: Device, M> PredictModel for AdaBoostModel<M>
where
    M: PredictModel<Input = Tensor<Dev>, Output = Tensor<Dev>>,
{
    type Input = M::Input;
    type Output = M::Output;

    /// ## Return
    /// - `y`: (n_samples,)
    ///
    /// AdaBoost.R2 预测 = 各弱模型预测的**加权中位数**（权重 \alpha_i）。
    /// 注意不能直接求加权和：\alpha 通常远大于 1，多轮累加会把预测放大到离谱的量级。
    fn predict(&self, x: &Self::Input) -> MlResult<Self::Output> {
        if self.estimators.is_empty() {
            luma_tensor::bail!("no estimators! first round error >= 0.5");
        }

        let (n_samples, _) = x.dims2()?;
        let mut preds = Vec::with_capacity(self.estimators.len());
        for (model, _) in &self.estimators {
            preds.push(model.predict(x)?.to_vec()?);
        }
        let total_alpha: f64 = self.estimators.iter().map(|(_, alpha)| alpha).sum();

        // 对每个样本：按预测值排序，累计 \alpha 权重，取累计权重首次过半的预测值
        let mut result = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let mut pairs: Vec<(f64, f64)> = preds.iter().zip(&self.estimators).map(|(p, (_, alpha))| (p[i], *alpha)).collect();
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let mut acc = 0.0;
            for (value, alpha) in pairs {
                acc += alpha;
                if acc >= 0.5 * total_alpha {
                    result.push(value);
                    break;
                }
            }
        }

        // Tensor::new 默认 F64，to_dtype 对齐输入 x 的 dtype（如默认 F32）
        Ok(Tensor::<Dev>::new(result, x.device())?.to_dtype(x.dtype())?)
    }
}

impl<F> AdaBoostRegressor<F> {
    pub fn fit<Dev: Device>(&self, x: &Tensor<Dev>, y: &Tensor<Dev>) -> MlResult<AdaBoostModel<F::Model>>
    where
        F: PredictFitWithWeight<Tensor<Dev>, Output = Tensor<Dev>, Weight = Tensor<Dev>>,
    {
        let (n_samples, _) = utils::validate_xy_shapes(x, y, None)?;

        // 初始化权重
        let mut weights = Tensor::ones((n_samples,), (x.device(), x.dtype()))?.div_(n_samples as f64)?;
        let mut estimators = Vec::new();

        // 依次训练模型
        for _ in 0..self.n_estimators {
            // 使用当前 weight 训练模型
            let model = self.fiter.fit_with_weight(x, y, &weights)?;

            // 计算该模型的误差
            let y_pred = model.predict(x)?;
            let errors = (y_pred - y).abs()?;

            // 计算相对误差，将 errors -> [0, 1]
            let max_error = errors.max_all()?.to_scalar()?;
            let rel_errors = errors / max_error;

            // 计算加权平均误差率
            // weights 表示样本的重要程度，将重要性体现到误差上，接着求和
            // 如果模型在权重很高的样本上算错了，误差会很大
            let avg_error = (&rel_errors * &weights).sum_all()?.to_scalar()?;
            // 算法要求 L < 0.5，如果误差超过 0.5，说明这个模型比随机猜测还差，此时会停止迭代。
            if avg_error >= 0.5 {
                break;
            }

            // 计算模型权重
            /*

                beta 衡量模型 “好坏” 的系数
                           L
                \beta = -------
                         1 - L

                - L -> 0, \beta -> 0
                - L == 0.5, \beta == 1

                                  1              1 - L
                \alpha = \ln (---------) = \ln (-------)
                                \beta              L

            */
            let beta = avg_error / (1.0 - avg_error);
            // learning_rate 收缩每个弱模型的贡献（与 sklearn 一致）
            let alpha = self.learning_rate * (1.0 / beta).ln();

            // 更新样本权重
            /*

                new_w = old_w * \beta^(1 - err)

                - 预测非常准确：err->0，更新因子为 \beta^1，而因为 \beta < 1，所以让权重减小
                - 预测非常糟糕：err->1,更新因子接近 1，样本权重保持不变

            */
            let new_weights = rel_errors
                .to_vec()?
                .into_iter()
                .zip(weights.to_vec()?)
                .map(|(e, old_w)| {
                    let factor = beta.powf(1.0 - e);
                    old_w * factor
                })
                .collect::<Vec<_>>();
            // Tensor::new 默认 F64，to_dtype 对齐输入 x 的 dtype，避免与 rel_errors（x 派生）运算时 DTypeMismatch
            weights = Tensor::<Dev>::new(new_weights, x.device())?.to_dtype(x.dtype())?;

            // // 归一化权重
            let sum_w = weights.sum_all()?.to_scalar()?;
            weights.div_(sum_w)?;

            estimators.push((model, alpha));
        }

        Ok(AdaBoostModel { estimators })
    }
}

impl<Dev: Device, F> PredictFit<Tensor<Dev>> for AdaBoostRegressor<F>
where
    F: PredictFitWithWeight<Tensor<Dev>, Output = Tensor<Dev>, Weight = Tensor<Dev>>,
{
    type Output = Tensor<Dev>;
    type Model = AdaBoostModel<F::Model>;

    fn fit(&self, x: &Tensor<Dev>, y: &Tensor<Dev>) -> MlResult<Self::Model> {
        // 委托给 inherent fit（UFCS 显式消除与 trait 方法的同名歧义）
        AdaBoostRegressor::fit(self, x, y)
    }
}

#[cfg(test)]
mod tests {
    use luma_tensor::{Cpu, Tensor};

    use crate::{
        core::{PredictFit, PredictModel},
        ensemble::AdaBoostRegressor,
        linear::LinearRegression,
    };

    #[test]
    fn test_ada_boost_regression() {
        const N_SAMPLES: usize = 100;
        // x 取 [-1, 1] 保证默认学习率 (0.01) 的线性回归收敛（x 尺度太大会发散）
        // 输入用默认 F32 验证内部 Tensor::new + to_dtype 的 dtype 对齐
        let x = Tensor::<Cpu>::rand(-1.0, 1.0, (N_SAMPLES,), &Cpu).unwrap();
        let y = 3.0 * &x + 2.0 + 0.1 * Tensor::<Cpu>::randn(0.0, 1.0, (N_SAMPLES,), &Cpu).unwrap();
        let x = x.unsqueeze(1).unwrap();

        let trainer = AdaBoostRegressor { fiter: LinearRegression::default(), n_estimators: 10, learning_rate: 1.0 };
        // inherent fit 与 PredictFit trait fit 两条路径都可训练（trait 路径可接入 Pipeline 等泛型接口）
        let _model_inherent = trainer.fit(&x, &y).unwrap();
        let model = PredictFit::fit(&trainer, &x, &y).unwrap();

        // 至少训练出一个弱模型；误差率 < 0.5 时 \alpha = ln((1-L)/L) > 0
        assert!(!model.estimators.is_empty());
        assert!(
            model.estimators.iter().all(|(_, alpha)| *alpha > 0.0),
            "alphas = {:?}",
            model.estimators.iter().map(|(_, a)| a).collect::<Vec<_>>()
        );

        // 10 轮 boosting 后训练误差应远小于 0.5（噪声标准差仅 0.1）
        let y_pred = model.predict(&x).unwrap();
        let mse = (y_pred - &y).sqr().unwrap().mean_all().unwrap().to_scalar().unwrap();
        assert!(mse < 0.5, "mse = {mse}");

        // learning_rate 应线性收缩 \alpha（训练过程与 learning_rate 无关，\alpha 恰好减半）
        let trainer_half = AdaBoostRegressor { fiter: LinearRegression::default(), n_estimators: 10, learning_rate: 0.5 };
        let model_half = PredictFit::fit(&trainer_half, &x, &y).unwrap();
        assert_eq!(model_half.estimators.len(), model.estimators.len());
        for ((_, alpha), (_, alpha_half)) in model.estimators.iter().zip(&model_half.estimators) {
            assert!((alpha_half - 0.5 * alpha).abs() < 1e-12, "alpha = {alpha}, alpha_half = {alpha_half}");
        }
    }
}
