use luma_tensor::{Device, IndexOp, Int, Tensor};

use crate::{
    core::{PredictFit, PredictModel},
    utils,
};

pub struct GaussianNB {
    pub var_smoothing: f64,
}

impl Default for GaussianNB {
    fn default() -> Self {
        Self { var_smoothing: 1e-9 }
    }
}

pub struct GaussianNBModel<Dev: Device> {
    pub class_log_prior: Tensor<Dev>, // (1, n_class)
    pub theta: Tensor<Dev>,           // 平均值 \mu: (n_class, n_features)
    pub var: Tensor<Dev>,             // 方差 \sigma^2: (n_class, n_features)
}

impl<Dev: Device> PredictFit<Tensor<Dev>> for GaussianNB {
    type Output = Tensor<Dev, Int>;
    type Model = GaussianNBModel<Dev>;

    /// ## Args
    /// - x: (n_samples, n_features): 连续型特征矩阵
    /// - y: (n_samples,)：标签 (0, 1, 2...)
    fn fit(&self, x: &Tensor<Dev>, y: &Tensor<Dev, Int>) -> crate::error::MlResult<Self::Model> {
        let (n_samples, _) = x.dims2()?;
        utils::validate_xy_shapes(x, y, None)?;

        let n_class = y.max_all()?.to_scalar()? + 1;

        let mut sample_counts = Vec::with_capacity(n_class as usize);
        let mut thetas = Vec::with_capacity(n_class as usize);
        let mut vars = Vec::with_capacity(n_class as usize);

        for c in 0..n_class {
            let mask = y.eq(c)?; // (n_samples,)

            // 1. 统计当前类别样本数
            let sample_count = mask.true_count()?;
            sample_counts.push(sample_count as f64);

            // 取出属于这个 class 的 X: (n_c, n_features)
            let cur_class_x = x.i(&mask)?;

            // 2. 计算平均值 \mu (沿样本维度求平均)
            let mu = cur_class_x.mean(0)?; // (n_features,)
            thetas.push(mu.clone());

            // 3. 计算方差 \sigma^2 = E[(X - \mu)^2]
            let diff = cur_class_x.broadcast_sub(&mu.unsqueeze(0)?)?;
            let sq_diff = diff.sqr()?;
            let variance = sq_diff.mean(0)?; // (n_features,)
            vars.push(variance);
        }

        // 4. 计算先验概率 \ln P(Y)
        // Tensor::new 默认 F64，to_dtype 对齐输入 x 的 dtype，避免与 theta/var（x 派生）运算时 DTypeMismatch
        let sample_counts = Tensor::<Dev>::new(sample_counts, x.device())?.to_dtype(x.dtype())?;
        let class_log_prior = (sample_counts / n_samples as f64).ln()?.unsqueeze(0)?; // (1, n_class)

        // 5. 组合参数矩阵并加上平滑项 (Variance Smoothing)
        let theta = Tensor::stack(&thetas, 0)?; // (n_class, n_features)

        let var = Tensor::stack(&vars, 0)?; // (n_class, n_features)
        var.add_(self.var_smoothing)?; // 防止方差为 0

        Ok(GaussianNBModel { class_log_prior, theta, var })
    }
}

impl<Dev: Device> PredictModel for GaussianNBModel<Dev> {
    type Input = Tensor<Dev>;
    type Output = Tensor<Dev, Int>;

    fn predict(&self, x: &Tensor<Dev>) -> crate::error::MlResult<Tensor<Dev, Int>> {
        let log_prob = self.predict_log_proba(x)?;
        let y = log_prob.argmax(1)?;
        Ok(y)
    }
}

impl<Dev: Device> GaussianNBModel<Dev> {
    pub fn predict_log_proba(&self, x: &Tensor<Dev>) -> crate::error::MlResult<Tensor<Dev>> {
        // 实现公式: X.matmul(W1.T) - (X^2).matmul(W2.T) + Intercept

        // W1 = \mu / \sigma^2  (n_class, n_features)
        let w1 = self.theta.div(&self.var)?;

        // W2 = 1.0 / (2 * \sigma^2)  (n_class, n_features)
        let w2 = (self.var.mul_scalar(2.0)?).recip()?;

        // 计算 Intercept 中的 \sum_i ( \ln(2\pi\sigma_i^2) + \mu_i^2 / \sigma_i^2 )
        let two_pi_var_ln = self.var.mul_scalar(2.0 * std::f64::consts::PI)?.ln()?;
        let mu_sq_over_var = self.theta.sqr()?.div(&self.var)?;
        let sum_term = (two_pi_var_ln + mu_sq_over_var).sum_keepdim(1)?; // (n_class, 1)

        // Intercept = \ln P(Y) - 0.5 * sum_term
        // 这里 class_log_prior 是 (1, n_class)，sum_term 是 (n_class, 1)
        // 需将其转置后相加，使其成为 (1, n_class) 用于后续 broadcast
        let intercept = &self.class_log_prior - &(sum_term.transpose_last()? * 0.5); // (1, n_class)

        // 计算预测矩阵部分
        // term1: X @ W1.T -> (n_samples, n_class)
        let term1 = x.matmul(&w1.transpose_last()?)?;

        // term2: X^2 @ W2.T -> (n_samples, n_class)
        let x_sq = x.sqr()?;
        let term2 = x_sq.matmul(&w2.transpose_last()?)?;

        // 最终 log probability: term1 - term2 + Intercept
        let log_prob = (term1 - term2).broadcast_add(&intercept)?; // (n_samples, n_class)

        Ok(log_prob)
    }
}

#[cfg(test)]
mod tests {
    use luma_tensor::{Cpu, Int, Tensor};

    use crate::{
        core::{PredictFit, PredictModel},
        datasets::load_iris,
        metrics::accuracy_score,
        naive_bayes::GaussianNB,
    };

    #[test]
    fn test_gaussian_nb_iris() {
        let device = Cpu::default();
        let iris = load_iris(&device).unwrap();
        let model = GaussianNB::default().fit(&iris.data, &iris.target).unwrap();

        // 参数形状: (n_class, n_features) / (1, n_class)
        assert_eq!(model.theta.dims(), [3, 4]);
        assert_eq!(model.var.dims(), [3, 4]);
        assert_eq!(model.class_log_prior.dims(), [1, 3]);

        let y_pred = model.predict(&iris.data).unwrap();
        let acc = accuracy_score(&iris.target, &y_pred).unwrap();
        assert!(acc > 0.9, "acc = {acc}");
    }

    #[test]
    fn test_gaussian_nb_toy_params_and_log_proba() {
        let device = Cpu::default();
        // 2 类 × 2 特征，每类 2 个样本，特征可分
        let x = Tensor::<Cpu>::new(vec![0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0, 11.0], &device).unwrap().reshape((4, 2)).unwrap();
        let y = Tensor::<Cpu, Int>::new(vec![0i64, 0, 1, 1], &device).unwrap();

        let nb = GaussianNB { var_smoothing: 1e-9 };
        let model = nb.fit(&x, &y).unwrap();

        // 手算参数: \mu = [0.5, 0.5] / [10.5, 10.5]，\sigma^2 = 0.25 + 平滑
        let theta = model.theta.to_vec().unwrap();
        let var = model.var.to_vec().unwrap();
        for c in 0..2 {
            assert!((theta[2 * c] - (0.5 + 10.0 * c as f64)).abs() < 1e-12);
            assert!((theta[2 * c + 1] - (0.5 + 10.0 * c as f64)).abs() < 1e-12);
            assert!((var[2 * c] - (0.25 + 1e-9)).abs() < 1e-15);
            assert!((var[2 * c + 1] - (0.25 + 1e-9)).abs() < 1e-15);
        }

        // 独立重算 \ln P(x|Y) + \ln P(Y)，验证 predict_log_proba 的矩阵展开公式
        let lp = model.predict_log_proba(&x).unwrap().to_vec().unwrap();
        let ln_prior = 0.5f64.ln();
        let e = |xi: f64, mu: f64, var: f64| -0.5 * (2.0 * std::f64::consts::PI * var).ln() - (xi - mu).powi(2) / (2.0 * var);
        // 样本 [0, 0] 属于类别 0 的对数概率
        let expected_0 = e(0.0, theta[0], var[0]) + e(0.0, theta[1], var[1]) + ln_prior;
        assert!((lp[0] - expected_0).abs() < 1e-9, "lp0 = {}, expected = {}", lp[0], expected_0);
        // 样本 [0, 0] 属于类别 1 的对数概率（应该非常小）
        let expected_1 = e(0.0, theta[2], var[2]) + e(0.0, theta[3], var[3]) + ln_prior;
        assert!((lp[1] - expected_1).abs() < 1e-9);

        // 训练集上应完美分类
        let y_pred = model.predict(&x).unwrap();
        assert_eq!(y_pred.to_vec().unwrap(), vec![0, 0, 1, 1]);
    }

    #[test]
    fn test_gaussian_nb_single_sample_per_class() {
        let device = Cpu::default();
        // 每类只有一个样本 => 方差为 0，依赖 var_smoothing 兜底（否则 ln(0) = -inf）
        let x = Tensor::<Cpu>::new(vec![1.0, 100.0], &device).unwrap().reshape((2, 1)).unwrap();
        let y = Tensor::<Cpu, Int>::new(vec![0i64, 1], &device).unwrap();

        let model = GaussianNB::default().fit(&x, &y).unwrap();
        let y_pred = model.predict(&x).unwrap();
        assert_eq!(y_pred.to_vec().unwrap(), vec![0, 1]);
    }

    #[test]
    fn test_gaussian_nb_f32_input() {
        // 默认 F32 输入：验证 fit 内部 Tensor::new + to_dtype 的 dtype 对齐
        let x = Tensor::<Cpu>::from_slice(
            &[0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0, 11.0],
            (4, 2),
            (Cpu::default(), luma_tensor::FloatDType::F32),
        )
        .unwrap();
        let y = Tensor::<Cpu, Int>::new(vec![0i64, 0, 1, 1], &Cpu::default()).unwrap();

        let model = GaussianNB::default().fit(&x, &y).unwrap();
        assert_eq!(model.theta.dtype(), luma_tensor::FloatDType::F32);

        let y_pred = model.predict(&x).unwrap();
        assert_eq!(y_pred.to_vec().unwrap(), vec![0, 0, 1, 1]);
    }
}
