use luma_tensor::{Device, Tensor};

pub fn mean_squar_error<Dev: Device>(y_true: &Tensor<Dev>, y_pred: &Tensor<Dev>) -> luma_tensor::Result<f64> {
    (y_pred - y_true).pow(2.0)?.sum_all()?.to_scalar()
}

pub fn mean_squared_error<Dev: Device>(y_true: &Tensor<Dev>, y_pred: &Tensor<Dev>) -> luma_tensor::Result<f64> {
    Ok((y_true - y_pred).sqr()?.sum_all()?.to_scalar()?.sqrt())
}

pub fn mean_absolute_error<Dev: Device>(y_true: &Tensor<Dev>, y_pred: &Tensor<Dev>) -> luma_tensor::Result<f64> {
    (y_true - y_pred).abs()?.sum_all()?.to_scalar()
}

pub fn r2_score<Dev: Device>(y_true: &Tensor<Dev>, y_pred: &Tensor<Dev>) -> luma_tensor::Result<f64> {
    let y_mean = y_true.mean_all()?.to_scalar()?;

    Ok(1.0 - ((y_true - y_pred).sqr()?.sum_all()?.to_scalar()?) / ((y_true - y_mean).sqr()?.sum_all()?.to_scalar()?))
}

#[cfg(test)]
mod tests {
    use luma_tensor::{Cpu, Tensor};

    use super::{mean_absolute_error, mean_squar_error, mean_squared_error, r2_score};

    #[test]
    fn test_regression_metrics_hand_computed() {
        let y_true = Tensor::<Cpu>::new(vec![1.0, 2.0, 3.0, 4.0], &Cpu).unwrap();
        let y_pred = Tensor::<Cpu>::new(vec![1.5, 2.0, 2.5, 4.5], &Cpu).unwrap();

        // 平方误差之和 = 0.25 + 0 + 0.25 + 0.25
        assert!((mean_squar_error(&y_true, &y_pred).unwrap() - 0.75).abs() < 1e-12);
        // 注意：mean_squared_error 实际返回的是 RMSE（先求和再开方）
        assert!((mean_squared_error(&y_true, &y_pred).unwrap() - 0.75f64.sqrt()).abs() < 1e-12);
        // 绝对误差之和 = 0.5 + 0 + 0.5 + 0.5
        assert!((mean_absolute_error(&y_true, &y_pred).unwrap() - 1.5).abs() < 1e-12);
        // r2 = 1 - SSE/SST, SST = Σ(y_true - y_mean)^2（sklearn 标准公式）
        // SSE = 0.75, y_mean = 2.5, SST = 2.25 + 0.25 + 0.25 + 2.25 = 5.0
        let expected_r2 = 1.0 - 0.75 / 5.0;
        assert!((r2_score(&y_true, &y_pred).unwrap() - expected_r2).abs() < 1e-12);
    }

    #[test]
    fn test_regression_metrics_extremes() {
        // 完美预测: SSE = 0（分母 SST 非 0，r2 = 1.0）
        let y = Tensor::<Cpu>::new(vec![1.0, 2.0, 3.0, 4.0], &Cpu).unwrap();
        assert!((mean_squar_error(&y, &y).unwrap()).abs() < 1e-12);
        assert!((mean_squared_error(&y, &y).unwrap()).abs() < 1e-12);
        assert!((mean_absolute_error(&y, &y).unwrap()).abs() < 1e-12);
        assert!((r2_score(&y, &y).unwrap() - 1.0).abs() < 1e-12);

        // 近似完美预测: r2 -> 1
        let y_near = Tensor::<Cpu>::new(vec![1.001, 2.001, 3.001, 4.001], &Cpu).unwrap();
        let r2 = r2_score(&y, &y_near).unwrap();
        assert!(r2 > 0.999, "r2 = {r2}");

        // y 恒定时分子分母同时为 0 -> NaN（r2 公式的已知边界，见迁移记录）
        let y_const = Tensor::<Cpu>::new(vec![2.0, 2.0, 2.0], &Cpu).unwrap();
        assert!(r2_score(&y_const, &y_const).unwrap().is_nan());
    }
}
