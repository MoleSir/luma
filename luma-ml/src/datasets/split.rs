use std::collections::HashSet;
use luma_tensor::{ops::BaseOpsDTypeKind, Bool, Device, IndexOp, Tensor};

/// X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
pub fn train_test_split<Dev, K1, K2>(
    x: &Tensor<Dev, K1>, y: &Tensor<Dev, K2>, test_ratio: f64,
) -> luma_tensor::Result<(Tensor<Dev, K1>, Tensor<Dev, K1>, Tensor<Dev, K2>, Tensor<Dev, K2>)> 
where
    Dev: Device,
    K1: BaseOpsDTypeKind<Dev>,
    K2: BaseOpsDTypeKind<Dev>,
{
    if test_ratio < 0.0 || test_ratio >= 1.0 {
        luma_tensor::bail!("invalid test_ratio {test_ratio}");
    }

    let n_samples = x.dims()[0];
    let n_samples_y = y.dims()[0];
    if n_samples != n_samples_y {
        luma_tensor::bail!("x n_samples != y n_samples");
    }

    let test_count = (n_samples as f64 * test_ratio) as usize;
    let mut test_indexs = HashSet::new();
    let mut test_mask = vec![false; n_samples];
    while test_indexs.len() < test_count {
        let index = rand::random_range(0..n_samples);
        if test_indexs.insert(index) {
            test_mask[index] = true;
        }
    }
    let test_mask = Tensor::<Dev, Bool>::new(test_mask, x.device())?;
    let train_mask = test_mask.not()?;

    let x_train = x.i(&train_mask)?;
    let y_train = y.i(&train_mask)?;
    
    let x_test = x.i(&test_mask)?;
    let y_test = y.i(&test_mask)?;

    Ok((x_train, x_test, y_train, y_test))
}

#[cfg(test)]
mod tests {
    use luma_tensor::{Cpu, Tensor};

    use super::train_test_split;

    #[test]
    fn test_split_shapes() {
        let x = Tensor::<Cpu>::rand(0.0, 1.0, (100, 4), &Cpu).unwrap();
        let y = Tensor::<Cpu>::rand(0.0, 1.0, (100,), &Cpu).unwrap();

        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.3).unwrap();
        assert_eq!(x_train.dims(), [70, 4]);
        assert_eq!(x_test.dims(), [30, 4]);
        assert_eq!(y_train.dims(), [70]);
        assert_eq!(y_test.dims(), [30]);
    }

    #[test]
    fn test_split_train_and_test_disjoint() {
        let x = Tensor::<Cpu>::rand(0.0, 1.0, (100, 1), &Cpu).unwrap();
        let y = Tensor::<Cpu>::rand(0.0, 1.0, (100,), &Cpu).unwrap();

        let (x_train, x_test, _, _) = train_test_split(&x, &y, 0.2).unwrap();
        // train 与 test 的样本不应重合（按第一列的值精确匹配）
        let train_rows: Vec<f64> = x_train.to_vec().unwrap();
        let test_rows: Vec<f64> = x_test.to_vec().unwrap();
        for v in &test_rows {
            assert!(!train_rows.contains(v), "test sample {v} also in train!");
        }
    }

    #[test]
    fn test_split_invalid_args() {
        let x = Tensor::<Cpu>::rand(0.0, 1.0, (10, 2), &Cpu).unwrap();
        let y = Tensor::<Cpu>::rand(0.0, 1.0, (10,), &Cpu).unwrap();
        let y9 = Tensor::<Cpu>::rand(0.0, 1.0, (9,), &Cpu).unwrap();

        // 0.0 是允许的边界（测试集为空）
        let (x_train, x_test, _, _) = train_test_split(&x, &y, 0.0).unwrap();
        assert_eq!(x_train.dims(), [10, 2]);
        assert_eq!(x_test.dims(), [0, 2]);

        assert!(train_test_split(&x, &y, 1.0).is_err(), "ratio 1.0 should be rejected");
        assert!(train_test_split(&x, &y, -0.1).is_err(), "negative ratio should be rejected");
        assert!(train_test_split(&x, &y9, 0.3).is_err(), "n_samples mismatch should be rejected");
    }
}