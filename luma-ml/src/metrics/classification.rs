use luma_tensor::{DTypeKind, Device, Float, IndexOp, Int, Tensor};

/// 统一的取数接口：luma 的 `to_vec` 是分 kind 实现的，用本 trait 让 metrics 保持对 kind 泛型。
#[doc(hidden)]
pub trait TensorToVec<T> {
    fn to_vec_local(&self) -> luma_tensor::Result<Vec<T>>;
}

impl<Dev: Device> TensorToVec<f64> for Tensor<Dev, Float> {
    fn to_vec_local(&self) -> luma_tensor::Result<Vec<f64>> {
        self.to_vec()
    }
}

impl<Dev: Device> TensorToVec<i64> for Tensor<Dev, Int> {
    fn to_vec_local(&self) -> luma_tensor::Result<Vec<i64>> {
        self.to_vec()
    }
}

/// ## Args
/// - `y_ture`: (n_samples,)
/// - `y_pred`: (n_samples,)
pub fn confusion_matrix<Dev: Device>(y_true: &Tensor<Dev, Int>, y_pred: &Tensor<Dev, Int>) -> luma_tensor::Result<Tensor<Dev, Int>> {
    let true_count = y_true.dims1()?;
    let pred_count = y_pred.dims1()?;
    if true_count != pred_count {
        luma_tensor::bail!("true count != pred count");
    }
    if true_count == 0 {
        luma_tensor::bail!("empty samples");
    }

    // TODO: 统计分类数量
    let n_class1 = y_true.max_all()?.to_scalar()? as usize + 1;
    let n_class2 = y_pred.max_all()?.to_scalar()? as usize + 1;
    let n_class = n_class1.max(n_class2);
    if n_class <= 1 {
        luma_tensor::bail!("only one class");
    }

    // 收集数据方便统计
    let pred_true_pairs: Vec<(i64, i64)> = y_pred.to_vec()?.into_iter().zip(y_true.to_vec()?).collect();

    // N x N
    let mut matrix = vec![0i64; n_class * n_class];
    for row in 0..n_class {
        for col in 0..n_class {
            // row: 预测值为 row 的样本
            // col: 真实值为 col 的样本
            let count = pred_true_pairs.iter().filter(|(pred, true_)| *pred == row as i64 && *true_ == col as i64).count() as i64;
            matrix[row * n_class + col] = count;
        }
    }

    Ok(Tensor::<Dev, Int>::from_vec_i64(matrix, (n_class, n_class), y_true.device())?)
}

pub fn accuracy_score<Dev, K, S>(y_true: &Tensor<Dev, K>, y_pred: &Tensor<Dev, K>) -> luma_tensor::Result<f64>
where
    Dev: Device,
    K: DTypeKind<Dev>,
    Tensor<Dev, K>: TensorToVec<S>,
    S: PartialEq,
{
    if y_true.dims() != y_pred.dims() {
        luma_tensor::bail!("y_true shape {:?} != y_pred shaoe {:?}", y_true.dims(), y_pred.dims());
    }

    let n_samples = y_true.element_count();
    if n_samples == 0 {
        luma_tensor::bail!("no samples!");
    }

    let n_correct = y_true.to_vec_local()?.into_iter().zip(y_pred.to_vec_local()?).filter(|(t, p)| t == p).count();

    Ok(n_correct as f64 / n_samples as f64)
}

pub fn precision_score<Dev: Device>(y_true: &Tensor<Dev, Int>, y_pred: &Tensor<Dev, Int>) -> luma_tensor::Result<f64> {
    let cm = confusion_matrix(y_true, y_pred)?;
    let (n_class, _) = cm.dims2().expect("confusion matrix must matrix!");

    if n_class == 2 {
        let tp = cm.i((0, 0))?.to_scalar()? as f64;
        let fp = cm.i((0, 1))?.to_scalar()? as f64;
        Ok(tp / (tp + fp))
    } else {
        // 为每个特征计算
        let mut scores = 0.0;
        for class in 0..n_class {
            let tp = cm.i((class, class))?.to_scalar()? as f64;
            let all_samples = cm.i(class)?.sum_all()?.to_scalar()? as f64;
            scores += tp / all_samples;
        }

        Ok(scores / n_class as f64)
    }
}

pub fn recall_score<Dev: Device>(y_true: &Tensor<Dev, Int>, y_pred: &Tensor<Dev, Int>) -> luma_tensor::Result<f64> {
    let cm = confusion_matrix(y_true, y_pred)?;
    let (n_class, _) = cm.dims2().expect("confusion matrix must matrix!");

    if n_class == 2 {
        let tp = cm.i((0, 0))?.to_scalar()? as f64;
        let fn_ = cm.i((1, 0))?.to_scalar()? as f64;
        Ok(tp / (tp + fn_))
    } else {
        let mut scores = 0.0;
        for class in 0..n_class {
            let tp = cm.i((class, class))?.to_scalar()? as f64;
            let all_samples = cm.i((.., class))?.sum_all()?.to_scalar()? as f64;
            scores += tp / all_samples;
        }

        Ok(scores / n_class as f64)
    }
}

pub fn f1_score<Dev: Device>(y_true: &Tensor<Dev, Int>, y_pred: &Tensor<Dev, Int>) -> luma_tensor::Result<f64> {
    let precision = precision_score(y_true, y_pred)?;
    let recall = recall_score(y_true, y_pred)?;
    Ok(2.0 * (precision * recall) / (precision + recall))
}

#[cfg(test)]
mod tests {
    use luma_tensor::{Cpu, Int, Tensor};

    use super::precision_score;
    use crate::metrics::{confusion_matrix, f1_score, recall_score};

    #[test]
    fn test_confusion_binary_matrix() {
        let device = Cpu;
        let y_true = Tensor::<Cpu, Int>::new(vec![1i64, 0, 1, 1, 0, 0, 1, 0, 0, 1], &device).unwrap();
        let y_pred = Tensor::<Cpu, Int>::new(vec![1i64, 0, 0, 1, 0, 0, 1, 1, 0, 1], &device).unwrap();
        let cm = confusion_matrix(&y_true, &y_pred).unwrap();
        println!("{}", cm);
    }

    #[test]
    fn test_binary_scores() {
        let device = Cpu;
        let y_true = Tensor::<Cpu, Int>::new(vec![1i64, 0, 1, 1, 0, 0, 1, 0, 0, 1], &device).unwrap();
        let y_pred = Tensor::<Cpu, Int>::new(vec![1i64, 0, 0, 1, 0, 0, 1, 1, 0, 1], &device).unwrap();
        let pr = precision_score(&y_true, &y_pred).unwrap();
        assert_eq!(pr, 0.8);
        let recall = recall_score(&y_true, &y_pred).unwrap();
        assert_eq!(recall, 0.8);
        let f1 = f1_score(&y_true, &y_pred).unwrap();
        assert_eq!(f1, 0.8000000000000002);
    }
}
