use std::collections::HashMap;

use luma_tensor::{Device, IndexOp, Int, IntDType, Tensor, ops::BaseOpsDTypeKind};

use crate::{
    core::{PredictFit, PredictModel},
    error::{MlError, MlResult},
    utils,
};

// =========================================================================================== //
//              Knn Regression
// =========================================================================================== //

pub struct KnnRegression {
    pub n_neighbors: usize,
}

pub struct KnnRegressionModel<Dev: Device> {
    pub n_neighbors: usize,
    pub n_features: usize,
    pub x_train: Tensor<Dev>,
    pub y_train: Tensor<Dev>,
}

impl KnnRegression {
    pub fn new(n_neighbors: usize) -> Self {
        Self { n_neighbors }
    }
}

impl<Dev: Device> PredictFit<Tensor<Dev>> for KnnRegression {
    type Output = Tensor<Dev>;
    type Model = KnnRegressionModel<Dev>;

    /// ## Args
    /// - `x_train`: (n_samples, n_features)
    /// - `y_train`: (n_samples,)
    fn fit(&self, x: &Tensor<Dev>, y: &Tensor<Dev>) -> MlResult<Self::Model> {
        let (n_samples, n_features) = utils::validate_xy_shapes(x, y, None)?;
        if n_samples < self.n_neighbors {
            thiserrorctx::bail!(MlError::Knn(format!("not enough samples! < k {}", self.n_neighbors)));
        }

        Ok(KnnRegressionModel { n_neighbors: self.n_neighbors, n_features, x_train: x.clone(), y_train: y.clone() })
    }
}

impl<Dev: Device> PredictModel for KnnRegressionModel<Dev> {
    type Input = Tensor<Dev>;
    type Output = Tensor<Dev>;

    /// ## Args
    /// - `x`: (n_test_samples, n_features)
    ///
    /// ## Return
    /// - `prediction`: (n_test_samples,)
    fn predict(&self, x: &Tensor<Dev>) -> MlResult<Tensor<Dev>> {
        let neighbors = find_closed_n_neighbors(&self.x_train, &self.y_train, x, self.n_neighbors)?;

        let prediction = neighbors.mean(1)?;

        Ok(prediction)
    }
}

// =========================================================================================== //
//              Knn Classifier
// =========================================================================================== //

pub struct KnnClassifier {
    pub n_neighbors: usize,
}

pub struct KnnClassifierModel<Dev: Device> {
    pub n_neighbors: usize,
    pub n_features: usize,
    pub x_train: Tensor<Dev>,
    pub y_train: Tensor<Dev, Int>,
}

impl KnnClassifier {
    pub fn new(n_neighbors: usize) -> Self {
        Self { n_neighbors }
    }
}

impl<Dev: Device> PredictFit<Tensor<Dev>> for KnnClassifier {
    type Output = Tensor<Dev, Int>;
    type Model = KnnClassifierModel<Dev>;

    /// ## Args
    /// - `x_train`: (n_samples, n_features)
    /// - `y_train`: (n_samples,)
    fn fit(&self, x: &Tensor<Dev>, y: &Tensor<Dev, Int>) -> MlResult<Self::Model> {
        let (n_samples, n_features) = utils::validate_xy_shapes(x, y, None)?;
        if n_samples < self.n_neighbors {
            thiserrorctx::bail!(MlError::Knn(format!("not enough samples! < k {}", self.n_neighbors)));
        }

        Ok(KnnClassifierModel { n_neighbors: self.n_neighbors, n_features, x_train: x.clone(), y_train: y.clone() })
    }
}

impl<Dev: Device> PredictModel for KnnClassifierModel<Dev> {
    type Input = Tensor<Dev>;
    type Output = Tensor<Dev, Int>;

    /// ## Args
    /// - `x`: (n_test_samples, n_features)
    ///
    /// ## Return
    /// - `prediction`: (n_test_samples,)
    fn predict(&self, x: &Tensor<Dev>) -> MlResult<Tensor<Dev, Int>> {
        let (n_test_samples, n_features) = x.dims2()?;
        if self.n_features != n_features {
            luma_tensor::bail!("expect n_fetures {}, not got {}", self.n_features, n_features);
        }

        let neighbor_labels = find_closed_n_neighbors(&self.x_train, &self.y_train, x, self.n_neighbors)?;

        let mut labels = Vec::with_capacity(n_test_samples);
        for n in 0..n_test_samples {
            let sample_labels = neighbor_labels.i(n)?.to_vec()?;
            let mut counters = HashMap::new();

            for label in sample_labels {
                *counters.entry(label).or_insert(0) += 1;
            }

            let majority_label = counters.into_iter().max_by_key(|&(_, count)| count).map(|(label, _)| label).unwrap();

            labels.push(majority_label);
        }

        Ok(Tensor::<Dev, Int>::new(labels, x.device())?)
    }
}

fn find_closed_n_neighbors<Dev, K2>(
    x_train: &Tensor<Dev>,
    y_train: &Tensor<Dev, K2>,
    x_test: &Tensor<Dev>,
    n_neighbors: usize,
) -> luma_tensor::Result<Tensor<Dev, K2>>
where
    Dev: Device,
    K2: BaseOpsDTypeKind<Dev>,
{
    let (n_test_samples, _) = x_test.dims2()?;
    let (n_train, _) = x_train.dims2()?;

    // (n_test_samples, 1, n_features) - (1, n_samples, n_features) => (n_test_samples, n_samples, n_features)
    let delta_features = x_test.unsqueeze(1)?.broadcast_sub(&x_train.unsqueeze(0)?)?;
    // (n_test_samples, n_samples, n_features) => (n_test_samples, n_samples)
    let neg_distances = delta_features.sqr()?.sum(2)?.neg()?;

    // topk: 重复 argmax 并屏蔽已选位置, 等价于 neg_distances.topk(n_neighbors, 1)
    // (n_test_samples, n_samples) => (n_test_samples, k)
    let arange = Tensor::<Dev, Int>::arange(0, n_train as i64, 1, (x_test.device(), IntDType::U32))?;
    let mut remaining = neg_distances;
    let mut idxs = Vec::with_capacity(n_neighbors);
    for _ in 0..n_neighbors {
        let idx = remaining.argmax(1)?; // (n_test_samples,)
        let mask = arange.broadcast_eq(&idx.unsqueeze(1)?)?; // (n_test_samples, n_samples)
        remaining = mask.pick_true(f64::NEG_INFINITY, &remaining)?; // 屏蔽已选位置
        idxs.push(idx.unsqueeze(1)?); // (n_test_samples, 1)
    }
    let idx = Tensor::cat(&idxs, 1)?;

    // (n_test_samples * k)
    let flat_idx = idx.flatten_all()?;
    let flat_neighbors = y_train.index_select(&flat_idx, 0)?;
    let neighbors = flat_neighbors.reshape((n_test_samples, n_neighbors))?;

    Ok(neighbors)
}

#[cfg(test)]
mod tests {
    use luma_tensor::{Cpu, Tensor};

    use crate::{
        core::{PredictFit, PredictModel},
        datasets::{load_iris, train_test_split},
        metrics::accuracy_score,
        neighbor::{KnnClassifier, KnnRegression},
    };

    #[test]
    fn test_knn_regression() {
        const N_SAMPLES: usize = 100;
        let device = Cpu::default();
        let x = Tensor::<Cpu>::rand(0.0, 10.0, (N_SAMPLES,), &device).unwrap();
        let y = x.sin().unwrap() + (0.1 * Tensor::<Cpu>::randn(0.0, 1.0, (N_SAMPLES,), &device).unwrap());
        let x = x.unsqueeze(1).unwrap();

        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.3).unwrap();

        let trainer = KnnRegression::new(5);
        let model = trainer.fit(&x_train, &y_train).unwrap();

        let y_pred = model.predict(&x_test).unwrap();

        // k 近邻对平滑的 sin 曲线拟合很好（噪声 0.1），MAE 应远小于 0.5
        let mae = (y_pred.clone() - &y_test).abs().unwrap().mean_all().unwrap().to_scalar().unwrap();
        assert!(mae < 0.5, "mae = {mae}");

        for (pred, real) in y_pred.to_vec().unwrap().into_iter().zip(y_test.to_vec().unwrap()) {
            println!("pred: {pred:.2} vs real: {real:.2}")
        }
    }

    #[test]
    fn test_knn_classifier_iris() {
        let iris = load_iris(&Cpu::default()).unwrap();
        let (x_train, x_test, y_train, y_test) = train_test_split(&iris.data, &iris.target, 0.3).unwrap();

        let model = KnnClassifier::new(5).fit(&x_train, &y_train).unwrap();
        let y_pred = model.predict(&x_test).unwrap();

        let acc = accuracy_score(&y_test, &y_pred).unwrap();
        // 随机切分偶尔落在困难测试集上，0.85 防 flake；模型故障会是 0.33~0.5
        assert!(acc > 0.85, "acc = {acc}");
    }
}
