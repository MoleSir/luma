use std::collections::HashMap;

use crate::{
    core::{PredictFit, PredictModel},
    error::MlResult,
    tree::{DecisionTreeClassifier, DecisionTreeClassifierModel},
};
use luma_tensor::{Device, IndexOp, Int, Tensor};
use rand::Rng;

pub struct RandomForestClassifier {
    pub n_estimators: usize,
    pub max_depth: usize,
}

pub struct RandomForestClassifierModel<Dev: Device> {
    pub trees: Vec<DecisionTreeClassifierModel<Dev>>,
}

impl<Dev: Device> PredictFit<Tensor<Dev>> for RandomForestClassifier {
    type Output = Tensor<Dev, Int>;
    type Model = RandomForestClassifierModel<Dev>;

    fn fit(&self, x: &Tensor<Dev>, y: &Tensor<Dev, Int>) -> MlResult<Self::Model> {
        let mut trees = Vec::with_capacity(self.n_estimators);
        let (n_samples, _) = x.dims2()?;

        let tree_trainer = DecisionTreeClassifier::new(self.max_depth);
        for _ in 0..self.n_estimators {
            let (x_boot, y_boot) = Self::bootstrap_sample(x, y, n_samples)?;
            let tree = tree_trainer.fit(&x_boot, &y_boot)?;
            trees.push(tree);
        }

        Ok(RandomForestClassifierModel { trees })
    }
}

impl<Dev: Device> PredictModel for RandomForestClassifierModel<Dev> {
    type Input = Tensor<Dev>;
    type Output = Tensor<Dev, Int>;

    fn predict(&self, x: &Tensor<Dev>) -> MlResult<Tensor<Dev, Int>> {
        let (n_samples, _) = x.dims2()?;
        let mut all_predictions = Vec::with_capacity(self.trees.len());

        for tree in &self.trees {
            let preds = tree.predict(x)?; // (n_samples)
            all_predictions.push(preds.to_vec()?);
        }

        let mut final_preds = Vec::with_capacity(n_samples);
        // for each samples, votes a best
        for i in 0..n_samples {
            let mut counter = HashMap::new();
            for tree_preds in &all_predictions {
                let label = tree_preds[i];
                *counter.entry(label).or_insert(0) += 1;
            }

            let majority_label = *counter.iter().max_by_key(|entry| entry.1).unwrap().0;
            final_preds.push(majority_label);
        }

        Ok(Tensor::<Dev, Int>::new(final_preds, x.device())?)
    }
}

impl RandomForestClassifier {
    pub fn new(n_estimators: usize, max_depth: usize) -> Self {
        Self { n_estimators, max_depth }
    }

    fn bootstrap_sample<Dev: Device>(x: &Tensor<Dev>, y: &Tensor<Dev, Int>, n_samples: usize) -> MlResult<(Tensor<Dev>, Tensor<Dev, Int>)> {
        let mut x_boots = Vec::with_capacity(n_samples);
        let mut y_boot = Vec::with_capacity(n_samples);

        let mut rng = rand::rng();
        for _ in 0..n_samples {
            let idx = rng.random_range(0..n_samples);
            let x_row = x.i(idx)?;
            let y_row = y.i(idx)?;
            x_boots.push(x_row);
            y_boot.push(y_row.to_scalar()?);
        }

        let x_boot = Tensor::stack(&x_boots, 0)?;
        let y_boot = Tensor::<Dev, Int>::new(y_boot, x.device())?;

        Ok((x_boot, y_boot))
    }
}

#[cfg(test)]
mod tests {
    use luma_tensor::Cpu;

    use crate::{
        core::{PredictFit, PredictModel},
        datasets::{load_iris, train_test_split},
        ensemble::RandomForestClassifier,
        metrics::accuracy_score,
    };

    #[test]
    fn test_random_forest_iris() {
        let iris = load_iris(&Cpu::default()).unwrap();
        let (x_train, x_test, y_train, y_test) = train_test_split(&iris.data, &iris.target, 0.3).unwrap();

        let rf = RandomForestClassifier::new(10, 10);
        let model = rf.fit(&x_train, &y_train).unwrap();
        assert_eq!(model.trees.len(), 10);

        let y_pred = model.predict(&x_test).unwrap();
        let acc = accuracy_score(&y_test, &y_pred).unwrap();
        // 随机切分偶尔落在困难测试集上（实测最低 0.889），0.85 防 flake；模型故障会是 0.33~0.5
        assert!(acc > 0.85, "acc = {acc}");
    }
}
