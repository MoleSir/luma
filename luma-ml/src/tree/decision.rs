use std::collections::HashMap;

use luma_tensor::{Device, DTypeKind, Float, IndexOp, Int, Tensor};

use crate::{error::MlResult, core::{PredictFit, PredictModel}};

pub enum DecisionTree<Dev: Device, V: DTypeKind<Dev>> {
    Leaf(V::Scalar),
    Node {
        feature_id: usize,
        threshold: f64,
        /// features[feature_id] <= threshold
        left: Box<DecisionTree<Dev, V>>,
        /// features[feature_id] > threshold
        right: Box<DecisionTree<Dev, V>>,
    }
}

impl<Dev: Device, V: DTypeKind<Dev>> DecisionTree<Dev, V> {
    pub fn depth(&self) -> usize {
        self.get_depth(1)
    }

    fn get_depth(&self, d: usize) -> usize{
        match self {
            Self::Leaf(_) => d,
            Self::Node { feature_id: _, threshold: _, left, right } => {
                let left_d = left.get_depth(d+1);
                let right_d = right.get_depth(d+1);
                left_d.max(right_d)
            }
        }
    }

    fn predict_single(&self, x: &Tensor<Dev>) -> MlResult<V::Scalar> {
        match self {
            Self::Leaf(v) => Ok(v.clone()),
            Self::Node { feature_id, threshold, left, right } => {
                let value = x.i(*feature_id)?.to_scalar()?;
                if value <= *threshold {
                    left.predict_single(x)
                } else {
                    right.predict_single(x)
                }
            }
        }
    }
}

impl<Dev: Device> DecisionTree<Dev, Float> {
    /// ## Args
    /// - `x`: (n_samples, n_features)
    ///
    /// ## Return
    /// - `prediction`: (n_samples,)
    pub fn predict(&self, x: &Tensor<Dev>) -> MlResult<Tensor<Dev, Float>> {
        let mut results = vec![];
        let (n_samples, _) = x.dims2()?;
        for n in 0..n_samples {
            results.push(self.predict_single(&x.i(n)?)?);
        }
        // Tensor::new 默认 F64，用 to_dtype 对齐输入 x 的 dtype（如默认 F32），
        // 避免下游与输入派生的张量做 in-place 运算时 DTypeMismatch
        Ok(Tensor::<Dev, Float>::new(results, x.device())?.to_dtype(x.dtype())?)
    }
}

impl<Dev: Device> DecisionTree<Dev, Int> {
    /// ## Args
    /// - `x`: (n_samples, n_features)
    ///
    /// ## Return
    /// - `prediction`: (n_samples,)
    pub fn predict(&self, x: &Tensor<Dev>) -> MlResult<Tensor<Dev, Int>> {
        let mut results = vec![];
        let (n_samples, _) = x.dims2()?;
        for n in 0..n_samples {
            results.push(self.predict_single(&x.i(n)?)?);
        }
        Ok(Tensor::<Dev, Int>::new(results, x.device())?)
    }
}

// ==================================================================================== //
//                      DecisionTreeClassifierModel
// ==================================================================================== //

pub struct DecisionTreeClassifier {
    pub max_depth: usize,
}

pub struct DecisionTreeClassifierModel<Dev: Device> {
    pub root: Box<DecisionTree<Dev, Int>>,
}

impl DecisionTreeClassifier {
    pub fn new(max_depth: usize) -> Self {
        Self { max_depth }
    }
}

impl<Dev: Device> PredictFit<Tensor<Dev>> for DecisionTreeClassifier {
    type Output = Tensor<Dev, Int>;
    type Model = DecisionTreeClassifierModel<Dev>;

    /// ## Args
    /// - `x`: (n_samples, n_features)
    /// - `y`: (n_samples)
    fn fit(&self, x: &Tensor<Dev>, y: &Tensor<Dev, Int>) -> MlResult<Self::Model> {
        let (n_samples, _) = x.dims2()?;
        let n_samples_y = y.dims1()?;
        if n_samples != n_samples_y {
            luma_tensor::bail!("x samples {} != y samples {}", n_samples, n_samples_y);
        }

        let n_class = y.max(0)?.to_scalar()? as usize + 1;
        let root = self.build_tree(0, x, y, n_class)?;

        Ok(DecisionTreeClassifierModel { root})
    }
}

impl<Dev: Device> PredictModel for DecisionTreeClassifierModel<Dev> {
    type Input = Tensor<Dev>;
    type Output = Tensor<Dev, Int>;

    fn predict(&self, x: &Tensor<Dev>) -> MlResult<Tensor<Dev, Int>> {
        self.root.predict(x)
    }
}

impl DecisionTreeClassifier {
    /// - `x`: (n_samples, n_features)
    /// - `y`: (n_samples)
    fn build_tree<Dev: Device>(&self, depth: usize, x: &Tensor<Dev>, y: &Tensor<Dev, Int>, n_class: usize) -> luma_tensor::Result<Box<DecisionTree<Dev, Int>>> {
        let (n_samples, n_features) = x.dims2()?;
        let mut counter = HashMap::new();
        for label in y.to_vec()? {
            *counter.entry(label).or_insert(0) += 1;
        }

        let majority_label = *counter.iter().max_by_key(|e| e.1).unwrap().0;
        /*
            1. is max depth?
            2. only one label(pure)
            3. no enough samples
        */
        if depth >= self.max_depth || counter.len() <= 1 || n_samples < 2 {
            return Ok(Box::new(DecisionTree::Leaf(majority_label)));
        }

        // find best split
        let mut best_record = (f64::MAX, 0, 0.0);
        for feature_id in 0..n_features {
            let features = x.i((.., feature_id))?;
            let best_cur_feat = self.find_best_split_for_feature(&features, y, n_class)?;
            if best_cur_feat.0 < best_record.0 {
                best_record.0 = best_cur_feat.0;
                best_record.1 = feature_id;
                best_record.2 = best_cur_feat.1;
            }
        }
        if best_record.0 == f64::MAX {
            return Ok(Box::new(DecisionTree::Leaf(majority_label)));
        }

        // split it !
        let (left_mask, right_mask) = Self::split_mask(x, best_record.1, best_record.2)?;
        let left_samples = left_mask.true_count()?;
        let right_samples = right_mask.true_count()?;
        if left_samples == 0 || right_samples == 0 {
            return Ok(Box::new(DecisionTree::Leaf(majority_label)));
        }

        let left_node = self.build_tree(depth + 1, &x.i(&left_mask)?, &y.i(&left_mask)?, n_class)?;
        let right_node = self.build_tree(depth + 1, &x.i(&right_mask)?, &y.i(&right_mask)?, n_class)?;

        Ok(Box::new(DecisionTree::Node {
            feature_id: best_record.1,
            threshold: best_record.2,
            left: left_node,
            right: right_node
        }))
    }

    fn split_mask<Dev: Device>(x: &Tensor<Dev>, feature_id: usize, threshold: f64) -> luma_tensor::Result<(Tensor<Dev, luma_tensor::Bool>, Tensor<Dev, luma_tensor::Bool>)> {
        let left_mask = x.i((.., feature_id))?.le(threshold)?;
        let right_mask = left_mask.not()?;

        Ok((left_mask, right_mask))
    }

    fn find_best_split_for_feature<Dev: Device>(
        &self,
        x_feature: &Tensor<Dev>,
        y_labels: &Tensor<Dev, Int>,
        n_class: usize
    ) -> luma_tensor::Result<(f64, f64)> {
        let n_samples = x_feature.dims1()?;
        if n_samples < 2 { return Ok((f64::MAX, 0.0)); }

        // 1. 将特征和标签配对并排序
        let mut samples: Vec<(f64, i64)> = x_feature.to_vec()?.into_iter().zip(y_labels.to_vec()?).collect();
        samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // 2. 初始化统计信息
        // 初始状态：所有样本都在右侧
        let mut left_counts = vec![0usize; n_class];
        let mut right_counts = vec![0usize; n_class];
        for &(_, label) in &samples {
            right_counts[label as usize] += 1;
        }

        let mut best_gini = f64::MAX;
        let mut best_threshold = 0.0;
        let mut n_left = 0;
        let mut n_right = n_samples;

        // 3. 遍历所有可能的切分点
        for i in 0..(n_samples - 1) {
            let (val_curr, label_curr) = samples[i];
            let (val_next, _) = samples[i + 1];

            // 更新计数器：将当前样本从右边移到左边
            left_counts[label_curr as usize] += 1;
            right_counts[label_curr as usize] -= 1;
            n_left += 1;
            n_right -= 1;

            // 如果两个相邻特征值相等，不能在此处切分，否则无法区分左右
            if val_curr == val_next {
                continue;
            }

            // 计算当前切分点的 Gini
            // 阈值取两数中间
            let threshold = (val_curr + val_next) / 2.0;

            let gini_left = Self::calculate_gini(&left_counts, n_left);
            let gini_right = Self::calculate_gini(&right_counts, n_right);

            let weighted_gini = (n_left as f64 / n_samples as f64) * gini_left
                              + (n_right as f64 / n_samples as f64) * gini_right;

            if weighted_gini < best_gini {
                best_gini = weighted_gini;
                best_threshold = threshold;
            }
        }

        Ok((best_gini, best_threshold))
    }

    fn calculate_gini(counts: &[usize], total: usize) -> f64 {
        if total == 0 { return 0.0; }
        let mut sum_sq = 0.0;
        let total_f = total as f64;
        for &count in counts {
            if count == 0 { continue; }
            let p = count as f64 / total_f;
            sum_sq += p * p;
        }
        1.0 - sum_sq
    }
}

impl<Dev: Device> DecisionTreeClassifierModel<Dev> {
    pub fn depth(&self) -> usize {
        self.root.depth()
    }
}

// ==================================================================================== //
//                      DecisionTreeRegressorModel
// ==================================================================================== //

pub struct DecisionTreeRegressor {
    pub max_depth: usize,
}

pub struct DecisionTreeRegressorModel<Dev: Device> {
    pub root: Box<DecisionTree<Dev, Float>>,
}

impl<Dev: Device> PredictFit<Tensor<Dev>> for DecisionTreeRegressor {
    type Output = Tensor<Dev>;
    type Model = DecisionTreeRegressorModel<Dev>;

    /// ## Args
    /// - `x`: (n_samples, n_features)
    /// - `y`: (n_samples) 注意：回归树的 y 现在是连续的浮点数 Tensor
    fn fit(&self, x: &Tensor<Dev>, y: &Tensor<Dev>) -> MlResult<Self::Model> {
        let (n_samples, _) = x.dims2()?;
        let n_samples_y = y.dims1()?;
        if n_samples != n_samples_y {
            luma_tensor::bail!("x samples {} != y samples {}", n_samples, n_samples_y);
        }

        let root = self.build_tree(0, x, y)?;

        Ok(DecisionTreeRegressorModel { root })
    }
}

impl<Dev: Device> PredictModel for DecisionTreeRegressorModel<Dev> {
    type Input = Tensor<Dev>;
    type Output = Tensor<Dev>;
    fn predict(&self, x: &Tensor<Dev>) -> MlResult<Tensor<Dev>> {
        self.root.predict(x)
    }
}

impl DecisionTreeRegressor {
    pub fn new(max_depth: usize) -> Self {
        Self { max_depth }
    }

    /// 构建回归树
    fn build_tree<Dev: Device>(&self, depth: usize, x: &Tensor<Dev>, y: &Tensor<Dev>) -> luma_tensor::Result<Box<DecisionTree<Dev, Float>>> {
        let (n_samples, n_features) = x.dims2()?;

        // 1. 计算当前节点的平均值
        let mut sum_y = 0.0;
        for val in y.to_vec()? {
            sum_y += val;
        }
        let mean_y = sum_y / n_samples as f64;

        /*
            停止条件：
            1. 达到最大深度
            2. 样本数太少不足以切分
        */
        if depth >= self.max_depth || n_samples < 2 {
            return Ok(Box::new(DecisionTree::Leaf(mean_y)));
        }

        // 2. 寻找最佳切分点
        let mut best_record: Option<(f64, usize, f64)> = None;

        for feature_id in 0..n_features {
            let features = x.i((.., feature_id))?;

            if let Some((sse, threshold)) = self.find_best_split_for_feature(&features, y)? {
                match best_record {
                    None => {
                        best_record = Some((sse, feature_id, threshold));
                    },
                    Some((min_sse, _, _)) if sse < min_sse => {
                        best_record = Some((sse, feature_id, threshold));
                    },
                    _ => {}
                }
            }
        }

        // 如果找不到可以降低误差的切分点
        let (_best_sse, best_feature_id, best_threshold) = match best_record {
            Some(record) => record,
            None => return Ok(Box::new(DecisionTree::Leaf(mean_y))),
        };

        // 3. 切分数据
        let (left_mask, right_mask) = Self::split_mask(x, best_feature_id, best_threshold)?;
        let left_samples = left_mask.true_count()?;
        let right_samples = right_mask.true_count()?;

        // 如果切分后有一边为空，说明无法有效切分，返回叶子节点
        if left_samples == 0 || right_samples == 0 {
            return Ok(Box::new(DecisionTree::Leaf(mean_y)));
        }

        // 4. 递归构建左右子树
        let left_node = self.build_tree(depth + 1, &x.i(&left_mask)?, &y.i(&left_mask)?)?;
        let right_node = self.build_tree(depth + 1, &x.i(&right_mask)?, &y.i(&right_mask)?)?;

        Ok(Box::new(DecisionTree::Node {
            feature_id: best_feature_id,
            threshold: best_threshold,
            left: left_node,
            right: right_node
        }))
    }

    fn split_mask<Dev: Device>(x: &Tensor<Dev>, feature_id: usize, threshold: f64) -> luma_tensor::Result<(Tensor<Dev, luma_tensor::Bool>, Tensor<Dev, luma_tensor::Bool>)> {
        let left_mask = x.i((.., feature_id))?.le(threshold)?;
        let right_mask = left_mask.not()?;
        Ok((left_mask, right_mask))
    }

    /// 为单个特征寻找最佳的 MSE (均方误差) 切分点
    fn find_best_split_for_feature<Dev: Device>(
        &self,
        x_feature: &Tensor<Dev>,
        y_targets: &Tensor<Dev>
    ) -> luma_tensor::Result<Option<(f64, f64)>> {
        let n_samples = x_feature.dims1()?;
        if n_samples < 2 { return Ok(None); }

        // 1. 将特征和目标值配对并按特征值排序
        let mut samples: Vec<(f64, f64)> = x_feature.to_vec()?.into_iter().zip(y_targets.to_vec()?).collect();
        samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // 2. 初始化统计信息：初始状态所有样本都在右侧
        let mut sum_right = 0.0;
        let mut sum_sq_right = 0.0;
        for &(_, y_val) in &samples {
            sum_right += y_val;
            sum_sq_right += y_val * y_val;
        }

        let mut sum_left = 0.0;
        let mut sum_sq_left = 0.0;

        let mut n_left = 0.0;
        let mut n_right = n_samples as f64;

        let mut best_sse: Option<f64> = None;
        let mut best_threshold = 0.0;

        // 3. 遍历所有可能的切分点
        // 利用动态规划的思想，每次将一个样本从右边移到左边，O(N) 算出每一步的平方误差
        for i in 0..(n_samples - 1) {
            let (val_curr, y_curr) = samples[i];
            let (val_next, _) = samples[i + 1];

            // 更新统计：将当前样本从右侧移到左侧
            sum_left += y_curr;
            sum_sq_left += y_curr * y_curr;
            n_left += 1.0;

            sum_right -= y_curr;
            sum_sq_right -= y_curr * y_curr;
            n_right -= 1.0;

            // 如果两个相邻特征值相等，不能在此处切分
            if val_curr == val_next {
                continue;
            }

            // 计算左右子集的 SSE (Sum of Squared Errors)
            // SSE = Sum(y^2) - (Sum(y))^2 / N
            let sse_left = sum_sq_left - (sum_left * sum_left) / n_left;
            let sse_right = sum_sq_right - (sum_right * sum_right) / n_right;
            let total_sse = sse_left + sse_right;

            let threshold = (val_curr + val_next) / 2.0;

            // 更新最佳 SSE
            match best_sse {
                None => {
                    best_sse = Some(total_sse);
                    best_threshold = threshold;
                },
                Some(min_sse) if total_sse < min_sse => {
                    best_sse = Some(total_sse);
                    best_threshold = threshold;
                },
                _ => {}
            }
        }

        Ok(best_sse.map(|sse| (sse, best_threshold)))
    }
}

impl<Dev: Device> DecisionTreeRegressorModel<Dev> {
    pub fn depth(&self) -> usize {
        self.root.depth()
    }
}

// ==================================================================================== //
//                      test
// ==================================================================================== //

#[cfg(test)]
mod tests {
    use luma_tensor::{Cpu, Int, Tensor};

    use crate::{datasets::{load_diabetes, load_iris, train_test_split}, metrics::accuracy_score, core::{PredictFit, PredictModel}, tree::{DecisionTreeClassifier, DecisionTreeRegressor}};

    #[test]
    fn test_class_iris() {
        let device = Cpu;
        let iris = load_iris(&device).unwrap();
        let x = iris.data;
        let y = iris.target;
        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.3).unwrap();

        let trainer = DecisionTreeClassifier::new(10);
        let model = trainer.fit(&x_train, &y_train).unwrap();

        let y_pred = model.predict(&x_test).unwrap();

        println!("Depth: {}", model.depth());
        let acc = accuracy_score(&y_test, &y_pred).unwrap();
        println!("Acc: {acc}");
        // 随机切分偶尔落在困难测试集上（实测最低 0.889），0.85 防 flake；模型故障会是 0.33~0.5
        assert!(acc > 0.85, "acc = {acc}");
    }

    #[test]
    fn test_diabetes() {
        let device = Cpu;
        let diabetes = load_diabetes(&device).unwrap();
        let x = diabetes.data;
        let y = diabetes.target;
        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.3).unwrap();

        let trainer = DecisionTreeRegressor::new(5);
        let model = trainer.fit(&x_train, &y_train).unwrap();
        let y_pred = model.predict(&x_test).unwrap();

        assert_eq!(y_pred.dims(), y_test.dims());
        // 糖尿病 target 范围约 [25, 346]，深度 5 的回归树 MAE 应远小于 80
        let mae = (y_pred - &y_test).abs().unwrap().mean_all().unwrap().to_scalar().unwrap();
        assert!(mae < 80.0, "mae = {mae}");
    }

    #[test]
    fn test_decision_tree_toy_exact_fit() {
        // 手造可完美切分的数据：特征 < 5.5 => 类别 0，否则类别 1
        let x = Tensor::<Cpu>::new(vec![0.0, 1.0, 10.0, 11.0], &Cpu).unwrap().reshape((4, 1)).unwrap();
        let y = Tensor::<Cpu, Int>::new(vec![0i64, 0, 1, 1], &Cpu).unwrap();

        let model = DecisionTreeClassifier::new(3).fit(&x, &y).unwrap();
        let y_pred = model.predict(&x).unwrap();
        assert_eq!(y_pred.to_vec().unwrap(), vec![0, 0, 1, 1]);
        // 根节点切一刀，左右都是叶子 => 深度 2
        assert_eq!(model.depth(), 2);
    }
}
