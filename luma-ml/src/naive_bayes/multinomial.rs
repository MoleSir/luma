use luma_tensor::{Device, FloatDType, IndexOp, Int, Tensor};

use crate::core::{PredictFit, PredictModel};

pub struct MultinomialNB {
    pub alpha: f64,
}

pub struct MultinomialNBModel<Dev: Device> {
    pub class_log_prior: Tensor<Dev>,
    pub feature_log_prob: Tensor<Dev>,
}

impl MultinomialNB {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl<Dev: Device> PredictFit<Tensor<Dev, Int>> for MultinomialNB {
    type Output = Tensor<Dev, Int>;
    type Model = MultinomialNBModel<Dev>;

    /// ## Args
    /// - x: (n_samples, n_features): 每个样本在每个特征上出现的次数
    /// - y: (n_samples,)：每个样本的分类标签
    fn fit(&self, x: &Tensor<Dev, Int>, y: &Tensor<Dev, Int>) -> crate::error::MlResult<Self::Model> {
        let (n_samples, _) = x.dims2()?;
        let n_samples_y = y.dims1()?;
        if n_samples != n_samples_y {
            luma_tensor::bail!("x samples {} != y samples {}", n_samples, n_samples_y);
        }

        let x = x.cast(FloatDType::F64)?;

        // 获取分类数量 TODO: 应该使用 unique
        let n_class = y.max_all()?.to_scalar()? + 1;

        // 统计每个类型的 “样本数量” + “每个特征出现的次数”
        let mut sample_counts = Vec::with_capacity(n_class as usize);
        let mut feature_counts = Vec::with_capacity(n_class as usize);

        for n in 0..n_class {
            let mask = y.eq(n)?; // (n_samples,)

            // 计算样本数量
            let sample_count = mask.true_count()?;
            sample_counts.push(sample_count as f64);

            // 取出属于这个 class 的特征信息，统计每个 feature 出现的总数
            let cur_class_x = x.i(&mask)?; // (n, n_features)
            let cur_feature_counts = cur_class_x.sum(0)?; // (n_features,)
            feature_counts.push(cur_feature_counts);
        }

        // 计算每个类型的概率
        let sample_counts = Tensor::<Dev>::new(sample_counts, x.device())?;
        let class_log_prior = (sample_counts / (n_samples as f64)).ln()?.unsqueeze(0)?; // (1, n_class)

        // 计算每个分类，每个feature的概率
        let feature_counts = Tensor::stack(&feature_counts, 0)?; // (n_class, n_features)
        feature_counts.add_(self.alpha)?;
        let feature_total_count = feature_counts.sum_keepdim(1)?; // (n_class, 1)
        let feature_prob = feature_counts.broadcast_div(&feature_total_count)?; // (n_class, n_features)
        let feature_log_prob = feature_prob.ln()?; // (n_class, n_features)

        Ok(MultinomialNBModel { class_log_prior, feature_log_prob })
    }
}

impl<Dev: Device> PredictModel for MultinomialNBModel<Dev> {
    type Input = Tensor<Dev, Int>;
    type Output = Tensor<Dev, Int>;

    /// ## Args
    /// - x: (n_samples, n_features)
    ///
    /// ## Returns
    /// - y: (n_samples)
    fn predict(&self, x: &Tensor<Dev, Int>) -> crate::error::MlResult<Tensor<Dev, Int>> {
        let prob = self.predict_proba(x)?; // (n_samples, n_class)
        let y = prob.argmax(1)?;
        Ok(y)
    }
}

impl<Dev: Device> MultinomialNBModel<Dev> {
    /// ## Args
    /// - x: (n_samples, n_features)
    ///
    /// ## Return
    /// - prob: (n_samples, n_class)
    pub fn predict_proba(&self, x: &Tensor<Dev, Int>) -> luma_tensor::Result<Tensor<Dev>> {
        let x = x.cast(FloatDType::F64)?;

        // \ln P(Y) + \sum_i x_i \ln P(x_i|Y)
        // (n_samples, n_features) @ (n_class, n_features).T
        // 相当于让 x 的每一行：一个样本与 feature_log_prob.T 向量矩阵乘，即：feature_log_prob @ 一个 sample 的特征
        // 正好相当于这个 sample 分别和 feature_log_prob 的每行（每行表示一个 class 的特征概率）点积，恰好就是上述公式的后半部分
        let prob = x.matmul(&self.feature_log_prob.transpose_last()?)?; // (n_samples, n_class)
        // 利用广播，给每个样本加上 \ln P(Y)
        let prob = prob.broadcast_add(&self.class_log_prior)?; // (n_samples, n_class)

        return Ok(prob);
    }
}

#[cfg(test)]
mod tests {
    use luma_tensor::{Cpu, Int, Tensor};

    use crate::error::MlResult;

    use super::*;

    #[test]
    fn test_multinomial_nb_basic() -> MlResult<()> {
        let device = Cpu::default();
        // 数据集：
        // 特征 0: "apple" 出现的次数
        // 特征 1: "macbook" 出现的次数
        // 样本 1: [2, 0] -> 标签 0 (水果)
        // 样本 2: [0, 2] -> 标签 1 (科技)
        let x = Tensor::<Cpu, Int>::new(vec![2i64, 0, 0, 2], &device)?.reshape((2, 2))?;

        let y = Tensor::<Cpu, Int>::new(vec![0i64, 1], &device)?;

        let nb = MultinomialNB::new(1.0); // alpha = 1.0 (拉普拉斯平滑)
        let model = nb.fit(&x, &y)?;

        // 测试推理
        // 一个包含很多 "apple" 的新样本 [3, 0] 应该预测为 0
        let test_x = Tensor::<Cpu, Int>::new(vec![3i64, 0], &device)?.reshape((1, 2))?;
        let prediction = model.predict(&test_x)?;

        assert_eq!(prediction.to_vec()?[0], 0);

        // 一个包含很多 "macbook" 的新样本 [0, 5] 应该预测为 1
        let test_x_2 = Tensor::<Cpu, Int>::new(vec![0i64, 5], &device)?.reshape((1, 2))?;
        let prediction_2 = model.predict(&test_x_2)?;

        assert_eq!(prediction_2.to_vec()?[0], 1);

        Ok(())
    }

    #[test]
    fn test_against_sklearn_values() -> crate::error::MlResult<()> {
        let device = Cpu::default();
        let x = Tensor::<Cpu, Int>::new(vec![1i64, 2, 2, 1, 3, 4, 4, 3], &device)?.reshape((4, 2))?;
        let y = Tensor::<Cpu, Int>::new(vec![0i64, 0, 1, 1], &device)?;

        let nb = MultinomialNB::new(1.0);
        let model = nb.fit(&x, &y)?;

        println!("{}", model.class_log_prior);
        println!("{}", model.feature_log_prob);

        Ok(())
    }
}

/*
    原始贝叶斯公式：

    $$
                P(Y) * P(X|Y)
    P(Y|X) = -----------------------

                    P(X)
    $$

    对分类任务，Y 表示样本属于某个分类，而 X 表示若干特征的出现的数量，而一般特征有多个；

    可以理解为：对一个样本，根据其特征数量，判断属于哪个分类

    所以其实 X 是多个特征同时出现的总事件，由于我们认为这些特征彼此无关，所以可以重写为：

    $$
                P(Y) * \prod_i P(x_i|Y)^x_i
    P(Y|X) = -----------------------------------

                    \prod_i P(x_i)^x_i
    $$

    ^x_i 是因为这些表示的都是出现次数，一个特征出现的概率是 P(x_i)，所以出现多次就要累乘多次（P(x_i) 中的 x_i 只是没有代表具体数字，只是代码这个特征是哪个）

    我们的目标就根据这个公式，输入一个样本 X（一些列特征的数量），计算每个分类 Y 的条件概率 P(Y|X)，我们选择最大的作为预测分类

    所以我们只要比较不同 P(Y|X) 谁更大即可，那么可以做一些简化，例如分母对所有 Y 都相同可以去掉，只要比较

    P(Y) * \prod_i P(x_i|Y)^x_i

    而因为只比较数值大小，加一个单调递增函数 ln：

    $$
    \ln [P(Y) * \prod_i P(x_i|Y)^x_i] = \ln P(Y) + \sum_i x_i \ln P(x_i|Y)
    $$

    这就是我们要比较的目标。其中我们要从训练集学习：

    $$
    1. \ln P(Y)：每个分类在训练集中出现的概率（就是统计个数/总数）
    2. \ln P(x_i|Y)：在每个分类中，每个特征出现的概率
    $$
*/
