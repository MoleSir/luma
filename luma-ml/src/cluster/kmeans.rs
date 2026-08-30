use std::collections::HashSet;

use luma_tensor::{D, Device, IndexOp, Tensor, no_grad, tensor::IntTensor};
use rand::Rng;

use crate::{MlResult, TransformFit, TransformModel};

pub struct KMeans {
    pub k: usize,
    pub init_policy: KMeansInitPolicy,
    pub max_iters: usize,
    pub eps: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KMeansInitPolicy {
    Random,
}

pub struct KMeansModel<Dev: Device> {
    pub centers: Tensor<Dev>,
}

impl<Dev: Device> TransformFit<Tensor<Dev>> for KMeans {
    type Output = IntTensor<Dev>;
    type Model = KMeansModel<Dev>;

    /// ## Args
    /// - `x`: (n_samples, n_features)
    fn fit(&self, x: &Tensor<Dev>) -> MlResult<Self::Model> {
        let (centers, _) = self.do_fit(x)?;
        Ok(KMeansModel { centers })
    }

    /// ## Args
    /// - `x`: (n_samples, n_features)
    ///
    /// ## Return
    /// - `labels`: (n_samples,)
    fn fit_transform(&self, x: &Tensor<Dev>) -> MlResult<IntTensor<Dev>> {
        let (_, labels) = self.do_fit(x)?;
        Ok(labels)
    }
}

impl<Dev: Device> TransformModel for KMeansModel<Dev> {
    type Input = Tensor<Dev>;
    type Output = IntTensor<Dev>;

    /// ## Args
    /// - `x`: (n_samples, n_features)
    ///
    /// ## Return
    /// - `labels`: (n_samples,)
    fn transform(&self, x: &Tensor<Dev>) -> MlResult<IntTensor<Dev>> {
        find_closest_center(x, &self.centers)
    }
}

impl KMeans {
    /// Kmeans fit
    ///
    /// ## Args
    /// - `x`: (n_samples, n_features)
    ///
    /// ## Returns
    /// - `centers`: (k, n_features)
    /// - `indexs`: (n_samples,)
    ///
    /// ## Flow
    ///
    /// 1. init k centers: (k, n_features)
    /// 2. for _ in range(max_iter):
    ///     1. for each samples, calculate distance for each center, and select min one(new index)
    ///     2. each sample now belong to a center, for each center update center values by (n_sample_in_center, n_features)
    ///
    fn do_fit<Dev: Device>(&self, x: &Tensor<Dev>) -> MlResult<(Tensor<Dev>, IntTensor<Dev>)> {
        no_grad!();

        let mut final_labels = None;
        if self.max_iters == 0 {
            luma_tensor::bail!("no iter!");
        }

        let mut centers = self.init_centers(x)?; // (k, n_features)
        let device = x.device();
        assert!(device.same_device(centers.device()));

        for _ in 0..self.max_iters {
            // 1. choise min distance center for each sample => (n_samples, k, n_features)
            let labels = find_closest_center(x, &centers)?;
            final_labels = Some(labels.clone());

            // 2. update centers
            let mut new_centers = vec![];
            for i in 0..self.k {
                // select the sample for ith centers
                let mask = labels.eq(i as i64)?; // (n_samples,) bool/fase
                if mask.true_count()? == 0 {
                    // no any samples in this cluster!
                    new_centers.push(centers.i(i)?);
                    continue;
                }

                let center_x = x.i(mask)?; // (n_true, n_features)
                let new_center = center_x.mean(0)?; // (n_features,)
                new_centers.push(new_center); // [(n_features); ]
            }
            // 注意：这里是新增一个维度堆叠成 (k, n_features)，不能用 cat（cat 会把 (n_features,) 拼成 (k*n_features,)）
            let new_centers = Tensor::stack(&new_centers, 0)?;

            let delta = (&new_centers - &centers).abs()?.mean_all()?.to_scalar()?;
            if delta < self.eps {
                break;
            }

            centers = new_centers;
        }

        Ok((centers, final_labels.expect("must not None")))
    }

    fn init_centers<Dev: Device>(&self, x: &Tensor<Dev>) -> MlResult<Tensor<Dev>> {
        match self.init_policy {
            KMeansInitPolicy::Random => {
                let (n_samples, _) = x.dims2()?;
                let mut indexs = HashSet::new();
                let mut rng = rand::rng();

                let mut centers = vec![];
                loop {
                    if indexs.len() == self.k {
                        break;
                    }

                    let index = rng.random_range(0..n_samples);
                    if indexs.insert(index) {
                        // a new index
                        centers.push(x.i(index)?);
                    } else {
                        continue;
                    }
                }

                let centers = Tensor::stack(&centers, 0)?;
                Ok(centers)
            }
        }
    }
}

fn find_closest_center<Dev: Device>(x: &Tensor<Dev>, centers: &Tensor<Dev>) -> MlResult<IntTensor<Dev>> {
    let delta = x.unsqueeze(1)?.broadcast_sub(&centers.unsqueeze(0)?)?;
    // (n_samples, k, n_features) => (n_samples, k)
    let distances = delta.pow_(2.0)?.sum_keepdim(D::Minus1)?;
    // (n_samples, k) => (n_samples,)
    let labels = distances.argmin(1)?;
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use luma_tensor::{Cpu, Tensor};

    use crate::{
        TransformFit, TransformModel,
        cluster::{KMeans, KMeansInitPolicy, KMeansModel},
    };

    /// 两个 5x5 的网格团：一个在 (0..4, 0..4)，一个在 (10..14, 10..14)，相距足够远保证分类清晰
    fn two_blobs() -> Tensor<Cpu> {
        let mut data = vec![];
        for i in 0..25 {
            data.push((i % 5) as f64);
            data.push((i / 5) as f64);
        }
        for i in 0..25 {
            data.push(10.0 + (i % 5) as f64);
            data.push(10.0 + (i / 5) as f64);
        }
        Tensor::from_vec_f64(data, (50, 2), &Cpu).unwrap()
    }

    #[test]
    fn test_kmeans_fit_two_blobs() {
        let x = two_blobs();
        let kmeans = KMeans { k: 2, init_policy: KMeansInitPolicy::Random, max_iters: 100, eps: 1e-6 };

        // 随机初始化可能让两个中心落在同一团：Lloyd 有时收敛到 30/20 之类的局部最优。
        // 用纯度（聚类标签与真实两团的对齐率）衡量质量，重试 20 次把 flake 概率压到极低。
        let mut purity = 0.0f64;
        for _ in 0..20 {
            let labels = kmeans.fit_transform(&x).unwrap().to_vec().unwrap();
            // 真实标签：前 25 个是团 A，后 25 个是团 B；聚类标签 0/1 可能互换，取较好对齐
            let n_correct = labels.iter().enumerate().filter(|&(i, &l)| (l == labels[0]) == (i < 25)).count();
            purity = purity.max(n_correct as f64 / 50.0);
            if purity >= 0.9 {
                break;
            }
        }
        assert!(purity >= 0.9, "purity = {purity}");
    }

    #[test]
    fn test_kmeans_transform_with_given_centers() {
        // 手设 centers 验证 transform 的最近中心分配逻辑（完全确定性）
        let centers = Tensor::<Cpu>::new(vec![0.0, 0.0, 10.0, 10.0], &Cpu).unwrap().reshape((2, 2)).unwrap();
        let model = KMeansModel { centers };

        let x = two_blobs();
        let labels = model.transform(&x).unwrap().to_vec().unwrap();
        for l in &labels[..25] {
            assert_eq!(*l, 0, "blob 1 point should belong to center 0");
        }
        for l in &labels[25..] {
            assert_eq!(*l, 1, "blob 2 point should belong to center 1");
        }
    }
}
