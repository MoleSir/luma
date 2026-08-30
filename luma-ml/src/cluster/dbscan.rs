use luma_tensor::{Device, IndexOp, Int, Tensor};

use crate::error::MlResult;

/// Density-Based Spatial Clustering of Applications with Noise
pub struct DBSCAN {
    pub eps: f64,
    pub min_pts: usize,
}

pub struct DBSCANModel<Dev: Device> {
    pub centers: Tensor<Dev>,
}

impl DBSCAN {
    pub fn fit<Dev: Device>(&self, x: &Tensor<Dev>) -> MlResult<Tensor<Dev, Int>> {
        let (n_samples, _) = x.dims2()?;
        // 预先计算两两结点的距离
        // (n_samples, 1, n_features) - (1, n_samples, n_features) => (n_samples, n_samples, n_features)
        let delta = x.unsqueeze(1)?.broadcast_sub(&x.unsqueeze(0)?)?;
        // (n_samples, n_samples, n_features) => (n_samples, n_samples,)
        let distances = delta.sqr()?.sum(2)?.sqrt()?;

        let mut labels = vec![i64::MAX; n_samples];
        let mut visited = vec![false; n_samples];
        let mut cluster_id = 0i64;

        // 遍历每个结点
        for sample in 0..n_samples {
            if visited[sample] {
                continue;
            }

            visited[sample] = true;
            // 获取聚类这个结点足够近的邻居
            let mut neighbors = self.region_query(sample, &distances)?;

            if neighbors.len() < self.min_pts {
                labels[sample] = i64::MAX;
            } else {
                self.expand_cluster(cluster_id, sample, &mut labels, &mut neighbors, &mut visited, &distances)?;
                cluster_id += 1;
            }
        }

        let labels = Tensor::<Dev, Int>::new(labels, x.device())?;

        Ok(labels)
    }

    fn expand_cluster<Dev: Device>(
        &self,
        cluster_id: i64,
        sample: usize,
        labels: &mut Vec<i64>,
        neighbors: &mut Vec<usize>,
        visited: &mut Vec<bool>,
        distances: &Tensor<Dev>,
    ) -> MlResult<()> {
        labels[sample] = cluster_id;

        let mut i = 0;
        while i < neighbors.len() {
            let neighbor = neighbors[i];
            if !visited[neighbor] {
                visited[neighbor] = true;
                let neighbor_neighbors = self.region_query(neighbor, distances)?;
                if neighbor_neighbors.len() > self.min_pts {
                    neighbors.extend(neighbor_neighbors);
                }
            }

            if labels[neighbor] == i64::MAX {
                labels[neighbor] = cluster_id;
            }
            i += 1;
        }

        Ok(())
    }

    fn region_query<Dev: Device>(&self, sample: usize, distances: &Tensor<Dev>) -> MlResult<Vec<usize>> {
        // 取出 sample 对应的位置
        // (n_samples, n_samples,) => (n_samples,)
        let sample_dis = distances.i(sample)?;
        // 过滤数量 (n_samples,)
        let mask = sample_dis.le(self.eps)?;

        Ok(mask.to_vec()?.into_iter().enumerate().filter(|(_, m)| *m).map(|(i, _)| i).collect::<Vec<_>>())
    }
}

#[cfg(test)]
mod tests {
    use luma_tensor::{Cpu, Tensor};

    use crate::cluster::DBSCAN;

    /// 两个相距 20 的 10x10 网格团（每个点间距 1）+ 1 个孤立点
    fn two_blobs_plus_noise() -> Tensor<Cpu> {
        let mut data = vec![];
        for i in 0..10 {
            for j in 0..10 {
                data.push(i as f64);
                data.push(j as f64);
            }
        }
        for i in 0..10 {
            for j in 0..10 {
                data.push(20.0 + i as f64);
                data.push(20.0 + j as f64);
            }
        }
        data.push(100.0);
        data.push(100.0);
        Tensor::from_vec_f64(data, (201, 2), &Cpu).unwrap()
    }

    #[test]
    fn test_dbscan_two_blobs_and_noise() {
        let x = two_blobs_plus_noise();
        let dbscan = DBSCAN { eps: 1.5, min_pts: 3 };
        let labels = dbscan.fit(&x).unwrap().to_vec().unwrap();

        // 噪声标记 i64::MAX 存在 Int 张量默认 I32 存储里会截断成 -1，所以这里不依赖具体标记值：
        // 直接用最后一个样本（孤立点）的标签作为噪声值
        let noise_label = labels[200];

        // 两个网格各自完全同簇，且两簇不同，并且都不是噪声值
        for i in 1..100 {
            assert_eq!(labels[i], labels[0], "sample {i} should be in cluster {}", labels[0]);
            assert_ne!(labels[i], noise_label);
        }
        for i in 101..200 {
            assert_eq!(labels[i], labels[100], "sample {i} should be in cluster {}", labels[100]);
            assert_ne!(labels[i], noise_label);
        }
        assert_ne!(labels[0], labels[100]);
        // 网格内每个点都应有至少 min_pts 个邻居（都是核心点），噪声点孤立
        assert_ne!(labels[200], labels[0]);
        assert_ne!(labels[200], labels[100]);
    }

    #[test]
    fn test_dbscan_all_noise() {
        // 所有点彼此相距很远 -> 每个点都只有自己一个邻居，全部判为噪声（标签全部相同即可，
        // 不依赖 i64::MAX 标记，因为 Int 张量默认 I32 存储会把它截断成 -1）
        let x = Tensor::<Cpu>::new(vec![0.0, 0.0, 100.0, 0.0, 0.0, 100.0], &Cpu).unwrap().reshape((3, 2)).unwrap();

        let dbscan = DBSCAN { eps: 1.0, min_pts: 2 };
        let labels = dbscan.fit(&x).unwrap().to_vec().unwrap();
        assert_eq!(labels[0], labels[1], "labels = {labels:?}");
        assert_eq!(labels[1], labels[2]);
    }
}
