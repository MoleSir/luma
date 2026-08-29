use std::path::PathBuf;
use luma_tensor::{Device, Int, Tensor};
use thiserrorctx::Context;

use crate::error::MlResult;

// ==================================================================================== //
//                      IRIS
// ==================================================================================== //

pub struct IrisDataset<Dev: Device> {
    pub data: Tensor<Dev>,
    pub target: Tensor<Dev, Int>,
}

impl<Dev: Device> IrisDataset<Dev> {
    pub fn new(device: &Dev) -> MlResult<Self> {
        load_iris(device)
    }
}

pub const IRIS_N_FEATURES: usize = 4;
pub const IRIS_N_SAMPLES: usize = 150;

pub fn load_iris<Dev: Device>(device: &Dev) -> MlResult<IrisDataset<Dev>> {
    let file_path = dataset_file_path("iris.csv");
    let content = std::fs::read_to_string(file_path).context("read build in iris.csv")?;

    let mut x = vec![];
    let mut y = vec![];

    for line in content.lines() {
        if line.is_empty() {
            continue;
        }
        let mut tokens = line.split(',');
        for _ in 0..IRIS_N_FEATURES {
            let v: f64 = tokens
                .next().expect("invalid iris data: no enough features!")
                .parse::<f64>().expect("invalid iris data: not number");
            x.push(v);
        }
        let label = tokens.next().expect("invalid iris data: no label!");
        y.push(str_to_label(label));
    }

    assert_eq!(x.len(), IRIS_N_FEATURES * IRIS_N_SAMPLES);
    assert_eq!(y.len(), IRIS_N_SAMPLES);

    let x = Tensor::from_vec_f64(x, (IRIS_N_SAMPLES, IRIS_N_FEATURES), device)?;
    let y = Tensor::<Dev, Int>::new(y, device)?;

    Ok( IrisDataset { data: x, target: y } )
}

fn str_to_label(s: &str) -> u32 {
    match s {
        "Iris-setosa" => 0,
        "Iris-versicolor" => 1,
        "Iris-virginica" => 2,
        _ => unreachable!("un support iris label")
    }
}

fn dataset_file_path(file_name: &str) -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut file_path = PathBuf::from(manifest_dir);
    file_path.push("data");
    file_path.push(file_name);
    file_path
}

// ==================================================================================== //
//                      test
// ==================================================================================== //

#[cfg(test)]
mod tests {
    use luma_tensor::Cpu;

    use crate::datasets::local::{IRIS_N_FEATURES, IRIS_N_SAMPLES};
    use super::load_iris;

    #[test]
    fn test_load_iris() {
        let iris = load_iris(&Cpu).unwrap();
        let x = iris.data;
        let y = iris.target;
        assert_eq!(x.dims(), [IRIS_N_SAMPLES, IRIS_N_FEATURES]);
        assert_eq!(y.dims(), [IRIS_N_SAMPLES,]);
    }

}