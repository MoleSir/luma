use luma_tensor::{Device, Int, Tensor};
use std::path::PathBuf;
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
            let v: f64 =
                tokens.next().expect("invalid iris data: no enough features!").parse::<f64>().expect("invalid iris data: not number");
            x.push(v);
        }
        let label = tokens.next().expect("invalid iris data: no label!");
        y.push(str_to_label(label));
    }

    assert_eq!(x.len(), IRIS_N_FEATURES * IRIS_N_SAMPLES);
    assert_eq!(y.len(), IRIS_N_SAMPLES);

    let x = Tensor::from_vec_f64(x, (IRIS_N_SAMPLES, IRIS_N_FEATURES), device)?;
    let y = Tensor::<Dev, Int>::new(y, device)?;

    Ok(IrisDataset { data: x, target: y })
}

fn str_to_label(s: &str) -> u32 {
    match s {
        "Iris-setosa" => 0,
        "Iris-versicolor" => 1,
        "Iris-virginica" => 2,
        _ => unreachable!("un support iris label"),
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
//                      Diabetes
// ==================================================================================== //

pub struct DiabetesDataset<Dev: Device> {
    pub headers: Vec<String>,
    pub data: Tensor<Dev>,
    pub target: Tensor<Dev>,
}

impl<Dev: Device> DiabetesDataset<Dev> {
    #[inline]
    pub fn new(device: &Dev) -> MlResult<Self> {
        load_diabetes(device)
    }
}

pub const DIABETES_N_FEATURES: usize = 10;
pub const DIABETES_N_SAMPLES: usize = 442;

pub fn load_diabetes<Dev: Device>(device: &Dev) -> MlResult<DiabetesDataset<Dev>> {
    let file_path = dataset_file_path("diabetes.csv");
    let content = std::fs::read_to_string(file_path).context("read build in diabetes.csv")?;

    let mut lines = content.lines();
    let headers = lines.next().expect("invalid diabetes: no headers!");
    let headers: Vec<String> = headers.split(',').map(|s| s.to_string()).collect();

    let mut x = vec![];
    let mut y = vec![];

    while let Some(line) = lines.next() {
        if line.is_empty() {
            continue;
        }

        let mut tokens = line.split(',');
        for _ in 0..DIABETES_N_FEATURES {
            let token = tokens.next().expect("invalid diabetes data: not enough features!");
            let v: f64 = token.trim().parse::<f64>().expect("invalid diabetes data: feature is not a number");
            x.push(v);
        }

        let label_token = tokens.next().expect("invalid diabetes data: no target!");
        let target_val: f64 = label_token.trim().parse::<f64>().expect("invalid diabetes data: target is not a number");
        y.push(target_val);
    }

    assert_eq!(x.len(), DIABETES_N_FEATURES * DIABETES_N_SAMPLES, "X size mismatch");
    assert_eq!(y.len(), DIABETES_N_SAMPLES, "Y size mismatch");

    let x = Tensor::from_vec_f64(x, (DIABETES_N_SAMPLES, DIABETES_N_FEATURES), device)?;
    let y = Tensor::from_vec_f64(y, (DIABETES_N_SAMPLES,), device)?; // Y 此时是一维的浮点 Tensor

    Ok(DiabetesDataset { headers, data: x, target: y })
}

// ==================================================================================== //
//                      test
// ==================================================================================== //

#[cfg(test)]
mod tests {
    use luma_tensor::Cpu;

    use super::load_iris;
    use crate::datasets::local::{IRIS_N_FEATURES, IRIS_N_SAMPLES};
    use crate::datasets::{DIABETES_N_FEATURES, DIABETES_N_SAMPLES, load_diabetes};

    #[test]
    fn test_load_iris() {
        let iris = load_iris(&Cpu).unwrap();
        let x = iris.data;
        let y = iris.target;
        assert_eq!(x.dims(), [IRIS_N_SAMPLES, IRIS_N_FEATURES]);
        assert_eq!(y.dims(), [IRIS_N_SAMPLES,]);
    }

    #[test]
    fn test_load_diabetes() {
        let diabetes = load_diabetes(&Cpu).unwrap();
        let x = diabetes.data;
        let y = diabetes.target;
        assert_eq!(x.dims(), [DIABETES_N_SAMPLES, DIABETES_N_FEATURES]);
        assert_eq!(y.dims(), [DIABETES_N_SAMPLES,]);
    }
}
