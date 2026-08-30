use super::{PredictFit, TransformFit};
use crate::{
    linear::LinearRegression,
    pipelines,
    preprocessing::{MinMaxScaler, StandardScaler},
};
use luma_tensor::{Cpu, Tensor};

#[test]
fn test_pipe_transform() {
    let device = Cpu;
    let f1 = StandardScaler::default();
    let f2 = MinMaxScaler::default();

    let fit = pipelines!(f1, f2);

    let x = Tensor::<Cpu>::rand(-10.0, 10.0, (100, 25), &device).unwrap();
    fit.fit_transform(&x).unwrap();
}

#[test]
fn test_stand_linear() {
    let device = Cpu;
    let pre = StandardScaler::default();
    let linear = LinearRegression::default();
    let fit = pipelines!(pre, linear);

    const N_SAMPLES: usize = 50;
    const N_FEATURES: usize = 5;
    let weight = Tensor::<Cpu>::rand(-2.0, 3.0, (N_FEATURES, 1), &device).unwrap();
    const B: f64 = 2.5;

    let x_train = Tensor::<Cpu>::rand(-1.0, 1.0, (N_SAMPLES, N_FEATURES), &device).unwrap();
    let y_train = x_train.matmul(&weight).unwrap().squeeze(1).unwrap() + B;
    let y_train = y_train + 0.1 * Tensor::<Cpu>::randn(0.0, 1.0, (N_SAMPLES,), &device).unwrap();
    fit.fit(&x_train, &y_train).unwrap();
}

#[test]
fn test_pipeline3() {
    let f1 = StandardScaler::default();
    let f2 = StandardScaler::default();
    let f3 = StandardScaler::default();
    pipelines!(f1, f2, f3);
}
