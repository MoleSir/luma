use std::marker::PhantomData;

use luma_macros::Module;
use luma_tensor::{Device, Float, Tensor};

use crate::NnResult;
use crate::functional;

/// Mean squared error loss: `mean((pred - target)²)`.
#[derive(Module, Clone, Default)]
pub struct MSELoss<D: Device> {
    #[module(skip)]
    _marker: PhantomData<D>,
}

impl<D: Device> MSELoss<D> {
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }

    pub fn forward(&self, pred: &Tensor<D, Float>, target: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
        functional::mse_loss(pred, target)
    }
}
