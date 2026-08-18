use std::marker::PhantomData;

use luma_macros::Module;
use luma_tensor::{Device, Float, Tensor};

use crate::NnResult;
use crate::functional;

/// Binary cross-entropy loss: `-mean(target*ln(pred) + (1-target)*ln(1-pred))`.
#[derive(Module, Clone, Default)]
pub struct BCELoss<D: Device> {
    #[module(skip)]
    _marker: PhantomData<D>,
}

impl<D: Device> BCELoss<D> {
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }

    pub fn forward(&self, pred: &Tensor<D, Float>, target: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
        functional::bce_loss(pred, target)
    }
}
