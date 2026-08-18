use std::marker::PhantomData;

use luma_macros::Module;
use luma_tensor::{Device, Float, Int, Tensor};

use crate::NnResult;
use crate::functional;

/// Cross-entropy loss: `nll_loss(log_softmax(pred), target)`.
#[derive(Module, Clone, Default)]
pub struct CrossEntropyLoss<D: Device> {
    #[module(skip)]
    _marker: PhantomData<D>,
}

impl<D: Device> CrossEntropyLoss<D> {
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }

    pub fn forward(&self, pred: &Tensor<D, Float>, target: &Tensor<D, Int>) -> NnResult<Tensor<D, Float>> {
        functional::cross_entropy_loss(pred, target)
    }
}
