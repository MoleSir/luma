use std::marker::PhantomData;

use luma_macros::Module;
use luma_tensor::{Device, Float, Tensor};

use crate::NnResult;
use crate::functional;

/// Randomly zeroes elements with probability `p` during training.
///
/// During evaluation ([`Module::eval`]) this is a no-op.
#[derive(Module, Clone)]
#[module(train = "set_training")]
pub struct Dropout<D: Device> {
    #[module(skip)]
    pub p: f64,
    #[module(skip)]
    training: bool,
    #[module(skip)]
    _marker: PhantomData<D>,
}

impl<D: Device> Dropout<D> {
    pub fn new(p: f64) -> Self {
        Self { p, training: true, _marker: PhantomData }
    }

    pub fn forward(&self, input: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
        functional::dropout(input, self.p, self.training)
    }

    fn set_training(&mut self, mode: bool) {
        self.training = mode;
    }
}
