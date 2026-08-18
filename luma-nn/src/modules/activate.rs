use std::marker::PhantomData;

use luma_macros::Module;
use luma_tensor::{Device, Float, Tensor};

use crate::NnResult;
use crate::functional;

// ============================================================================
//   ReLU
// ============================================================================

#[derive(Module, Clone, Default)]
pub struct ReLU<D: Device> {
    #[module(skip)]
    _marker: PhantomData<D>,
}

impl<D: Device> ReLU<D> {
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }

    pub fn forward(&self, input: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
        functional::relu(input)
    }
}

// ============================================================================
//   LeakyReLU
// ============================================================================

#[derive(Module, Clone)]
pub struct LeakyReLU<D: Device> {
    #[module(skip)]
    pub negative_slope: f64,
    #[module(skip)]
    _marker: PhantomData<D>,
}

impl<D: Device> LeakyReLU<D> {
    pub fn new(negative_slope: f64) -> Self {
        Self { negative_slope, _marker: PhantomData }
    }

    pub fn forward(&self, input: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
        functional::leaky_relu(input, self.negative_slope)
    }
}

impl<D: Device> Default for LeakyReLU<D> {
    fn default() -> Self {
        Self { negative_slope: 0.01, _marker: PhantomData }
    }
}

// ============================================================================
//   Sigmoid
// ============================================================================

#[derive(Module, Clone, Default)]
pub struct Sigmoid<D: Device> {
    #[module(skip)]
    _marker: PhantomData<D>,
}

impl<D: Device> Sigmoid<D> {
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }

    pub fn forward(&self, input: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
        functional::sigmoid(input)
    }
}

// ============================================================================
//   Tanh
// ============================================================================

#[derive(Module, Clone, Default)]
pub struct Tanh<D: Device> {
    #[module(skip)]
    _marker: PhantomData<D>,
}

impl<D: Device> Tanh<D> {
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }

    pub fn forward(&self, input: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
        functional::tanh(input)
    }
}

// ============================================================================
//   GELU
// ============================================================================

#[derive(Module, Clone, Default)]
pub struct GELU<D: Device> {
    #[module(skip)]
    _marker: PhantomData<D>,
}

impl<D: Device> GELU<D> {
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }

    pub fn forward(&self, input: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
        functional::gelu(input)
    }
}

// ============================================================================
//   SiLU
// ============================================================================

#[derive(Module, Clone, Default)]
pub struct SiLU<D: Device> {
    #[module(skip)]
    _marker: PhantomData<D>,
}

impl<D: Device> SiLU<D> {
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }

    pub fn forward(&self, input: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
        functional::silu(input)
    }
}
