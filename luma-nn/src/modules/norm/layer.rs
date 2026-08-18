use luma_macros::Module;
use luma_tensor::ops::construct::TensorCreationOptions;
use luma_tensor::{Device, Float, Tensor};

use crate::functional as F;
use crate::init::Init;
use crate::{NnResult, Parameter};

// ============================================================================
//   LayerNormConfig
// ============================================================================

#[derive(Debug, Clone)]
pub struct LayerNormConfig {
    pub normalized_shape: usize,
    pub eps: f64,
    pub weight_init: Init,
    pub bias_init: Init,
}

impl LayerNormConfig {
    pub fn new(normalized_shape: usize) -> Self {
        Self { normalized_shape, eps: 1e-5, weight_init: Init::Ones, bias_init: Init::Zeros }
    }
}

// ============================================================================
//   LayerNorm
// ============================================================================

#[derive(Module)]
#[module(display = "display")]
pub struct LayerNorm<Dev: Device> {
    pub weight: Parameter<Dev>,
    pub bias: Parameter<Dev>,

    #[module(skip)]
    pub normalized_shape: usize,
    #[module(skip)]
    pub eps: f64,
}

impl<Dev: Device> LayerNorm<Dev> {
    pub fn new(normalized_shape: usize, options: impl Into<TensorCreationOptions<Dev, Float>>) -> NnResult<Self> {
        let config = LayerNormConfig::new(normalized_shape);
        Self::from_config(&config, options)
    }

    pub fn from_config(config: &LayerNormConfig, options: impl Into<TensorCreationOptions<Dev, Float>>) -> NnResult<Self> {
        let options: TensorCreationOptions<Dev, Float> = options.into();
        let opts = (&options.device, options.dtype);

        let weight = config.weight_init.init_param((config.normalized_shape,), opts)?;
        let bias = config.bias_init.init_param((config.normalized_shape,), opts)?;

        Ok(Self { weight, bias, normalized_shape: config.normalized_shape, eps: config.eps })
    }

    pub fn display(&self) -> String {
        format!("{}", self.normalized_shape)
    }

    #[inline]
    pub fn forward(&self, input: &Tensor<Dev>) -> NnResult<Tensor<Dev>> {
        F::layer_norm(input, Some(&self.weight), Some(&self.bias), self.eps)
    }
}
