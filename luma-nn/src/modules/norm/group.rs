use luma_macros::Module;
use luma_tensor::ops::construct::TensorCreationOptions;
use luma_tensor::{Device, Float, Tensor};

use crate::functional as F;
use crate::init::Init;
use crate::{NnError, NnResult, Parameter};

// ============================================================================
//   GroupNormConfig
// ============================================================================

#[derive(Debug, Clone)]
pub struct GroupNormConfig {
    pub num_groups: usize,
    pub num_channels: usize,
    pub eps: f64,
    pub affine: bool,
    pub weight_init: Init,
    pub bias_init: Init,
}

impl GroupNormConfig {
    pub fn new(num_groups: usize, num_channels: usize) -> Self {
        Self { num_groups, num_channels, eps: 1e-5, affine: true, weight_init: Init::Ones, bias_init: Init::Zeros }
    }
}

// ============================================================================
//   GroupNorm
// ============================================================================

/// Applies Group Normalization over `(C/G, ...)`.
#[derive(Module)]
#[module(display = "display")]
pub struct GroupNorm<Dev: Device> {
    pub weight: Option<Parameter<Dev>>,
    pub bias: Option<Parameter<Dev>>,

    #[module(skip)]
    pub num_groups: usize,
    #[module(skip)]
    pub num_channels: usize,
    #[module(skip)]
    pub eps: f64,
}

impl<Dev: Device> GroupNorm<Dev> {
    pub fn new(
        num_groups: usize,
        num_channels: usize,
        options: impl Into<TensorCreationOptions<Dev, Float>>,
    ) -> NnResult<Self> {
        let config = GroupNormConfig::new(num_groups, num_channels);
        Self::from_config(&config, options)
    }

    pub fn from_config(config: &GroupNormConfig, options: impl Into<TensorCreationOptions<Dev, Float>>) -> NnResult<Self> {
        if config.num_groups == 0 || !config.num_channels.is_multiple_of(config.num_groups) {
            return Err(NnError::Core(luma_tensor::Error::Msg(
                "group_norm: num_groups must divide num_channels".into(),
            )));
        }

        let options: TensorCreationOptions<Dev, Float> = options.into();
        let opts = (&options.device, options.dtype);

        let (weight, bias) = if config.affine {
            (
                Some(config.weight_init.init_param((config.num_channels,), opts)?),
                Some(config.bias_init.init_param((config.num_channels,), opts)?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            weight,
            bias,
            num_groups: config.num_groups,
            num_channels: config.num_channels,
            eps: config.eps,
        })
    }

    pub fn display(&self) -> String {
        format!("groups={}, channels={}", self.num_groups, self.num_channels)
    }

    #[inline]
    pub fn forward(&self, input: &Tensor<Dev>) -> NnResult<Tensor<Dev>> {
        F::group_norm(input, self.num_groups, self.weight.as_deref(), self.bias.as_deref(), self.eps)
    }
}
