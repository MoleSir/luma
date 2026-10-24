use luma_macros::Module;
use luma_tensor::ops::construct::TensorCreationOptions;
use luma_tensor::{Device, Float, Tensor};

use crate::functional as F;
use crate::init::{Init, NonLinearity};
use crate::{NnResult, Parameter};

// ============================================================================
//   Conv2dConfig
// ============================================================================

#[derive(Debug, Clone)]
pub struct Conv2dConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
    pub bias: bool,
    pub weight_init: Init,
    pub bias_init: Init,
}

impl Conv2dConfig {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: (usize, usize)) -> Self {
        let fan_in = in_channels * kernel_size.0 * kernel_size.1;
        let bound = 1.0 / (fan_in as f64).sqrt();
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            bias: true,
            weight_init: Init::kaiming_uniform(NonLinearity::Relu, false),
            bias_init: Init::uniform(-bound, bound),
        }
    }
}

// ============================================================================
//   Conv2d
// ============================================================================

/// 2-D convolution: `(N, C_in, H, W) -> (N, C_out, H_out, W_out)`.
#[derive(Module)]
#[module(display = "display")]
pub struct Conv2d<Dev: Device> {
    pub weight: Parameter<Dev>,       // (C_out, C_in, kH, kW)
    pub bias: Option<Parameter<Dev>>, // (C_out)

    #[module(skip)]
    pub in_channels: usize,
    #[module(skip)]
    pub out_channels: usize,
    #[module(skip)]
    pub kernel_size: (usize, usize),
    #[module(skip)]
    pub stride: (usize, usize),
    #[module(skip)]
    pub padding: (usize, usize),
    #[module(skip)]
    pub dilation: (usize, usize),
}

impl<Dev: Device> Conv2d<Dev> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        options: impl Into<TensorCreationOptions<Dev, Float>>,
    ) -> NnResult<Self> {
        let config = Conv2dConfig::new(in_channels, out_channels, kernel_size);
        Self::from_config(&config, options)
    }

    pub fn from_config(config: &Conv2dConfig, options: impl Into<TensorCreationOptions<Dev, Float>>) -> NnResult<Self> {
        let options: TensorCreationOptions<Dev, Float> = options.into();
        let opts = (&options.device, options.dtype);

        let (kh, kw) = config.kernel_size;
        let fan_in = config.in_channels * kh * kw;
        let fan_out = config.out_channels * kh * kw;
        let weight_shape = (config.out_channels, config.in_channels, kh, kw);
        let weight = config.weight_init.init_param_with(weight_shape, fan_in, fan_out, opts)?;
        let bias = if config.bias {
            Some(config.bias_init.init_param((config.out_channels,), opts)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            in_channels: config.in_channels,
            out_channels: config.out_channels,
            kernel_size: config.kernel_size,
            stride: config.stride,
            padding: config.padding,
            dilation: config.dilation,
        })
    }

    pub fn display(&self) -> String {
        format!(
            "in={}, out={}, kernel={:?}, stride={:?}, padding={:?}, dilation={:?}",
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation
        )
    }

    #[inline]
    pub fn forward(&self, input: &Tensor<Dev>) -> NnResult<Tensor<Dev>> {
        F::conv2d(
            input,
            &self.weight,
            self.bias.as_deref(),
            self.stride,
            self.padding,
            self.dilation,
        )
    }
}
