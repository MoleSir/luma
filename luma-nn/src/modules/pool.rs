use std::marker::PhantomData;

use luma_macros::Module;
use luma_tensor::{Device, Tensor};

use crate::NnResult;
use crate::functional as F;

// ============================================================================
//   MaxPool2d
// ============================================================================

#[derive(Debug, Clone)]
pub struct MaxPool2dConfig {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
    pub ceil_mode: bool,
}

impl MaxPool2dConfig {
    pub fn new(kernel_size: (usize, usize)) -> Self {
        Self { kernel_size, stride: kernel_size, padding: (0, 0), dilation: (1, 1), ceil_mode: false }
    }
}

/// 2-D max pooling: `(N, C, H, W) -> (N, C, H_out, W_out)`.
#[derive(Module, Clone)]
#[module(display = "display")]
pub struct MaxPool2d<Dev: Device> {
    #[module(skip)]
    pub kernel_size: (usize, usize),
    #[module(skip)]
    pub stride: (usize, usize),
    #[module(skip)]
    pub padding: (usize, usize),
    #[module(skip)]
    pub dilation: (usize, usize),
    #[module(skip)]
    pub ceil_mode: bool,
    #[module(skip)]
    _marker: PhantomData<Dev>,
}

impl<Dev: Device> MaxPool2d<Dev> {
    pub fn new(kernel_size: (usize, usize)) -> Self {
        Self::from_config(&MaxPool2dConfig::new(kernel_size))
    }

    pub fn from_config(config: &MaxPool2dConfig) -> Self {
        Self {
            kernel_size: config.kernel_size,
            stride: config.stride,
            padding: config.padding,
            dilation: config.dilation,
            ceil_mode: config.ceil_mode,
            _marker: PhantomData,
        }
    }

    pub fn display(&self) -> String {
        format!(
            "kernel={:?}, stride={:?}, padding={:?}, dilation={:?}, ceil={}",
            self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode
        )
    }

    #[inline]
    pub fn forward(&self, input: &Tensor<Dev>) -> NnResult<Tensor<Dev>> {
        F::max_pool2d(input, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)
    }
}

// ============================================================================
//   AvgPool2d
// ============================================================================

#[derive(Debug, Clone)]
pub struct AvgPool2dConfig {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub ceil_mode: bool,
    pub count_include_pad: bool,
}

impl AvgPool2dConfig {
    pub fn new(kernel_size: (usize, usize)) -> Self {
        Self { kernel_size, stride: kernel_size, padding: (0, 0), ceil_mode: false, count_include_pad: true }
    }
}

/// 2-D average pooling: `(N, C, H, W) -> (N, C, H_out, W_out)`.
#[derive(Module, Clone)]
#[module(display = "display")]
pub struct AvgPool2d<Dev: Device> {
    #[module(skip)]
    pub kernel_size: (usize, usize),
    #[module(skip)]
    pub stride: (usize, usize),
    #[module(skip)]
    pub padding: (usize, usize),
    #[module(skip)]
    pub ceil_mode: bool,
    #[module(skip)]
    pub count_include_pad: bool,
    #[module(skip)]
    _marker: PhantomData<Dev>,
}

impl<Dev: Device> AvgPool2d<Dev> {
    pub fn new(kernel_size: (usize, usize)) -> Self {
        Self::from_config(&AvgPool2dConfig::new(kernel_size))
    }

    pub fn from_config(config: &AvgPool2dConfig) -> Self {
        Self {
            kernel_size: config.kernel_size,
            stride: config.stride,
            padding: config.padding,
            ceil_mode: config.ceil_mode,
            count_include_pad: config.count_include_pad,
            _marker: PhantomData,
        }
    }

    pub fn display(&self) -> String {
        format!(
            "kernel={:?}, stride={:?}, padding={:?}, ceil={}, count_include_pad={}",
            self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad
        )
    }

    #[inline]
    pub fn forward(&self, input: &Tensor<Dev>) -> NnResult<Tensor<Dev>> {
        F::avg_pool2d(input, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)
    }
}

// ============================================================================
//   AdaptiveAvgPool2d
// ============================================================================

#[derive(Debug, Clone)]
pub struct AdaptiveAvgPool2dConfig {
    pub output_size: (usize, usize),
}

impl AdaptiveAvgPool2dConfig {
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }
}

/// Adaptive 2-D average pooling to a fixed output size.
#[derive(Module, Clone)]
#[module(display = "display")]
pub struct AdaptiveAvgPool2d<Dev: Device> {
    #[module(skip)]
    pub output_size: (usize, usize),
    #[module(skip)]
    _marker: PhantomData<Dev>,
}

impl<Dev: Device> AdaptiveAvgPool2d<Dev> {
    pub fn new(output_size: (usize, usize)) -> Self {
        Self::from_config(&AdaptiveAvgPool2dConfig::new(output_size))
    }

    pub fn from_config(config: &AdaptiveAvgPool2dConfig) -> Self {
        Self { output_size: config.output_size, _marker: PhantomData }
    }

    pub fn display(&self) -> String {
        format!("output_size={:?}", self.output_size)
    }

    #[inline]
    pub fn forward(&self, input: &Tensor<Dev>) -> NnResult<Tensor<Dev>> {
        F::adaptive_avg_pool2d(input, self.output_size)
    }
}
