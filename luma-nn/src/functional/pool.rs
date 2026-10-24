use luma_tensor::{Device, Float, Tensor};

use crate::NnResult;

pub fn max_pool2d<D: Device>(
    x: &Tensor<D, Float>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    ceil_mode: bool,
) -> NnResult<Tensor<D, Float>> {
    Ok(x.max_pool2d(kernel_size, stride, padding, dilation, ceil_mode)?)
}

pub fn avg_pool2d<D: Device>(
    x: &Tensor<D, Float>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    ceil_mode: bool,
    count_include_pad: bool,
) -> NnResult<Tensor<D, Float>> {
    Ok(x.avg_pool2d(kernel_size, stride, padding, ceil_mode, count_include_pad)?)
}

pub fn adaptive_avg_pool2d<D: Device>(x: &Tensor<D, Float>, output_size: (usize, usize)) -> NnResult<Tensor<D, Float>> {
    Ok(x.adaptive_avg_pool2d(output_size)?)
}
