use luma_tensor::{Device, Float, Tensor};
use crate::NnResult;

pub fn conv2d<D: Device>(
    x: &Tensor<D, Float>,
    weight: &Tensor<D, Float>,
    bias: Option<&Tensor<D, Float>>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> NnResult<Tensor<D, Float>> {
    Ok(x.conv2d(weight, bias, stride, padding, dilation)?)
}