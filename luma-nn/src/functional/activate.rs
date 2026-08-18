use luma_tensor::{Device, Float, Tensor};

use crate::NnResult;

pub fn relu<D: Device>(input: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
    Ok(input.relu()?)
}

pub fn leaky_relu<D: Device>(input: &Tensor<D, Float>, negative_slope: f64) -> NnResult<Tensor<D, Float>> {
    Ok(input.leaky_relu(negative_slope)?)
}

pub fn sigmoid<D: Device>(input: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
    Ok(input.sigmoid()?)
}

pub fn tanh<D: Device>(input: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
    Ok(input.tanh()?)
}

pub fn gelu<D: Device>(input: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
    Ok(input.gelu()?)
}

pub fn silu<D: Device>(input: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
    Ok(input.silu()?)
}

pub fn exp<D: Device>(input: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
    Ok(input.exp()?)
}

pub fn ln<D: Device>(input: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
    Ok(input.ln()?)
}

pub fn sqr<D: Device>(input: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
    Ok(input.sqr()?)
}

pub fn sqrt<D: Device>(input: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
    Ok(input.sqrt()?)
}
