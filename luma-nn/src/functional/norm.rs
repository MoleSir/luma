use crate::NnResult;
use luma_tensor::{Device, Tensor};

/// Applies Layer Normalization over a mini-batch of inputs.
///
/// $\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$
///
/// The mean and standard-deviation are calculated over the last dimension.
///
/// ## Arguments
///
/// * `input` - Input tensor of shape `(..., normalized_shape)`.
/// * `weight` - Optional scale tensor ($\gamma$) of shape `(normalized_shape)`.
/// * `bias` - Optional shift tensor ($\beta$) of shape `(normalized_shape)`.
/// * `eps` - A value added to the denominator for numerical stability.
pub fn layer_norm<Dev: Device>(
    input: &Tensor<Dev>,
    weight: Option<&Tensor<Dev>>,
    bias: Option<&Tensor<Dev>>,
    eps: f64,
) -> NnResult<Tensor<Dev>> {
    Ok(input.layer_norm(weight, bias, eps)?)
}

/// Applies Group Normalization over `(C/G, ...)` with `num_groups` groups.
///
/// ## Arguments
///
/// * `input` - Input tensor of shape `(N, C, ...)`.
/// * `num_groups` - Number of groups; must divide the channel count.
/// * `weight` - Optional per-channel scale tensor of shape `(C,)`.
/// * `bias` - Optional per-channel shift tensor of shape `(C,)`.
/// * `eps` - A value added to the denominator for numerical stability.
pub fn group_norm<Dev: Device>(
    input: &Tensor<Dev>,
    num_groups: usize,
    weight: Option<&Tensor<Dev>>,
    bias: Option<&Tensor<Dev>>,
    eps: f64,
) -> NnResult<Tensor<Dev>> {
    Ok(input.group_norm(num_groups, weight, bias, eps)?)
}

/// Applies Root Mean Square Normalization.
///
/// $\text{RMS}(x) = \frac{x}{\sqrt{\text{Mean}(x^2) + \epsilon}} \cdot \gamma$
///
/// Commonly used in LLMs (e.g., LLaMA). It normalizes inputs based on the root mean square
/// without centering the mean.
///
/// ## Arguments
///
/// * `input` - Input tensor.
/// * `weight` - Optional scale tensor ($\gamma$).
/// * `bias` - Optional shift tensor ($\beta$).
/// * `eps` - A value added to the denominator for numerical stability.
pub fn rms_norm<Dev: Device>(
    input: &Tensor<Dev>,
    weight: Option<&Tensor<Dev>>,
    bias: Option<&Tensor<Dev>>,
    eps: f64,
) -> NnResult<Tensor<Dev>> {
    Ok(input.rms_norm(weight, bias, eps)?)
}
