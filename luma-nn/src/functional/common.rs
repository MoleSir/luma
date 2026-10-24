use luma_tensor::{Device, Dim, Float, Int, Tensor};

use crate::NnResult;

/// Applies a linear transformation: `input @ weight^T + bias`.
///
/// - `input` shape: `(..., in_features)`
/// - `weight` shape: `(out_features, in_features)`
/// - `bias` shape: `(out_features,)`
pub fn linear<D: Device>(
    input: &Tensor<D, Float>,
    weight: &Tensor<D, Float>,
    bias: Option<&Tensor<D, Float>>,
) -> NnResult<Tensor<D, Float>> {
    Ok(input.linear(weight, bias)?)
}

/// Lookup embeddings for integer indices.
///
/// `weight` shape: `(num_embeddings, embedding_dim)`.
/// `indices` shape: `(...)` — arbitrary.
/// Returns shape: `(..., embedding_dim)`.
pub fn embedding<D: Device>(weight: &Tensor<D, Float>, indices: &Tensor<D, Int>) -> NnResult<Tensor<D, Float>> {
    Ok(weight.embedding(indices)?)
}

/// During training, randomly zeroes elements with probability `p` and scales
/// the remainder by `1/(1-p)`. During evaluation this is a no-op.
pub fn dropout<D: Device>(input: &Tensor<D, Float>, p: f64, training: bool) -> NnResult<Tensor<D, Float>> {
    Ok(input.dropout(p, training)?)
}

/// Softmax along `dim`.
pub fn softmax<D: Device, Dm: Dim>(input: &Tensor<D, Float>, dim: Dm) -> NnResult<Tensor<D, Float>> {
    Ok(input.softmax(dim)?)
}
