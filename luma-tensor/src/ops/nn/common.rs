use super::NnOpDefault;
use crate::{Device, Float, Int, Result, Tensor};

impl NnOpDefault {
    /// Applies a linear transformation: `input @ weight^T + bias`.
    ///
    /// `input`: `(..., in_features)`; `weight`: `(out_features, in_features)`;
    /// `bias`: `(out_features,)`.
    pub fn linear_forward<D: Device>(
        input: &Tensor<D, Float>,
        weight: &Tensor<D, Float>,
        bias: Option<&Tensor<D, Float>>,
    ) -> Result<Tensor<D, Float>> {
        let rank = input.rank();

        // `matmul` requires operands of equal rank; flatten leading dims of a
        // batched input so `(..., in_features) @ (out_features, in_features)^T`
        // reduces to a plain 2D matmul, then reshape the result back.
        let (flat_input, out_shape) = if rank > 2 {
            let flat = input.flatten(0, rank - 2)?;
            let mut shape = input.dims()[..rank - 1].to_vec();
            shape.push(weight.dims()[0]);
            (flat, Some(shape))
        } else {
            (input.clone(), None)
        };

        let mut output = flat_input.matmul(&weight.transpose_last()?)?;
        if let Some(b) = bias {
            output = output.broadcast_add(b)?;
        }
        if let Some(shape) = out_shape {
            output = output.reshape(shape)?;
        }
        Ok(output)
    }

    /// Lookup embeddings for integer indices.
    ///
    /// `weight`: `(num_embeddings, embedding_dim)`; `indices`: arbitrary shape.
    /// Returns `(..., embedding_dim)`.
    pub fn embedding_forward<D: Device>(weight: &Tensor<D, Float>, indices: &Tensor<D, Int>) -> Result<Tensor<D, Float>> {
        let out_dims = indices.element_count();
        let out = weight.index_select(&indices.reshape((out_dims,))?, 0)?;
        let mut shape = indices.dims().to_vec();
        shape.push(weight.dims()[1]);
        Ok(out.reshape(shape)?)
    }

    /// During training, randomly zeroes elements with probability `p` and
    /// scales the remainder by `1/(1-p)`. During evaluation this is a no-op.
    pub fn dropout_forward<D: Device>(input: &Tensor<D, Float>, p: f64, training: bool) -> Result<Tensor<D, Float>> {
        if !training || p == 0.0 {
            return Ok(input.clone());
        }
        let scale = 1.0 / (1.0 - p);
        let mask = input.rand_like(0.0, 1.0)?.gt_scalar(p)?; // true = keep
        let scaled = input.mul_scalar(scale)?;
        Ok(mask.pick_false(&scaled, 0.0)?)
    }
}
