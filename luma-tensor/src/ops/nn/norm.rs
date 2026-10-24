use super::{NnOpDefault, NormBackward};
use crate::{Device, Float, Result, Shape, Tensor};

impl NnOpDefault {
    // ------------------------------------------------------------------
    // RMSNorm
    // ------------------------------------------------------------------

    /// RMSNorm over the last dim: `x / sqrt(mean(x^2) + eps) * weight + bias`.
    ///
    /// `weight` / `bias` are the optional elementwise affine parameters.
    pub fn rms_norm_forward<D: Device>(
        x: &Tensor<D, Float>,
        weight: Option<&Tensor<D, Float>>,
        bias: Option<&Tensor<D, Float>>,
        eps: f64,
    ) -> Result<Tensor<D, Float>> {
        let dim = last_dim(x)?;
        let mean_sq = x.sqr()?.mean_keepdim(dim)?;
        let rms = mean_sq.add_scalar(eps)?.sqrt()?;
        let mut y = x.broadcast_div(&rms)?;
        if let Some(weight) = weight {
            y = y.broadcast_mul(weight)?;
        }
        if let Some(bias) = bias {
            y = y.broadcast_add(bias)?;
        }
        Ok(y)
    }

    /// Backward of [`Self::rms_norm_forward`].
    ///
    /// Returns `(dx, dweight, dbias)`; the affine gradients are `None` when
    /// the corresponding forward argument was `None`. `weight` / `bias` are
    /// expected to be 1-D over the last dim.
    pub fn rms_norm_backward<D: Device>(
        x: &Tensor<D, Float>,
        weight: Option<&Tensor<D, Float>>,
        bias: Option<&Tensor<D, Float>>,
        y: &Tensor<D, Float>,
        grad: &Tensor<D, Float>,
        eps: f64,
    ) -> Result<NormBackward<D>> {
        let dim = last_dim(x)?;
        let n = x.dims()[dim] as f64;

        // r = sqrt(mean(x^2) + eps), x_hat = x / r, z = w * x_hat
        let mean_sq = x.sqr()?.mean_keepdim(dim)?;
        let r = mean_sq.add_scalar(eps)?.sqrt()?;
        let x_hat = x.broadcast_div(&r)?;
        let z = match bias {
            Some(bias) => y.broadcast_sub(bias)?,
            None => y.clone(),
        };

        // dx = (g * w - x_hat * mean(g * z)) / r
        let s = grad.mul(&z)?.sum_keepdim(dim)?.div_scalar(n)?;
        let gw = match weight {
            Some(weight) => grad.broadcast_mul(weight)?,
            None => grad.clone(),
        };
        let dx = gw.sub(&x_hat.broadcast_mul(&s)?)?.broadcast_div(&r)?;

        let dweight = match weight {
            Some(_) => Some(reduce_leading(&grad.mul(&x_hat)?, dim)?),
            None => None,
        };
        let dbias = match bias {
            Some(_) => Some(reduce_leading(grad, dim)?),
            None => None,
        };

        Ok((dx, dweight, dbias))
    }

    // ------------------------------------------------------------------
    // LayerNorm
    // ------------------------------------------------------------------

    /// Layer normalization over the last dimension.
    ///
    /// `x`: `(..., normalized)`; `weight` / `bias`: `(normalized,)` or `None`.
    pub fn layer_norm_forward<D: Device>(
        x: &Tensor<D, Float>,
        weight: Option<&Tensor<D, Float>>,
        bias: Option<&Tensor<D, Float>>,
        eps: f64,
    ) -> Result<Tensor<D, Float>> {
        let dim = last_dim(x)?;
        let mean = x.mean_keepdim(dim)?;
        let var = x.var_keepdim(dim)?;
        let r = var.add_scalar(eps)?.sqrt()?;
        let mut y = x.broadcast_sub(&mean)?.broadcast_div(&r)?;
        if let Some(weight) = weight {
            y = y.broadcast_mul(weight)?;
        }
        if let Some(bias) = bias {
            y = y.broadcast_add(bias)?;
        }
        Ok(y)
    }

    /// Backward of [`Self::layer_norm_forward`].
    ///
    /// Returns `(dx, dweight, dbias)`; the latter two are `None` when the
    /// corresponding forward argument was `None`.
    pub fn layer_norm_backward<D: Device>(
        x: &Tensor<D, Float>,
        weight: Option<&Tensor<D, Float>>,
        bias: Option<&Tensor<D, Float>>,
        _y: &Tensor<D, Float>,
        grad: &Tensor<D, Float>,
        eps: f64,
    ) -> Result<NormBackward<D>> {
        let dim = last_dim(x)?;
        let mean = x.mean_keepdim(dim)?;
        let var = x.var_keepdim(dim)?;
        let r = var.add_scalar(eps)?.sqrt()?;
        let x_hat = x.broadcast_sub(&mean)?.broadcast_div(&r)?;

        let z = match weight {
            Some(weight) => grad.broadcast_mul(weight)?,
            None => grad.clone(),
        };
        let mean_z = z.mean_keepdim(dim)?;
        let mean_zx = z.mul(&x_hat)?.mean_keepdim(dim)?;
        let dx = z
            .broadcast_sub(&mean_z)?
            .broadcast_sub(&x_hat.broadcast_mul(&mean_zx)?)?
            .broadcast_div(&r)?;

        let dweight = match weight {
            Some(_) => Some(reduce_leading(&grad.mul(&x_hat)?, dim)?),
            None => None,
        };
        let dbias = match bias {
            Some(_) => Some(reduce_leading(grad, dim)?),
            None => None,
        };

        Ok((dx, dweight, dbias))
    }

    // ------------------------------------------------------------------
    // GroupNorm
    // ------------------------------------------------------------------

    /// Group normalization over `(C/G, ...)`.
    ///
    /// `x`: `(N, C, ...)`; `weight` / `bias`: `(C,)` or `None`.
    pub fn group_norm_forward<D: Device>(
        x: &Tensor<D, Float>,
        num_groups: usize,
        weight: Option<&Tensor<D, Float>>,
        bias: Option<&Tensor<D, Float>>,
        eps: f64,
    ) -> Result<Tensor<D, Float>> {
        let (n, c, groups, group_elems) = group_geometry(x, num_groups)?;
        let xg = x.contiguous()?.reshape((n, groups, group_elems))?;
        let mean = xg.mean_keepdim(2)?;
        let var = xg.var_keepdim(2)?;
        let r = var.add_scalar(eps)?.sqrt()?;
        let mut y = xg.broadcast_sub(&mean)?.broadcast_div(&r)?.reshape(x.shape().clone())?;

        if let Some(weight) = weight {
            y = y.broadcast_mul(&weight.reshape(affine_shape(x.rank(), c))?)?;
        }
        if let Some(bias) = bias {
            y = y.broadcast_add(&bias.reshape(affine_shape(x.rank(), c))?)?;
        }
        Ok(y)
    }

    /// Backward of [`Self::group_norm_forward`].
    ///
    /// Returns `(dx, dweight, dbias)`; the latter two are `None` when the
    /// corresponding forward argument was `None`.
    pub fn group_norm_backward<D: Device>(
        x: &Tensor<D, Float>,
        num_groups: usize,
        weight: Option<&Tensor<D, Float>>,
        bias: Option<&Tensor<D, Float>>,
        _y: &Tensor<D, Float>,
        grad: &Tensor<D, Float>,
        eps: f64,
    ) -> Result<NormBackward<D>> {
        let (n, c, groups, group_elems) = group_geometry(x, num_groups)?;
        let rank = x.rank();

        let gw = match weight {
            Some(weight) => grad.broadcast_mul(&weight.reshape(affine_shape(rank, c))?)?,
            None => grad.clone(),
        };

        let xg = x.contiguous()?.reshape((n, groups, group_elems))?;
        let zg = gw.contiguous()?.reshape((n, groups, group_elems))?;
        let mean = xg.mean_keepdim(2)?;
        let var = xg.var_keepdim(2)?;
        let r = var.add_scalar(eps)?.sqrt()?;
        let x_hat = xg.broadcast_sub(&mean)?.broadcast_div(&r)?;

        let mean_z = zg.mean_keepdim(2)?;
        let mean_zx = zg.mul(&x_hat)?.mean_keepdim(2)?;
        let dxg = zg
            .broadcast_sub(&mean_z)?
            .broadcast_sub(&x_hat.broadcast_mul(&mean_zx)?)?
            .broadcast_div(&r)?;
        let dx = dxg.reshape(x.shape().clone())?;

        // Parameter gradients sum over N and every spatial axis.
        let reduce_dims: Vec<usize> = (0..rank).filter(|&d| d != 1).collect();
        let x_hat_full = x_hat.reshape(x.shape().clone())?;
        let dweight = match weight {
            Some(_) => Some(grad.mul(&x_hat_full)?.sum_dims(reduce_dims.as_slice(), false)?),
            None => None,
        };
        let dbias = match bias {
            Some(_) => Some(grad.sum_dims(reduce_dims.as_slice(), false)?),
            None => None,
        };

        Ok((dx, dweight, dbias))
    }
}

fn last_dim<D: Device>(x: &Tensor<D, Float>) -> Result<usize> {
    x.rank()
        .checked_sub(1)
        .ok_or_else(|| crate::Error::Msg("expected input rank >= 1".into()))
}

/// Sum `t` over every axis except the last (`dim`), yielding a 1-D parameter
/// gradient of shape `(t.dims()[dim],)`.
fn reduce_leading<D: Device>(t: &Tensor<D, Float>, dim: usize) -> Result<Tensor<D, Float>> {
    if t.rank() > 1 {
        let leading: Vec<usize> = (0..dim).collect();
        t.sum_dims(leading.as_slice(), false)
    } else {
        Ok(t.clone())
    }
}

/// Shape `[1, C, 1, ..., 1]` used to broadcast per-channel affine parameters
/// over `(N, C, ...)`.
fn affine_shape(rank: usize, channels: usize) -> Shape {
    let mut dims = vec![1usize; rank];
    dims[1] = channels;
    Shape::from(dims)
}

fn group_geometry<D: Device>(x: &Tensor<D, Float>, num_groups: usize) -> Result<(usize, usize, usize, usize)> {
    let rank = x.rank();
    if rank < 2 {
        return Err(crate::Error::Msg("group_norm: expected rank >= 2".into()));
    }
    let n = x.dims()[0];
    let c = x.dims()[1];
    if num_groups == 0 || !c.is_multiple_of(num_groups) {
        return Err(crate::Error::Msg("group_norm: num_groups must divide channels".into()));
    }
    let spatial: usize = x.dims()[2..].iter().product();
    Ok((n, c, num_groups, (c / num_groups) * spatial))
}
