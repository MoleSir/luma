use super::NnOpDefault;
use crate::{Device, Float, Result, Tensor};

impl NnOpDefault {
    /// Numerically-stable log-sum-exp along `dim`.
    pub fn logsumexp_forward<D: Device>(x: &Tensor<D, Float>, dim: usize, keepdim: bool) -> Result<Tensor<D, Float>> {
        let m = x.max_keepdim(dim)?;
        let e = x.broadcast_sub(&m)?.exp()?;
        let lse = e.sum_keepdim(dim)?.ln()?.add(&m)?;
        // `keepdim=false` is just a view-level squeeze; the reduction always
        // runs with the reduced axis kept.
        if keepdim { Ok(lse) } else { lse.squeeze(dim) }
    }

    /// Softmax along `dim`.
    ///
    /// Softmax preserves the axis by definition; when `keepdim` is `false` a
    /// size-1 axis is removed with a view-level squeeze.
    pub fn softmax_forward<D: Device>(x: &Tensor<D, Float>, dim: usize, keepdim: bool) -> Result<Tensor<D, Float>> {
        let lse = x.logsumexp_keepdim(dim)?;
        let y = x.broadcast_sub(&lse)?.exp()?;
        if keepdim { Ok(y) } else { y.squeeze(dim) }
    }

    /// Backward of [`Self::softmax_forward`].
    ///
    /// `y` is the softmax output and `grad` the upstream gradient:
    /// `dx = y * (grad - sum(grad * y, dim, keepdim))`.
    /// When `keepdim` is `false` the squeezed axis is restored with `unsqueeze`.
    pub fn softmax_backward<D: Device>(
        y: &Tensor<D, Float>,
        grad: &Tensor<D, Float>,
        dim: usize,
        keepdim: bool,
    ) -> Result<Tensor<D, Float>> {
        let (y, grad) = if keepdim {
            (y.clone(), grad.clone())
        } else {
            (y.unsqueeze(dim)?, grad.unsqueeze(dim)?)
        };
        let gy = grad.mul(&y)?;
        let s = gy.sum_keepdim(dim)?;
        y.mul(&grad.sub(&s.broadcast_as(grad.shape().clone())?)?)
    }

    /// Log-softmax along `dim`: `x - logsumexp(x, dim)`.
    ///
    /// Like softmax, the axis is preserved; `keepdim=false` only squeezes a
    /// size-1 axis from the result.
    pub fn log_softmax_forward<D: Device>(x: &Tensor<D, Float>, dim: usize, keepdim: bool) -> Result<Tensor<D, Float>> {
        let lse = x.logsumexp_keepdim(dim)?;
        let y = x.broadcast_sub(&lse)?;
        if keepdim { Ok(y) } else { y.squeeze(dim) }
    }
}
