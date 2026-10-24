use super::{Conv2dBackward, NnOpDefault};
use crate::{Device, Float, Result, Tensor};

impl NnOpDefault {
    /// 2-D convolution (no groups).
    ///
    /// `x`: `(N, C_in, H, W)`; `weight`: `(C_out, C_in, kH, kW)`;
    /// `bias`: `(C_out,)`; output: `(N, C_out, H_out, W_out)`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_forward<D: Device>(
        x: &Tensor<D, Float>,
        weight: &Tensor<D, Float>,
        bias: Option<&Tensor<D, Float>>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<Tensor<D, Float>> {
        let (n, c_in, h, w) = x.dims4()?;
        let (c_out, c_w, kh, kw) = weight.dims4()?;
        if c_w != c_in {
            return Err(crate::Error::Msg(format!(
                "conv2d: weight has {c_w} input channels, expected {c_in}"
            )));
        }
        let h_out = out_size(h, padding.0, dilation.0, kh, stride.0, "conv2d")?;
        let w_out = out_size(w, padding.1, dilation.1, kw, stride.1, "conv2d")?;

        let x_pad = pad2d(x, padding.0, padding.1)?;
        let patches = im2col(&x_pad, kh, kw, stride, dilation, h_out, w_out)?;
        let k = c_in * kh * kw;
        let l = h_out * w_out;

        // Flatten the kernel in im2col order: (C_out, kH, kW, C_in).
        let w_flat = weight
            .permute([0usize, 2, 3, 1])?
            .contiguous()?
            .reshape((c_out, k))?;

        let out = patches
            .contiguous()?
            .reshape((n * l, k))?
            .matmul(&w_flat.transpose(0, 1)?)?
            .reshape((n, l, c_out))?;
        let mut out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((n, c_out, h_out, w_out))?;

        if let Some(bias) = bias {
            out = out.broadcast_add(&bias.reshape((1, c_out, 1, 1))?)?;
        }
        Ok(out)
    }

    /// Backward of [`Self::conv2d_forward`].
    ///
    /// Returns `(dx, dweight, dbias)`; `dbias` is `None` when the forward bias
    /// was `None`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_backward<D: Device>(
        x: &Tensor<D, Float>,
        weight: &Tensor<D, Float>,
        bias: Option<&Tensor<D, Float>>,
        _y: &Tensor<D, Float>,
        grad: &Tensor<D, Float>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<Conv2dBackward<D>> {
        let (n, c_in, h, w) = x.dims4()?;
        let (c_out, _, kh, kw) = weight.dims4()?;
        let h_out = out_size(h, padding.0, dilation.0, kh, stride.0, "conv2d")?;
        let w_out = out_size(w, padding.1, dilation.1, kw, stride.1, "conv2d")?;
        let l = h_out * w_out;
        let k = c_in * kh * kw;

        let x_pad = pad2d(x, padding.0, padding.1)?;
        let patches = im2col(&x_pad, kh, kw, stride, dilation, h_out, w_out)?; // (N,L,K)
        let w_flat = weight
            .permute([0usize, 2, 3, 1])?
            .contiguous()?
            .reshape((c_out, k))?;

        let g = grad
            .contiguous()?
            .reshape((n, c_out, l))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((n * l, c_out))?;
        let p = patches.contiguous()?.reshape((n * l, k))?;

        let dweight_flat = g.transpose(0, 1)?.matmul(&p)?; // (C_out, K)
        let dweight = dweight_flat
            .reshape((c_out, kh, kw, c_in))?
            .permute([0usize, 3, 1, 2])?
            .contiguous()?;

        let dpatches = g.matmul(&w_flat)?.reshape((n, l, k))?;
        let dx_pad = Tensor::<D, Float>::zeros(x_pad.shape().clone(), (x.device(), x.dtype()))?;
        for i in 0..kh {
            for j in 0..kw {
                let t = i * kw + j;
                let d = dpatches
                    .narrow(2, t * c_in, c_in)?
                    .transpose(1, 2)?
                    .contiguous()?
                    .reshape((n, c_in, h_out, w_out))?;
                let h0 = i * dilation.0;
                let w0 = j * dilation.1;
                let h1 = h0 + stride.0 * (h_out - 1) + 1;
                let w1 = w0 + stride.1 * (w_out - 1) + 1;
                let target = dx_pad.slice(2, h0, h1, stride.0)?.slice(3, w0, w1, stride.1)?;
                target.add_(&d)?;
            }
        }

        let dx = if padding.0 == 0 && padding.1 == 0 {
            dx_pad
        } else {
            dx_pad
                .slice(2, padding.0, padding.0 + h, 1)?
                .slice(3, padding.1, padding.1 + w, 1)?
                .contiguous()?
        };

        let dbias = match bias {
            Some(_) => Some(grad.sum_dims([0usize, 2, 3], false)?),
            None => None,
        };

        Ok((dx, dweight, dbias))
    }
}

fn out_size(in_size: usize, padding: usize, dilation: usize, kernel: usize, stride: usize, op: &'static str) -> Result<usize> {
    let padded = in_size + 2 * padding;
    let effective = dilation
        .checked_mul(kernel.saturating_sub(1))
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| crate::Error::Msg(format!("{op}: kernel geometry overflow")))?;
    if padded < effective {
        return Err(crate::Error::Msg(format!("{op}: input is smaller than the effective kernel")));
    }
    Ok((padded - effective) / stride + 1)
}

fn pad2d<D: Device>(x: &Tensor<D, Float>, ph: usize, pw: usize) -> Result<Tensor<D, Float>> {
    if ph == 0 && pw == 0 {
        return Ok(x.clone());
    }
    let (n, c, _, w) = x.dims4()?;
    let mut y = x.clone();
    if ph > 0 {
        let zeros = Tensor::<D, Float>::zeros((n, c, ph, w), (x.device(), x.dtype()))?;
        y = Tensor::cat(&[&zeros, &y, &zeros], 2)?;
    }
    if pw > 0 {
        let (n, c, h, _) = y.dims4()?;
        let zeros = Tensor::<D, Float>::zeros((n, c, h, pw), (y.device(), y.dtype()))?;
        y = Tensor::cat(&[&zeros, &y, &zeros], 3)?;
    }
    Ok(y)
}

/// `(N, C, H, W)` -> `(N, H_out * W_out, C * kH * kW)` in `(kh, kw, c)` order.
fn im2col<D: Device>(
    x_pad: &Tensor<D, Float>,
    kh: usize,
    kw: usize,
    stride: (usize, usize),
    dilation: (usize, usize),
    h_out: usize,
    w_out: usize,
) -> Result<Tensor<D, Float>> {
    let (n, c, _, _) = x_pad.dims4()?;
    let mut cols: Vec<Tensor<D, Float>> = Vec::with_capacity(kh * kw);
    for i in 0..kh {
        for j in 0..kw {
            let h0 = i * dilation.0;
            let w0 = j * dilation.1;
            let h1 = h0 + stride.0 * (h_out - 1) + 1;
            let w1 = w0 + stride.1 * (w_out - 1) + 1;
            let v = x_pad.slice(2, h0, h1, stride.0)?.slice(3, w0, w1, stride.1)?;
            let v = v
                .contiguous()?
                .reshape((n, c, h_out * w_out))?
                .transpose(1, 2)?;
            cols.push(v);
        }
    }
    let refs: Vec<&Tensor<D, Float>> = cols.iter().collect();
    Tensor::cat(&refs, 2)
}
