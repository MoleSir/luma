use super::NnOpDefault;
use crate::{Device, Float, Result, Tensor};

impl NnOpDefault {
    // ------------------------------------------------------------------
    // MaxPool2d
    // ------------------------------------------------------------------

    /// 2-D max pooling.
    ///
    /// `x`: `(N, C, H, W)`; output: `(N, C, H_out, W_out)`. Padding is filled
    /// with `-inf`, matching PyTorch's `max_pool2d`.
    pub fn max_pool2d_forward<D: Device>(
        x: &Tensor<D, Float>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        ceil_mode: bool,
    ) -> Result<Tensor<D, Float>> {
        let (n, c, h, w) = x.dims4()?;
        let geom = PoolGeometry::new(h, w, kernel_size, stride, padding, dilation, ceil_mode, "max_pool2d")?;
        let x_pad = pad2d_asym(x, padding.0, padding.0 + geom.extra_h, padding.1, padding.1 + geom.extra_w, f64::NEG_INFINITY)?;

        let mut out = window(&x_pad, 0, 0, stride, dilation, geom.h_out, geom.w_out)?;
        for i in 0..kernel_size.0 {
            for j in 0..kernel_size.1 {
                if i == 0 && j == 0 {
                    continue;
                }
                let v = window(&x_pad, i, j, stride, dilation, geom.h_out, geom.w_out)?;
                out = out.maximum(&v)?;
            }
        }
        debug_assert_eq!(out.dims(), &[n, c, geom.h_out, geom.w_out]);
        Ok(out)
    }

    /// Backward of [`Self::max_pool2d_forward`].
    ///
    /// Gradients are routed to every element equal to the window maximum. When
    /// several elements tie, the gradient is split evenly between them.
    pub fn max_pool2d_backward<D: Device>(
        x: &Tensor<D, Float>,
        y: &Tensor<D, Float>,
        grad: &Tensor<D, Float>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        ceil_mode: bool,
    ) -> Result<Tensor<D, Float>> {
        let (n, c, h, w) = x.dims4()?;
        let geom = PoolGeometry::new(h, w, kernel_size, stride, padding, dilation, ceil_mode, "max_pool2d")?;
        let x_pad = pad2d_asym(x, padding.0, padding.0 + geom.extra_h, padding.1, padding.1 + geom.extra_w, f64::NEG_INFINITY)?;
        let counts = Tensor::<D, Float>::zeros((n, c, geom.h_out, geom.w_out), (x.device(), x.dtype()))?;
        for i in 0..kernel_size.0 {
            for j in 0..kernel_size.1 {
                let v = window(&x_pad, i, j, stride, dilation, geom.h_out, geom.w_out)?;
                let m = y.eq(&v)?.cast_float(y.dtype())?;
                counts.add_(&m)?;
            }
        }

        let dx_pad = Tensor::<D, Float>::zeros(x_pad.shape().clone(), (x.device(), x.dtype()))?;
        for i in 0..kernel_size.0 {
            for j in 0..kernel_size.1 {
                let v = window(&x_pad, i, j, stride, dilation, geom.h_out, geom.w_out)?;
                let m = y.eq(&v)?.cast_float(y.dtype())?;
                let contrib = grad.mul(&m)?.div(&counts)?;
                dx_pad
                    .slice(2, i * dilation.0, i * dilation.0 + stride.0 * (geom.h_out - 1) + 1, stride.0)?
                    .slice(3, j * dilation.1, j * dilation.1 + stride.1 * (geom.w_out - 1) + 1, stride.1)?
                    .add_(&contrib)?;
            }
        }

        crop(&dx_pad, padding, h, w)
    }

    // ------------------------------------------------------------------
    // AvgPool2d
    // ------------------------------------------------------------------

    /// 2-D average pooling.
    ///
    /// `x`: `(N, C, H, W)`; output: `(N, C, H_out, W_out)`.
    pub fn avg_pool2d_forward<D: Device>(
        x: &Tensor<D, Float>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Result<Tensor<D, Float>> {
        let (n, c, h, w) = x.dims4()?;
        let geom = PoolGeometry::new(h, w, kernel_size, stride, padding, (1, 1), ceil_mode, "avg_pool2d")?;
        let x_pad = pad2d_asym(x, padding.0, padding.0 + geom.extra_h, padding.1, padding.1 + geom.extra_w, 0.0)?;

        let mut sum = window(&x_pad, 0, 0, stride, (1, 1), geom.h_out, geom.w_out)?;
        for i in 0..kernel_size.0 {
            for j in 0..kernel_size.1 {
                if i == 0 && j == 0 {
                    continue;
                }
                sum = sum.add(&window(&x_pad, i, j, stride, (1, 1), geom.h_out, geom.w_out)?)?;
            }
        }

        let denom = avg_denominator(x, &geom, kernel_size, stride, padding, count_include_pad)?;
        let out = match denom {
            AvgDenominator::Scalar(d) => sum.div_scalar(d)?,
            AvgDenominator::Tensor(d) => sum.broadcast_div(&d)?,
        };
        debug_assert_eq!(out.dims(), &[n, c, geom.h_out, geom.w_out]);
        Ok(out)
    }

    /// Backward of [`Self::avg_pool2d_forward`].
    pub fn avg_pool2d_backward<D: Device>(
        x: &Tensor<D, Float>,
        _y: &Tensor<D, Float>,
        grad: &Tensor<D, Float>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Result<Tensor<D, Float>> {
        let (n, c, h, w) = x.dims4()?;
        let geom = PoolGeometry::new(h, w, kernel_size, stride, padding, (1, 1), ceil_mode, "avg_pool2d")?;
        let denom = avg_denominator(x, &geom, kernel_size, stride, padding, count_include_pad)?;
        let step = match denom {
            AvgDenominator::Scalar(d) => grad.div_scalar(d)?,
            AvgDenominator::Tensor(d) => grad.broadcast_div(&d)?,
        };

        let dx_pad = Tensor::<D, Float>::zeros((n, c, geom.h_full, geom.w_full), (x.device(), x.dtype()))?;
        for i in 0..kernel_size.0 {
            for j in 0..kernel_size.1 {
                dx_pad
                    .slice(2, i, i + stride.0 * (geom.h_out - 1) + 1, stride.0)?
                    .slice(3, j, j + stride.1 * (geom.w_out - 1) + 1, stride.1)?
                    .add_(&step)?;
            }
        }

        crop(&dx_pad, padding, h, w)
    }

    // ------------------------------------------------------------------
    // AdaptiveAvgPool2d
    // ------------------------------------------------------------------

    /// Adaptive average pooling to `output_size`.
    ///
    /// Windows are `[floor(i * H / H_out), ceil((i + 1) * H / H_out))`, so they
    /// may overlap when the output is larger than the input.
    pub fn adaptive_avg_pool2d_forward<D: Device>(x: &Tensor<D, Float>, output_size: (usize, usize)) -> Result<Tensor<D, Float>> {
        let (n, c, h, w) = x.dims4()?;
        if output_size.0 == 0 || output_size.1 == 0 {
            return Err(crate::Error::Msg("adaptive_avg_pool2d: output size must be positive".into()));
        }
        let (oh, ow) = output_size;
        let mut rows: Vec<Tensor<D, Float>> = Vec::with_capacity(oh);
        for i in 0..oh {
            let h0 = i * h / oh;
            let h1 = ((i + 1) * h + oh - 1) / oh;
            let mut cols: Vec<Tensor<D, Float>> = Vec::with_capacity(ow);
            for j in 0..ow {
                let w0 = j * w / ow;
                let w1 = ((j + 1) * w + ow - 1) / ow;
                let v = x.narrow(2, h0, h1 - h0)?.narrow(3, w0, w1 - w0)?.mean_keepdim(2)?.mean_keepdim(3)?;
                cols.push(v);
            }
            let refs: Vec<&Tensor<D, Float>> = cols.iter().collect();
            rows.push(Tensor::cat(&refs, 3)?);
        }
        let refs: Vec<&Tensor<D, Float>> = rows.iter().collect();
        let out = Tensor::cat(&refs, 2)?;
        debug_assert_eq!(out.dims(), &[n, c, oh, ow]);
        Ok(out)
    }

    /// Backward of [`Self::adaptive_avg_pool2d_forward`].
    pub fn adaptive_avg_pool2d_backward<D: Device>(
        x: &Tensor<D, Float>,
        _y: &Tensor<D, Float>,
        grad: &Tensor<D, Float>,
        output_size: (usize, usize),
    ) -> Result<Tensor<D, Float>> {
        let (n, c, h, w) = x.dims4()?;
        if output_size.0 == 0 || output_size.1 == 0 {
            return Err(crate::Error::Msg("adaptive_avg_pool2d: output size must be positive".into()));
        }
        let (oh, ow) = output_size;
        let dx = Tensor::<D, Float>::zeros((n, c, h, w), (x.device(), x.dtype()))?;
        for i in 0..oh {
            let h0 = i * h / oh;
            let h1 = ((i + 1) * h + oh - 1) / oh;
            for j in 0..ow {
                let w0 = j * w / ow;
                let w1 = ((j + 1) * w + ow - 1) / ow;
                let area = ((h1 - h0) * (w1 - w0)).max(1) as f64;
                let g = grad.narrow(2, i, 1)?.narrow(3, j, 1)?.div_scalar(area)?;
                let contrib = g.broadcast_as((n, c, h1 - h0, w1 - w0))?;
                dx.slice(2, h0, h1, 1)?.slice(3, w0, w1, 1)?.add_(&contrib)?;
            }
        }
        Ok(dx)
    }
}

// ============================================================================
//   Geometry and helpers
// ============================================================================

struct PoolGeometry {
    h_out: usize,
    w_out: usize,
    extra_h: usize,
    extra_w: usize,
    h_full: usize,
    w_full: usize,
}

impl PoolGeometry {
    #[allow(clippy::too_many_arguments)]
    fn new(
        h: usize,
        w: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        ceil_mode: bool,
        op: &'static str,
    ) -> Result<Self> {
        if kernel.0 == 0 || kernel.1 == 0 {
            return Err(crate::Error::Msg(format!("{op}: kernel size must be positive")));
        }
        if stride.0 == 0 || stride.1 == 0 {
            return Err(crate::Error::Msg(format!("{op}: stride must be positive")));
        }
        if dilation.0 == 0 || dilation.1 == 0 {
            return Err(crate::Error::Msg(format!("{op}: dilation must be positive")));
        }

        let effective_h = dilation.0 * (kernel.0 - 1) + 1;
        let effective_w = dilation.1 * (kernel.1 - 1) + 1;
        let padded_h = h + 2 * padding.0;
        let padded_w = w + 2 * padding.1;
        if padded_h < effective_h || padded_w < effective_w {
            return Err(crate::Error::Msg(format!("{op}: padded input is smaller than the effective kernel")));
        }

        let h_out = pool_out_size(h, padding.0, effective_h, stride.0, ceil_mode);
        let w_out = pool_out_size(w, padding.1, effective_w, stride.1, ceil_mode);
        if h_out == 0 || w_out == 0 {
            return Err(crate::Error::Msg(format!("{op}: output size is zero")));
        }

        let needed_h = effective_h + stride.0 * (h_out - 1);
        let needed_w = effective_w + stride.1 * (w_out - 1);
        let extra_h = needed_h.saturating_sub(padded_h);
        let extra_w = needed_w.saturating_sub(padded_w);

        Ok(Self { h_out, w_out, extra_h, extra_w, h_full: padded_h + extra_h, w_full: padded_w + extra_w })
    }
}

fn pool_out_size(input: usize, padding: usize, effective: usize, stride: usize, ceil_mode: bool) -> usize {
    let padded = input + 2 * padding;
    let round = if ceil_mode { stride - 1 } else { 0 };
    let mut out = (padded - effective + round) / stride + 1;
    if ceil_mode && (out - 1) * stride >= input + padding {
        out -= 1;
    }
    out
}

fn window<D: Device>(
    x: &Tensor<D, Float>,
    i: usize,
    j: usize,
    stride: (usize, usize),
    dilation: (usize, usize),
    h_out: usize,
    w_out: usize,
) -> Result<Tensor<D, Float>> {
    let h0 = i * dilation.0;
    let w0 = j * dilation.1;
    let h1 = h0 + stride.0 * (h_out - 1) + 1;
    let w1 = w0 + stride.1 * (w_out - 1) + 1;
    x.slice(2, h0, h1, stride.0)?.slice(3, w0, w1, stride.1)
}

enum AvgDenominator<D: Device> {
    Scalar(f64),
    Tensor(Tensor<D, Float>),
}

#[allow(clippy::too_many_arguments)]
fn avg_denominator<D: Device>(
    x: &Tensor<D, Float>,
    geom: &PoolGeometry,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    count_include_pad: bool,
) -> Result<AvgDenominator<D>> {
    let count = (kernel.0 * kernel.1) as f64;
    if (count_include_pad || padding == (0, 0)) && geom.extra_h == 0 && geom.extra_w == 0 {
        return Ok(AvgDenominator::Scalar(count));
    }

    // A separable 0/1 mask over the padded input. `count_include_pad` counts the
    // original padding but never the extra cells added for `ceil_mode`.
    let (top, left, allow_h, allow_w) = if count_include_pad {
        (0, 0, geom.h_full - geom.extra_h, geom.w_full - geom.extra_w)
    } else {
        let (_, _, h, w) = x.dims4()?;
        (padding.0, padding.1, h, w)
    };
    let bottom = geom.h_full - top - allow_h;
    let right = geom.w_full - left - allow_w;
    let ones = Tensor::<D, Float>::ones((1, 1, allow_h, allow_w), (x.device(), x.dtype()))?;
    let mask = pad2d_asym(&ones, top, bottom, left, right, 0.0)?;

    let counts = Tensor::<D, Float>::zeros((1, 1, geom.h_out, geom.w_out), (x.device(), x.dtype()))?;
    for i in 0..kernel.0 {
        for j in 0..kernel.1 {
            let v = window(&mask, i, j, stride, (1, 1), geom.h_out, geom.w_out)?;
            counts.add_(&v)?;
        }
    }
    Ok(AvgDenominator::Tensor(counts))
}

fn crop<D: Device>(padded: &Tensor<D, Float>, padding: (usize, usize), h: usize, w: usize) -> Result<Tensor<D, Float>> {
    if padding == (0, 0) && padded.dims()[2] == h && padded.dims()[3] == w {
        Ok(padded.clone())
    } else {
        Ok(padded.slice(2, padding.0, padding.0 + h, 1)?.slice(3, padding.1, padding.1 + w, 1)?.contiguous()?)
    }
}

fn pad2d_asym<D: Device>(
    x: &Tensor<D, Float>,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
    value: f64,
) -> Result<Tensor<D, Float>> {
    if top == 0 && bottom == 0 && left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let (n, c, _, _) = x.dims4()?;
    let mut y = x.clone();
    if top > 0 {
        let pad = Tensor::<D, Float>::full((n, c, top, x.dims()[3]), value, (x.device(), x.dtype()))?;
        y = Tensor::cat(&[&pad, &y], 2)?;
    }
    if bottom > 0 {
        let (_, _, _, w) = y.dims4()?;
        let pad = Tensor::<D, Float>::full((n, c, bottom, w), value, (x.device(), x.dtype()))?;
        y = Tensor::cat(&[&y, &pad], 2)?;
    }
    if left > 0 {
        let (_, _, h, _) = y.dims4()?;
        let pad = Tensor::<D, Float>::full((n, c, h, left), value, (x.device(), x.dtype()))?;
        y = Tensor::cat(&[&pad, &y], 3)?;
    }
    if right > 0 {
        let (_, _, h, _) = y.dims4()?;
        let pad = Tensor::<D, Float>::full((n, c, h, right), value, (x.device(), x.dtype()))?;
        y = Tensor::cat(&[&y, &pad], 3)?;
    }
    Ok(y)
}
