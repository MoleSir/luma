use super::{AttentionBackward, NnOpDefault};
use crate::{Bool, BoolDType, Device, Float, Int, IntDType, Result, Tensor};

impl NnOpDefault {
    /// Scaled dot-product attention.
    ///
    /// `q`: `(batch, q_len, q_heads, head_size)`;
    /// `k` / `v`: `(batch, kv_len, kv_heads, head_size)`;
    /// output: `(batch, q_len, q_heads, head_size)`.
    #[allow(clippy::too_many_arguments)]
    pub fn scaled_dot_product_attention_forward<D: Device>(
        q: &Tensor<D, Float>,
        k: &Tensor<D, Float>,
        v: &Tensor<D, Float>,
        scale: Option<f64>,
        causal: bool,
        start_pos: usize,
    ) -> Result<Tensor<D, Float>> {
        let (b, lq, hq, d) = q.dims4()?;
        let (bk, lk, hkv, dk) = k.dims4()?;
        let (bv, lv, hv, dv) = v.dims4()?;
        if bk != b || bv != b || lk != lv || hkv != hv || dk != d || dv != d {
            return Err(crate::Error::Msg("scaled_dot_product_attention: incompatible q/k/v shapes".into()));
        }
        if hkv == 0 || hq % hkv != 0 {
            return Err(crate::Error::Msg(
                "scaled_dot_product_attention: q_heads must be a multiple of kv_heads".into(),
            ));
        }
        let group = hq / hkv;

        let qh = q.transpose(1, 2)?.contiguous()?; // (B, Hq, Lq, D)
        let kh = expand_heads(k, group)?.transpose(1, 2)?.contiguous()?; // (B, Hq, Lk, D)
        let vh = expand_heads(v, group)?.transpose(1, 2)?.contiguous()?;

        let scale = scale.unwrap_or(1.0 / (d as f64).sqrt());
        let mut scores = qh.matmul(&kh.transpose(2, 3)?)?.mul_scalar(scale)?;
        if causal {
            let mask = causal_mask(q, lq, lk, start_pos)?;
            let mask = mask.broadcast_as(scores.shape().clone())?;
            scores = mask.pick_false(&scores, f64::NEG_INFINITY)?;
        }
        let attn = scores.softmax(3)?;
        attn.matmul(&vh)?.transpose(1, 2)
    }

    /// Backward of [`Self::scaled_dot_product_attention_forward`].
    ///
    /// Returns `(dq, dk, dv)`.
    #[allow(clippy::too_many_arguments)]
    pub fn scaled_dot_product_attention_backward<D: Device>(
        q: &Tensor<D, Float>,
        k: &Tensor<D, Float>,
        v: &Tensor<D, Float>,
        _y: &Tensor<D, Float>,
        grad: &Tensor<D, Float>,
        scale: Option<f64>,
        causal: bool,
        start_pos: usize,
    ) -> Result<AttentionBackward<D>> {
        let (_b, lq, hq, d) = q.dims4()?;
        let (_, lk, hkv, _) = k.dims4()?;
        let group = hq / hkv;

        let qh = q.transpose(1, 2)?.contiguous()?;
        let kh = expand_heads(k, group)?.transpose(1, 2)?.contiguous()?;
        let vh = expand_heads(v, group)?.transpose(1, 2)?.contiguous()?;
        let gh = grad.transpose(1, 2)?.contiguous()?;

        let scale = scale.unwrap_or(1.0 / (d as f64).sqrt());
        let mut scores = qh.matmul(&kh.transpose(2, 3)?)?.mul_scalar(scale)?;
        if causal {
            let mask = causal_mask(q, lq, lk, start_pos)?;
            let mask = mask.broadcast_as(scores.shape().clone())?;
            scores = mask.pick_false(&scores, f64::NEG_INFINITY)?;
        }
        let attn = scores.softmax(3)?;

        // dv = attn^T @ grad, dattn = grad @ v^T
        let dv_h = attn.transpose(2, 3)?.contiguous()?.matmul(&gh)?;
        let dattn = gh.matmul(&vh.transpose(2, 3)?.contiguous()?)?;

        // softmax backward: dscores = attn * (dattn - sum(dattn * attn))
        let dscores = attn.mul(&dattn.broadcast_sub(&dattn.mul(&attn)?.sum_keepdim(3)?)?)?;

        let dq = dscores.matmul(&kh)?.mul_scalar(scale)?.transpose(1, 2)?;
        let dk = reduce_heads(&dscores.transpose(2, 3)?.contiguous()?.matmul(&qh)?.mul_scalar(scale)?, group)?.transpose(1, 2)?;
        let dv = reduce_heads(&dv_h, group)?.transpose(1, 2)?;

        Ok((dq, dk, dv))
    }
}

/// Repeat every KV head `repeats` times: `(B, L, Hkv, D)` -> `(B, L, Hq, D)`.
fn expand_heads<D: Device>(x: &Tensor<D, Float>, repeats: usize) -> Result<Tensor<D, Float>> {
    if repeats == 1 {
        return Ok(x.clone());
    }
    let (_b, _l, h, _d) = x.dims4()?;
    let indices: Vec<i64> = (0..h).flat_map(|i| std::iter::repeat_n(i as i64, repeats)).collect();
    let idx = Tensor::<D, Int>::from_slice(&indices, indices.len(), (x.device(), IntDType::I32))?;
    x.index_select(&idx, 2)
}

/// Sum groups of repeated query heads back into KV heads:
/// `(B, Hq, L, D)` -> `(B, Hkv, L, D)`.
fn reduce_heads<D: Device>(x: &Tensor<D, Float>, group: usize) -> Result<Tensor<D, Float>> {
    if group == 1 {
        return Ok(x.clone());
    }
    let (b, h, l, d) = x.dims4()?;
    let x = x.reshape((b, h / group, group, l, d))?;
    x.sum_dims([2usize], false)
}

/// Causal mask where query `i` (global position `start_pos + i`) sees key
/// `j <= start_pos + i`.
fn causal_mask<D: Device>(q: &Tensor<D, Float>, lq: usize, lk: usize, start_pos: usize) -> Result<Tensor<D, Bool>> {
    let mut data = vec![false; lq * lk];
    for i in 0..lq {
        let limit = start_pos + i;
        for j in 0..lk {
            data[i * lk + j] = j <= limit;
        }
    }
    Tensor::<D, Bool>::from_slice(&data, (lq, lk), (q.device(), BoolDType::Bool))
}
