use luma_tensor::tensor::IntTensor;
use luma_tensor::{D, Device, Float, IntDType, Shape, Tensor};

#[cfg(feature = "cuda")]
use luma_tensor::is_grad_enabled;

use crate::{NnError, NnResult};

/// Which implementation [`scaled_dot_product_attention`] should use.
///
/// There is deliberately no `Auto`: the caller must state the intent. `Flash`
/// is only available on CUDA and reports an error otherwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionImpl {
    /// Portable, differentiable path built from core tensor ops.
    Math,
    /// Fused flash-attention kernel (CUDA, f32, `head_size` 32/64/128, causal).
    Flash,
}

/// Configuration for [`scaled_dot_product_attention`].
#[derive(Clone, Copy, Debug)]
pub struct AttentionConfig {
    pub imp: AttentionImpl,
    /// Apply a causal mask: Q token at global position `g` sees K index `j <= g`.
    pub causal: bool,
    /// Global position of the first Q token (KV-cache decoding).
    pub start_pos: usize,
    /// Attention scale. `None` means `1 / sqrt(head_size)`.
    pub scale: Option<f64>,
}

impl AttentionConfig {
    pub fn math() -> Self {
        Self { imp: AttentionImpl::Math, causal: true, start_pos: 0, scale: None }
    }

    pub fn flash() -> Self {
        Self { imp: AttentionImpl::Flash, causal: true, start_pos: 0, scale: None }
    }

    pub fn causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    pub fn start_pos(mut self, start_pos: usize) -> Self {
        self.start_pos = start_pos;
        self
    }

    pub fn scale(mut self, scale: f64) -> Self {
        self.scale = Some(scale);
        self
    }
}

/// Scaled dot-product attention:
///
/// `Attention(Q, K, V) = softmax(Q Kᵀ * scale) V`
///
/// Shapes:
/// - `q`: `(batch, q_seq_len, q_num_heads, head_size)`
/// - `k`: `(batch, kv_seq_len, kv_num_heads, head_size)`
/// - `v`: `(batch, kv_seq_len, kv_num_heads, head_size)`
/// - returns: `(batch, q_seq_len, q_num_heads, head_size)`
///
/// The implementation is selected by [`AttentionConfig::imp`]. `Math` works on
/// any [`Device`]; `Flash` requires CUDA and errors on any other device.
pub fn scaled_dot_product_attention<Dev: Device>(
    q: &Tensor<Dev, Float>,
    k: &Tensor<Dev, Float>,
    v: &Tensor<Dev, Float>,
    cfg: &AttentionConfig,
) -> NnResult<Tensor<Dev, Float>> {
    match cfg.imp {
        AttentionImpl::Math => attention_math(q, k, v, cfg),
        AttentionImpl::Flash => {
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (q, k, v, cfg);
                Err(NnError::FlashAttentionRequiresCuda)
            }
            #[cfg(feature = "cuda")]
            {
                if !cfg.causal {
                    return Err(NnError::FlashAttentionUnsupported(
                        "flash attention 仅支持 causal=true".into(),
                    ));
                }
                // Grad must be checked *before* `custom_op3` (which runs the op's
                // forward under `no_grad`), because the flash backward is not
                // implemented yet.
                if is_grad_enabled() && (q.requires_grad() || k.requires_grad() || v.requires_grad()) {
                    return Err(NnError::FlashAttentionUnsupported(
                        "flash attention backward 尚未实现，训练请使用 AttentionConfig::math()".into(),
                    ));
                }
                // `FlashAttnFuncOp` is generic over `Device` and downcasts to CUDA
                // at runtime, so this keeps the generic entry point without Rust
                // specialisation.
                q.custom_op3(
                    k,
                    v,
                    Box::new(luma_flash_attn::FlashAttnFuncOp::with_start_pos(
                        cfg.scale,
                        cfg.causal,
                        cfg.start_pos,
                    )),
                )
                .map_err(map_flash_error)
            }
        }
    }
}

fn nn_msg(msg: impl Into<String>) -> NnError {
    NnError::Core(luma_tensor::Error::Msg(msg.into()))
}

fn shape_err(lhs: &Shape, rhs: &Shape, op: &'static str) -> NnError {
    NnError::Core(luma_tensor::Error::ShapeMismatchBinaryOp { lhs: lhs.clone(), rhs: rhs.clone(), op })
}

// ---------------------------------------------------------------------------
// Math path: composed from core, differentiable tensor ops.
// ---------------------------------------------------------------------------

fn attention_math<Dev: Device>(
    q: &Tensor<Dev, Float>,
    k: &Tensor<Dev, Float>,
    v: &Tensor<Dev, Float>,
    cfg: &AttentionConfig,
) -> NnResult<Tensor<Dev, Float>> {
    let (_batch, _q_seq_len, q_num_heads, q_head_size) = q.dims4()?;
    let (_batch, _kv_seq_len, kv_num_heads, head_size) = k.dims4()?;

    if q_head_size != head_size {
        return Err(shape_err(q.shape(), k.shape(), "scaled_dot_product_attention"));
    }
    if kv_num_heads == 0 || q_num_heads % kv_num_heads != 0 {
        return Err(nn_msg(format!(
            "q_num_heads {q_num_heads} must be a multiple of kv_num_heads {kv_num_heads}"
        )));
    }
    if k.shape() != v.shape() {
        return Err(shape_err(k.shape(), v.shape(), "scaled_dot_product_attention"));
    }
    let num_repeat = q_num_heads / kv_num_heads;

    // (batch, seq, heads, head_size) -> (batch, heads, seq, head_size)
    let q = q.transpose(1, 2)?.contiguous()?;
    let k = repeat_kv(&k.transpose(1, 2)?.contiguous()?, num_repeat)?;
    let v = repeat_kv(&v.transpose(1, 2)?.contiguous()?, num_repeat)?;

    let attn_weight = q.matmul(&k.transpose_last()?)?; // (batch, heads, seq, tseq)
    let scale = cfg.scale.unwrap_or_else(|| 1.0 / (head_size as f64).sqrt());
    let attn_weight = attn_weight.mul_scalar(scale)?;

    let attn_weight = if cfg.causal {
        let (_, _, seq, tseq) = attn_weight.dims4()?;
        let q_pos = IntTensor::<Dev>::arange(
            cfg.start_pos as i64,
            (cfg.start_pos + seq) as i64,
            1,
            (q.device(), IntDType::U32),
        )?;
        let k_pos = IntTensor::<Dev>::arange(0, tseq as i64, 1, (q.device(), IntDType::U32))?;
        let mask = q_pos.unsqueeze(1)?.broadcast_lt(&k_pos.unsqueeze(0)?)?; // (seq, tseq)
        let mask = mask.unsqueeze(0)?.unsqueeze(0)?.broadcast_as(attn_weight.shape())?;
        mask.pick_true(f64::NEG_INFINITY, &attn_weight)?
    } else {
        attn_weight
    };

    let attn_weight = crate::functional::softmax(&attn_weight, D::Minus1)?;
    let attn_scores = attn_weight.matmul(&v)?;
    Ok(attn_scores.transpose(1, 2)?.contiguous()?)
}

fn repeat_kv<Dev: Device>(t: &Tensor<Dev, Float>, num_repeat: usize) -> NnResult<Tensor<Dev, Float>> {
    if num_repeat == 1 {
        return Ok(t.clone());
    }
    let (batch, num_heads, seq, head_size) = t.dims4()?;
    let t = t
        .unsqueeze(2)?
        .repeat_dim(2, num_repeat)?
        .reshape((batch, num_heads * num_repeat, seq, head_size))?;
    Ok(t)
}

// ---------------------------------------------------------------------------
// Flash path: the op itself lives in `luma-flash-attn`; here we only map its
// `luma_tensor::Error` back into a user-facing `NnError`.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
fn map_flash_error(e: luma_tensor::Error) -> NnError {
    match e {
        luma_tensor::Error::CustomOp(c) => NnError::FlashAttentionUnsupported(c.to_string()),
        other => NnError::Core(other),
    }
}
