use luma_tensor::{Device, Float, Tensor};

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
        AttentionImpl::Math => Ok(q.scaled_dot_product_attention(k, v, cfg.scale, cfg.causal, cfg.start_pos)?.contiguous()?),
        AttentionImpl::Flash => {
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (q, k, v, cfg);
                Err(NnError::FlashAttentionRequiresCuda)
            }
            #[cfg(feature = "cuda")]
            {
                if !cfg.causal {
                    return Err(NnError::FlashAttentionUnsupported("flash attention 仅支持 causal=true".into()));
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
                q.custom_op3(k, v, Box::new(luma_flash_attn::FlashAttnFuncOp::with_start_pos(cfg.scale, cfg.causal, cfg.start_pos)))
                    .map_err(map_flash_error)
            }
        }
    }
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
