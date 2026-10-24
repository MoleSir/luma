use luma_macros::Module;
use luma_tensor::ops::construct::TensorCreationOptions;
use luma_tensor::{Device, Float, Tensor};

use crate::functional::{self as F, AttentionConfig, AttentionImpl};
use crate::{Linear, NnError, NnResult};

// ============================================================================
//   MultiheadAttentionConfig
// ============================================================================

#[derive(Debug, Clone)]
pub struct MultiheadAttentionConfig {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub bias: bool,
    /// Apply a causal mask.
    pub causal: bool,
    /// Attention scale; `None` means `1 / sqrt(head_dim)`.
    pub scale: Option<f64>,
    pub imp: AttentionImpl,
}

impl MultiheadAttentionConfig {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        Self { embed_dim, num_heads, bias: true, causal: true, scale: None, imp: AttentionImpl::Math }
    }
}

// ============================================================================
//   MultiheadAttention
// ============================================================================

/// Multi-head self-attention with `Q`/`K`/`V`/output projections.
///
/// Input / output: `(batch, seq_len, embed_dim)`.
#[derive(Module)]
#[module(display = "display")]
pub struct MultiheadAttention<Dev: Device> {
    pub q_proj: Linear<Dev>,
    pub k_proj: Linear<Dev>,
    pub v_proj: Linear<Dev>,
    pub out_proj: Linear<Dev>,

    #[module(skip)]
    pub embed_dim: usize,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
    #[module(skip)]
    pub causal: bool,
    #[module(skip)]
    pub scale: Option<f64>,
    #[module(skip)]
    pub imp: AttentionImpl,
}

impl<Dev: Device> MultiheadAttention<Dev> {
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        options: impl Into<TensorCreationOptions<Dev, Float>>,
    ) -> NnResult<Self> {
        let config = MultiheadAttentionConfig::new(embed_dim, num_heads);
        Self::from_config(&config, options)
    }

    pub fn from_config(
        config: &MultiheadAttentionConfig,
        options: impl Into<TensorCreationOptions<Dev, Float>>,
    ) -> NnResult<Self> {
        if config.num_heads == 0 || !config.embed_dim.is_multiple_of(config.num_heads) {
            return Err(NnError::HeadSizeCannotDivideByNumhead(config.embed_dim, config.num_heads));
        }

        let options: TensorCreationOptions<Dev, Float> = options.into();
        let opts = (&options.device, options.dtype);

        let q_proj = Linear::new(config.embed_dim, config.embed_dim, config.bias, opts)?;
        let k_proj = Linear::new(config.embed_dim, config.embed_dim, config.bias, opts)?;
        let v_proj = Linear::new(config.embed_dim, config.embed_dim, config.bias, opts)?;
        let out_proj = Linear::new(config.embed_dim, config.embed_dim, config.bias, opts)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            embed_dim: config.embed_dim,
            num_heads: config.num_heads,
            head_dim: config.embed_dim / config.num_heads,
            causal: config.causal,
            scale: config.scale,
            imp: config.imp,
        })
    }

    pub fn display(&self) -> String {
        format!("embed_dim={}, heads={}, causal={}", self.embed_dim, self.num_heads, self.causal)
    }

    #[inline]
    pub fn forward(&self, input: &Tensor<Dev>) -> NnResult<Tensor<Dev>> {
        self.forward_with_start_pos(input, 0)
    }

    /// Like [`Self::forward`], with an explicit global position for KV-cache
    /// decoding.
    pub fn forward_with_start_pos(&self, input: &Tensor<Dev>, start_pos: usize) -> NnResult<Tensor<Dev>> {
        let (batch, seq, embed) = input.dims3()?;
        if embed != self.embed_dim {
            return Err(NnError::Core(luma_tensor::Error::Msg(format!(
                "multihead_attention: expected embed_dim {}, got {embed}",
                self.embed_dim
            ))));
        }

        let q = self.q_proj.forward(input)?.reshape((batch, seq, self.num_heads, self.head_dim))?;
        let k = self.k_proj.forward(input)?.reshape((batch, seq, self.num_heads, self.head_dim))?;
        let v = self.v_proj.forward(input)?.reshape((batch, seq, self.num_heads, self.head_dim))?;

        let cfg = AttentionConfig { imp: self.imp, causal: self.causal, start_pos, scale: self.scale };
        let out = F::scaled_dot_product_attention(&q, &k, &v, &cfg)?;
        let out = out.contiguous()?.reshape((batch, seq, self.embed_dim))?;
        self.out_proj.forward(&out)
    }
}
