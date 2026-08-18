use luma_macros::Module;
use luma_tensor::ops::construct::TensorCreationOptions;
use luma_tensor::{Device, Int, Tensor};

use crate::init::Init;
use crate::{NnResult, Parameter};

// ============================================================================
//   EmbeddingConfig
// ============================================================================

/// Configuration for [`Embedding`].
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub padding_idx: Option<usize>,
    pub weight_init: Init,
}

impl EmbeddingConfig {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        Self { num_embeddings, embedding_dim, padding_idx: None, weight_init: Init::normal(0.0, 1.0) }
    }
}

// ============================================================================
//   Embedding
// ============================================================================

/// A lookup table that maps integer indices to dense vectors.
///
/// `weight` shape: `(num_embeddings, embedding_dim)`.
#[derive(Module, Clone)]
#[module(display = "display")]
pub struct Embedding<D: Device> {
    pub weight: Parameter<D>, // (num_embeddings, embedding_dim)

    #[module(skip)]
    pub num_embeddings: usize,
    #[module(skip)]
    pub embedding_dim: usize,
}

impl<D: Device> Embedding<D> {
    /// Shortcut constructor with default initialisation (N(0, 1)).
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        options: impl Into<TensorCreationOptions<D, luma_tensor::Float>>,
    ) -> NnResult<Self> {
        let config = EmbeddingConfig::new(num_embeddings, embedding_dim);
        Self::from_config(&config, options)
    }

    /// Full-control constructor from an [`EmbeddingConfig`].
    pub fn from_config(config: &EmbeddingConfig, options: impl Into<TensorCreationOptions<D, luma_tensor::Float>>) -> NnResult<Self> {
        let options: TensorCreationOptions<D, luma_tensor::Float> = options.into();
        let opts = (&options.device, options.dtype);

        let weight = config.weight_init.init_param((config.num_embeddings, config.embedding_dim), opts)?;

        Ok(Self { weight, num_embeddings: config.num_embeddings, embedding_dim: config.embedding_dim })
    }

    /// Custom display — called by `Module::extra_display` via `#[module(display = "display")]`.
    pub fn display(&self) -> String {
        format!("{}x{}", self.num_embeddings, self.embedding_dim)
    }

    /// Lookup embeddings for integer indices.
    ///
    /// `indices` shape: `(...)`  →  returns `(..., embedding_dim)`.
    pub fn forward(&self, indices: &Tensor<D, Int>) -> NnResult<Tensor<D, luma_tensor::Float>> {
        crate::functional::embedding(&self.weight, indices)
    }
}
