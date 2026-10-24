//! High-level neural-network tensor operators.
//!
//! The user-facing methods live here; the default, device-agnostic forward /
//! backward implementations are grouped by category in the submodules.

mod attention;
mod common;
mod conv;
mod loss;
mod norm;
mod pool;
mod softmax;

use crate::{Device, Dim, Float, Int, Tensor};

/// `(input_grad, weight_grad, bias_grad)` for a normalization backward pass.
pub type NormBackward<D> = (Tensor<D, Float>, Option<Tensor<D, Float>>, Option<Tensor<D, Float>>);

/// `(input_grad, weight_grad, bias_grad)` for a convolution backward pass.
pub type Conv2dBackward<D> = (Tensor<D, Float>, Tensor<D, Float>, Option<Tensor<D, Float>>);

/// `(q_grad, k_grad, v_grad)` for an attention backward pass.
pub type AttentionBackward<D> = (Tensor<D, Float>, Tensor<D, Float>, Tensor<D, Float>);

impl<D: Device> Tensor<D, Float> {
    /// Log-sum-exp along `dim`, numerically stable.
    pub fn logsumexp<Dm: Dim>(&self, dim: Dm) -> crate::Result<Self> {
        let dim = dim.to_index(self.shape(), "logsumexp")?;
        D::logsumexp(self, dim, false)
    }

    /// Log-sum-exp along `dim`, keeping the reduced axis.
    pub fn logsumexp_keepdim<Dm: Dim>(&self, dim: Dm) -> crate::Result<Self> {
        let dim = dim.to_index(self.shape(), "logsumexp_keepdim")?;
        D::logsumexp(self, dim, true)
    }

    /// Softmax over `dim`.
    ///
    /// Dispatches through [`crate::NnProvide`], so a device may substitute a fused
    /// implementation; the default path is composed from primitive ops.
    pub fn softmax<Dm: Dim>(&self, dim: Dm) -> crate::Result<Self> {
        let dim = dim.to_index(self.shape(), "softmax")?;
        D::softmax(self, dim, true)
    }

    /// Log-softmax over `dim`: `x - logsumexp(x, dim)`.
    ///
    /// Dispatches through [`crate::NnProvide`] (see [`Tensor::softmax`]).
    pub fn log_softmax<Dm: Dim>(&self, dim: Dm) -> crate::Result<Self> {
        let dim = dim.to_index(self.shape(), "log_softmax")?;
        D::log_softmax(self, dim, true)
    }

    /// Applies a linear transformation: `input @ weight^T + bias`.
    ///
    /// `input`: `(..., in_features)`; `weight`: `(out_features, in_features)`;
    /// `bias`: `(out_features,)`.
    pub fn linear(&self, weight: &Self, bias: Option<&Self>) -> crate::Result<Self> {
        D::linear(self, weight, bias)
    }

    /// Lookup embeddings for integer indices.
    ///
    /// `self`: `(num_embeddings, embedding_dim)`; `indices`: arbitrary shape.
    pub fn embedding(&self, indices: &Tensor<D, Int>) -> crate::Result<Self> {
        D::embedding(self, indices)
    }

    /// Randomly zeroes elements with probability `p` during training and scales
    /// the remainder by `1/(1-p)`. A no-op during evaluation.
    pub fn dropout(&self, p: f64, training: bool) -> crate::Result<Self> {
        D::dropout(self, p, training)
    }

    /// RMSNorm over the last dim: `x / sqrt(mean(x^2)+eps) * weight + bias`.
    ///
    /// `weight` and `bias` are optional elementwise affine parameters.
    pub fn rms_norm(&self, weight: Option<&Self>, bias: Option<&Self>, eps: f64) -> crate::Result<Self> {
        D::rms_norm(self, weight, bias, eps)
    }

    /// Layer normalization over the last dimension.
    ///
    /// `weight` / `bias` are optional affine parameters shaped like the
    /// normalized trailing dimension(s).
    pub fn layer_norm(&self, weight: Option<&Self>, bias: Option<&Self>, eps: f64) -> crate::Result<Self> {
        D::layer_norm(self, weight, bias, eps)
    }

    /// Group normalization over `(C/G, ...)` with `num_groups` groups.
    pub fn group_norm(&self, num_groups: usize, weight: Option<&Self>, bias: Option<&Self>, eps: f64) -> crate::Result<Self> {
        D::group_norm(self, num_groups, weight, bias, eps)
    }

    /// 2-D convolution.
    ///
    /// `x`: `(N, C_in, H, W)`; `weight`: `(C_out, C_in, kH, kW)`;
    /// `bias`: `(C_out,)`; output: `(N, C_out, H_out, W_out)`.
    pub fn conv2d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> crate::Result<Self> {
        D::conv2d(self, weight, bias, stride, padding, dilation)
    }

    /// 2-D max pooling.
    ///
    /// `x`: `(N, C, H, W)`; output: `(N, C, H_out, W_out)`.
    pub fn max_pool2d(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        ceil_mode: bool,
    ) -> crate::Result<Self> {
        D::max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode)
    }

    /// 2-D average pooling.
    ///
    /// `x`: `(N, C, H, W)`; output: `(N, C, H_out, W_out)`.
    pub fn avg_pool2d(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> crate::Result<Self> {
        D::avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad)
    }

    /// Adaptive average pooling to `output_size`.
    pub fn adaptive_avg_pool2d(&self, output_size: (usize, usize)) -> crate::Result<Self> {
        D::adaptive_avg_pool2d(self, output_size)
    }

    /// Mean squared error: `mean((pred - target)^2)`.
    pub fn mse_loss(&self, target: &Self) -> crate::Result<Self> {
        D::mse_loss(self, target)
    }

    /// Negative log-likelihood loss. `self` is `(..., C)` log-probabilities and
    /// `target` contains class indices.
    pub fn nll_loss(&self, target: &Tensor<D, Int>) -> crate::Result<Self> {
        D::nll_loss(self, target)
    }

    /// Cross-entropy loss: `nll_loss(log_softmax(self), target)`.
    pub fn cross_entropy_loss(&self, target: &Tensor<D, Int>) -> crate::Result<Self> {
        D::cross_entropy_loss(self, target)
    }

    /// Binary cross-entropy loss.
    pub fn bce_loss(&self, target: &Self) -> crate::Result<Self> {
        D::bce_loss(self, target)
    }

    /// Scaled dot-product attention.
    ///
    /// `q`: `(batch, q_len, q_heads, head_size)`;
    /// `k` / `v`: `(batch, kv_len, kv_heads, head_size)`.
    pub fn scaled_dot_product_attention(
        &self,
        k: &Self,
        v: &Self,
        scale: Option<f64>,
        causal: bool,
        start_pos: usize,
    ) -> crate::Result<Self> {
        D::scaled_dot_product_attention(self, k, v, scale, causal, start_pos)
    }
}

/// Default, device-agnostic implementations of the high-level NN operators.
pub struct NnOpDefault;
