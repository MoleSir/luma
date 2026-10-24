use crate::{Float, Int, Result, Tensor};
use crate::ops::NnOpDefault;

/// Device-provided implementations of high-level NN operators.
///
/// Every method has a default implementation built from primitive ops (see
/// [`NnOpDefault`]). A new device can get correct behaviour by providing an
/// empty impl:
///
/// ```ignore
/// impl NnProvide<MyDevice> for MyDevice {}
/// ```
///
/// A device that has a fused kernel overrides the corresponding method, e.g.
/// returning a [`CustomOp`](crate::CustomOp1)-backed tensor so autograd keeps
/// working.
pub trait NnProvide<D: super::Device> {
    fn logsumexp(x: &Tensor<D, Float>, dim: usize, keepdim: bool) -> Result<Tensor<D, Float>> {
        NnOpDefault::logsumexp_forward(x, dim, keepdim)
    }

    fn softmax(x: &Tensor<D, Float>, dim: usize, keepdim: bool) -> Result<Tensor<D, Float>> {
        NnOpDefault::softmax_forward(x, dim, keepdim)
    }

    fn log_softmax(x: &Tensor<D, Float>, dim: usize, keepdim: bool) -> Result<Tensor<D, Float>> {
        NnOpDefault::log_softmax_forward(x, dim, keepdim)
    }

    fn rms_norm(
        x: &Tensor<D, Float>,
        weight: Option<&Tensor<D, Float>>,
        bias: Option<&Tensor<D, Float>>,
        eps: f64,
    ) -> Result<Tensor<D, Float>> {
        NnOpDefault::rms_norm_forward(x, weight, bias, eps)
    }

    /// Layer normalization over the last dimension.
    fn layer_norm(
        x: &Tensor<D, Float>,
        weight: Option<&Tensor<D, Float>>,
        bias: Option<&Tensor<D, Float>>,
        eps: f64,
    ) -> Result<Tensor<D, Float>> {
        NnOpDefault::layer_norm_forward(x, weight, bias, eps)
    }

    /// Group normalization over `(C/G, ...)`.
    fn group_norm(
        x: &Tensor<D, Float>,
        num_groups: usize,
        weight: Option<&Tensor<D, Float>>,
        bias: Option<&Tensor<D, Float>>,
        eps: f64,
    ) -> Result<Tensor<D, Float>> {
        NnOpDefault::group_norm_forward(x, num_groups, weight, bias, eps)
    }

    /// 2-D convolution.
    #[allow(clippy::too_many_arguments)]
    fn conv2d(
        x: &Tensor<D, Float>,
        weight: &Tensor<D, Float>,
        bias: Option<&Tensor<D, Float>>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<Tensor<D, Float>> {
        NnOpDefault::conv2d_forward(x, weight, bias, stride, padding, dilation)
    }

    /// Applies a linear transformation: `input @ weight^T + bias`.
    fn linear(x: &Tensor<D, Float>, weight: &Tensor<D, Float>, bias: Option<&Tensor<D, Float>>) -> Result<Tensor<D, Float>> {
        NnOpDefault::linear_forward(x, weight, bias)
    }

    /// Lookup embeddings for integer indices.
    fn embedding(weight: &Tensor<D, Float>, indices: &Tensor<D, Int>) -> Result<Tensor<D, Float>> {
        NnOpDefault::embedding_forward(weight, indices)
    }

    /// Dropout: zero with probability `p` and scale by `1/(1-p)` during training.
    fn dropout(x: &Tensor<D, Float>, p: f64, training: bool) -> Result<Tensor<D, Float>> {
        NnOpDefault::dropout_forward(x, p, training)
    }

    /// Mean squared error: `mean((pred - target)^2)`.
    fn mse_loss(pred: &Tensor<D, Float>, target: &Tensor<D, Float>) -> Result<Tensor<D, Float>> {
        NnOpDefault::mse_loss_forward(pred, target)
    }

    /// Negative log-likelihood loss.
    fn nll_loss(log_probs: &Tensor<D, Float>, target: &Tensor<D, Int>) -> Result<Tensor<D, Float>> {
        NnOpDefault::nll_loss_forward(log_probs, target)
    }

    /// Cross-entropy loss: `nll_loss(log_softmax(pred), target)`.
    fn cross_entropy_loss(pred: &Tensor<D, Float>, target: &Tensor<D, Int>) -> Result<Tensor<D, Float>> {
        NnOpDefault::cross_entropy_loss_forward(pred, target)
    }

    /// Binary cross-entropy loss.
    fn bce_loss(pred: &Tensor<D, Float>, target: &Tensor<D, Float>) -> Result<Tensor<D, Float>> {
        NnOpDefault::bce_loss_forward(pred, target)
    }

    /// 2-D max pooling.
    #[allow(clippy::too_many_arguments)]
    fn max_pool2d(
        x: &Tensor<D, Float>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        ceil_mode: bool,
    ) -> Result<Tensor<D, Float>> {
        NnOpDefault::max_pool2d_forward(x, kernel_size, stride, padding, dilation, ceil_mode)
    }

    /// 2-D average pooling.
    #[allow(clippy::too_many_arguments)]
    fn avg_pool2d(
        x: &Tensor<D, Float>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Result<Tensor<D, Float>> {
        NnOpDefault::avg_pool2d_forward(x, kernel_size, stride, padding, ceil_mode, count_include_pad)
    }

    /// Adaptive average pooling to `output_size`.
    fn adaptive_avg_pool2d(x: &Tensor<D, Float>, output_size: (usize, usize)) -> Result<Tensor<D, Float>> {
        NnOpDefault::adaptive_avg_pool2d_forward(x, output_size)
    }

    /// Scaled dot-product attention.
    #[allow(clippy::too_many_arguments)]
    fn scaled_dot_product_attention(
        q: &Tensor<D, Float>,
        k: &Tensor<D, Float>,
        v: &Tensor<D, Float>,
        scale: Option<f64>,
        causal: bool,
        start_pos: usize,
    ) -> Result<Tensor<D, Float>> {
        NnOpDefault::scaled_dot_product_attention_forward(q, k, v, scale, causal, start_pos)
    }
}
