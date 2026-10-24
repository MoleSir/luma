use luma_tensor::{Device, Dim, Float, Int, Tensor};

use crate::NnResult;

// ============================================================================
//   MSE
// ============================================================================

/// Mean squared error: `mean((pred - target)^2)`.
pub fn mse_loss<D: Device>(pred: &Tensor<D, Float>, target: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
    Ok(pred.mse_loss(target)?)
}

// ============================================================================
//   NLL
// ============================================================================

/// Negative log-likelihood loss.
///
/// `log_probs` shape: `(..., C)` — log-probabilities.
/// `target` shape: `(...)` — class indices (Int).
///
/// Returns `-mean(log_probs[..., target])`.
pub fn nll_loss<D: Device>(log_probs: &Tensor<D, Float>, target: &Tensor<D, Int>) -> NnResult<Tensor<D, Float>> {
    Ok(log_probs.nll_loss(target)?)
}

// ============================================================================
//   CrossEntropy
// ============================================================================

/// Cross-entropy loss: `nll_loss(log_softmax(pred), target)`.
///
/// `pred` shape: `(..., C)` — raw logits.
/// `target` shape: `(...)` — class indices (Int).
pub fn cross_entropy_loss<D: Device>(pred: &Tensor<D, Float>, target: &Tensor<D, Int>) -> NnResult<Tensor<D, Float>> {
    Ok(pred.cross_entropy_loss(target)?)
}

/// Numerically stable log-softmax: `x - logsumexp(x, dim)`.
pub fn log_softmax<Dev: Device, D: Dim>(pred: &Tensor<Dev, Float>, dim: D) -> NnResult<Tensor<Dev, Float>> {
    Ok(pred.log_softmax(dim)?)
}

// ============================================================================
//   BCE
// ============================================================================

/// Binary cross-entropy loss: `-mean(target*ln(pred) + (1-target)*ln(1-pred))`.
///
/// `pred` is clamped to `[eps, 1-eps]` for numerical stability.
pub fn bce_loss<D: Device>(pred: &Tensor<D, Float>, target: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
    Ok(pred.bce_loss(target)?)
}
