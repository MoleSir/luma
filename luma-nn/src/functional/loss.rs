use luma_tensor::{Device, Dim, Float, Int, Tensor};

use crate::NnResult;

// ============================================================================
//   MSE
// ============================================================================

/// Mean squared error: `mean((pred - target)²)`.
pub fn mse_loss<D: Device>(pred: &Tensor<D, Float>, target: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
    Ok(pred.sub(target)?.sqr()?.mean_all()?)
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
    let rank = log_probs.rank();
    let class_dim = rank - 1;

    // Reshape target to have the same rank as log_probs if needed.
    let target = if target.rank() == rank - 1 {
        let mut shape = target.dims().to_vec();
        shape.push(1);
        target.reshape(shape)?
    } else {
        target.clone()
    };

    let selected = log_probs.gather(&target, class_dim)?;
    Ok(-&selected.mean_all()?)
}

// ============================================================================
//   CrossEntropy
// ============================================================================

/// Cross-entropy loss: `nll_loss(log_softmax(pred), target)`.
///
/// `pred` shape: `(..., C)` — raw logits.
/// `target` shape: `(...)` — class indices (Int).
pub fn cross_entropy_loss<D: Device>(pred: &Tensor<D, Float>, target: &Tensor<D, Int>) -> NnResult<Tensor<D, Float>> {
    let class_dim = pred.rank().saturating_sub(1);
    let log_softmax = log_softmax(pred, class_dim)?;
    nll_loss(&log_softmax, target)
}

/// Numerically stable log-softmax: `x - logsumexp(x, dim)`.
pub fn log_softmax<Dev: Device, D: Dim>(pred: &Tensor<Dev, Float>, dim: D) -> NnResult<Tensor<Dev, Float>> {
    let lse = pred.logsumexp_keepdim(dim)?;
    Ok(pred.broadcast_sub(&lse)?)
}

// ============================================================================
//   BCE
// ============================================================================

/// Binary cross-entropy loss: `-mean(target*ln(pred) + (1-target)*ln(1-pred))`.
///
/// `pred` is clamped to `[eps, 1-eps]` for numerical stability.
pub fn bce_loss<D: Device>(pred: &Tensor<D, Float>, target: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
    let eps = 1e-7;
    let pred = pred.clamp(Some(eps), Some(1.0 - eps))?;

    let left = target * &pred.ln()?;
    let one_minus_target = 1.0 - target;
    let one_minus_pred = 1.0 - &pred;
    let right = one_minus_target * &one_minus_pred.ln()?;

    let bce = -&(left + right);
    Ok(bce.mean_all()?)
}
