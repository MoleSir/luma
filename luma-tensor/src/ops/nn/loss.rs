use super::NnOpDefault;
use crate::{Device, Float, Int, Result, Tensor};

impl NnOpDefault {
    /// Mean squared error: `mean((pred - target)^2)`.
    pub fn mse_loss_forward<D: Device>(pred: &Tensor<D, Float>, target: &Tensor<D, Float>) -> Result<Tensor<D, Float>> {
        Ok(pred.sub(target)?.sqr()?.mean_all()?)
    }

    /// Negative log-likelihood loss.
    ///
    /// `log_probs`: `(..., C)` log-probabilities; `target`: `(...)` class
    /// indices. Returns `-mean(log_probs[..., target])`.
    pub fn nll_loss_forward<D: Device>(log_probs: &Tensor<D, Float>, target: &Tensor<D, Int>) -> Result<Tensor<D, Float>> {
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

    /// Cross-entropy loss: `nll_loss(log_softmax(pred), target)`.
    ///
    /// `pred`: `(..., C)` raw logits; `target`: `(...)` class indices.
    pub fn cross_entropy_loss_forward<D: Device>(pred: &Tensor<D, Float>, target: &Tensor<D, Int>) -> Result<Tensor<D, Float>> {
        let class_dim = pred.rank().saturating_sub(1);
        let log_softmax = pred.log_softmax(class_dim)?;
        log_softmax.nll_loss(target)
    }

    /// Binary cross-entropy loss:
    /// `-mean(target*ln(pred) + (1-target)*ln(1-pred))`.
    ///
    /// `pred` is clamped to `[eps, 1-eps]` for numerical stability.
    pub fn bce_loss_forward<D: Device>(pred: &Tensor<D, Float>, target: &Tensor<D, Float>) -> Result<Tensor<D, Float>> {
        let eps = 1e-7;
        let pred = pred.clamp(Some(eps), Some(1.0 - eps))?;

        let left = target.mul(&pred.ln()?)?;
        let one_minus_target = target.neg()?.add_scalar(1.0)?;
        let one_minus_pred = pred.neg()?.add_scalar(1.0)?;
        let right = one_minus_target.mul(&one_minus_pred.ln()?)?;

        let bce = left.add(&right)?.neg()?;
        Ok(bce.mean_all()?)
    }
}
