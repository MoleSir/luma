use luma_io::lpk::LumaPack;
use luma_tensor::{Device, DynTensor, GradStore, Scalar, Tensor, no_grad};

use super::Optimizer;

// ============================================================================
//   Adam
// ============================================================================

#[derive(Clone, Debug)]
pub struct AdamConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self { lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8 }
    }
}

struct AdamParam<D: Device> {
    param: Tensor<D>,
    first_moment: Tensor<D>,  // m
    second_moment: Tensor<D>, // v
}

pub struct Adam<D: Device> {
    params: Vec<AdamParam<D>>,
    step_t: usize,
    config: AdamConfig,
}

impl<D: Device> Adam<D> {
    pub fn new(params: impl Into<Vec<Tensor<D>>>, config: AdamConfig) -> luma_tensor::Result<Self> {
        let params = params
            .into()
            .into_iter()
            .map(|param| {
                let first_moment = param.zeros_like()?;
                let second_moment = param.zeros_like()?;
                Ok(AdamParam { param, first_moment, second_moment })
            })
            .collect::<luma_tensor::Result<Vec<_>>>()?;
        Ok(Self { params, step_t: 0, config })
    }
}

impl<D: Device> Optimizer for Adam<D> {
    type Device = D;

    fn get_lr(&self) -> f64 {
        self.config.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.lr = lr;
    }

    /// ```text
    ///   m = β₁·m + (1-β₁)·g
    ///   v = β₂·v + (1-β₂)·g²
    ///   m̂ = m / (1-β₁ᵗ)    v̂ = v / (1-β₂ᵗ)
    ///   param -= lr * m̂ / (√v̂ + ε)
    /// ```
    fn step(&mut self, grads: &GradStore<Self::Device>) -> luma_tensor::Result<()> {
        no_grad!();
        self.step_t += 1;

        let lr = self.config.lr;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;

        let bias_m = 1.0 - beta1.powi(self.step_t as i32);
        let bias_v = 1.0 - beta2.powi(self.step_t as i32);

        for AdamParam { param, first_moment, second_moment } in self.params.iter_mut() {
            if let Some(g) = grads.get(&param) {
                let g = g.clone();

                // m = β₁·m + (1-β₁)·g
                first_moment.mul_scalar_(beta1)?;
                first_moment.add_(&g.mul_scalar(1.0 - beta1)?)?;

                // v = β₂·v + (1-β₂)·g²
                second_moment.mul_scalar_(beta2)?;
                second_moment.add_(&g.pow(2.0)?.mul_scalar(1.0 - beta2)?)?;

                // bias-corrected estimates
                let m_hat = first_moment.mul_scalar(1.0 / bias_m)?;
                let v_hat = second_moment.mul_scalar(1.0 / bias_v)?;
                let denom = v_hat.sqrt()?.add_scalar(eps)?;
                param.sub_(&m_hat.div(&denom)?.mul_scalar(lr)?)?;
            }
        }

        Ok(())
    }

    fn state_dict(&self) -> luma_tensor::Result<LumaPack<Self::Device>> {
        let mut pack = LumaPack::new();
        for (i, p) in self.params.iter().enumerate() {
            pack.tensors.insert(format!("{i}.first_moment"), DynTensor::Float(p.first_moment.clone()));
            pack.tensors.insert(format!("{i}.second_moment"), DynTensor::Float(p.second_moment.clone()));
        }
        pack.scalars.insert("lr".into(), Scalar::F64(self.config.lr));
        pack.scalars.insert("beta1".into(), Scalar::F64(self.config.beta1));
        pack.scalars.insert("beta2".into(), Scalar::F64(self.config.beta2));
        pack.scalars.insert("eps".into(), Scalar::F64(self.config.eps));
        pack.scalars.insert("step_t".into(), Scalar::I32(self.step_t as i32));
        Ok(pack)
    }

    fn load_state_dict(&mut self, pack: &LumaPack<Self::Device>) -> luma_tensor::Result<()> {
        if let Some(v) = pack.scalars.get("lr").and_then(|s| s.to_f64()) {
            self.config.lr = v;
        }
        if let Some(v) = pack.scalars.get("beta1").and_then(|s| s.to_f64()) {
            self.config.beta1 = v;
        }
        if let Some(v) = pack.scalars.get("beta2").and_then(|s| s.to_f64()) {
            self.config.beta2 = v;
        }
        if let Some(v) = pack.scalars.get("eps").and_then(|s| s.to_f64()) {
            self.config.eps = v;
        }
        if let Some(v) = pack.scalars.get("step_t").and_then(|s| s.to_i64()) {
            self.step_t = v as usize;
        }
        for (i, p) in self.params.iter_mut().enumerate() {
            if let Some(dt) = pack.tensors.get(&format!("{i}.first_moment")) {
                if let Some(src) = dt.as_float() {
                    p.first_moment.copy_(src)?;
                }
            }
            if let Some(dt) = pack.tensors.get(&format!("{i}.second_moment")) {
                if let Some(src) = dt.as_float() {
                    p.second_moment.copy_(src)?;
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
//   AdamW
// ============================================================================

#[derive(Clone, Debug)]
pub struct AdamWConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self { lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 1e-2 }
    }
}

struct AdamWParam<D: Device> {
    param: Tensor<D>,
    first_moment: Tensor<D>,  // m
    second_moment: Tensor<D>, // v
}

pub struct AdamW<D: Device> {
    params: Vec<AdamWParam<D>>,
    step_t: usize,
    config: AdamWConfig,
}

impl<D: Device> AdamW<D> {
    pub fn new(params: impl Into<Vec<Tensor<D>>>, config: AdamWConfig) -> luma_tensor::Result<Self> {
        let params = params
            .into()
            .into_iter()
            .map(|param| {
                let first_moment = param.zeros_like()?;
                let second_moment = param.zeros_like()?;
                Ok(AdamWParam { param, first_moment, second_moment })
            })
            .collect::<luma_tensor::Result<Vec<_>>>()?;
        Ok(Self { params, step_t: 0, config })
    }
}

impl<D: Device> Optimizer for AdamW<D> {
    type Device = D;

    fn get_lr(&self) -> f64 {
        self.config.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.lr = lr;
    }

    /// ```text
    ///   param -= lr * weight_decay * param          (decoupled)
    ///   m = β₁·m + (1-β₁)·g
    ///   v = β₂·v + (1-β₂)·g²
    ///   m̂ = m / (1-β₁ᵗ)    v̂ = v / (1-β₂ᵗ)
    ///   param -= lr * m̂ / (√v̂ + ε)
    /// ```
    fn step(&mut self, grads: &GradStore<Self::Device>) -> luma_tensor::Result<()> {
        no_grad!();
        self.step_t += 1;

        let lr = self.config.lr;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;
        let weight_decay = self.config.weight_decay;

        let bias_m = 1.0 - beta1.powi(self.step_t as i32);
        let bias_v = 1.0 - beta2.powi(self.step_t as i32);

        for AdamWParam { param, first_moment, second_moment } in self.params.iter_mut() {
            if let Some(g) = grads.get(&param) {
                let g = g.clone();

                // decoupled weight decay
                if weight_decay != 0.0 {
                    param.sub_(&param.mul_scalar(lr * weight_decay)?)?;
                }

                // m = β₁·m + (1-β₁)·g
                first_moment.mul_scalar_(beta1)?;
                first_moment.add_(&g.mul_scalar(1.0 - beta1)?)?;

                // v = β₂·v + (1-β₂)·g²
                second_moment.mul_scalar_(beta2)?;
                second_moment.add_(&g.pow(2.0)?.mul_scalar(1.0 - beta2)?)?;

                // bias-corrected estimates
                let m_hat = first_moment.mul_scalar(1.0 / bias_m)?;
                let v_hat = second_moment.mul_scalar(1.0 / bias_v)?;
                let denom = v_hat.sqrt()?.add_scalar(eps)?;
                param.sub_(&m_hat.div(&denom)?.mul_scalar(lr)?)?;
            }
        }

        Ok(())
    }

    fn state_dict(&self) -> luma_tensor::Result<LumaPack<Self::Device>> {
        let mut pack = LumaPack::new();
        for (i, p) in self.params.iter().enumerate() {
            pack.tensors.insert(format!("{i}.first_moment"), DynTensor::Float(p.first_moment.clone()));
            pack.tensors.insert(format!("{i}.second_moment"), DynTensor::Float(p.second_moment.clone()));
        }
        pack.scalars.insert("lr".into(), Scalar::F64(self.config.lr));
        pack.scalars.insert("beta1".into(), Scalar::F64(self.config.beta1));
        pack.scalars.insert("beta2".into(), Scalar::F64(self.config.beta2));
        pack.scalars.insert("eps".into(), Scalar::F64(self.config.eps));
        pack.scalars.insert("weight_decay".into(), Scalar::F64(self.config.weight_decay));
        pack.scalars.insert("step_t".into(), Scalar::I32(self.step_t as i32));
        Ok(pack)
    }

    fn load_state_dict(&mut self, pack: &LumaPack<Self::Device>) -> luma_tensor::Result<()> {
        if let Some(v) = pack.scalars.get("lr").and_then(|s| s.to_f64()) {
            self.config.lr = v;
        }
        if let Some(v) = pack.scalars.get("beta1").and_then(|s| s.to_f64()) {
            self.config.beta1 = v;
        }
        if let Some(v) = pack.scalars.get("beta2").and_then(|s| s.to_f64()) {
            self.config.beta2 = v;
        }
        if let Some(v) = pack.scalars.get("eps").and_then(|s| s.to_f64()) {
            self.config.eps = v;
        }
        if let Some(v) = pack.scalars.get("weight_decay").and_then(|s| s.to_f64()) {
            self.config.weight_decay = v;
        }
        if let Some(v) = pack.scalars.get("step_t").and_then(|s| s.to_i64()) {
            self.step_t = v as usize;
        }
        for (i, p) in self.params.iter_mut().enumerate() {
            if let Some(dt) = pack.tensors.get(&format!("{i}.first_moment")) {
                if let Some(src) = dt.as_float() {
                    p.first_moment.copy_(src)?;
                }
            }
            if let Some(dt) = pack.tensors.get(&format!("{i}.second_moment")) {
                if let Some(src) = dt.as_float() {
                    p.second_moment.copy_(src)?;
                }
            }
        }
        Ok(())
    }
}
