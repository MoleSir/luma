use luma_io::lpk::LumaPack;
use luma_tensor::{Device, DynTensor, Scalar, Tensor, no_grad};

use super::Optimizer;

#[derive(Clone, Debug)]
pub struct MomentumConfig {
    pub lr: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub dampening: f64,
    pub nesterov: bool,
}

impl Default for MomentumConfig {
    fn default() -> Self {
        Self { lr: 1e-3, momentum: 0.9, weight_decay: 0.0, dampening: 0.0, nesterov: false }
    }
}

struct MomentumParam<D: Device> {
    param: Tensor<D>,
    velocity: Tensor<D>,
}

pub struct Momentum<D: Device> {
    params: Vec<MomentumParam<D>>,
    config: MomentumConfig,
}

impl<D: Device> Momentum<D> {
    pub fn new(params: impl Into<Vec<Tensor<D>>>, config: MomentumConfig) -> luma_tensor::Result<Self> {
        let params = params
            .into()
            .into_iter()
            .map(|param| {
                let velocity = param.zeros_like()?;
                Ok(MomentumParam { param, velocity })
            })
            .collect::<luma_tensor::Result<Vec<_>>>()?;
        Ok(Self { params, config })
    }
}

impl<D: Device> Optimizer for Momentum<D> {
    type Device = D;

    fn get_lr(&self) -> f64 {
        self.config.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.lr = lr;
    }

    fn step(&mut self, grads: &luma_tensor::GradStore<Self::Device>) -> luma_tensor::Result<()> {
        no_grad!();

        let lr = self.config.lr;
        let momentum = self.config.momentum;
        let weight_decay = self.config.weight_decay;
        let dampening = self.config.dampening;
        let nesterov = self.config.nesterov;

        for MomentumParam { param, velocity } in self.params.iter_mut() {
            if let Some(g) = grads.get(&param) {
                let mut g = g.clone();
                if weight_decay != 0.0 {
                    /*
                        origin: w_t = w_{t-1} - lr*g
                        weight_decay: w_t = (1-lr*eta)w_{t-1} - lr*g
                        equal to: w_t = w_{t-1} - lr*(g + eta*w)
                    */
                    g.add_(&param.mul_scalar(weight_decay)?)?;
                }

                // v_new = momentum * v_old + (1 - dampening) * g
                if momentum != 0.0 {
                    if dampening != 0.0 {
                        g.mul_scalar_(1. - dampening)?;
                    }

                    // v = m*v + g
                    velocity.mul_scalar_(momentum)?;
                    velocity.add_(&g)?;

                    if nesterov {
                        g.add_(&velocity.mul_scalar(momentum)?)?;
                    } else {
                        g = velocity.clone();
                    }
                }

                param.sub_(&g.mul_scalar(lr)?)?;
            }
        }

        Ok(())
    }

    fn state_dict(&self) -> luma_tensor::Result<LumaPack<Self::Device>> {
        let mut pack = LumaPack::new();
        for (i, p) in self.params.iter().enumerate() {
            pack.tensors.insert(format!("{i}.velocity"), DynTensor::Float(p.velocity.clone()));
        }
        pack.scalars.insert("lr".into(), Scalar::F64(self.config.lr));
        pack.scalars.insert("momentum".into(), Scalar::F64(self.config.momentum));
        pack.scalars.insert("weight_decay".into(), Scalar::F64(self.config.weight_decay));
        pack.scalars.insert("dampening".into(), Scalar::F64(self.config.dampening));
        pack.scalars.insert("nesterov".into(), Scalar::Bool(self.config.nesterov));
        Ok(pack)
    }

    fn load_state_dict(&mut self, pack: &LumaPack<Self::Device>) -> luma_tensor::Result<()> {
        if let Some(v) = pack.scalars.get("lr").and_then(|s| s.to_f64()) {
            self.config.lr = v;
        }
        if let Some(v) = pack.scalars.get("momentum").and_then(|s| s.to_f64()) {
            self.config.momentum = v;
        }
        if let Some(v) = pack.scalars.get("weight_decay").and_then(|s| s.to_f64()) {
            self.config.weight_decay = v;
        }
        if let Some(v) = pack.scalars.get("dampening").and_then(|s| s.to_f64()) {
            self.config.dampening = v;
        }
        if let Some(v) = pack.scalars.get("nesterov").and_then(|s| s.to_bool()) {
            self.config.nesterov = v;
        }
        for (i, p) in self.params.iter_mut().enumerate() {
            let key = format!("{i}.velocity");
            if let Some(dt) = pack.tensors.get(&key) {
                if let Some(src) = dt.as_float() {
                    p.velocity.copy_(src)?;
                }
            }
        }
        Ok(())
    }
}
