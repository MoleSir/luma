use luma_io::lpk::LumaPack;
use luma_tensor::{Device, DynTensor, GradStore, Scalar, Tensor, no_grad};

use super::Optimizer;

#[derive(Clone, Debug)]
pub struct RMSPropConfig {
    pub lr: f64,
    pub alpha: f64, // 平滑常数（通常设为 0.99）
    pub eps: f64,   // 极小值（通常设为 1e-8）
    pub weight_decay: f64,
}

impl Default for RMSPropConfig {
    fn default() -> Self {
        Self { lr: 1e-2, alpha: 0.99, eps: 1e-8, weight_decay: 0.0 }
    }
}

struct RMSPropParam<D: Device> {
    param: Tensor<D>,
    square_avg: Tensor<D>,
}

pub struct RMSProp<D: Device> {
    params: Vec<RMSPropParam<D>>,
    config: RMSPropConfig,
}

impl<D: Device> RMSProp<D> {
    pub fn new(params: impl Into<Vec<Tensor<D>>>, config: RMSPropConfig) -> luma_tensor::Result<Self> {
        let params = params
            .into()
            .into_iter()
            .map(|param| {
                let square_avg = param.zeros_like()?;
                Ok(RMSPropParam { param, square_avg })
            })
            .collect::<luma_tensor::Result<Vec<_>>>()?;
        Ok(Self { params, config })
    }
}

impl<D: Device> Optimizer for RMSProp<D> {
    type Device = D;

    fn set_lr(&mut self, lr: f64) {
        self.config.lr = lr;
    }

    fn get_lr(&self) -> f64 {
        self.config.lr
    }

    fn step(&mut self, grads: &GradStore<Self::Device>) -> luma_tensor::Result<()> {
        no_grad!();
        let lr = self.config.lr;
        let alpha = self.config.alpha;
        let eps = self.config.eps;
        let weight_decay = self.config.weight_decay;

        for RMSPropParam { param, square_avg } in self.params.iter_mut() {
            if let Some(g) = grads.get(&param) {
                let g = g.clone();

                /*
                    1. weight decay

                    w_t = w_{t-1} - lr*g
                    w_t = (1-lr*eta)w_{t-1} - lr*g
                    w_t = w_{t-1} - lr*(g+eta*w_{t+1})

                    g = g+eta*w_{t+1}
                */
                if weight_decay != 0.0 {
                    g.add_(&param.mul_scalar(weight_decay)?)?;
                }

                // 2. update square_avg
                // square_avg = alpha*square_avg + (1-alpha)*g^2
                square_avg.mul_scalar_(alpha)?;
                square_avg.add_(&g.pow(2.0)?.mul_scalar(1. - alpha)?)?;

                // 3. update param
                let denom = square_avg.sqrt()?.add_scalar(eps)?;
                param.sub_(&g.div(&denom)?.mul_scalar(lr)?)?;
            }
        }

        Ok(())
    }

    fn state_dict(&self) -> luma_tensor::Result<LumaPack<Self::Device>> {
        let mut pack = LumaPack::new();
        for (i, p) in self.params.iter().enumerate() {
            pack.tensors.insert(format!("{i}.square_avg"), DynTensor::Float(p.square_avg.clone()));
        }
        pack.scalars.insert("lr".into(), Scalar::F64(self.config.lr));
        pack.scalars.insert("alpha".into(), Scalar::F64(self.config.alpha));
        pack.scalars.insert("eps".into(), Scalar::F64(self.config.eps));
        pack.scalars.insert("weight_decay".into(), Scalar::F64(self.config.weight_decay));
        Ok(pack)
    }

    fn load_state_dict(&mut self, pack: &LumaPack<Self::Device>) -> luma_tensor::Result<()> {
        if let Some(v) = pack.scalars.get("lr").and_then(|s| s.to_f64()) {
            self.config.lr = v;
        }
        if let Some(v) = pack.scalars.get("alpha").and_then(|s| s.to_f64()) {
            self.config.alpha = v;
        }
        if let Some(v) = pack.scalars.get("eps").and_then(|s| s.to_f64()) {
            self.config.eps = v;
        }
        if let Some(v) = pack.scalars.get("weight_decay").and_then(|s| s.to_f64()) {
            self.config.weight_decay = v;
        }
        for (i, p) in self.params.iter_mut().enumerate() {
            let key = format!("{i}.square_avg");
            if let Some(dt) = pack.tensors.get(&key) {
                if let Some(src) = dt.as_float() {
                    p.square_avg.copy_(src)?;
                }
            }
        }
        Ok(())
    }
}
