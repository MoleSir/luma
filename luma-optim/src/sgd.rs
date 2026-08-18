use luma_io::lpk::LumaPack;
use luma_tensor::{Device, GradStore, Scalar, Tensor, no_grad};

use super::Optimizer;

pub struct SGD<D: Device> {
    pub params: Vec<Tensor<D>>,
    pub learning_rate: f64,
}

impl<D: Device> SGD<D> {
    pub fn new(params: impl Into<Vec<Tensor<D>>>, learning_rate: f64) -> Self {
        Self { params: params.into(), learning_rate }
    }
}

impl<D: Device> Optimizer for SGD<D> {
    type Device = D;

    fn get_lr(&self) -> f64 {
        self.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    /// w_t = w_{t-1} - lr * g
    fn step(&mut self, grads: &GradStore<Self::Device>) -> luma_tensor::Result<()> {
        no_grad!();
        for var in self.params.iter() {
            if let Some(grad) = grads.get(var) {
                var.sub_(&grad.mul_scalar(self.learning_rate)?)?;
            }
        }
        Ok(())
    }

    fn state_dict(&self) -> luma_tensor::Result<LumaPack<Self::Device>> {
        let mut pack = LumaPack::new();
        pack.scalars.insert("lr".into(), Scalar::F64(self.learning_rate));
        Ok(pack)
    }

    fn load_state_dict(&mut self, pack: &LumaPack<Self::Device>) -> luma_tensor::Result<()> {
        if let Some(lr) = pack.scalars.get("lr").and_then(|s| s.to_f64()) {
            self.learning_rate = lr;
        }
        Ok(())
    }
}

pub struct SGDM<D: Device> {
    pub params: Vec<SGDMParam<D>>,
    pub learning_rate: f64,
    pub momentum: f64,
}

pub struct SGDMParam<D: Device> {
    pub param: Tensor<D>,
    pub velocity: Tensor<D>,
}

impl<D: Device> Optimizer for SGDM<D> {
    type Device = D;

    fn get_lr(&self) -> f64 {
        self.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    ///
    /// v = m*v + grad
    ///
    /// w_t = w_{t-1} - lr * v
    ///
    fn step(&mut self, grads: &GradStore<Self::Device>) -> luma_tensor::Result<()> {
        no_grad!();
        for SGDMParam { param, velocity } in self.params.iter_mut() {
            if let Some(grad) = grads.get(&param) {
                // update v: v = m * v + grad
                velocity.mul_scalar_(self.momentum)?;
                velocity.add_(grad)?;
                param.sub_(&velocity.mul_scalar(self.learning_rate)?)?;
            }
        }

        Ok(())
    }

    fn state_dict(&self) -> luma_tensor::Result<LumaPack<Self::Device>> {
        let mut pack = LumaPack::new();
        for (i, p) in self.params.iter().enumerate() {
            pack.tensors.insert(format!("{i}.velocity"), luma_tensor::DynTensor::Float(p.velocity.clone()));
        }
        pack.scalars.insert("lr".into(), Scalar::F64(self.learning_rate));
        pack.scalars.insert("momentum".into(), Scalar::F64(self.momentum));
        Ok(pack)
    }

    fn load_state_dict(&mut self, pack: &LumaPack<Self::Device>) -> luma_tensor::Result<()> {
        if let Some(lr) = pack.scalars.get("lr").and_then(|s| s.to_f64()) {
            self.learning_rate = lr;
        }
        if let Some(m) = pack.scalars.get("momentum").and_then(|s| s.to_f64()) {
            self.momentum = m;
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
