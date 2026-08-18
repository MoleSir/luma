mod adam;
mod momentum;
mod rms_prop;
mod sgd;

use luma_io::lpk::LumaPack;
use luma_tensor::{Device, GradStore};

pub use adam::*;
pub use momentum::*;
pub use rms_prop::*;
pub use sgd::*;

pub trait Optimizer {
    type Device: Device;

    fn get_lr(&self) -> f64;
    fn set_lr(&mut self, lr: f64);
    fn step(&mut self, grads: &GradStore<Self::Device>) -> luma_tensor::Result<()>;
    fn state_dict(&self) -> luma_tensor::Result<LumaPack<Self::Device>>;
    fn load_state_dict(&mut self, pack: &LumaPack<Self::Device>) -> luma_tensor::Result<()>;
}
