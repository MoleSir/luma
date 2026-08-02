pub mod bool_ops;
pub mod cpu;
pub mod float_ops;
pub mod int_ops;

pub use bool_ops::BoolOps;
pub use cpu::Cpu;
pub use float_ops::FloatOps;
pub use int_ops::IntOps;

use crate::{dtype::Storage, Bool, Float, Int};

pub trait Device: 
    'static + Copy + Clone + 
    Send + Sync + 
    Default +
    FloatOps<Self> + 
    IntOps<Self> +
    BoolOps<Self> 
{
    type FloatStorage: Storage<Self, Float>;
    type IntStorage: Storage<Self, Int>;
    type BoolStorage: Storage<Self, Bool>;

    fn name(&self) -> String;
}
