mod common;
pub mod mem;
use crate::Graph;
pub use common::{GraphExecutor, Step};
use luma_tensor::{Device, DynTensor};

pub trait Executor: Sized {
    type Dev: Device;
    type Err: std::error::Error;

    fn compile(graph: &Graph, device: &Self::Dev) -> Result<Self, Self::Err>;
    fn run(&mut self, inputs: &[DynTensor<Self::Dev>]) -> Result<Vec<DynTensor<Self::Dev>>, Self::Err>;
}
