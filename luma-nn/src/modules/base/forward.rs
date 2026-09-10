use luma_tensor::{Device, Float, Int, Tensor};

use super::module::Module;
use crate::activate::{GELU, LeakyReLU, ReLU, SiLU, Sigmoid, Tanh};
use crate::loss::{BCELoss, CrossEntropyLoss, MSELoss};
use crate::{BatchNorm1d, Dropout, Embedding, LayerNorm, Linear, NnResult, RMSNorm};

// ============================================================================================ //
//                        ModuleForward
// ============================================================================================ //

/// A uniform `forward` interface over [`Module`]s.
///
/// `Module<D>` deliberately has no `forward` (every module defines it as an
/// inherent method with its own signature). This trait pins a module's
/// input/output types so generic machinery — e.g. `luma_compile::trace` — can feed
/// an example input through any module:
///
/// - single-input modules use `Input = Tensor<D, K>`;
/// - multi-input modules (losses) use a tuple, e.g.
///   `Input = (Tensor<D, Float>, Tensor<D, Int>)`.
pub trait ModuleForward<D: Device>: Module<D> {
    type Input;
    type Output;

    fn forward(&self, input: &Self::Input) -> NnResult<Self::Output>;
}

/// Delegate impls for modules whose `forward` takes a single `Float` tensor.
macro_rules! impl_forward_single_float {
    ($($ty:ident),* $(,)?) => {
        $(
            impl<D: Device> ModuleForward<D> for $ty<D> {
                type Input = Tensor<D, Float>;
                type Output = Tensor<D, Float>;

                fn forward(&self, input: &Self::Input) -> NnResult<Self::Output> {
                    // Inherent method wins over the trait method being defined.
                    Self::forward(self, input)
                }
            }
        )*
    };
}

impl_forward_single_float!(Linear, BatchNorm1d, RMSNorm, LayerNorm, Dropout, ReLU, LeakyReLU, Sigmoid, Tanh, GELU, SiLU,);

impl<D: Device> ModuleForward<D> for Embedding<D> {
    type Input = Tensor<D, Int>;
    type Output = Tensor<D, Float>;

    fn forward(&self, input: &Self::Input) -> NnResult<Self::Output> {
        Self::forward(self, input)
    }
}

macro_rules! impl_forward_two_float {
    ($($ty:ident),* $(,)?) => {
        $(
            impl<D: Device> ModuleForward<D> for $ty<D> {
                type Input = (Tensor<D, Float>, Tensor<D, Float>);
                type Output = Tensor<D, Float>;

                fn forward(&self, input: &Self::Input) -> NnResult<Self::Output> {
                    Self::forward(self, &input.0, &input.1)
                }
            }
        )*
    };
}

impl_forward_two_float!(MSELoss, BCELoss);

impl<D: Device> ModuleForward<D> for CrossEntropyLoss<D> {
    type Input = (Tensor<D, Float>, Tensor<D, Int>);
    type Output = Tensor<D, Float>;

    fn forward(&self, input: &Self::Input) -> NnResult<Self::Output> {
        Self::forward(self, &input.0, &input.1)
    }
}
