use std::ops::{Deref, DerefMut};

use luma_tensor::{Device, Float, Tensor};

use super::module::Module;
use super::visitor::{TensorVisitor, TensorVisitorMut};

// ============================================================================================ //
//                        Parameter
// ============================================================================================ //

/// A trainable `Float` tensor registered as a *parameter*.
///
/// During `train(true)` / `eval()` the parameter's `requires_grad` flag is
/// toggled so that the autograd engine only tracks it in training mode.
#[derive(Clone)]
pub struct Parameter<D: Device>(pub Tensor<D, Float>);

impl<D: Device> Deref for Parameter<D> {
    type Target = Tensor<D, Float>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.tensor()
    }
}

impl<D: Device> DerefMut for Parameter<D> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.tensor_mut()
    }
}

impl<D: Device> Parameter<D> {
    pub fn new(tensor: Tensor<D, Float>) -> Self {
        tensor.set_requires_grad(true);
        Self(tensor)
    }

    pub fn tensor(&self) -> &Tensor<D, Float> {
        &self.0
    }

    pub fn tensor_mut(&mut self) -> &mut Tensor<D, Float> {
        &mut self.0
    }
}

impl<D: Device> Module<D> for Parameter<D> {
    #[inline]
    fn visit_param<Visitor: TensorVisitor<D>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        visitor.visit_float(&self.0)
    }

    #[inline]
    fn visit_param_mut<Visitor: TensorVisitorMut<D>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        visitor.visit_float_mut(&mut self.0)
    }

    #[inline]
    fn visit_state<Visitor: TensorVisitor<D>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        visitor.visit_float(&self.0)
    }

    #[inline]
    fn visit_state_mut<Visitor: TensorVisitorMut<D>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        visitor.visit_float_mut(&mut self.0)
    }

    #[inline]
    fn set_train(&mut self, mode: bool) {
        self.0.set_requires_grad(mode);
    }
}
