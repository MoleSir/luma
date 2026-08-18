use std::ops::{Deref, DerefMut};

use luma_tensor::{Bool, DTypeKind, Device, Float, Int, Tensor};

use super::module::Module;
use super::visitor::{TensorVisitor, TensorVisitorMut};

// ============================================================================================ //
//                        VisitorDispatch — maps kind → visitor method
// ============================================================================================ //

/// Maps a compile-time [`DTypeKind`] marker to the correct `visit_*` method.
///
/// Implemented for [`Float`], [`Int`], and [`Bool`] so that [`Buffer`] can
/// delegate to the correct visitor method without runtime branching.
trait VisitorDispatch<D: Device>: DTypeKind<D> {
    fn visit_tensor<V: TensorVisitor<D>>(tensor: &Tensor<D, Self>, visitor: &mut V) -> Result<(), V::Error>;

    fn visit_tensor_mut<V: TensorVisitorMut<D>>(tensor: &mut Tensor<D, Self>, visitor: &mut V) -> Result<(), V::Error>;
}

impl<D: Device> VisitorDispatch<D> for Float {
    fn visit_tensor<V: TensorVisitor<D>>(tensor: &Tensor<D, Self>, visitor: &mut V) -> Result<(), V::Error> {
        visitor.visit_float(tensor)
    }
    fn visit_tensor_mut<V: TensorVisitorMut<D>>(tensor: &mut Tensor<D, Self>, visitor: &mut V) -> Result<(), V::Error> {
        visitor.visit_float_mut(tensor)
    }
}

impl<D: Device> VisitorDispatch<D> for Int {
    fn visit_tensor<V: TensorVisitor<D>>(tensor: &Tensor<D, Self>, visitor: &mut V) -> Result<(), V::Error> {
        visitor.visit_int(tensor)
    }
    fn visit_tensor_mut<V: TensorVisitorMut<D>>(tensor: &mut Tensor<D, Self>, visitor: &mut V) -> Result<(), V::Error> {
        visitor.visit_int_mut(tensor)
    }
}

impl<D: Device> VisitorDispatch<D> for Bool {
    fn visit_tensor<V: TensorVisitor<D>>(tensor: &Tensor<D, Self>, visitor: &mut V) -> Result<(), V::Error> {
        visitor.visit_bool(tensor)
    }
    fn visit_tensor_mut<V: TensorVisitorMut<D>>(tensor: &mut Tensor<D, Self>, visitor: &mut V) -> Result<(), V::Error> {
        visitor.visit_bool_mut(tensor)
    }
}

// ============================================================================================ //
//                        Buffer
// ============================================================================================ //

/// A non-trainable tensor registered as a *buffer* (e.g. running-mean in
/// BatchNorm).  Buffers are part of the module's persisted state but are
/// never touched by the optimiser.
///
/// The default kind is [`Float`]; use `Buffer<D, Int>` for integer buffers
/// (e.g. `num_batches_tracked`).
#[derive(Clone)]
pub struct Buffer<D: Device, K: DTypeKind<D> = Float>(pub Tensor<D, K>);

impl<D: Device, K: DTypeKind<D>> Deref for Buffer<D, K> {
    type Target = Tensor<D, K>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.tensor()
    }
}

impl<D: Device, K: DTypeKind<D>> DerefMut for Buffer<D, K> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.tensor_mut()
    }
}

impl<D: Device> Buffer<D, Float> {
    pub fn new(tensor: Tensor<D, Float>) -> Self {
        tensor.set_requires_grad(false);
        Self(tensor)
    }
}

impl<D: Device> Buffer<D, Bool> {
    pub fn new(tensor: Tensor<D, Bool>) -> Self {
        Self(tensor)
    }
}

impl<D: Device> Buffer<D, Int> {
    pub fn new(tensor: Tensor<D, Int>) -> Self {
        Self(tensor)
    }
}

impl<D: Device, K: DTypeKind<D>> Buffer<D, K> {
    pub fn tensor(&self) -> &Tensor<D, K> {
        &self.0
    }

    pub fn tensor_mut(&mut self) -> &mut Tensor<D, K> {
        &mut self.0
    }
}

impl<D: Device, K: VisitorDispatch<D>> Module<D> for Buffer<D, K> {
    #[inline]
    fn visit_buffer<Visitor: TensorVisitor<D>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        K::visit_tensor(&self.0, visitor)
    }

    #[inline]
    fn visit_buffer_mut<Visitor: TensorVisitorMut<D>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        K::visit_tensor_mut(&mut self.0, visitor)
    }

    #[inline]
    fn visit_state<Visitor: TensorVisitor<D>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        K::visit_tensor(&self.0, visitor)
    }

    #[inline]
    fn visit_state_mut<Visitor: TensorVisitorMut<D>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        K::visit_tensor_mut(&mut self.0, visitor)
    }
}
