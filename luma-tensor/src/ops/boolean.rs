//! Logical ops, reductions, and masked selection for `Bool` tensors. No autograd.

use crate::{Bool, Device, Float, FloatMeta, Tensor};

impl<D: Device> Tensor<D, Bool> {
    pub fn and(&self, rhs: &Self) -> crate::Result<Self> {
        let shape = self.same_shape(rhs, "and")?.clone();
        let storage = D::b_and(&*self.storage_read()?, self.layout(), &*rhs.storage_read()?, rhs.layout())?;
        Ok(Self::from_storage(storage, shape, ()))
    }

    pub fn or(&self, rhs: &Self) -> crate::Result<Self> {
        let shape = self.same_shape(rhs, "or")?.clone();
        let storage = D::b_or(&*self.storage_read()?, self.layout(), &*rhs.storage_read()?, rhs.layout())?;
        Ok(Self::from_storage(storage, shape, ()))
    }

    pub fn xor(&self, rhs: &Self) -> crate::Result<Self> {
        let shape = self.same_shape(rhs, "xor")?.clone();
        let storage = D::b_xor(&*self.storage_read()?, self.layout(), &*rhs.storage_read()?, rhs.layout())?;
        Ok(Self::from_storage(storage, shape, ()))
    }

    pub fn not(&self) -> crate::Result<Self> {
        let storage = D::b_not(&*self.storage_read()?, self.layout())?;
        Ok(Self::from_storage(storage, self.shape().clone(), ()))
    }

    pub fn all_all(&self) -> crate::Result<bool> {
        // "all true" == true_count equals element_count
        Ok(self.true_count()? == self.element_count())
    }

    pub fn any_all(&self) -> crate::Result<bool> {
        Ok(self.true_count()? > 0)
    }

    pub fn true_count(&self) -> crate::Result<usize> {
        D::b_true_count(&*self.storage_read()?, self.layout())
    }
}

impl<D: Device> Tensor<D, Bool> {
    /// Elementwise select: `mask ? on_true : on_false`. Records `Op::IfElse`.
    pub fn if_else(&self, on_true: &Tensor<D, Float>, on_false: &Tensor<D, Float>) -> crate::Result<Tensor<D, Float>> {
        let storage = D::f_if_else(
            &*self.storage_read()?,
            self.layout(),
            &*on_true.storage_read()?,
            on_true.layout(),
            &*on_false.storage_read()?,
            on_false.layout(),
        )?;
        let meta = FloatMeta::on_if_else(self, Some(on_true), Some(on_false));
        Ok(Tensor::<D, Float>::from_storage(storage, on_true.shape().clone(), meta))
    }

    /// `mask ? value : on_false` with a scalar true-value.
    pub fn if_else_scalar_true(&self, value: f64, on_false: &Tensor<D, Float>) -> crate::Result<Tensor<D, Float>> {
        let on_true = Tensor::<D, Float>::full(on_false.shape().clone(), value, on_false.dtype())?;
        self.if_else(&on_true, on_false)
    }

    /// `mask ? on_true : value` with a scalar false-value.
    pub fn if_else_scalar_false(&self, on_true: &Tensor<D, Float>, value: f64) -> crate::Result<Tensor<D, Float>> {
        let on_false = Tensor::<D, Float>::full(on_true.shape().clone(), value, on_true.dtype())?;
        self.if_else(on_true, &on_false)
    }
}
