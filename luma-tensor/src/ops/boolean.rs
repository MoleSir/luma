//! Logical ops, reductions, and masked selection for `Bool` tensors. No autograd.

use crate::{Bool, DTypeKind, Device, Float, Int, Layout, Tensor, TensorMeta};

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
        Ok(self.true_count()? == self.element_count())
    }

    pub fn any_all(&self) -> crate::Result<bool> {
        Ok(self.true_count()? > 0)
    }

    pub fn true_count(&self) -> crate::Result<usize> {
        D::b_true_count(&*self.storage_read()?, self.layout())
    }
}

pub trait PickDTypeKind<D: Device>: DTypeKind<D> {
    fn pick_dispatch(
        mask: &D::BoolStorage,
        mask_l: &Layout,
        on_true: &Self::Storage,
        true_l: &Layout,
        on_false: &Self::Storage,
        false_l: &Layout,
    ) -> crate::Result<Self::Storage>;

    fn pick_true_dispatch(
        mask: &D::BoolStorage,
        mask_l: &Layout,
        value: Self::Scalar,
        on_false: &Self::Storage,
        false_l: &Layout,
    ) -> crate::Result<Self::Storage>;

    fn pick_false_dispatch(
        mask: &D::BoolStorage,
        mask_l: &Layout,
        on_true: &Self::Storage,
        true_l: &Layout,
        value: Self::Scalar,
    ) -> crate::Result<Self::Storage>;
}

impl<D: Device> PickDTypeKind<D> for Float {
    fn pick_dispatch(
        mask: &<D as Device>::BoolStorage,
        mask_l: &Layout,
        on_true: &Self::Storage,
        true_l: &Layout,
        on_false: &Self::Storage,
        false_l: &Layout,
    ) -> crate::Result<Self::Storage> {
        D::f_pick(mask, mask_l, on_true, true_l, on_false, false_l)
    }

    fn pick_true_dispatch(
        mask: &<D as Device>::BoolStorage,
        mask_l: &Layout,
        value: Self::Scalar,
        on_false: &Self::Storage,
        false_l: &Layout,
    ) -> crate::Result<Self::Storage> {
        D::f_pick_true(mask, mask_l, value, on_false, false_l)
    }

    fn pick_false_dispatch(
        mask: &<D as Device>::BoolStorage,
        mask_l: &Layout,
        on_true: &Self::Storage,
        true_l: &Layout,
        value: Self::Scalar,
    ) -> crate::Result<Self::Storage> {
        D::f_pick_false(mask, mask_l, on_true, true_l, value)
    }
}

impl<D: Device> PickDTypeKind<D> for Int {
    fn pick_dispatch(
        mask: &<D as Device>::BoolStorage,
        mask_l: &Layout,
        on_true: &Self::Storage,
        true_l: &Layout,
        on_false: &Self::Storage,
        false_l: &Layout,
    ) -> crate::Result<Self::Storage> {
        D::i_pick(mask, mask_l, on_true, true_l, on_false, false_l)
    }

    fn pick_true_dispatch(
        mask: &<D as Device>::BoolStorage,
        mask_l: &Layout,
        value: Self::Scalar,
        on_false: &Self::Storage,
        false_l: &Layout,
    ) -> crate::Result<Self::Storage> {
        D::i_pick_true(mask, mask_l, value, on_false, false_l)
    }

    fn pick_false_dispatch(
        mask: &<D as Device>::BoolStorage,
        mask_l: &Layout,
        on_true: &Self::Storage,
        true_l: &Layout,
        value: Self::Scalar,
    ) -> crate::Result<Self::Storage> {
        D::i_pick_false(mask, mask_l, on_true, true_l, value)
    }
}

impl<D: Device> PickDTypeKind<D> for Bool {
    fn pick_dispatch(
        mask: &<D as Device>::BoolStorage,
        mask_l: &Layout,
        on_true: &Self::Storage,
        true_l: &Layout,
        on_false: &Self::Storage,
        false_l: &Layout,
    ) -> crate::Result<Self::Storage> {
        D::b_pick(mask, mask_l, on_true, true_l, on_false, false_l)
    }

    fn pick_true_dispatch(
        mask: &<D as Device>::BoolStorage,
        mask_l: &Layout,
        value: Self::Scalar,
        on_false: &Self::Storage,
        false_l: &Layout,
    ) -> crate::Result<Self::Storage> {
        D::b_pick_true(mask, mask_l, value, on_false, false_l)
    }

    fn pick_false_dispatch(
        mask: &<D as Device>::BoolStorage,
        mask_l: &Layout,
        on_true: &Self::Storage,
        true_l: &Layout,
        value: Self::Scalar,
    ) -> crate::Result<Self::Storage> {
        D::b_pick_false(mask, mask_l, on_true, true_l, value)
    }
}

impl<D: Device> Tensor<D, Bool> {
    pub fn pick<K: PickDTypeKind<D>>(&self, on_true: &Tensor<D, K>, on_false: &Tensor<D, K>) -> crate::Result<Tensor<D, K>> {
        let storage = K::pick_dispatch(
            &*self.storage_read()?,
            self.layout(),
            &*on_true.storage_read()?,
            on_true.layout(),
            &*on_false.storage_read()?,
            on_false.layout(),
        )?;
        let meta = K::Meta::on_pick(self, Some(on_true), Some(on_false));
        Ok(Tensor::from_storage(storage, on_true.shape().clone(), meta))
    }

    /// `mask ? value : on_false` with a scalar true-value.
    pub fn pick_true<K: PickDTypeKind<D>>(&self, value: K::Scalar, on_false: &Tensor<D, K>) -> crate::Result<Tensor<D, K>> {
        let storage = K::pick_true_dispatch(
            &*self.storage_read()?,
            self.layout(),
            value,
            &*on_false.storage_read()?,
            on_false.layout(),
        )?;
        let meta = K::Meta::on_pick(self, None, Some(on_false));
        Ok(Tensor::from_storage(storage, on_false.shape().clone(), meta))
    }

    /// `mask ? on_true : value` with a scalar false-value.
    pub fn pick_false<K: PickDTypeKind<D>>(&self, on_true: &Tensor<D, K>, value: K::Scalar) -> crate::Result<Tensor<D, K>> {
        let storage = K::pick_false_dispatch(
            &*self.storage_read()?,
            self.layout(),
            &*on_true.storage_read()?,
            on_true.layout(),
            value,
        )?;
        let meta = K::Meta::on_pick(self, Some(on_true), None);
        Ok(Tensor::from_storage(storage, on_true.shape().clone(), meta))
    }
}
