//! Type conversion between kinds and precisions.
//!
//! `Float::cast` (precision change within the float kind) records `Op::Cast` so
//! its gradient flows back — the capability the old typed design could not express.

use crate::{
    Bool, DTypeKind, Device, Float, FloatMeta, Int, Tensor,
    dtype::{BoolDType, FloatDType, IntDType},
};

pub trait Cast<D: Device, K: DTypeKind<D>> {
    type Output: DTypeKind<D>;
    fn cast(tensor: &Tensor<D, K>, dtype: Self) -> crate::Result<Tensor<D, Self::Output>>;
}

impl<D: Device, K: DTypeKind<D>> Tensor<D, K> {
    pub fn cast<T: Cast<D, K>>(&self, dtype: T) -> crate::Result<Tensor<D, T::Output>> {
        T::cast(self, dtype)
    }
}

impl<D: Device, K: CastDTypeKind<D>> Cast<D, K> for FloatDType {
    type Output = Float;
    fn cast(tensor: &Tensor<D, K>, dtype: Self) -> crate::Result<Tensor<D, Self::Output>> {
        tensor.cast_float(dtype)
    }
}

impl<D: Device, K: CastDTypeKind<D>> Cast<D, K> for IntDType {
    type Output = Int;
    fn cast(tensor: &Tensor<D, K>, dtype: Self) -> crate::Result<Tensor<D, Self::Output>> {
        tensor.cast_int(dtype)
    }
}

impl<D: Device, K: CastDTypeKind<D>> Cast<D, K> for BoolDType {
    type Output = Bool;
    fn cast(tensor: &Tensor<D, K>, dtype: Self) -> crate::Result<Tensor<D, Self::Output>> {
        tensor.cast_bool(dtype)
    }
}

pub trait CastDTypeKind<D: Device>: DTypeKind<D> {
    fn cast_float(tensor: &Tensor<D, Self>, to: FloatDType) -> crate::Result<Tensor<D, Float>>;
    fn cast_int(tensor: &Tensor<D, Self>, to: IntDType) -> crate::Result<Tensor<D, Int>>;
    fn cast_bool(tensor: &Tensor<D, Self>, to: BoolDType) -> crate::Result<Tensor<D, Bool>>;
}

impl<D: Device> CastDTypeKind<D> for Float {
    fn cast_float(tensor: &Tensor<D, Self>, to: FloatDType) -> crate::Result<Tensor<D, Float>> {
        let storage = D::f_cast_float(&*tensor.storage_read()?, tensor.layout(), to)?;
        let meta = FloatMeta::on_cast(tensor);
        Ok(Tensor::from_storage(storage, tensor.shape().clone(), meta))
    }

    fn cast_int(tensor: &Tensor<D, Self>, to: IntDType) -> crate::Result<Tensor<D, Int>> {
        let storage = D::f_cast_int(&*tensor.storage_read()?, tensor.layout(), to)?;
        Ok(Tensor::from_storage(storage, tensor.shape().clone(), ()))
    }

    fn cast_bool(tensor: &Tensor<D, Self>, to: BoolDType) -> crate::Result<Tensor<D, Bool>> {
        let storage = D::f_cast_bool(&*tensor.storage_read()?, tensor.layout(), to)?;
        Ok(Tensor::from_storage(storage, tensor.shape().clone(), ()))
    }
}

impl<D: Device> CastDTypeKind<D> for Int {
    fn cast_float(tensor: &Tensor<D, Self>, to: FloatDType) -> crate::Result<Tensor<D, Float>> {
        let storage = D::i_cast_float(&*tensor.storage_read()?, tensor.layout(), to)?;
        Ok(Tensor::from_storage(storage, tensor.shape().clone(), FloatMeta::val()))
    }

    fn cast_int(tensor: &Tensor<D, Self>, to: IntDType) -> crate::Result<Tensor<D, Int>> {
        let storage = D::i_cast_int(&*tensor.storage_read()?, tensor.layout(), to)?;
        Ok(Tensor::from_storage(storage, tensor.shape().clone(), ()))
    }

    fn cast_bool(tensor: &Tensor<D, Self>, to: BoolDType) -> crate::Result<Tensor<D, Bool>> {
        let storage = D::i_cast_bool(&*tensor.storage_read()?, tensor.layout(), to)?;
        Ok(Tensor::from_storage(storage, tensor.shape().clone(), ()))
    }
}

impl<D: Device> CastDTypeKind<D> for Bool {
    fn cast_float(tensor: &Tensor<D, Self>, to: FloatDType) -> crate::Result<Tensor<D, Float>> {
        let storage = D::b_cast_float(&*tensor.storage_read()?, tensor.layout(), to)?;
        Ok(Tensor::from_storage(storage, tensor.shape().clone(), FloatMeta::val()))
    }

    fn cast_int(tensor: &Tensor<D, Self>, to: IntDType) -> crate::Result<Tensor<D, Int>> {
        let storage = D::b_cast_int(&*tensor.storage_read()?, tensor.layout(), to)?;
        Ok(Tensor::from_storage(storage, tensor.shape().clone(), ()))
    }

    fn cast_bool(tensor: &Tensor<D, Self>, to: BoolDType) -> crate::Result<Tensor<D, Bool>> {
        let storage = D::b_cast_bool(&*tensor.storage_read()?, tensor.layout(), to)?;
        Ok(Tensor::from_storage(storage, tensor.shape().clone(), ()))
    }
}

impl<D: Device, K: CastDTypeKind<D>> Tensor<D, K> {
    #[inline]
    pub fn cast_float(&self, to: FloatDType) -> crate::Result<Tensor<D, Float>> {
        K::cast_float(self, to)
    }

    #[inline]
    pub fn cast_int(&self, to: IntDType) -> crate::Result<Tensor<D, Int>> {
        K::cast_int(self, to)
    }

    #[inline]
    pub fn cast_bool(&self, to: BoolDType) -> crate::Result<Tensor<D, Bool>> {
        K::cast_bool(self, to)
    }
}
