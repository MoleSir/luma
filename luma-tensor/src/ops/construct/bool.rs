use std::borrow::Cow;

use crate::{Bool, Device, Result, Shape, Tensor};

use super::helpers;
use super::into_tensor::IntoTensor;
use super::options::TensorCreationOptions;

impl<D: Device> Tensor<D, Bool> {
    pub fn new(data: impl IntoTensor<D, Bool>, device: &D) -> Result<Self> {
        let shape = data.shape()?;
        let storage = data.into_storage(device)?;
        Ok(Self::from_storage(storage, shape, ()))
    }

    pub fn from_vec_bool<'a, S: Into<Shape>>(data: impl Into<Cow<'a, [bool]>>, shape: S, device: &D) -> Result<Self> {
        let shape = shape.into();
        let storage = D::b_from_bool(data, device)?;
        Ok(Self::from_storage(storage, shape, ()))
    }

    pub fn falses<S: Into<Shape>>(shape: S, options: impl Into<TensorCreationOptions<D, Bool>>) -> Result<Self> {
        let options: TensorCreationOptions<D, Bool> = options.into();
        let shape = shape.into();
        let storage = D::b_falses(&shape, &options.device, options.dtype)?;
        Ok(Self::from_storage(storage, shape, ()))
    }

    pub fn trues<S: Into<Shape>>(shape: S, options: impl Into<TensorCreationOptions<D, Bool>>) -> Result<Self> {
        let options: TensorCreationOptions<D, Bool> = options.into();
        let shape = shape.into();
        let storage = D::b_trues(&shape, &options.device, options.dtype)?;
        Ok(Self::from_storage(storage, shape, ()))
    }

    pub fn from_slice<S: Into<Shape>>(data: &[bool], shape: S, options: impl Into<TensorCreationOptions<D, Bool>>) -> Result<Self> {
        let options: TensorCreationOptions<D, Bool> = options.into();
        let shape = shape.into();
        if shape.element_count() != data.len() {
            return Err(crate::Error::ElementSizeMismatch { expected: data.len(), got: shape.element_count(), op: "from_slice" });
        }
        let storage = D::b_from_bool(data, &options.device)?;
        Ok(Self::from_storage(storage, shape, ()))
    }

    pub fn to_vec(&self) -> crate::Result<Vec<bool>> {
        D::b_to_vec(&*self.storage_read()?, self.layout())
    }

    pub fn eye(n: usize, options: impl Into<TensorCreationOptions<D, Bool>>) -> Result<Self> {
        let options = options.into();
        Self::new(helpers::fill_eye::<bool>(n), &options.device)
    }

    pub fn diag(diag: &[bool], options: impl Into<TensorCreationOptions<D, Bool>>) -> Result<Self> {
        let options = options.into();
        let n = diag.len();
        let mut v = vec![false; n * n];
        for i in 0..n {
            v[i * n + i] = diag[i];
        }
        Self::new(v, &options.device)
    }

    pub fn tril(n: usize, diagonal: bool, options: impl Into<TensorCreationOptions<D, Bool>>) -> Result<Self> {
        let options = options.into();
        Self::new(helpers::fill_tril::<bool>(n, diagonal), &options.device)
    }

    pub fn triu(n: usize, diagonal: bool, options: impl Into<TensorCreationOptions<D, Bool>>) -> Result<Self> {
        let options = options.into();
        Self::new(helpers::fill_triu::<bool>(n, diagonal), &options.device)
    }
}
