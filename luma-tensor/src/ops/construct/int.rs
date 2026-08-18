use std::borrow::Cow;

use crate::{Device, Int, Result, Shape, Tensor, dtype::IntDType};

use super::helpers;
use super::into_tensor::IntoTensor;
use super::options::TensorCreationOptions;

impl<D: Device> Tensor<D, Int> {
    pub fn new(data: impl IntoTensor<D, Int>, device: &D) -> Result<Self> {
        let shape = data.shape()?;
        let storage = data.into_storage(device)?;
        Ok(Self::from_storage(storage, shape, ()))
    }

    pub fn from_vec_i64<'a, S: Into<Shape>>(data: impl Into<Cow<'a, [i64]>>, shape: S, device: &D) -> Result<Self> {
        let shape = shape.into();
        let storage = D::i_from_i64(data, device)?;
        Ok(Self::from_storage(storage, shape, ()))
    }

    pub fn from_vec_i32<'a, S: Into<Shape>>(data: impl Into<Cow<'a, [i32]>>, shape: S, device: &D) -> Result<Self> {
        let shape = shape.into();
        let storage = D::i_from_i32(data, device)?;
        Ok(Self::from_storage(storage, shape, ()))
    }

    pub fn from_vec_u32<'a, S: Into<Shape>>(data: impl Into<Cow<'a, [u32]>>, shape: S, device: &D) -> Result<Self> {
        let shape = shape.into();
        let storage = D::i_from_u32(data, device)?;
        Ok(Self::from_storage(storage, shape, ()))
    }

    pub fn from_vec_u8<'a, S: Into<Shape>>(data: impl Into<Cow<'a, [u8]>>, shape: S, device: &D) -> Result<Self> {
        let shape = shape.into();
        let storage = D::i_from_u8(data, device)?;
        Ok(Self::from_storage(storage, shape, ()))
    }

    pub fn from_slice<S: Into<Shape>>(data: &[i64], shape: S, options: impl Into<TensorCreationOptions<D, Int>>) -> Result<Self> {
        let options: TensorCreationOptions<D, Int> = options.into();
        let shape = shape.into();
        if shape.element_count() != data.len() {
            return Err(crate::Error::ElementSizeMismatch { expected: data.len(), got: shape.element_count(), op: "from_slice" });
        }
        let storage = match options.dtype {
            IntDType::I32 => {
                let v: Vec<i32> = data.iter().map(|&x| x as i32).collect();
                D::i_from_bytes(bytemuck::cast_slice(&v), &options.device, IntDType::I32)?
            }
            IntDType::U32 => {
                let v: Vec<u32> = data.iter().map(|&x| x as u32).collect();
                D::i_from_bytes(bytemuck::cast_slice(&v), &options.device, IntDType::U32)?
            }
            IntDType::U8 => {
                let v: Vec<u8> = data.iter().map(|&x| x as u8).collect();
                D::i_from_bytes(&v, &options.device, IntDType::U8)?
            }
        };
        Ok(Self::from_storage(storage, shape, ()))
    }

    pub fn to_vec(&self) -> crate::Result<Vec<i64>> {
        D::i_to_vec(&*self.storage_read()?, self.layout())
    }

    pub fn arange(start: i64, end: i64, step: i64, options: impl Into<TensorCreationOptions<D, Int>>) -> Result<Self> {
        let options: TensorCreationOptions<D, Int> = options.into();
        let (storage, n) = D::i_arange(start, end, step, &options.device, options.dtype)?;
        Ok(Self::from_storage(storage, Shape::from(n), ()))
    }

    pub fn eye(n: usize, options: impl Into<TensorCreationOptions<D, Int>>) -> Result<Self> {
        let options = options.into();
        match options.dtype {
            IntDType::I32 => Self::new(helpers::fill_eye::<i32>(n), &options.device),
            IntDType::U32 => Self::new(helpers::fill_eye::<u32>(n), &options.device),
            IntDType::U8 => Self::new(helpers::fill_eye::<u8>(n), &options.device),
        }
    }

    pub fn diag(diag: &[i64], options: impl Into<TensorCreationOptions<D, Int>>) -> Result<Self> {
        let options = options.into();
        let n = diag.len();
        match options.dtype {
            IntDType::I32 => {
                let mut v = vec![0i32; n * n];
                for i in 0..n {
                    v[i * n + i] = diag[i] as i32;
                }
                Self::new(v, &options.device)
            }
            IntDType::U32 => {
                let mut v = vec![0u32; n * n];
                for i in 0..n {
                    v[i * n + i] = diag[i] as u32;
                }
                Self::new(v, &options.device)
            }
            IntDType::U8 => {
                let mut v = vec![0u8; n * n];
                for i in 0..n {
                    v[i * n + i] = diag[i] as u8;
                }
                Self::new(v, &options.device)
            }
        }
    }

    pub fn tril(n: usize, diagonal: bool, options: impl Into<TensorCreationOptions<D, Int>>) -> Result<Self> {
        let options = options.into();
        match options.dtype {
            IntDType::I32 => Self::new(helpers::fill_tril::<i32>(n, diagonal), &options.device),
            IntDType::U32 => Self::new(helpers::fill_tril::<u32>(n, diagonal), &options.device),
            IntDType::U8 => Self::new(helpers::fill_tril::<u8>(n, diagonal), &options.device),
        }
    }

    pub fn triu(n: usize, diagonal: bool, options: impl Into<TensorCreationOptions<D, Int>>) -> Result<Self> {
        let options = options.into();
        match options.dtype {
            IntDType::I32 => Self::new(helpers::fill_triu::<i32>(n, diagonal), &options.device),
            IntDType::U32 => Self::new(helpers::fill_triu::<u32>(n, diagonal), &options.device),
            IntDType::U8 => Self::new(helpers::fill_triu::<u8>(n, diagonal), &options.device),
        }
    }
}
