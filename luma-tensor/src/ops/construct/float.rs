use std::borrow::Cow;

use crate::{Device, Float, FloatMeta, Result, Shape, Tensor, dtype::FloatDType};

use super::helpers;
use super::into_tensor::IntoTensor;
use super::options::TensorCreationOptions;

impl<D: Device> Tensor<D, Float> {
    pub fn new(data: impl IntoTensor<D, Float>, device: &D) -> Result<Self> {
        let shape = data.shape()?;
        let storage = data.into_storage(device)?;
        Ok(Self::from_storage(storage, shape, FloatMeta::val()))
    }

    pub fn from_vec_f64<'a, S: Into<Shape>>(data: impl Into<Cow<'a, [f64]>>, shape: S, device: &D) -> Result<Self> {
        let shape = shape.into();
        let storage = D::f_from_f64(data, device)?;
        Ok(Self::from_storage(storage, shape, FloatMeta::val()))
    }

    pub fn from_vec_f32<'a, S: Into<Shape>>(data: impl Into<Cow<'a, [f32]>>, shape: S, device: &D) -> Result<Self> {
        let shape = shape.into();
        let storage = D::f_from_f32(data, device)?;
        Ok(Self::from_storage(storage, shape, FloatMeta::val()))
    }

    pub fn from_slice<S: Into<Shape>>(data: &[f64], shape: S, options: impl Into<TensorCreationOptions<D, Float>>) -> Result<Self> {
        let options: TensorCreationOptions<D, Float> = options.into();
        let shape = shape.into();
        if shape.element_count() != data.len() {
            return Err(crate::Error::ElementSizeMismatch { expected: data.len(), got: shape.element_count(), op: "from_slice" });
        }
        let storage = match options.dtype {
            FloatDType::F64 => D::f_from_f64(data, &options.device)?,
            FloatDType::F32 => {
                let v: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                D::f_from_f32(&v, &options.device)?
            }
        };
        Ok(Self::from_storage(storage, shape, FloatMeta::val()))
    }

    pub fn to_vec(&self) -> crate::Result<Vec<f64>> {
        D::f_to_vec(&*self.storage_read()?, self.layout())
    }

    pub fn randn<S: Into<Shape>>(mean: f64, std: f64, shape: S, options: impl Into<TensorCreationOptions<D, Float>>) -> Result<Self> {
        let options: TensorCreationOptions<D, Float> = options.into();
        let shape = shape.into();
        let storage = D::f_rand_normal(&shape, mean, std, &options.device, options.dtype)?;
        Ok(Self::from_storage(storage, shape, FloatMeta::val()))
    }

    pub fn rand<S: Into<Shape>>(lo: f64, hi: f64, shape: S, options: impl Into<TensorCreationOptions<D, Float>>) -> Result<Self> {
        let options: TensorCreationOptions<D, Float> = options.into();
        let shape = shape.into();
        let storage = D::f_rand_uniform(&shape, lo, hi, &options.device, options.dtype)?;
        Ok(Self::from_storage(storage, shape, FloatMeta::val()))
    }

    pub fn rand_like(&self, lo: f64, hi: f64) -> Result<Self> {
        Self::rand(lo, hi, self.shape().clone(), self.dtype())
    }

    pub fn randn_like(&self, mean: f64, std: f64) -> Result<Self> {
        Self::randn(mean, std, self.shape().clone(), self.dtype())
    }

    pub fn eye(n: usize, options: impl Into<TensorCreationOptions<D, Float>>) -> Result<Self> {
        let options = options.into();
        match options.dtype {
            FloatDType::F64 => Self::new(helpers::fill_eye::<f64>(n), &options.device),
            FloatDType::F32 => Self::new(helpers::fill_eye::<f32>(n), &options.device),
        }
    }

    pub fn diag(diag: &[f64], options: impl Into<TensorCreationOptions<D, Float>>) -> Result<Self> {
        let options = options.into();
        let n = diag.len();
        match options.dtype {
            FloatDType::F64 => {
                let mut v = vec![0.0f64; n * n];
                for i in 0..n {
                    v[i * n + i] = diag[i];
                }
                Self::new(v, &options.device)
            }
            FloatDType::F32 => {
                let mut v = vec![0.0f32; n * n];
                for i in 0..n {
                    v[i * n + i] = diag[i] as f32;
                }
                Self::new(v, &options.device)
            }
        }
    }

    pub fn tril(n: usize, diagonal: bool, options: impl Into<TensorCreationOptions<D, Float>>) -> Result<Self> {
        let options = options.into();
        match options.dtype {
            FloatDType::F64 => Self::new(helpers::fill_tril::<f64>(n, diagonal), &options.device),
            FloatDType::F32 => Self::new(helpers::fill_tril::<f32>(n, diagonal), &options.device),
        }
    }

    pub fn triu(n: usize, diagonal: bool, options: impl Into<TensorCreationOptions<D, Float>>) -> Result<Self> {
        let options = options.into();
        match options.dtype {
            FloatDType::F64 => Self::new(helpers::fill_triu::<f64>(n, diagonal), &options.device),
            FloatDType::F32 => Self::new(helpers::fill_triu::<f32>(n, diagonal), &options.device),
        }
    }

    pub fn linspace(start: f64, stop: f64, n: usize, options: impl Into<TensorCreationOptions<D, Float>>) -> Result<Self> {
        let options = options.into();
        let step = if n > 1 { (stop - start) / (n as f64 - 1.0) } else { 0.0 };
        match options.dtype {
            FloatDType::F64 => {
                let v: Vec<f64> = (0..n).map(|i| start + step * i as f64).collect();
                Self::new(v, &options.device)
            }
            FloatDType::F32 => {
                let v: Vec<f32> = (0..n).map(|i| (start + step * i as f64) as f32).collect();
                Self::new(v, &options.device)
            }
        }
    }
}
