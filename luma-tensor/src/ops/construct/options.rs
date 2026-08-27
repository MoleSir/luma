use std::borrow::Cow;

use crate::{
    Bool, DTypeKind, Device, Float, Int, Layout, Shape, Tensor,
    dtype::{BoolDType, FloatDType, IntDType},
};

pub struct TensorCreationOptions<D: Device, K: DTypeKind<D>> {
    pub device: D,
    pub dtype: K::DType,
}

impl<D: Device, K: DTypeKind<D>> From<&D> for TensorCreationOptions<D, K> {
    fn from(device: &D) -> Self {
        Self { device: device.clone(), dtype: K::DType::default() }
    }
}

impl<D: Device, K: DTypeKind<D>> From<D> for TensorCreationOptions<D, K> {
    fn from(device: D) -> Self {
        Self { device, dtype: K::DType::default() }
    }
}

impl<D: Device> From<FloatDType> for TensorCreationOptions<D, Float> {
    fn from(dtype: FloatDType) -> Self {
        Self { device: D::default(), dtype }
    }
}

impl<D: Device> From<IntDType> for TensorCreationOptions<D, Int> {
    fn from(dtype: IntDType) -> Self {
        Self { device: D::default(), dtype }
    }
}

impl<D: Device> From<BoolDType> for TensorCreationOptions<D, Bool> {
    fn from(dtype: BoolDType) -> Self {
        Self { device: D::default(), dtype }
    }
}

impl<D: Device, K: DTypeKind<D>> From<()> for TensorCreationOptions<D, K> {
    fn from(_: ()) -> Self {
        Self { device: D::default(), dtype: K::DType::default() }
    }
}

impl<D: Device, K: DTypeKind<D>> From<(&D, K::DType)> for TensorCreationOptions<D, K> {
    fn from((device, dtype): (&D, K::DType)) -> Self {
        Self { device: device.clone(), dtype }
    }
}

impl<D: Device, K: DTypeKind<D>> From<(D, K::DType)> for TensorCreationOptions<D, K> {
    fn from((device, dtype): (D, K::DType)) -> Self {
        Self { device, dtype }
    }
}

pub trait ConstructDTypeKind<D: Device>: DTypeKind<D> {
    fn zeros_dispatch(shape: &Shape, device: &D, dtype: Self::DType) -> crate::Result<Self::Storage>;
    fn ones_dispatch(shape: &Shape, device: &D, dtype: Self::DType) -> crate::Result<Self::Storage>;
    fn full_dispatch(shape: &Shape, value: Self::Scalar, device: &D, dtype: Self::DType) -> crate::Result<Self::Storage>;
}

impl<D: Device> ConstructDTypeKind<D> for Float {
    fn zeros_dispatch(shape: &Shape, device: &D, dtype: FloatDType) -> crate::Result<Self::Storage> {
        D::f_zeros(shape, device, dtype)
    }

    fn ones_dispatch(shape: &Shape, device: &D, dtype: FloatDType) -> crate::Result<Self::Storage> {
        D::f_ones(shape, device, dtype)
    }

    fn full_dispatch(shape: &Shape, value: f64, device: &D, dtype: FloatDType) -> crate::Result<Self::Storage> {
        D::f_full(shape, value, device, dtype)
    }
}

impl<D: Device> ConstructDTypeKind<D> for Int {
    fn zeros_dispatch(shape: &Shape, device: &D, dtype: IntDType) -> crate::Result<Self::Storage> {
        D::i_zeros(shape, device, dtype)
    }

    fn ones_dispatch(shape: &Shape, device: &D, dtype: IntDType) -> crate::Result<Self::Storage> {
        D::i_ones(shape, device, dtype)
    }

    fn full_dispatch(shape: &Shape, value: Self::Scalar, device: &D, dtype: IntDType) -> crate::Result<Self::Storage> {
        D::i_full(shape, value, device, dtype)
    }
}

impl<D: Device, K: DTypeKind<D>> Tensor<D, K> {
    pub fn phantom<S: Into<Shape>>(shape: S, options: impl Into<TensorCreationOptions<D, K>>) -> crate::Result<Self> {
        let options: TensorCreationOptions<D, K> = options.into();
        let shape = shape.into();
        Ok(Self::phantom_storage(shape, options.dtype, options.device))
    }
}

impl<D: Device, K: ConstructDTypeKind<D>> Tensor<D, K> {
    pub fn zeros<S: Into<Shape>>(shape: S, options: impl Into<TensorCreationOptions<D, K>>) -> crate::Result<Self> {
        let options: TensorCreationOptions<D, K> = options.into();
        let shape = shape.into();
        let storage = K::zeros_dispatch(&shape, &options.device, options.dtype)?;
        Ok(Self::from_storage(storage, shape, K::Meta::default()))
    }

    pub fn ones<S: Into<Shape>>(shape: S, options: impl Into<TensorCreationOptions<D, K>>) -> crate::Result<Self> {
        let options: TensorCreationOptions<D, K> = options.into();
        let shape = shape.into();
        let storage = K::ones_dispatch(&shape, &options.device, options.dtype)?;
        Ok(Self::from_storage(storage, shape, K::Meta::default()))
    }

    pub fn full<S: Into<Shape>>(shape: S, value: K::Scalar, options: impl Into<TensorCreationOptions<D, K>>) -> crate::Result<Self> {
        let options: TensorCreationOptions<D, K> = options.into();
        let shape = shape.into();
        let storage = K::full_dispatch(&shape, value, &options.device, options.dtype)?;
        Ok(Self::from_storage(storage, shape, K::Meta::default()))
    }

    pub fn zeros_like(&self) -> crate::Result<Self> {
        Self::zeros(self.shape().clone(), (self.device(), self.dtype()))
    }

    pub fn ones_like(&self) -> crate::Result<Self> {
        Self::ones(self.shape().clone(), (self.device(), self.dtype()))
    }
}

// ============================================================================
//    BytesDTypeKind: kind dispatch for raw-byte I/O
// ============================================================================

/// Kind-level dispatch for `from_bytes` / `to_bytes`.
///
/// Each impl forwards to the corresponding method on the device ops trait
/// (`FloatOps::f_from_bytes` / `f_to_bytes`, etc.).
pub trait BytesDTypeKind<D: Device>: DTypeKind<D> {
    fn from_bytes_dispatch(bytes: &[u8], shape: &Shape, device: &D, dtype: Self::DType) -> crate::Result<Self::Storage>;
    fn to_bytes_dispatch<'a>(storage: &'a Self::Storage, layout: &Layout) -> crate::Result<Cow<'a, [u8]>>;
}

impl<D: Device> BytesDTypeKind<D> for Float {
    fn from_bytes_dispatch(bytes: &[u8], shape: &Shape, device: &D, dtype: FloatDType) -> crate::Result<Self::Storage> {
        D::f_from_bytes(bytes, shape, device, dtype)
    }

    fn to_bytes_dispatch<'a>(storage: &'a Self::Storage, layout: &Layout) -> crate::Result<Cow<'a, [u8]>> {
        D::f_to_bytes(storage, layout)
    }
}

impl<D: Device> BytesDTypeKind<D> for Int {
    fn from_bytes_dispatch(bytes: &[u8], shape: &Shape, device: &D, dtype: IntDType) -> crate::Result<Self::Storage> {
        D::i_from_bytes(bytes, shape, device, dtype)
    }

    fn to_bytes_dispatch<'a>(storage: &'a Self::Storage, layout: &Layout) -> crate::Result<Cow<'a, [u8]>> {
        D::i_to_bytes(storage, layout)
    }
}

impl<D: Device> BytesDTypeKind<D> for Bool {
    fn from_bytes_dispatch(bytes: &[u8], shape: &Shape, device: &D, _dtype: BoolDType) -> crate::Result<Self::Storage> {
        D::b_from_bytes(bytes, shape, device, BoolDType::Bool)
    }

    fn to_bytes_dispatch<'a>(storage: &'a Self::Storage, layout: &Layout) -> crate::Result<Cow<'a, [u8]>> {
        D::b_to_bytes(storage, layout)
    }
}

impl<D: Device, K: BytesDTypeKind<D>> Tensor<D, K> {
    /// Create a tensor from raw little-endian bytes.
    ///
    /// The byte slice length must equal `shape.element_count() * dtype.size_in_bytes()`.
    pub fn from_bytes<'a>(
        bytes: impl Into<Cow<'a, [u8]>>,
        shape: impl Into<Shape>,
        options: impl Into<TensorCreationOptions<D, K>>,
    ) -> crate::Result<Self> {
        let options: TensorCreationOptions<D, K> = options.into();
        let shape = shape.into();
        let storage = K::from_bytes_dispatch(&bytes.into(), &shape, &options.device, options.dtype)?;
        Ok(Self::from_storage(storage, shape, K::Meta::default()))
    }

    /// Read raw little-endian bytes in logical (layout) order.
    ///
    /// Always returns owned bytes (the internal `RwLock` prevents zero-copy
    /// borrowing at this level; the device trait internally uses `Cow` for
    /// contiguous-optimised paths).
    pub fn to_bytes(&self) -> crate::Result<Vec<u8>> {
        let guard = self.storage_read()?;
        Ok(K::to_bytes_dispatch(&*guard, self.layout())?.into_owned())
    }
}
