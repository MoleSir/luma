use crate::{dtype::{BoolDType, FloatDType, IntDType}, Bool, DType, DTypeKind, Device, Float, FloatMeta, Int, Result, Shape, Tensor};

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

/// Factory operations for constructing tensors.
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

impl<D: Device> Tensor<D, Float> {
    /// Create a tensor from a Rust scalar, array, slice, or Vec — auto-inferring shape.
    ///
    /// ```rust
    /// use luma_tensor::{Cpu, Tensor};
    ///
    /// let s = Tensor::<Cpu>::new(3.14).unwrap();            // scalar
    /// let v = Tensor::<Cpu>::new(&[1.0, 2.0, 3.0]).unwrap(); // 1-D
    /// let m = Tensor::<Cpu>::new(&[[1.0, 2.0], [3.0, 4.0]]).unwrap(); // 2-D
    /// ```
    ///
    /// Uses default device and `F32` precision. For explicit control use [`from_slice`](Self::from_slice).
    pub fn new(data: impl IntoTensor<D, Float>) -> Result<Self> {
        let device = D::default();
        let shape = data.shape()?;
        let storage = data.into_storage(&device, FloatDType::default())?;
        Ok(Self::from_storage(storage, shape, FloatMeta::val()))
    }

    pub fn from_slice<S: Into<Shape>>(data: &[f64], shape: S, options: impl Into<TensorCreationOptions<D, Float>>) -> Result<Self> {
        let options: TensorCreationOptions<D, Float> = options.into();
        let shape = shape.into();
        if shape.element_count() != data.len() {
            return Err(crate::Error::ElementSizeMismatch { expected: data.len(), got: shape.element_count(), op: "from_slice" });
        }
        let storage = D::f_from_f64(data, options.dtype)?;
        Ok(Self::from_storage(storage, shape, FloatMeta::val()))
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

    /// Create an `n`×`n` identity matrix.
    pub fn eye(n: usize) -> Result<Self> {
        let mut v = vec![0.0f64; n * n];
        for i in 0..n { v[i * n + i] = 1.0; }
        Self::from_slice(&v, (n, n), FloatDType::default())
    }

    /// Create a diagonal matrix from a 1-D slice.
    pub fn diag(diag: &[f64]) -> Result<Self> {
        let n = diag.len();
        let mut v = vec![0.0f64; n * n];
        for i in 0..n { v[i * n + i] = diag[i]; }
        Self::from_slice(&v, (n, n), FloatDType::default())
    }

    /// Lower-triangular matrix of ones (includes diagonal when `diagonal` is true).
    pub fn tril(n: usize, diagonal: bool) -> Result<Self> {
        let mut v = vec![1.0f64; n * n];
        for i in 0..n {
            let end = if diagonal { i + 1 } else { i };
            for j in end..n {
                v[i * n + j] = 0.0;
            }
        }
        Self::from_slice(&v, (n, n), FloatDType::default())
    }

    /// Upper-triangular matrix of ones (includes diagonal when `diagonal` is true).
    pub fn triu(n: usize, diagonal: bool) -> Result<Self> {
        let mut v = vec![1.0f64; n * n];
        for i in 0..n {
            let start = if diagonal { i } else { i + 1 };
            for j in 0..start {
                v[i * n + j] = 0.0;
            }
        }
        Self::from_slice(&v, (n, n), FloatDType::default())
    }

    /// Linearly spaced values from `start` to (including) `stop`.
    pub fn linspace(start: f64, stop: f64, n: usize) -> Result<Self> {
        let step = if n > 1 { (stop - start) / (n as f64 - 1.0) } else { 0.0 };
        let v: Vec<f64> = (0..n).map(|i| start + step * i as f64).collect();
        Self::from_slice(&v, n, FloatDType::default())
    }
}

impl<D: Device> Tensor<D, Int> {
    /// Create an integer tensor from a Rust scalar, array, slice, or Vec.
    ///
    /// Uses default device and `I32` precision. For explicit control use [`from_slice`](Self::from_slice).
    pub fn new(data: impl IntoTensor<D, Int>) -> Result<Self> {
        let device = D::default();
        let shape = data.shape()?;
        let storage = data.into_storage(&device, IntDType::default())?;
        Ok(Self::from_storage(storage, shape, ()))
    }

    pub fn from_slice<S: Into<Shape>>(data: &[i64], shape: S, options: impl Into<TensorCreationOptions<D, Int>>) -> Result<Self> {
        let options: TensorCreationOptions<D, Int> = options.into();
        let shape = shape.into();
        if shape.element_count() != data.len() {
            return Err(crate::Error::ElementSizeMismatch { expected: data.len(), got: shape.element_count(), op: "from_slice" });
        }
        let storage = D::i_from_i64(data, &options.device, options.dtype)?;
        Ok(Self::from_storage(storage, shape, ()))
    }

    pub fn arange(start: i64, end: i64, step: i64, options: impl Into<TensorCreationOptions<D, Int>>) -> Result<Self> {
        let options: TensorCreationOptions<D, Int> = options.into();
        let (storage, n) = D::i_arange(start, end, step, &options.device, options.dtype)?;
        Ok(Self::from_storage(storage, Shape::from(n), ()))
    }

    /// Create an `n`×`n` integer identity matrix.
    pub fn eye(n: usize) -> Result<Self> {
        let mut v = vec![0i64; n * n];
        for i in 0..n { v[i * n + i] = 1; }
        Self::from_slice(&v, (n, n), IntDType::default())
    }

    /// Create a diagonal integer matrix from a 1-D slice.
    pub fn diag(diag: &[i64]) -> Result<Self> {
        let n = diag.len();
        let mut v = vec![0i64; n * n];
        for i in 0..n { v[i * n + i] = diag[i]; }
        Self::from_slice(&v, (n, n), IntDType::default())
    }

    /// Lower-triangular integer matrix of ones.
    pub fn tril(n: usize, diagonal: bool) -> Result<Self> {
        let mut v = vec![1i64; n * n];
        for i in 0..n {
            let end = if diagonal { i + 1 } else { i };
            for j in end..n { v[i * n + j] = 0; }
        }
        Self::from_slice(&v, (n, n), IntDType::default())
    }

    /// Upper-triangular integer matrix of ones.
    pub fn triu(n: usize, diagonal: bool) -> Result<Self> {
        let mut v = vec![1i64; n * n];
        for i in 0..n {
            let start = if diagonal { i } else { i + 1 };
            for j in 0..start { v[i * n + j] = 0; }
        }
        Self::from_slice(&v, (n, n), IntDType::default())
    }
}

impl<D: Device> Tensor<D, Bool> {
    /// Create a boolean tensor from a Rust scalar, array, slice, or Vec.
    ///
    /// Uses default device and `Bool` precision.
    pub fn new(data: impl IntoTensor<D, Bool>) -> Result<Self> {
        let device = D::default();
        let shape = data.shape()?;
        let storage = data.into_storage(&device, BoolDType::default())?;
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
        let storage = D::b_from_bool(data, &options.device, options.dtype)?;
        Ok(Self::from_storage(storage, shape, ()))
    }
}

// ---- IntoTensor trait ---------------------------------------------------------

/// Converts a Rust value (scalar, array, slice, Vec) into a [`Tensor`].
///
/// # Implementations
///
/// | Input | Shape |
/// |-------|-------|
/// | scalar (`f64`/`i64`/`bool`) | `()` |
/// | `&[T]`, `Vec<T>` | `(len,)` |
/// | `[T; N]`, `&[T; N]` | `(N,)` |
/// | `&[[T; C]; R]` | `(R, C)` |
/// | `&[[[T; D]; C]; R]` | `(R, C, D)` |
/// | `&[[[[T; D4]; D3]; D2]; D1]` | `(D1, D2, D3, D4)` |
///
/// See [`Tensor::new`].
pub trait IntoTensor<D: Device, K: DTypeKind<D>> {
    fn shape(&self) -> Result<Shape>;
    fn into_storage(self, device: &D, dtype: K::DType) -> Result<K::Storage>;
}

// ---- Float impls --------------------------------------------------------------

impl<D: Device> IntoTensor<D, Float> for f64 {
    fn shape(&self) -> Result<Shape> { Ok(Shape::scalar()) }
    fn into_storage(self, _device: &D, dtype: FloatDType) -> Result<D::FloatStorage> {
        D::f_from_f64(&[self], dtype)
    }
}

impl<D: Device> IntoTensor<D, Float> for &[f64] {
    fn shape(&self) -> Result<Shape> { Ok(Shape::from(self.len())) }
    fn into_storage(self, _device: &D, dtype: FloatDType) -> Result<D::FloatStorage> {
        D::f_from_f64(self, dtype)
    }
}

impl<D: Device> IntoTensor<D, Float> for Vec<f64> {
    fn shape(&self) -> Result<Shape> { Ok(Shape::from(self.len())) }
    fn into_storage(self, _device: &D, dtype: FloatDType) -> Result<D::FloatStorage> {
        D::f_from_f64(&self, dtype)
    }
}

impl<D: Device, const N: usize> IntoTensor<D, Float> for [f64; N] {
    fn shape(&self) -> Result<Shape> { Ok(Shape::from(N)) }
    fn into_storage(self, _device: &D, dtype: FloatDType) -> Result<D::FloatStorage> {
        D::f_from_f64(&self, dtype)
    }
}

impl<D: Device, const N: usize> IntoTensor<D, Float> for &[f64; N] {
    fn shape(&self) -> Result<Shape> { Ok(Shape::from(N)) }
    fn into_storage(self, _device: &D, dtype: FloatDType) -> Result<D::FloatStorage> {
        D::f_from_f64(self.as_slice(), dtype)
    }
}

impl<D: Device, const R: usize, const C: usize> IntoTensor<D, Float> for &[[f64; C]; R] {
    fn shape(&self) -> Result<Shape> { Ok(Shape::from((R, C))) }
    fn into_storage(self, _device: &D, dtype: FloatDType) -> Result<D::FloatStorage> {
        D::f_from_f64(&self.concat(), dtype)
    }
}

impl<D: Device, const D1: usize, const D2: usize, const D3: usize> IntoTensor<D, Float>
    for &[[[f64; D3]; D2]; D1]
{
    fn shape(&self) -> Result<Shape> { Ok(Shape::from((D1, D2, D3))) }
    fn into_storage(self, _device: &D, dtype: FloatDType) -> Result<D::FloatStorage> {
        let mut v = Vec::with_capacity(D1 * D2 * D3);
        for i1 in 0..D1 {
            for i2 in 0..D2 {
                v.extend_from_slice(&self[i1][i2]);
            }
        }
        D::f_from_f64(&v, dtype)
    }
}

impl<D: Device, const D1: usize, const D2: usize, const D3: usize, const D4: usize> IntoTensor<D, Float>
    for &[[[[f64; D4]; D3]; D2]; D1]
{
    fn shape(&self) -> Result<Shape> { Ok(Shape::from((D1, D2, D3, D4))) }
    fn into_storage(self, _device: &D, dtype: FloatDType) -> Result<D::FloatStorage> {
        let mut v = Vec::with_capacity(D1 * D2 * D3 * D4);
        for i1 in 0..D1 {
            for i2 in 0..D2 {
                for i3 in 0..D3 {
                    v.extend_from_slice(&self[i1][i2][i3]);
                }
            }
        }
        D::f_from_f64(&v, dtype)
    }
}

// ---- Int impls ----------------------------------------------------------------

impl<D: Device> IntoTensor<D, Int> for i64 {
    fn shape(&self) -> Result<Shape> { Ok(Shape::scalar()) }
    fn into_storage(self, device: &D, dtype: IntDType) -> Result<D::IntStorage> {
        D::i_from_i64(&[self], device, dtype)
    }
}

impl<D: Device> IntoTensor<D, Int> for &[i64] {
    fn shape(&self) -> Result<Shape> { Ok(Shape::from(self.len())) }
    fn into_storage(self, device: &D, dtype: IntDType) -> Result<D::IntStorage> {
        D::i_from_i64(self, device, dtype)
    }
}

impl<D: Device> IntoTensor<D, Int> for Vec<i64> {
    fn shape(&self) -> Result<Shape> { Ok(Shape::from(self.len())) }
    fn into_storage(self, device: &D, dtype: IntDType) -> Result<D::IntStorage> {
        D::i_from_i64(&self, device, dtype)
    }
}

impl<D: Device, const N: usize> IntoTensor<D, Int> for [i64; N] {
    fn shape(&self) -> Result<Shape> { Ok(Shape::from(N)) }
    fn into_storage(self, device: &D, dtype: IntDType) -> Result<D::IntStorage> {
        D::i_from_i64(&self, device, dtype)
    }
}

impl<D: Device, const N: usize> IntoTensor<D, Int> for &[i64; N] {
    fn shape(&self) -> Result<Shape> { Ok(Shape::from(N)) }
    fn into_storage(self, device: &D, dtype: IntDType) -> Result<D::IntStorage> {
        D::i_from_i64(self.as_slice(), device, dtype)
    }
}

impl<D: Device, const R: usize, const C: usize> IntoTensor<D, Int> for &[[i64; C]; R] {
    fn shape(&self) -> Result<Shape> { Ok(Shape::from((R, C))) }
    fn into_storage(self, device: &D, dtype: IntDType) -> Result<D::IntStorage> {
        D::i_from_i64(&self.concat(), device, dtype)
    }
}

impl<D: Device, const D1: usize, const D2: usize, const D3: usize> IntoTensor<D, Int>
    for &[[[i64; D3]; D2]; D1]
{
    fn shape(&self) -> Result<Shape> { Ok(Shape::from((D1, D2, D3))) }
    fn into_storage(self, device: &D, dtype: IntDType) -> Result<D::IntStorage> {
        let mut v = Vec::with_capacity(D1 * D2 * D3);
        for i1 in 0..D1 {
            for i2 in 0..D2 {
                v.extend_from_slice(&self[i1][i2]);
            }
        }
        D::i_from_i64(&v, device, dtype)
    }
}

impl<D: Device, const D1: usize, const D2: usize, const D3: usize, const D4: usize> IntoTensor<D, Int>
    for &[[[[i64; D4]; D3]; D2]; D1]
{
    fn shape(&self) -> Result<Shape> { Ok(Shape::from((D1, D2, D3, D4))) }
    fn into_storage(self, device: &D, dtype: IntDType) -> Result<D::IntStorage> {
        let mut v = Vec::with_capacity(D1 * D2 * D3 * D4);
        for i1 in 0..D1 {
            for i2 in 0..D2 {
                for i3 in 0..D3 {
                    v.extend_from_slice(&self[i1][i2][i3]);
                }
            }
        }
        D::i_from_i64(&v, device, dtype)
    }
}

// ---- Bool impls ---------------------------------------------------------------

impl<D: Device> IntoTensor<D, Bool> for bool {
    fn shape(&self) -> Result<Shape> { Ok(Shape::scalar()) }
    fn into_storage(self, device: &D, dtype: BoolDType) -> Result<D::BoolStorage> {
        D::b_from_bool(&[self], device, dtype)
    }
}

impl<D: Device> IntoTensor<D, Bool> for &[bool] {
    fn shape(&self) -> Result<Shape> { Ok(Shape::from(self.len())) }
    fn into_storage(self, device: &D, dtype: BoolDType) -> Result<D::BoolStorage> {
        D::b_from_bool(self, device, dtype)
    }
}

impl<D: Device> IntoTensor<D, Bool> for Vec<bool> {
    fn shape(&self) -> Result<Shape> { Ok(Shape::from(self.len())) }
    fn into_storage(self, device: &D, dtype: BoolDType) -> Result<D::BoolStorage> {
        D::b_from_bool(&self, device, dtype)
    }
}

impl<D: Device, const N: usize> IntoTensor<D, Bool> for [bool; N] {
    fn shape(&self) -> Result<Shape> { Ok(Shape::from(N)) }
    fn into_storage(self, device: &D, dtype: BoolDType) -> Result<D::BoolStorage> {
        D::b_from_bool(&self, device, dtype)
    }
}

impl<D: Device, const N: usize> IntoTensor<D, Bool> for &[bool; N] {
    fn shape(&self) -> Result<Shape> { Ok(Shape::from(N)) }
    fn into_storage(self, device: &D, dtype: BoolDType) -> Result<D::BoolStorage> {
        D::b_from_bool(self.as_slice(), device, dtype)
    }
}

impl<D: Device, const R: usize, const C: usize> IntoTensor<D, Bool> for &[[bool; C]; R] {
    fn shape(&self) -> Result<Shape> { Ok(Shape::from((R, C))) }
    fn into_storage(self, device: &D, dtype: BoolDType) -> Result<D::BoolStorage> {
        D::b_from_bool(&self.concat(), device, dtype)
    }
}

impl<D: Device, const D1: usize, const D2: usize, const D3: usize> IntoTensor<D, Bool>
    for &[[[bool; D3]; D2]; D1]
{
    fn shape(&self) -> Result<Shape> { Ok(Shape::from((D1, D2, D3))) }
    fn into_storage(self, device: &D, dtype: BoolDType) -> Result<D::BoolStorage> {
        let mut v = Vec::with_capacity(D1 * D2 * D3);
        for i1 in 0..D1 {
            for i2 in 0..D2 {
                v.extend_from_slice(&self[i1][i2]);
            }
        }
        D::b_from_bool(&v, device, dtype)
    }
}

impl<D: Device, const D1: usize, const D2: usize, const D3: usize, const D4: usize> IntoTensor<D, Bool>
    for &[[[[bool; D4]; D3]; D2]; D1]
{
    fn shape(&self) -> Result<Shape> { Ok(Shape::from((D1, D2, D3, D4))) }
    fn into_storage(self, device: &D, dtype: BoolDType) -> Result<D::BoolStorage> {
        let mut v = Vec::with_capacity(D1 * D2 * D3 * D4);
        for i1 in 0..D1 {
            for i2 in 0..D2 {
                for i3 in 0..D3 {
                    v.extend_from_slice(&self[i1][i2][i3]);
                }
            }
        }
        D::b_from_bool(&v, device, dtype)
    }
}

/// Default float precision when unspecified.
pub const DEFAULT_FLOAT: DType = DType::F32;
/// Default int precision.
pub const DEFAULT_INT: DType = DType::I32;
