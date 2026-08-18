use super::from_element::{BoolFrom, FloatFrom, IntFrom};
use crate::{Bool, DTypeKind, Device, Float, Int, Result, Shape};

pub trait IntoTensor<D: Device, K: DTypeKind<D>> {
    fn shape(&self) -> Result<Shape>;
    fn into_storage(self, device: &D) -> Result<K::Storage>;
}

// ---- Float impls --------------------------------------------------------------

impl<D: Device, T: FloatFrom> IntoTensor<D, Float> for T {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::scalar())
    }
    fn into_storage(self, device: &D) -> Result<D::FloatStorage> {
        T::into_storage(&[self][..], device)
    }
}

impl<D: Device, T: FloatFrom> IntoTensor<D, Float> for &[T] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }
    fn into_storage(self, device: &D) -> Result<D::FloatStorage> {
        T::into_storage(self, device)
    }
}

impl<D: Device, T: FloatFrom> IntoTensor<D, Float> for Vec<T> {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }
    fn into_storage(self, device: &D) -> Result<D::FloatStorage> {
        T::into_storage(self, device)
    }
}

impl<D: Device, T: FloatFrom, const N: usize> IntoTensor<D, Float> for [T; N] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(N))
    }
    fn into_storage(self, device: &D) -> Result<D::FloatStorage> {
        T::into_storage(&self[..], device)
    }
}

impl<D: Device, T: FloatFrom, const N: usize> IntoTensor<D, Float> for &[T; N] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(N))
    }
    fn into_storage(self, device: &D) -> Result<D::FloatStorage> {
        T::into_storage(self.as_slice(), device)
    }
}

impl<D: Device, T: FloatFrom, const R: usize, const C: usize> IntoTensor<D, Float> for &[[T; C]; R] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((R, C)))
    }
    fn into_storage(self, device: &D) -> Result<D::FloatStorage> {
        let v = self.concat();
        T::into_storage(&v, device)
    }
}

impl<D: Device, T: FloatFrom, const D1: usize, const D2: usize, const D3: usize> IntoTensor<D, Float> for &[[[T; D3]; D2]; D1] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((D1, D2, D3)))
    }
    fn into_storage(self, device: &D) -> Result<D::FloatStorage> {
        let mut v = Vec::with_capacity(D1 * D2 * D3);
        for i1 in 0..D1 {
            for i2 in 0..D2 {
                v.extend_from_slice(&self[i1][i2]);
            }
        }
        T::into_storage(v, device)
    }
}

impl<D: Device, T: FloatFrom, const D1: usize, const D2: usize, const D3: usize, const D4: usize> IntoTensor<D, Float>
    for &[[[[T; D4]; D3]; D2]; D1]
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((D1, D2, D3, D4)))
    }
    fn into_storage(self, device: &D) -> Result<D::FloatStorage> {
        let mut v = Vec::with_capacity(D1 * D2 * D3 * D4);
        for i1 in 0..D1 {
            for i2 in 0..D2 {
                for i3 in 0..D3 {
                    v.extend_from_slice(&self[i1][i2][i3]);
                }
            }
        }
        T::into_storage(v, device)
    }
}

// ---- Int impls ----------------------------------------------------------------

impl<D: Device, T: IntFrom> IntoTensor<D, Int> for T {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::scalar())
    }
    fn into_storage(self, device: &D) -> Result<D::IntStorage> {
        T::into_storage(&[self][..], device)
    }
}

impl<D: Device, T: IntFrom> IntoTensor<D, Int> for &[T] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }
    fn into_storage(self, device: &D) -> Result<D::IntStorage> {
        T::into_storage(self, device)
    }
}

impl<D: Device, T: IntFrom> IntoTensor<D, Int> for Vec<T> {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }
    fn into_storage(self, device: &D) -> Result<D::IntStorage> {
        T::into_storage(self, device)
    }
}

impl<D: Device, T: IntFrom, const N: usize> IntoTensor<D, Int> for [T; N] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(N))
    }
    fn into_storage(self, device: &D) -> Result<D::IntStorage> {
        T::into_storage(&self[..], device)
    }
}

impl<D: Device, T: IntFrom, const N: usize> IntoTensor<D, Int> for &[T; N] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(N))
    }
    fn into_storage(self, device: &D) -> Result<D::IntStorage> {
        T::into_storage(self.as_slice(), device)
    }
}

impl<D: Device, T: IntFrom, const R: usize, const C: usize> IntoTensor<D, Int> for &[[T; C]; R] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((R, C)))
    }
    fn into_storage(self, device: &D) -> Result<D::IntStorage> {
        let v = self.concat();
        T::into_storage(&v, device)
    }
}

impl<D: Device, T: IntFrom, const D1: usize, const D2: usize, const D3: usize> IntoTensor<D, Int> for &[[[T; D3]; D2]; D1] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((D1, D2, D3)))
    }
    fn into_storage(self, device: &D) -> Result<D::IntStorage> {
        let mut v = Vec::with_capacity(D1 * D2 * D3);
        for i1 in 0..D1 {
            for i2 in 0..D2 {
                v.extend_from_slice(&self[i1][i2]);
            }
        }
        T::into_storage(v, device)
    }
}

impl<D: Device, T: IntFrom, const D1: usize, const D2: usize, const D3: usize, const D4: usize> IntoTensor<D, Int>
    for &[[[[T; D4]; D3]; D2]; D1]
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((D1, D2, D3, D4)))
    }
    fn into_storage(self, device: &D) -> Result<D::IntStorage> {
        let mut v = Vec::with_capacity(D1 * D2 * D3 * D4);
        for i1 in 0..D1 {
            for i2 in 0..D2 {
                for i3 in 0..D3 {
                    v.extend_from_slice(&self[i1][i2][i3]);
                }
            }
        }
        T::into_storage(v, device)
    }
}

// ---- Bool impls ---------------------------------------------------------------

impl<D: Device, T: BoolFrom> IntoTensor<D, Bool> for T {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::scalar())
    }
    fn into_storage(self, device: &D) -> Result<D::BoolStorage> {
        T::into_storage(&[self][..], device)
    }
}

impl<D: Device, T: BoolFrom> IntoTensor<D, Bool> for &[T] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }
    fn into_storage(self, device: &D) -> Result<D::BoolStorage> {
        T::into_storage(self, device)
    }
}

impl<D: Device, T: BoolFrom> IntoTensor<D, Bool> for Vec<T> {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }
    fn into_storage(self, device: &D) -> Result<D::BoolStorage> {
        T::into_storage(self, device)
    }
}

impl<D: Device, T: BoolFrom, const N: usize> IntoTensor<D, Bool> for [T; N] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(N))
    }
    fn into_storage(self, device: &D) -> Result<D::BoolStorage> {
        T::into_storage(&self[..], device)
    }
}

impl<D: Device, T: BoolFrom, const N: usize> IntoTensor<D, Bool> for &[T; N] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(N))
    }
    fn into_storage(self, device: &D) -> Result<D::BoolStorage> {
        T::into_storage(self.as_slice(), device)
    }
}

impl<D: Device, T: BoolFrom, const R: usize, const C: usize> IntoTensor<D, Bool> for &[[T; C]; R] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((R, C)))
    }
    fn into_storage(self, device: &D) -> Result<D::BoolStorage> {
        let v = self.concat();
        T::into_storage(&v, device)
    }
}

impl<D: Device, T: BoolFrom, const D1: usize, const D2: usize, const D3: usize> IntoTensor<D, Bool> for &[[[T; D3]; D2]; D1] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((D1, D2, D3)))
    }
    fn into_storage(self, device: &D) -> Result<D::BoolStorage> {
        let mut v = Vec::with_capacity(D1 * D2 * D3);
        for i1 in 0..D1 {
            for i2 in 0..D2 {
                v.extend_from_slice(&self[i1][i2]);
            }
        }
        T::into_storage(v, device)
    }
}

impl<D: Device, T: BoolFrom, const D1: usize, const D2: usize, const D3: usize, const D4: usize> IntoTensor<D, Bool>
    for &[[[[T; D4]; D3]; D2]; D1]
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((D1, D2, D3, D4)))
    }
    fn into_storage(self, device: &D) -> Result<D::BoolStorage> {
        let mut v = Vec::with_capacity(D1 * D2 * D3 * D4);
        for i1 in 0..D1 {
            for i2 in 0..D2 {
                for i3 in 0..D3 {
                    v.extend_from_slice(&self[i1][i2][i3]);
                }
            }
        }
        T::into_storage(v, device)
    }
}
