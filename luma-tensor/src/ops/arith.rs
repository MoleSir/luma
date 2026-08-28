//! `std::ops` trait implementations for [`Tensor`].

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::ops::numeric::NumericDTypeKind;
use crate::ops::shape::ShapeDTypeKind;
use crate::{Bool, DTypeKind, Device, Float, Int, Tensor};

// ============================================================================
//   TensorOrScalar — bridges concrete scalars to K::Scalar (like lumen)
// ============================================================================

pub enum TensorOrScalar<D: Device, K: DTypeKind<D>> {
    Tensor(Tensor<D, K>),
    Scalar(K::Scalar),
}

// Concrete scalar → TensorOrScalar (K::Scalar resolves here, not at trait boundary)
impl<D: Device> From<f64> for TensorOrScalar<D, Float> {
    fn from(v: f64) -> Self {
        Self::Scalar(v)
    }
}
impl<D: Device> From<i64> for TensorOrScalar<D, Int> {
    fn from(v: i64) -> Self {
        Self::Scalar(v)
    }
}
impl<D: Device> From<bool> for TensorOrScalar<D, Bool> {
    fn from(v: bool) -> Self {
        Self::Scalar(v)
    }
}
impl<D: Device, K: DTypeKind<D>> From<Tensor<D, K>> for TensorOrScalar<D, K> {
    fn from(t: Tensor<D, K>) -> Self {
        Self::Tensor(t)
    }
}
impl<D: Device, K: DTypeKind<D>> From<&Tensor<D, K>> for TensorOrScalar<D, K> {
    fn from(t: &Tensor<D, K>) -> Self {
        Self::Tensor(t.clone())
    }
}

// ============================================================================
//   &t OP rhs   (rhs: impl Into<TensorOrScalar<D, K>>)
// ============================================================================

macro_rules! impl_ref_op {
    ($Trait:ident, $method:ident, $scalar_method:ident) => {
        impl<D, K, R> $Trait<R> for &Tensor<D, K>
        where
            D: Device,
            K: NumericDTypeKind<D> + ShapeDTypeKind<D>,
            R: Into<TensorOrScalar<D, K>>,
        {
            type Output = Tensor<D, K>;
            fn $method(self, rhs: R) -> Self::Output {
                match rhs.into() {
                    TensorOrScalar::Tensor(t) => Tensor::$method(self, &t).unwrap(),
                    TensorOrScalar::Scalar(s) => Tensor::$scalar_method(self, s).unwrap(),
                }
            }
        }

        impl<D, K, R> $Trait<R> for Tensor<D, K>
        where
            D: Device,
            K: NumericDTypeKind<D> + ShapeDTypeKind<D>,
            R: Into<TensorOrScalar<D, K>>,
        {
            type Output = Tensor<D, K>;
            fn $method(self, rhs: R) -> Self::Output {
                match rhs.into() {
                    TensorOrScalar::Tensor(t) => Tensor::$method(&self, &t).unwrap(),
                    TensorOrScalar::Scalar(s) => Tensor::$scalar_method(&self, s).unwrap(),
                }
            }
        }
    };
}

impl_ref_op!(Add, add, add_scalar);
impl_ref_op!(Sub, sub, sub_scalar);
impl_ref_op!(Mul, mul, mul_scalar);
impl_ref_op!(Div, div, div_scalar);

// ============================================================================
//   s OP &t   (scalar-left — concrete per scalar type)
// ============================================================================

impl<D: Device> Add<&Tensor<D, Float>> for f64 {
    type Output = Tensor<D, Float>;
    fn add(self, rhs: &Tensor<D, Float>) -> Self::Output {
        Tensor::add_scalar(rhs, self).unwrap()
    }
}
impl<D: Device> Sub<&Tensor<D, Float>> for f64 {
    type Output = Tensor<D, Float>;
    fn sub(self, rhs: &Tensor<D, Float>) -> Self::Output {
        Tensor::sub_scalar_lhs(rhs, self).unwrap()
    }
}
impl<D: Device> Mul<&Tensor<D, Float>> for f64 {
    type Output = Tensor<D, Float>;
    fn mul(self, rhs: &Tensor<D, Float>) -> Self::Output {
        Tensor::mul_scalar(rhs, self).unwrap()
    }
}
impl<D: Device> Div<&Tensor<D, Float>> for f64 {
    type Output = Tensor<D, Float>;
    fn div(self, rhs: &Tensor<D, Float>) -> Self::Output {
        Tensor::div_scalar_lhs(rhs, self).unwrap()
    }
}

impl<D: Device> Add<Tensor<D, Float>> for f64 {
    type Output = Tensor<D, Float>;
    fn add(self, rhs: Tensor<D, Float>) -> Self::Output {
        Tensor::add_scalar(&rhs, self).unwrap()
    }
}
impl<D: Device> Sub<Tensor<D, Float>> for f64 {
    type Output = Tensor<D, Float>;
    fn sub(self, rhs: Tensor<D, Float>) -> Self::Output {
        Tensor::sub_scalar_lhs(&rhs, self).unwrap()
    }
}
impl<D: Device> Mul<Tensor<D, Float>> for f64 {
    type Output = Tensor<D, Float>;
    fn mul(self, rhs: Tensor<D, Float>) -> Self::Output {
        Tensor::mul_scalar(&rhs, self).unwrap()
    }
}
impl<D: Device> Div<Tensor<D, Float>> for f64 {
    type Output = Tensor<D, Float>;
    fn div(self, rhs: Tensor<D, Float>) -> Self::Output {
        Tensor::div_scalar_lhs(&rhs, self).unwrap()
    }
}

impl<D: Device> Add<&Tensor<D, Int>> for i64 {
    type Output = Tensor<D, Int>;
    fn add(self, rhs: &Tensor<D, Int>) -> Self::Output {
        Tensor::add_scalar(rhs, self).unwrap()
    }
}
impl<D: Device> Mul<&Tensor<D, Int>> for i64 {
    type Output = Tensor<D, Int>;
    fn mul(self, rhs: &Tensor<D, Int>) -> Self::Output {
        Tensor::mul_scalar(rhs, self).unwrap()
    }
}

impl<D: Device> Add<Tensor<D, Int>> for i64 {
    type Output = Tensor<D, Int>;
    fn add(self, rhs: Tensor<D, Int>) -> Self::Output {
        Tensor::add_scalar(&rhs, self).unwrap()
    }
}
impl<D: Device> Mul<Tensor<D, Int>> for i64 {
    type Output = Tensor<D, Int>;
    fn mul(self, rhs: Tensor<D, Int>) -> Self::Output {
        Tensor::mul_scalar(&rhs, self).unwrap()
    }
}

// ============================================================================
//   t OP= &t  +  -&t, -t
// ============================================================================

macro_rules! impl_assign_and_neg {
    ($kind:ty) => {
        impl<D: Device> AddAssign<&Tensor<D, $kind>> for Tensor<D, $kind>
        where
            $kind: NumericDTypeKind<D> + ShapeDTypeKind<D>,
        {
            fn add_assign(&mut self, rhs: &Tensor<D, $kind>) {
                Tensor::add_(self, rhs).unwrap();
            }
        }
        impl<D: Device> SubAssign<&Tensor<D, $kind>> for Tensor<D, $kind>
        where
            $kind: NumericDTypeKind<D> + ShapeDTypeKind<D>,
        {
            fn sub_assign(&mut self, rhs: &Tensor<D, $kind>) {
                Tensor::sub_(self, rhs).unwrap();
            }
        }
        impl<D: Device> MulAssign<&Tensor<D, $kind>> for Tensor<D, $kind>
        where
            $kind: NumericDTypeKind<D> + ShapeDTypeKind<D>,
        {
            fn mul_assign(&mut self, rhs: &Tensor<D, $kind>) {
                Tensor::mul_(self, rhs).unwrap();
            }
        }
        impl<D: Device> DivAssign<&Tensor<D, $kind>> for Tensor<D, $kind>
        where
            $kind: NumericDTypeKind<D> + ShapeDTypeKind<D>,
        {
            fn div_assign(&mut self, rhs: &Tensor<D, $kind>) {
                Tensor::div_(self, rhs).unwrap();
            }
        }

        impl<D: Device> Neg for &Tensor<D, $kind>
        where
            $kind: NumericDTypeKind<D> + ShapeDTypeKind<D>,
        {
            type Output = Tensor<D, $kind>;
            fn neg(self) -> Self::Output {
                Tensor::neg(self).unwrap()
            }
        }
        impl<D: Device> Neg for Tensor<D, $kind>
        where
            $kind: NumericDTypeKind<D> + ShapeDTypeKind<D>,
        {
            type Output = Tensor<D, $kind>;
            fn neg(self) -> Self::Output {
                -&self
            }
        }
    };
}

impl_assign_and_neg!(Float);
impl_assign_and_neg!(Int);
