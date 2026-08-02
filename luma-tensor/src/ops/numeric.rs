use crate::{BinaryOp, Bool, CmpOp, DTypeKind, Device, Float, FloatMeta, Int, Layout, Tensor, TensorMeta, UnaryOp, Storage};
use super::shape::ShapeDTypeKind;

// ============================================================================
//    Numeric dispatch
// ============================================================================

pub trait NumericDTypeKind<D: Device>: DTypeKind<D> + Sized {
    fn binary_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: &Self::Storage, rhs_l: &Layout, op: BinaryOp) -> crate::Result<Self::Storage>;
    fn binary_scalar_rhs_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: Self::Scalar, op: BinaryOp) -> crate::Result<Self::Storage>;
    fn binary_scalar_lhs_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: Self::Scalar, op: BinaryOp) -> crate::Result<Self::Storage>;

    fn cmp_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: &Self::Storage, rhs_l: &Layout, op: CmpOp) -> crate::Result<D::BoolStorage>;
    
    fn neg_dispatch(x: &Self::Storage, l: &Layout) -> crate::Result<Self::Storage>;
    fn abs_dispatch(x: &Self::Storage, l: &Layout) -> crate::Result<Self::Storage>;
}

impl<D: Device> NumericDTypeKind<D> for Float {
    fn binary_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: &Self::Storage, rhs_l: &Layout, op: BinaryOp) -> crate::Result<Self::Storage> {
        D::f_binary(lhs, lhs_l, rhs, rhs_l, op)
    }

    fn binary_scalar_rhs_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: Self::Scalar, op: BinaryOp) -> crate::Result<Self::Storage> {
        D::f_binary_scalar(lhs, lhs_l, rhs, op)
    }

    fn binary_scalar_lhs_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: Self::Scalar, op: BinaryOp) -> crate::Result<Self::Storage> {
        D::f_binary_scalar(lhs, lhs_l, rhs, op)
    }

    fn cmp_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: &Self::Storage, rhs_l: &Layout, op: CmpOp) -> crate::Result<D::BoolStorage> {
        D::f_cmp(lhs, lhs_l, rhs, rhs_l, op)
    }

    fn neg_dispatch(x: &Self::Storage, l: &Layout) -> crate::Result<Self::Storage> {
        D::f_unary(x, l, UnaryOp::Neg)
    }

    fn abs_dispatch(x: &Self::Storage, l: &Layout) -> crate::Result<Self::Storage> {
        D::f_unary(x, l, UnaryOp::Abs)
    }
}

impl<D: Device> NumericDTypeKind<D> for Int {
    fn binary_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: &Self::Storage, rhs_l: &Layout, op: BinaryOp) -> crate::Result<Self::Storage> {
        D::i_binary(lhs, lhs_l, rhs, rhs_l, op)
    }

    fn binary_scalar_rhs_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: Self::Scalar, op: BinaryOp) -> crate::Result<Self::Storage> {
        D::i_binary_scalar(lhs, lhs_l, rhs, op)
    }

    fn binary_scalar_lhs_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: Self::Scalar, op: BinaryOp) -> crate::Result<Self::Storage> {
        D::i_binary_scalar(lhs, lhs_l, rhs, op)
    }

    fn cmp_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: &Self::Storage, rhs_l: &Layout, op: CmpOp) -> crate::Result<D::BoolStorage> {
        D::i_cmp(lhs, lhs_l, rhs, rhs_l, op)
    }

    fn neg_dispatch(x: &Self::Storage, l: &Layout) -> crate::Result<Self::Storage> {
        D::i_neg(x, l)
    }

    fn abs_dispatch(x: &Self::Storage, l: &Layout) -> crate::Result<Self::Storage> {
        D::i_abs(x, l)
    }
}

// ============================================================================
//   impl binary op
// ============================================================================

impl<D: Device, K: NumericDTypeKind<D> + ShapeDTypeKind<D>> Tensor<D, K> {
    pub(crate) fn binary_impl(&self, rhs: &Self, op: BinaryOp, name: &'static str) -> crate::Result<Self> {
        let shape = self.same_shape(rhs, name)?.clone();
        let s = K::binary_dispatch(&*self.storage_read()?, self.layout(), &*rhs.storage_read()?, rhs.layout(), op)?;
        assert_eq!(self.dtype(), s.dtype());
        Ok(Self::from_storage(s, shape, K::Meta::on_binary(self, rhs, op)))
    }

    pub(crate) fn binary_broadcast_impl(&self, rhs: &Self, op: BinaryOp, name: &'static str) -> crate::Result<Self> {
        let out_shape = self.shape().broadcast_shape_binary_op(rhs.shape(), name)?;
        let lhs = self.broadcast_as(out_shape.clone())?;
        let rhs = rhs.broadcast_as(out_shape)?;
        lhs.binary_impl(&rhs, op, name)
    }

    pub(crate) fn binary_scalar_rhs_impl(&self, rhs: K::Scalar, op: BinaryOp) -> crate::Result<Self> {
        let storage = K::binary_scalar_rhs_dispatch(&*self.storage_read()?, self.layout(), rhs, op)?;
        let meta = K::Meta::on_binary_scalar_rhs(self, rhs, op);
        assert_eq!(self.dtype(), storage.dtype());
        Ok(Self::from_storage(storage, self.shape().clone(), meta))
    }

    pub(crate) fn binary_scalar_lhs_impl(&self, rhs: K::Scalar, op: BinaryOp) -> crate::Result<Self> {
        let storage = K::binary_scalar_rhs_dispatch(&*self.storage_read()?, self.layout(), rhs, op)?;
        let meta = K::Meta::on_binary_scalar_rhs(self, rhs, op);
        assert_eq!(self.dtype(), storage.dtype());
        Ok(Self::from_storage(storage, self.shape().clone(), meta))
    }
}

macro_rules! binary_impl {
    ($name:ident, $variant:ident) => {
        paste::paste! {
            #[inline]
            pub fn $name(&self, rhs: &Self) -> crate::Result<Self> {
                self.binary_impl(rhs, BinaryOp::$variant, stringify!($name))
            }

            #[inline]
            pub fn [<broadcast_ $name>](&self, rhs: &Self) -> crate::Result<Self> {
                self.binary_broadcast_impl(rhs, BinaryOp::$variant, stringify!([<broadcast_ $name>]))
            }
            
            #[inline]
            pub fn [<$name _scalar>](&self, rhs: K::Scalar) -> crate::Result<Self> {
                self.binary_scalar_rhs_impl(rhs, BinaryOp::$variant)
            }

            #[inline]
            pub fn [<$name _scalar_lhs>](&self, rhs: K::Scalar) -> crate::Result<Self> {
                self.binary_scalar_lhs_impl(rhs, BinaryOp::$variant)
            }
        }
    };
}

impl<D: Device, K: NumericDTypeKind<D> + ShapeDTypeKind<D>> Tensor<D, K> {
    binary_impl!(add, Add);
    binary_impl!(sub, Sub);
    binary_impl!(mul, Mul);
    binary_impl!(div, Div);
    binary_impl!(maximum, Maximum);
    binary_impl!(minimum, Minimum);

    pub fn neg(&self) -> crate::Result<Self> {
        let s = K::neg_dispatch(&*self.storage_read()?, self.layout())?;
        assert_eq!(self.dtype(), s.dtype());
        Ok(Self::from_storage(s, self.shape().clone(), K::Meta::on_unary(self, UnaryOp::Neg)))
    }

    pub fn abs(&self) -> crate::Result<Self> {
        let s = K::abs_dispatch(&*self.storage_read()?, self.layout())?;
        assert_eq!(self.dtype(), s.dtype());
        Ok(Self::from_storage(s, self.shape().clone(), K::Meta::on_unary(self, UnaryOp::Abs)))
    }
}

// ============================================================================
//   impl cmp op
// ============================================================================

impl<D: Device, K: NumericDTypeKind<D> + ShapeDTypeKind<D>> Tensor<D, K> {
    pub(crate) fn cmp_impl(&self, rhs: &Self, op: CmpOp, name: &'static str) -> crate::Result<Tensor<D, Bool>> {
        let shape = self.same_shape(rhs, name)?.clone();
        let s = K::cmp_dispatch(&*self.storage_read()?, self.layout(), &*rhs.storage_read()?, rhs.layout(), op)?;
        Ok(Tensor::<D, Bool>::from_storage(s, shape, ()))
    }

    pub(crate) fn broadcast_cmp_impl(&self, rhs: &Self, op: CmpOp, name: &'static str) -> crate::Result<Tensor<D, Bool>> {
        let out_shape = self.shape().broadcast_shape_binary_op(rhs.shape(), name)?;
        let lhs = self.broadcast_as(out_shape.clone())?;
        let rhs = rhs.broadcast_as(out_shape)?;
        lhs.cmp_impl(&rhs, op, name)
    }
}

macro_rules! cmp_impl {
    ($name:ident, $variant:ident) => {
        paste::paste! {
            #[inline]
            pub fn $name(&self, rhs: &Self) -> crate::Result<Tensor<D, Bool>> {
                self.cmp_impl(rhs, CmpOp::$variant, stringify!($name))
            }

            #[inline]
            pub fn [<broadcast_ $name>](&self, rhs: &Self) -> crate::Result<Tensor<D, Bool>> {
                self.broadcast_cmp_impl(rhs, CmpOp::$variant, stringify!([<broadcast_ $name>]))
            }
        }
    };
}

impl<D: Device, K: NumericDTypeKind<D> + ShapeDTypeKind<D>> Tensor<D, K> {
    cmp_impl!(eq, Eq);
    cmp_impl!(ne, Ne);
    cmp_impl!(lt, Lt);
    cmp_impl!(le, Le);
    cmp_impl!(gt, Gt);
    cmp_impl!(ge, Ge);
}

// ============================================================================
//   activate for float
// ============================================================================

macro_rules! unary_method {
    ($name:ident, $variant:ident) => {
        pub fn $name(&self) -> crate::Result<Self> {
            self.unary_impl(UnaryOp::$variant)
        }
    };
}

impl<D: Device> Tensor<D, Float> {
    fn unary_impl(&self, op: UnaryOp) -> crate::Result<Self> {
        let storage = D::f_unary(&*self.storage_read()?, self.layout(), op)?;
        let meta = FloatMeta::on_unary(self, op);
        Ok(Self::from_storage(storage, self.shape().clone(), meta))
    }

    unary_method!(exp, Exp);
    unary_method!(ln, Ln);
    unary_method!(sin, Sin);
    unary_method!(cos, Cos);
    unary_method!(tanh, Tanh);
    unary_method!(sqr, Sqr);
    unary_method!(sqrt, Sqrt);
    unary_method!(recip, Recip);
    unary_method!(gelu, Gelu);
    unary_method!(gelu_erf, GeluErf);
    unary_method!(erf, Erf);
    unary_method!(relu, Relu);
    unary_method!(silu, Silu);
    unary_method!(sigmoid, Sigmoid);
    unary_method!(floor, Floor);
    unary_method!(ceil, Ceil);
    unary_method!(round, Round);
    unary_method!(sign, Sign);

    pub fn leaky_relu(&self, negative_slope: f64) -> crate::Result<Self> {
        self.unary_impl(UnaryOp::LeakyRelu(negative_slope))
    }

    pub fn pow(&self, e: f64) -> crate::Result<Self> {
        self.unary_impl(UnaryOp::Pow(e))
    }

    pub fn affine(&self, mul: f64, add: f64) -> crate::Result<Self> {
        self.unary_impl(UnaryOp::Affine { mul, add })
    }

    /// Clamp values to `[min, max]`. Pass `None` to skip one bound.
    pub fn clamp(&self, min: Option<f64>, max: Option<f64>) -> crate::Result<Self> {
        self.unary_impl(UnaryOp::Clamp { min, max })
    }
}
