use super::{arith::TensorOrScalar, shape::ShapeDTypeKind};
use crate::{BinaryOp, Bool, CmpOp, DTypeKind, Device, Float, FloatMeta, FloatUnaryOp, Int, Layout, Storage, Tensor, TensorMeta, UnaryOp};

// ============================================================================
//    Numeric dispatch
// ============================================================================

pub trait NumericDTypeKind<D: Device>: DTypeKind<D> + Sized {
    fn binary_dispatch(
        lhs: &Self::Storage,
        lhs_l: &Layout,
        rhs: &Self::Storage,
        rhs_l: &Layout,
        op: BinaryOp,
    ) -> crate::Result<Self::Storage>;
    fn binary_inplace_dispatch(
        dst: &mut Self::Storage,
        dst_l: &Layout,
        src: &Self::Storage,
        src_l: &Layout,
        op: BinaryOp,
    ) -> crate::Result<()>;
    fn binary_scalar_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: Self::Scalar, op: BinaryOp) -> crate::Result<Self::Storage>;
    fn binary_scalar_inplace_dispatch(dst: &mut Self::Storage, dst_l: &Layout, rhs: Self::Scalar, op: BinaryOp) -> crate::Result<()>;
    fn binary_scalar_lhs_dispatch(scalar: Self::Scalar, rhs: &Self::Storage, rhs_l: &Layout, op: BinaryOp) -> crate::Result<Self::Storage>;

    fn cmp_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: &Self::Storage, rhs_l: &Layout, op: CmpOp) -> crate::Result<D::BoolStorage>;
    fn cmp_scalar_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: Self::Scalar, op: CmpOp) -> crate::Result<D::BoolStorage>;

    fn unary_dispatch(x: &Self::Storage, l: &Layout, op: UnaryOp<Self::Scalar>) -> crate::Result<Self::Storage>;
    fn unary_inplace_dispatch(dst: &mut Self::Storage, dst_l: &Layout, op: UnaryOp<Self::Scalar>) -> crate::Result<()>;
}

impl<D: Device> NumericDTypeKind<D> for Float {
    fn binary_dispatch(
        lhs: &Self::Storage,
        lhs_l: &Layout,
        rhs: &Self::Storage,
        rhs_l: &Layout,
        op: BinaryOp,
    ) -> crate::Result<Self::Storage> {
        D::f_binary(lhs, lhs_l, rhs, rhs_l, op)
    }
    fn binary_inplace_dispatch(
        dst: &mut Self::Storage,
        dst_l: &Layout,
        src: &Self::Storage,
        src_l: &Layout,
        op: BinaryOp,
    ) -> crate::Result<()> {
        D::f_binary_(dst, dst_l, src, src_l, op)
    }
    fn binary_scalar_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: Self::Scalar, op: BinaryOp) -> crate::Result<Self::Storage> {
        D::f_binary_scalar(lhs, lhs_l, rhs, op)
    }
    fn binary_scalar_inplace_dispatch(dst: &mut Self::Storage, dst_l: &Layout, rhs: Self::Scalar, op: BinaryOp) -> crate::Result<()> {
        D::f_binary_scalar_(dst, dst_l, rhs, op)
    }
    fn binary_scalar_lhs_dispatch(scalar: Self::Scalar, rhs: &Self::Storage, rhs_l: &Layout, op: BinaryOp) -> crate::Result<Self::Storage> {
        D::f_binary_scalar_lhs(scalar, rhs, rhs_l, op)
    }
    fn cmp_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: &Self::Storage, rhs_l: &Layout, op: CmpOp) -> crate::Result<D::BoolStorage> {
        D::f_cmp(lhs, lhs_l, rhs, rhs_l, op)
    }
    fn cmp_scalar_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: Self::Scalar, op: CmpOp) -> crate::Result<D::BoolStorage> {
        D::f_cmp_scalar(lhs, lhs_l, rhs, op)
    }
    fn unary_dispatch(x: &Self::Storage, l: &Layout, op: UnaryOp<Self::Scalar>) -> crate::Result<Self::Storage> {
        D::f_unary(x, l, op)
    }
    fn unary_inplace_dispatch(dst: &mut Self::Storage, dst_l: &Layout, op: UnaryOp<Self::Scalar>) -> crate::Result<()> {
        D::f_unary_(dst, dst_l, op)
    }
}

impl<D: Device> NumericDTypeKind<D> for Int {
    fn binary_dispatch(
        lhs: &Self::Storage,
        lhs_l: &Layout,
        rhs: &Self::Storage,
        rhs_l: &Layout,
        op: BinaryOp,
    ) -> crate::Result<Self::Storage> {
        D::i_binary(lhs, lhs_l, rhs, rhs_l, op)
    }
    fn binary_inplace_dispatch(
        dst: &mut Self::Storage,
        dst_l: &Layout,
        src: &Self::Storage,
        src_l: &Layout,
        op: BinaryOp,
    ) -> crate::Result<()> {
        D::i_binary_(dst, dst_l, src, src_l, op)
    }
    fn binary_scalar_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: Self::Scalar, op: BinaryOp) -> crate::Result<Self::Storage> {
        D::i_binary_scalar(lhs, lhs_l, rhs, op)
    }
    fn binary_scalar_inplace_dispatch(dst: &mut Self::Storage, dst_l: &Layout, rhs: Self::Scalar, op: BinaryOp) -> crate::Result<()> {
        D::i_binary_scalar_(dst, dst_l, rhs, op)
    }
    fn binary_scalar_lhs_dispatch(scalar: Self::Scalar, rhs: &Self::Storage, rhs_l: &Layout, op: BinaryOp) -> crate::Result<Self::Storage> {
        D::i_binary_scalar_lhs(scalar, rhs, rhs_l, op)
    }
    fn cmp_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: &Self::Storage, rhs_l: &Layout, op: CmpOp) -> crate::Result<D::BoolStorage> {
        D::i_cmp(lhs, lhs_l, rhs, rhs_l, op)
    }
    fn cmp_scalar_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: Self::Scalar, op: CmpOp) -> crate::Result<D::BoolStorage> {
        D::i_cmp_scalar(lhs, lhs_l, rhs, op)
    }
    fn unary_dispatch(x: &Self::Storage, l: &Layout, op: UnaryOp<Self::Scalar>) -> crate::Result<Self::Storage> {
        D::i_unary(x, l, op)
    }
    fn unary_inplace_dispatch(dst: &mut Self::Storage, dst_l: &Layout, op: UnaryOp<Self::Scalar>) -> crate::Result<()> {
        D::i_unary_(dst, dst_l, op)
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

    pub(crate) fn binary_scalar_impl(&self, rhs: K::Scalar, op: BinaryOp) -> crate::Result<Self> {
        let storage = K::binary_scalar_dispatch(&*self.storage_read()?, self.layout(), rhs, op)?;
        let meta = K::Meta::on_binary_scalar(self, rhs, op);
        assert_eq!(self.dtype(), storage.dtype());
        Ok(Self::from_storage(storage, self.shape().clone(), meta))
    }

    pub(crate) fn binary_inplace_impl(&self, rhs: &Self, op: BinaryOp) -> crate::Result<Self> {
        let mut s = self.storage_write()?;
        K::binary_inplace_dispatch(&mut s, self.layout(), &*rhs.storage_read()?, rhs.layout(), op)?;
        Ok(self.clone())
    }

    pub(crate) fn binary_scalar_inplace_impl(&self, rhs: K::Scalar, op: BinaryOp) -> crate::Result<Self> {
        let mut s = self.storage_write()?;
        K::binary_scalar_inplace_dispatch(&mut s, self.layout(), rhs, op)?;
        Ok(self.clone())
    }

    pub(crate) fn binary_broadcast_impl(&self, rhs: &Self, op: BinaryOp, name: &'static str) -> crate::Result<Self> {
        let out_shape = self.shape().broadcast_shape_binary_op(rhs.shape(), name)?;
        let lhs = self.broadcast_as(out_shape.clone())?;
        let rhs = rhs.broadcast_as(out_shape)?;
        lhs.binary_impl(&rhs, op, name)
    }
}

macro_rules! binary_impl {
    ($name:ident, $variant:ident) => {
        paste::paste! {
            pub fn $name(&self, rhs: impl Into<TensorOrScalar<D, K>>) -> crate::Result<Self> {
                let rhs: TensorOrScalar<D, K> = rhs.into();
                match rhs {
                    TensorOrScalar::Scalar(rhs) => self.binary_scalar_impl(rhs, BinaryOp::$variant),
                    TensorOrScalar::Tensor(rhs) => self.binary_impl(&rhs, BinaryOp::$variant, stringify!($name)),
                }
            }

            #[inline]
            pub fn [<$name _scalar>](&self, rhs: K::Scalar) -> crate::Result<Self> {
                self.binary_scalar_impl(rhs, BinaryOp::$variant)
            }

            pub fn [<$name _>](&self, rhs: impl Into<TensorOrScalar<D, K>>) -> crate::Result<Self> {
                let rhs: TensorOrScalar<D, K> = rhs.into();
                match rhs {
                    TensorOrScalar::Scalar(rhs) => self.binary_scalar_inplace_impl(rhs, BinaryOp::$variant),
                    TensorOrScalar::Tensor(rhs) => self.binary_inplace_impl(&rhs, BinaryOp::$variant),
                }
            }

            #[inline]
            pub fn [<$name _scalar_>](&self, rhs: K::Scalar) -> crate::Result<Self> {
                self.binary_scalar_inplace_impl(rhs, BinaryOp::$variant)
            }

            #[inline]
            pub fn [<broadcast_ $name>](&self, rhs: &Self) -> crate::Result<Self> {
                self.binary_broadcast_impl(rhs, BinaryOp::$variant, stringify!([<broadcast_ $name>]))
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

    pub(crate) fn cmp_scalar_imlp(&self, rhs: K::Scalar, op: CmpOp) -> crate::Result<Tensor<D, Bool>> {
        let s = K::cmp_scalar_dispatch(&*self.storage_read()?, self.layout(), rhs, op)?;
        Ok(Tensor::<D, Bool>::from_storage(s, self.shape(), ()))
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
            pub fn $name(&self, rhs: impl Into<TensorOrScalar<D, K>>) -> crate::Result<Tensor<D, Bool>> {
                let rhs: TensorOrScalar<D, K> = rhs.into();
                match rhs {
                    TensorOrScalar::Scalar(rhs) => self.cmp_scalar_imlp(rhs, CmpOp::$variant),
                    TensorOrScalar::Tensor(rhs) => self.cmp_impl(&rhs, CmpOp::$variant, stringify!($name)),
                }
            }

            #[inline]
            pub fn [<$name _scalar>](&self, rhs: K::Scalar) -> crate::Result<Tensor<D, Bool>> {
                self.cmp_scalar_imlp(rhs, CmpOp::$variant)
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
//   unary
// ============================================================================

impl<D: Device, K: NumericDTypeKind<D> + ShapeDTypeKind<D>> Tensor<D, K> {
    pub(crate) fn unary_impl(&self, op: UnaryOp<K::Scalar>) -> crate::Result<Self> {
        let storage = K::unary_dispatch(&*self.storage_read()?, self.layout(), op)?;
        let meta = K::Meta::on_unary(self, op);
        Ok(Self::from_storage(storage, self.shape().clone(), meta))
    }

    pub(crate) fn unary_inplace_impl(&self, op: UnaryOp<K::Scalar>) -> crate::Result<Self> {
        let mut s = self.storage_write()?;
        K::unary_inplace_dispatch(&mut s, self.layout(), op)?;
        Ok(self.clone())
    }
}

macro_rules! unary_method {
    ($name:ident, $op:tt) => {
        paste::paste! {
            #[inline]
            pub fn $name(&self) -> crate::Result<Self> {
                self.unary_impl(UnaryOp::$op)
            }

            #[inline]
            pub fn [<$name _>](&self) -> crate::Result<Self> {
                self.unary_inplace_impl(UnaryOp::$op)
            }
        }
    };
}

impl<D: Device, K: NumericDTypeKind<D> + ShapeDTypeKind<D>> Tensor<D, K> {
    unary_method!(neg, Neg);
    unary_method!(abs, Abs);
    unary_method!(sign, Sign);

    #[inline]
    pub fn affine(&self, mul: K::Scalar, add: K::Scalar) -> crate::Result<Self> {
        self.unary_impl(UnaryOp::Affine(mul, add))
    }

    #[inline]
    pub fn affine_(&self, mul: K::Scalar, add: K::Scalar) -> crate::Result<Self> {
        self.unary_inplace_impl(UnaryOp::Affine(mul, add))
    }

    #[inline]
    pub fn pow(&self, exp: K::Scalar) -> crate::Result<Self> {
        self.unary_impl(UnaryOp::Pow(exp))
    }

    #[inline]
    pub fn pow_(&self, exp: K::Scalar) -> crate::Result<Self> {
        self.unary_inplace_impl(UnaryOp::Pow(exp))
    }

    #[inline]
    pub fn clamp(&self, min: Option<K::Scalar>, max: Option<K::Scalar>) -> crate::Result<Self> {
        self.unary_impl(UnaryOp::Clamp(min, max))
    }

    #[inline]
    pub fn clamp_(&self, min: Option<K::Scalar>, max: Option<K::Scalar>) -> crate::Result<Self> {
        self.unary_inplace_impl(UnaryOp::Clamp(min, max))
    }

    /// `scalar - self` (scalar on the **left**).
    #[inline]
    pub fn sub_scalar_lhs(&self, lhs: K::Scalar) -> crate::Result<Self> {
        let storage = K::binary_scalar_lhs_dispatch(lhs, &*self.storage_read()?, self.layout(), BinaryOp::Sub)?;
        let meta = K::Meta::default();
        Ok(Self::from_storage(storage, self.shape().clone(), meta))
    }

    /// `scalar / self` (scalar on the **left**).
    #[inline]
    pub fn div_scalar_lhs(&self, lhs: K::Scalar) -> crate::Result<Self> {
        let storage = K::binary_scalar_lhs_dispatch(lhs, &*self.storage_read()?, self.layout(), BinaryOp::Div)?;
        let meta = K::Meta::default();
        Ok(Self::from_storage(storage, self.shape().clone(), meta))
    }
}

// ============================================================================
//   activate for float
// ============================================================================

macro_rules! float_unary_method {
    ($name:ident, $variant:ident) => {
        paste::paste! {
            #[inline]
            pub fn $name(&self) -> crate::Result<Self> {
                self.float_unary_impl(FloatUnaryOp::$variant)
            }

            #[inline]
            pub fn [<$name _>](&self) -> crate::Result<()> {
                self.float_unary_inplace_impl(FloatUnaryOp::$variant)
            }
        }
    };
}

impl<D: Device> Tensor<D, Float> {
    fn float_unary_impl(&self, op: FloatUnaryOp) -> crate::Result<Self> {
        let storage = D::f_float_unary(&*self.storage_read()?, self.layout(), op)?;
        let meta = FloatMeta::on_float_unary(self, op);
        Ok(Self::from_storage(storage, self.shape().clone(), meta))
    }

    fn float_unary_inplace_impl(&self, op: FloatUnaryOp) -> crate::Result<()> {
        let mut s = self.storage_write()?;
        D::f_float_unary_(&mut s, self.layout(), op)
    }

    float_unary_method!(exp, Exp);
    float_unary_method!(ln, Ln);
    float_unary_method!(sin, Sin);
    float_unary_method!(cos, Cos);
    float_unary_method!(tanh, Tanh);
    float_unary_method!(sqr, Sqr);
    float_unary_method!(sqrt, Sqrt);
    float_unary_method!(recip, Recip);
    float_unary_method!(gelu, Gelu);
    float_unary_method!(gelu_erf, GeluErf);
    float_unary_method!(erf, Erf);
    float_unary_method!(relu, Relu);
    float_unary_method!(silu, Silu);
    float_unary_method!(sigmoid, Sigmoid);
    float_unary_method!(floor, Floor);
    float_unary_method!(ceil, Ceil);
    float_unary_method!(round, Round);

    pub fn leaky_relu(&self, negative_slope: f64) -> crate::Result<Self> {
        self.float_unary_impl(FloatUnaryOp::LeakyRelu(negative_slope))
    }

    pub fn leaky_relu_(&self, negative_slope: f64) -> crate::Result<()> {
        self.float_unary_inplace_impl(FloatUnaryOp::LeakyRelu(negative_slope))
    }
}
