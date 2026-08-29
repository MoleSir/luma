//! Per-element numeric traits. Generic CPU kernels are written over these, then
//! the storage enums dispatch to the concrete `f32`/`f64`/`i32`/… instantiation.
//!
//! This is the CPU analogue of luma-core's `NumDType`/`FloatDType`/`IntDType`,
//! trimmed to exactly what the kernels need.

use std::iter::{Product, Sum};

/// Numeric element shared by float and int kinds.
pub trait CpuDType:
    Copy
    + PartialOrd
    + Send
    + Sync
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
}

pub trait CpuNum:
    CpuDType
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + Sum
    + Product
{
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn from_usize(v: usize) -> Self;
    fn to_usize(self) -> usize;

    fn minimum(a: Self, b: Self) -> Self {
        if a > b { b } else { a }
    }
    fn maximum(a: Self, b: Self) -> Self {
        if a < b { b } else { a }
    }
    fn abs(self) -> Self;
    fn signum(self) -> Self;
}

/// Float-only element: transcendental math and activations.
pub trait CpuFloat: CpuNum + std::ops::Neg<Output = Self> {
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tanh(self) -> Self;
    fn sqrt(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn powf(self, e: Self) -> Self;
    fn recip(self) -> Self;
    fn erf(self) -> Self;

    fn sqr(self) -> Self {
        self * self
    }
    fn relu(self) -> Self {
        Self::maximum(self, Self::ZERO)
    }
    fn leaky_relu(self, negative_slope: Self) -> Self {
        if self > Self::ZERO { self } else { self * negative_slope }
    }
    fn sigmoid(self) -> Self {
        Self::ONE / (Self::ONE + (-self).exp())
    }
    fn silu(self) -> Self {
        self / (Self::ONE + (-self).exp())
    }
    /// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
    fn gelu(self) -> Self {
        let sqrt_2_over_pi = Self::from_f64(0.797_884_560_802_865_4);
        let coef = Self::from_f64(0.044715);
        let half = Self::from_f64(0.5);
        let x3 = self * self * self;
        let inner = sqrt_2_over_pi * (self + coef * x3);
        half * self * (Self::ONE + inner.tanh())
    }
    /// 0.5 * x * (1 + erf(x / sqrt(2)))
    fn gelu_erf(self) -> Self {
        let frac_1_sqrt_2 = Self::from_f64(std::f64::consts::FRAC_1_SQRT_2);
        let half = Self::from_f64(0.5);
        half * self * (Self::ONE + (self * frac_1_sqrt_2).erf())
    }
}

/// Int-only element. `MAX` doubles as the padding sentinel used by indexing ops
/// (matching luma-core's `I::max_value()` convention).
pub trait CpuInt: CpuNum + Ord + Eq {
    const MAX: Self;
}

// ---- f32 / f64 ----
macro_rules! impl_cpu_float {
    ($t:ty, $erf:path) => {
        impl CpuDType for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
        }

        impl CpuNum for $t {
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn from_usize(v: usize) -> Self {
                v as $t
            }
            fn to_usize(self) -> usize {
                self as usize
            }
            fn abs(self) -> Self {
                <$t>::abs(self)
            }
            fn signum(self) -> Self {
                <$t>::signum(self)
            }
        }
        impl CpuFloat for $t {
            fn exp(self) -> Self {
                <$t>::exp(self)
            }
            fn ln(self) -> Self {
                <$t>::ln(self)
            }
            fn sin(self) -> Self {
                <$t>::sin(self)
            }
            fn cos(self) -> Self {
                <$t>::cos(self)
            }
            fn tanh(self) -> Self {
                <$t>::tanh(self)
            }
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            fn floor(self) -> Self {
                <$t>::floor(self)
            }
            fn ceil(self) -> Self {
                <$t>::ceil(self)
            }
            fn round(self) -> Self {
                <$t>::round(self)
            }
            fn powf(self, e: Self) -> Self {
                <$t>::powf(self, e)
            }
            fn recip(self) -> Self {
                <$t>::recip(self)
            }
            fn erf(self) -> Self {
                $erf(self)
            }
        }
    };
}

impl_cpu_float!(f32, libm::erff);
impl_cpu_float!(f64, libm::erf);

// ---- i32 / u32 / u8 ----
macro_rules! impl_cpu_int {
    ($t:ty) => {
        impl CpuDType for $t {
            const ZERO: Self = 0;
            const ONE: Self = 1;
        }

        impl CpuNum for $t {
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn from_usize(v: usize) -> Self {
                v as $t
            }
            fn to_usize(self) -> usize {
                self as usize
            }
            fn abs(self) -> Self {
                // unsigned types have no abs; wrap the signed case only.
                #[allow(unused_comparisons)]
                if self < 0 { Self::ZERO.wrapping_sub(self) } else { self }
            }
            fn signum(self) -> Self {
                #[allow(unused_comparisons)]
                if self > 0 {
                    Self::ONE
                } else if self < 0 {
                    // unreachable for unsigned; wrapping_sub avoids const-overflow.
                    Self::ZERO.wrapping_sub(Self::ONE)
                } else {
                    Self::ZERO
                }
            }
        }
        impl CpuInt for $t {
            const MAX: Self = <$t>::MAX;
        }
    };
}

impl_cpu_int!(i32);
impl_cpu_int!(u32);
impl_cpu_int!(u8);

impl CpuDType for bool {
    const ZERO: Self = false;
    const ONE: Self = true;
}
