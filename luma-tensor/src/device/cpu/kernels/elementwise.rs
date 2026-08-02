//! Generic elementwise kernels (binary, binary-scalar, unary, comparison,
//! in-place accumulate). Ported from luma-core's `arith.rs`, generalized over
//! the [`CpuNum`] element trait and returning plain `Vec`s.

use super::element::{CpuFloat, CpuNum};
use crate::{BinaryOp, CmpOp, Layout, StorageIndices, UnaryOp};

/// `out[i] = f(lhs[i], rhs[i])` over two (possibly non-contiguous) views of the
/// same logical shape. Mirrors luma-core `compute_binary_op`.
pub fn binary<T: Copy, U, F>(lhs: &[T], lhs_l: &Layout, rhs: &[T], rhs_l: &Layout, f: F) -> Vec<U>
where
    F: Fn(T, T) -> U,
{
    match (lhs_l.storage_indices(), rhs_l.storage_indices()) {
        (StorageIndices::Contiguous(li), StorageIndices::Contiguous(ri)) => {
            let lhs = &lhs[li.begin_index..li.end_index];
            let rhs = &rhs[ri.begin_index..ri.end_index];
            lhs.iter().zip(rhs.iter()).map(|(&l, &r)| f(l, r)).collect()
        }
        (StorageIndices::Contiguous(li), StorageIndices::Uncontiguous(ri)) => {
            let lhs = &lhs[li.begin_index..li.end_index];
            lhs.iter().zip(ri).map(|(&l, ri)| f(l, rhs[ri])).collect()
        }
        (StorageIndices::Uncontiguous(li), StorageIndices::Contiguous(ri)) => {
            let rhs = &rhs[ri.begin_index..ri.end_index];
            li.zip(rhs.iter()).map(|(li, &r)| f(lhs[li], r)).collect()
        }
        (StorageIndices::Uncontiguous(li), StorageIndices::Uncontiguous(ri)) => li.zip(ri).map(|(li, ri)| f(lhs[li], rhs[ri])).collect(),
    }
}

/// `out[i] = f(lhs[i], scalar)`.
pub fn binary_scalar<T: Copy, U, F>(lhs: &[T], lhs_l: &Layout, rhs: T, f: F) -> Vec<U>
where
    F: Fn(T, T) -> U,
{
    match lhs_l.storage_indices() {
        StorageIndices::Contiguous(i) => {
            let lhs = &lhs[i.begin_index..i.end_index];
            lhs.iter().map(|&v| f(v, rhs)).collect()
        }
        StorageIndices::Uncontiguous(i) => i.map(|i| f(lhs[i], rhs)).collect(),
    }
}

/// `out[i] = f(scalar, rhs[i])`.
pub fn scalar_binary<T: Copy, U, F>(lhs: T, rhs: &[T], rhs_l: &Layout, f: F) -> Vec<U>
where
    F: Fn(T, T) -> U,
{
    match rhs_l.storage_indices() {
        StorageIndices::Contiguous(i) => {
            let rhs = &rhs[i.begin_index..i.end_index];
            rhs.iter().map(|&v| f(lhs, v)).collect()
        }
        StorageIndices::Uncontiguous(i) => i.map(|i| f(lhs, rhs[i])).collect(),
    }
}

/// `out[i] = f(x[i])`.
pub fn unary<T: Copy, U, F>(x: &[T], layout: &Layout, f: F) -> Vec<U>
where
    F: Fn(T) -> U,
{
    match layout.storage_indices() {
        StorageIndices::Contiguous(i) => {
            let x = &x[i.begin_index..i.end_index];
            x.iter().map(|&v| f(v)).collect()
        }
        StorageIndices::Uncontiguous(i) => i.map(|i| f(x[i])).collect(),
    }
}

// ---- op-enum -> closure dispatchers (shared by float and int) ----

fn num_binary_fn<T: CpuNum>(op: BinaryOp) -> fn(T, T) -> T {
    match op {
        BinaryOp::Add => |a, b| a + b,
        BinaryOp::Sub => |a, b| a - b,
        BinaryOp::Mul => |a, b| a * b,
        BinaryOp::Div => |a, b| a / b,
        BinaryOp::Maximum => T::maximum,
        BinaryOp::Minimum => T::minimum,
    }
}

fn cmp_fn<T: CpuNum>(op: CmpOp) -> fn(T, T) -> bool {
    match op {
        CmpOp::Eq => |a, b| a.partial_cmp(&b) == Some(std::cmp::Ordering::Equal),
        CmpOp::Ne => |a, b| a.partial_cmp(&b) != Some(std::cmp::Ordering::Equal),
        CmpOp::Lt => |a, b| a < b,
        CmpOp::Le => |a, b| a <= b,
        CmpOp::Gt => |a, b| a > b,
        CmpOp::Ge => |a, b| a >= b,
    }
}

/// Dispatch a numeric binary op over two views.
pub fn num_binary<T: CpuNum>(lhs: &[T], lhs_l: &Layout, rhs: &[T], rhs_l: &Layout, op: BinaryOp) -> Vec<T> {
    binary(lhs, lhs_l, rhs, rhs_l, num_binary_fn::<T>(op))
}

/// Dispatch a numeric binary-scalar op over one view.
pub fn num_binary_scalar<T: CpuNum>(lhs: &[T], lhs_l: &Layout, rhs: T, op: BinaryOp) -> Vec<T> {
    binary_scalar(lhs, lhs_l, rhs, num_binary_fn::<T>(op))
}

/// Dispatch a comparison op over two views, producing a bool buffer.
pub fn num_cmp<T: CpuNum>(lhs: &[T], lhs_l: &Layout, rhs: &[T], rhs_l: &Layout, op: CmpOp) -> Vec<bool> {
    binary(lhs, lhs_l, rhs, rhs_l, cmp_fn::<T>(op))
}

/// Dispatch a float unary op over one view.
pub fn float_unary<T: CpuFloat>(x: &[T], layout: &Layout, op: UnaryOp) -> Vec<T> {
    match op {
        UnaryOp::Exp => unary(x, layout, |v| v.exp()),
        UnaryOp::Ln => unary(x, layout, |v| v.ln()),
        UnaryOp::Sin => unary(x, layout, |v| v.sin()),
        UnaryOp::Cos => unary(x, layout, |v| v.cos()),
        UnaryOp::Tanh => unary(x, layout, |v| v.tanh()),
        UnaryOp::Abs => unary(x, layout, |v| v.abs()),
        UnaryOp::Neg => unary(x, layout, |v| -v),
        UnaryOp::Sqr => unary(x, layout, |v| v.sqr()),
        UnaryOp::Sqrt => unary(x, layout, |v| v.sqrt()),
        UnaryOp::Recip => unary(x, layout, |v| v.recip()),
        UnaryOp::Gelu => unary(x, layout, |v| v.gelu()),
        UnaryOp::GeluErf => unary(x, layout, |v| v.gelu_erf()),
        UnaryOp::Erf => unary(x, layout, |v| v.erf()),
        UnaryOp::Relu => unary(x, layout, |v| v.relu()),
        UnaryOp::Silu => unary(x, layout, |v| v.silu()),
        UnaryOp::Sigmoid => unary(x, layout, |v| v.sigmoid()),
        UnaryOp::Floor => unary(x, layout, |v| v.floor()),
        UnaryOp::Ceil => unary(x, layout, |v| v.ceil()),
        UnaryOp::Round => unary(x, layout, |v| v.round()),
        UnaryOp::Sign => unary(x, layout, |v| v.signum()),
        UnaryOp::LeakyRelu(a) => {
            let a = T::from_f64(a);
            unary(x, layout, |v| v.leaky_relu(a))
        }
        UnaryOp::Pow(e) => {
            let e = T::from_f64(e);
            unary(x, layout, |v| v.powf(e))
        }
        UnaryOp::Affine { mul, add } => {
            let (mul, add) = (T::from_f64(mul), T::from_f64(add));
            unary(x, layout, |v| v * mul + add)
        }
        UnaryOp::Clamp { min, max } => {
            let lo = min.map(T::from_f64);
            let hi = max.map(T::from_f64);
            unary(x, layout, |v| {
                let mut val = v;
                if let Some(lo) = lo {
                    val = T::maximum(val, lo);
                }
                if let Some(hi) = hi {
                    val = T::minimum(val, hi);
                }
                val
            })
        }
    }
}

/// In-place `dst = f(dst, src)` respecting both layouts. Mirrors luma-core
/// `binary_inplace_op`; used by the backward pass to accumulate gradients.
pub fn binary_<T: Copy, F>(dst: &mut [T], dst_l: &Layout, src: &[T], src_l: &Layout, f: F)
where
    F: Fn(T, T) -> T,
{
    match (dst_l.storage_indices(), src_l.storage_indices()) {
        (StorageIndices::Contiguous(di), StorageIndices::Contiguous(si)) => {
            let dst = &mut dst[di.begin_index..di.end_index];
            let src = &src[si.begin_index..si.end_index];
            dst.iter_mut().zip(src.iter()).for_each(|(d, &s)| *d = f(*d, s));
        }
        (StorageIndices::Contiguous(di), StorageIndices::Uncontiguous(si)) => {
            let dst = &mut dst[di.begin_index..di.end_index];
            dst.iter_mut().zip(si).for_each(|(d, si)| *d = f(*d, src[si]));
        }
        (StorageIndices::Uncontiguous(di), StorageIndices::Contiguous(si)) => {
            let src = &src[si.begin_index..si.end_index];
            di.zip(src.iter()).for_each(|(di, &s)| dst[di] = f(dst[di], s));
        }
        (StorageIndices::Uncontiguous(di), StorageIndices::Uncontiguous(si)) => {
            di.zip(si).for_each(|(di, si)| dst[di] = f(dst[di], src[si]));
        }
    }
}
