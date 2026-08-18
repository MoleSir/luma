//! Generic elementwise kernels (binary, binary-scalar, unary, comparison,
//! in-place accumulate). Ported from luma-core's `arith.rs`, generalized over
//! the [`CpuNum`] element trait and returning plain `Vec`s.

use super::element::{CpuFloat, CpuNum};
use crate::{BinaryOp, CmpOp, FloatUnaryOp, Layout, StorageIndices};

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

pub fn num_binary_fn<T: CpuNum>(op: BinaryOp) -> fn(T, T) -> T {
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

/// Dispatch a numeric scalar-binary op (`scalar OP element`) over one view.
pub fn num_scalar_binary<T: CpuNum>(lhs: T, rhs: &[T], rhs_l: &Layout, op: BinaryOp) -> Vec<T> {
    scalar_binary(lhs, rhs, rhs_l, num_binary_fn::<T>(op))
}

/// Dispatch a comparison op over two views, producing a bool buffer.
pub fn num_cmp<T: CpuNum>(lhs: &[T], lhs_l: &Layout, rhs: &[T], rhs_l: &Layout, op: CmpOp) -> Vec<bool> {
    binary(lhs, lhs_l, rhs, rhs_l, cmp_fn::<T>(op))
}

/// Dispatch a float unary op over one view.
pub fn float_unary<T: CpuFloat>(x: &[T], layout: &Layout, op: FloatUnaryOp) -> Vec<T> {
    match op {
        FloatUnaryOp::Exp => unary(x, layout, |v| v.exp()),
        FloatUnaryOp::Ln => unary(x, layout, |v| v.ln()),
        FloatUnaryOp::Sin => unary(x, layout, |v| v.sin()),
        FloatUnaryOp::Cos => unary(x, layout, |v| v.cos()),
        FloatUnaryOp::Tanh => unary(x, layout, |v| v.tanh()),
        FloatUnaryOp::Sqr => unary(x, layout, |v| v.sqr()),
        FloatUnaryOp::Sqrt => unary(x, layout, |v| v.sqrt()),
        FloatUnaryOp::Recip => unary(x, layout, |v| v.recip()),
        FloatUnaryOp::Gelu => unary(x, layout, |v| v.gelu()),
        FloatUnaryOp::GeluErf => unary(x, layout, |v| v.gelu_erf()),
        FloatUnaryOp::Erf => unary(x, layout, |v| v.erf()),
        FloatUnaryOp::Relu => unary(x, layout, |v| v.relu()),
        FloatUnaryOp::Silu => unary(x, layout, |v| v.silu()),
        FloatUnaryOp::Sigmoid => unary(x, layout, |v| v.sigmoid()),
        FloatUnaryOp::Floor => unary(x, layout, |v| v.floor()),
        FloatUnaryOp::Ceil => unary(x, layout, |v| v.ceil()),
        FloatUnaryOp::Round => unary(x, layout, |v| v.round()),
        FloatUnaryOp::LeakyRelu(a) => {
            let a = T::from_f64(a);
            unary(x, layout, |v| v.leaky_relu(a))
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

/// In-place `dst[i] = f(dst[i], rhs)`.
pub fn binary_scalar_<T: Copy, F>(dst: &mut [T], dst_l: &Layout, rhs: T, f: F)
where
    F: Fn(T, T) -> T,
{
    match dst_l.storage_indices() {
        StorageIndices::Contiguous(di) => {
            let dst = &mut dst[di.begin_index..di.end_index];
            dst.iter_mut().for_each(|d| *d = f(*d, rhs));
        }
        StorageIndices::Uncontiguous(di) => {
            di.for_each(|di| dst[di] = f(dst[di], rhs));
        }
    }
}

/// In-place `dst[i] = f(dst[i])`.
pub fn unary_<T: Copy, F>(dst: &mut [T], dst_l: &Layout, f: F)
where
    F: Fn(T) -> T,
{
    match dst_l.storage_indices() {
        StorageIndices::Contiguous(di) => {
            let dst = &mut dst[di.begin_index..di.end_index];
            dst.iter_mut().for_each(|d| *d = f(*d));
        }
        StorageIndices::Uncontiguous(di) => {
            di.for_each(|di| dst[di] = f(dst[di]));
        }
    }
}

/// Out-of-place comparison against scalar: `out[i] = f(lhs[i], rhs)`.
pub fn cmp_scalar<T: CpuNum>(lhs: &[T], lhs_l: &Layout, rhs: T, op: CmpOp) -> Vec<bool> {
    let f = cmp_fn::<T>(op);
    binary_scalar(lhs, lhs_l, rhs, f)
}
