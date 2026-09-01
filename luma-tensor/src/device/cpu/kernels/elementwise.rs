//! Generic elementwise kernels (binary, binary-scalar, unary, comparison,
//! in-place accumulate). Ported from luma-core's `arith.rs`, generalized over
//! the [`CpuNum`] element trait.
//!
//! Allocating kernels take a [`Cpu`] and route their output buffer through its
//! allocator (`device.collect_alloc`) — the single interception point for a
//! pooling allocator. The `*_` in-place variants write into a caller-provided
//! `&mut [T]` and never allocate.

use super::element::{CpuFloat, CpuNum};
use crate::Cpu;
use crate::device::cpu::allocator::AllocVec;
use crate::{BinaryOp, CmpOp, FloatUnaryOp, Layout, StorageIndices};

/// `out[i] = f(lhs[i], rhs[i])` over two (possibly non-contiguous) views of the
/// same logical shape. Mirrors luma-core `compute_binary_op`.
pub fn binary<T: Copy, U: AllocVec, F>(lhs: &[T], lhs_l: &Layout, rhs: &[T], rhs_l: &Layout, f: F, device: &Cpu) -> Vec<U>
where
    F: Fn(T, T) -> U,
{
    match (lhs_l.storage_indices(), rhs_l.storage_indices()) {
        (StorageIndices::Contiguous(li), StorageIndices::Contiguous(ri)) => {
            let lhs = &lhs[li.begin_index..li.end_index];
            let rhs = &rhs[ri.begin_index..ri.end_index];
            device.collect_alloc(lhs.iter().zip(rhs.iter()).map(|(&l, &r)| f(l, r)))
        }
        (StorageIndices::Contiguous(li), StorageIndices::Uncontiguous(ri)) => {
            let lhs = &lhs[li.begin_index..li.end_index];
            device.collect_alloc(lhs.iter().zip(ri).map(|(&l, ri)| f(l, rhs[ri])))
        }
        (StorageIndices::Uncontiguous(li), StorageIndices::Contiguous(ri)) => {
            let rhs = &rhs[ri.begin_index..ri.end_index];
            device.collect_alloc(li.zip(rhs.iter()).map(|(li, &r)| f(lhs[li], r)))
        }
        (StorageIndices::Uncontiguous(li), StorageIndices::Uncontiguous(ri)) => {
            device.collect_alloc(li.zip(ri).map(|(li, ri)| f(lhs[li], rhs[ri])))
        }
    }
}

/// `out[i] = f(lhs[i], scalar)`.
pub fn binary_scalar<T: Copy, U: AllocVec, F>(lhs: &[T], lhs_l: &Layout, rhs: T, f: F, device: &Cpu) -> Vec<U>
where
    F: Fn(T, T) -> U,
{
    match lhs_l.storage_indices() {
        StorageIndices::Contiguous(i) => {
            let lhs = &lhs[i.begin_index..i.end_index];
            device.collect_alloc(lhs.iter().map(|&v| f(v, rhs)))
        }
        StorageIndices::Uncontiguous(i) => device.collect_alloc(i.map(|i| f(lhs[i], rhs))),
    }
}

/// `out[i] = f(scalar, rhs[i])`.
pub fn scalar_binary<T: Copy, U: AllocVec, F>(lhs: T, rhs: &[T], rhs_l: &Layout, f: F, device: &Cpu) -> Vec<U>
where
    F: Fn(T, T) -> U,
{
    match rhs_l.storage_indices() {
        StorageIndices::Contiguous(i) => {
            let rhs = &rhs[i.begin_index..i.end_index];
            device.collect_alloc(rhs.iter().map(|&v| f(lhs, v)))
        }
        StorageIndices::Uncontiguous(i) => device.collect_alloc(i.map(|i| f(lhs, rhs[i]))),
    }
}

/// `out[i] = f(x[i])`.
pub fn unary<T: Copy, U: AllocVec, F>(x: &[T], layout: &Layout, f: F, device: &Cpu) -> Vec<U>
where
    F: Fn(T) -> U,
{
    match layout.storage_indices() {
        StorageIndices::Contiguous(i) => {
            let x = &x[i.begin_index..i.end_index];
            device.collect_alloc(x.iter().map(|&v| f(v)))
        }
        StorageIndices::Uncontiguous(i) => device.collect_alloc(i.map(|i| f(x[i]))),
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
pub fn num_binary<T: CpuNum>(lhs: &[T], lhs_l: &Layout, rhs: &[T], rhs_l: &Layout, op: BinaryOp, device: &Cpu) -> Vec<T> {
    binary(lhs, lhs_l, rhs, rhs_l, num_binary_fn::<T>(op), device)
}

/// Dispatch a numeric binary-scalar op over one view.
pub fn num_binary_scalar<T: CpuNum>(lhs: &[T], lhs_l: &Layout, rhs: T, op: BinaryOp, device: &Cpu) -> Vec<T> {
    binary_scalar(lhs, lhs_l, rhs, num_binary_fn::<T>(op), device)
}

/// Dispatch a numeric scalar-binary op (`scalar OP element`) over one view.
pub fn num_scalar_binary<T: CpuNum>(lhs: T, rhs: &[T], rhs_l: &Layout, op: BinaryOp, device: &Cpu) -> Vec<T> {
    scalar_binary(lhs, rhs, rhs_l, num_binary_fn::<T>(op), device)
}

/// Dispatch a comparison op over two views, producing a bool buffer.
pub fn num_cmp<T: CpuNum>(lhs: &[T], lhs_l: &Layout, rhs: &[T], rhs_l: &Layout, op: CmpOp, device: &Cpu) -> Vec<bool> {
    binary(lhs, lhs_l, rhs, rhs_l, cmp_fn::<T>(op), device)
}

/// Dispatch a float unary op over one view.
pub fn float_unary<T: CpuFloat>(x: &[T], layout: &Layout, op: FloatUnaryOp, device: &Cpu) -> Vec<T> {
    match op {
        FloatUnaryOp::Exp => unary(x, layout, |v| v.exp(), device),
        FloatUnaryOp::Ln => unary(x, layout, |v| v.ln(), device),
        FloatUnaryOp::Sin => unary(x, layout, |v| v.sin(), device),
        FloatUnaryOp::Cos => unary(x, layout, |v| v.cos(), device),
        FloatUnaryOp::Tanh => unary(x, layout, |v| v.tanh(), device),
        FloatUnaryOp::Sqr => unary(x, layout, |v| v.sqr(), device),
        FloatUnaryOp::Sqrt => unary(x, layout, |v| v.sqrt(), device),
        FloatUnaryOp::Recip => unary(x, layout, |v| v.recip(), device),
        FloatUnaryOp::Gelu => unary(x, layout, |v| v.gelu(), device),
        FloatUnaryOp::GeluErf => unary(x, layout, |v| v.gelu_erf(), device),
        FloatUnaryOp::Erf => unary(x, layout, |v| v.erf(), device),
        FloatUnaryOp::Relu => unary(x, layout, |v| v.relu(), device),
        FloatUnaryOp::Silu => unary(x, layout, |v| v.silu(), device),
        FloatUnaryOp::Sigmoid => unary(x, layout, |v| v.sigmoid(), device),
        FloatUnaryOp::Floor => unary(x, layout, |v| v.floor(), device),
        FloatUnaryOp::Ceil => unary(x, layout, |v| v.ceil(), device),
        FloatUnaryOp::Round => unary(x, layout, |v| v.round(), device),
        FloatUnaryOp::LeakyRelu(a) => {
            let a = T::from_f64(a);
            unary(x, layout, |v| v.leaky_relu(a), device)
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
pub fn cmp_scalar<T: CpuNum>(lhs: &[T], lhs_l: &Layout, rhs: T, op: CmpOp, device: &Cpu) -> Vec<bool> {
    let f = cmp_fn::<T>(op);
    binary_scalar(lhs, lhs_l, rhs, f, device)
}
