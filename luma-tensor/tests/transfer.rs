//! CPU-only tests for device transfer.
//!
//! `Tensor::to_device` on the same device is a documented no-op, so the
//! bytes copy path (`TransferDTypeKind::transfer`) is exercised directly
//! here; the real `Cpu` <-> `Cuda` path lives in `tests/cuda.rs`.
//! Run with: cargo test --test transfer

mod common;

use common::*;
use luma_tensor::dtype::FloatDType;
use luma_tensor::{Cpu, Error, Float, Int, TransferDTypeKind};

// ---- same-device fast path (via the public API) ---------------------------

#[test]
fn transfer_identity() {
    common::transfer::test_to_device_identity_f32(&Cpu);
    common::transfer::test_to_device_identity_int(&Cpu);
    common::transfer::test_to_device_identity_bool(&Cpu);
    common::transfer::test_to_device_identity_requires_grad(&Cpu);
}

// ---- deep-copy path (transfer() directly, since same-device to_device is a no-op)

#[test]
fn transfer_copy_f32() {
    let src = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
    let dst = Float::transfer(&src, &Cpu).unwrap();

    assert_ne!(dst.id(), src.id(), "copy path must produce a fresh tensor");
    assert_eq!(dst.dtype(), src.dtype());
    assert_eq!(dst.shape(), src.shape());
    assert_close(&dst.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-6, 1e-6);
}

#[test]
fn transfer_copy_preserves_dtype() {
    let src = tensor_f64_dev(&[1.5, 2.5], (2,), &Cpu);
    let dst = Float::transfer(&src, &Cpu).unwrap();
    assert_eq!(dst.dtype(), FloatDType::F64);
    assert_close(&dst.to_vec().unwrap(), &[1.5, 2.5], 1e-12, 1e-12);
}

#[test]
fn transfer_copy_non_contiguous_becomes_contiguous() {
    let src = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
    let view = src.transpose(0usize, 1usize).unwrap();
    assert!(!view.is_contiguous());

    let dst = Float::transfer(&view, &Cpu).unwrap();
    assert!(dst.is_contiguous());
    // logical order of the transposed view
    assert_close(&dst.to_vec().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-6, 1e-6);
}

#[test]
fn transfer_copy_int_bool() {
    let src = tensor_i32(&[1, 2, 3], (3,));
    let dst = Int::transfer(&src, &Cpu).unwrap();
    assert_ne!(dst.id(), src.id());
    assert_eq!(dst.dtype(), src.dtype());
    assert_eq!(dst.to_vec().unwrap(), vec![1, 2, 3]);

    let srcb = tensor_bool_dev(&[true, false, true], (3,), &Cpu);
    let dstb = luma_tensor::Bool::transfer(&srcb, &Cpu).unwrap();
    assert_ne!(dstb.id(), srcb.id());
    assert_eq!(dstb.to_vec().unwrap(), vec![true, false, true]);
}

#[test]
fn transfer_copy_severs_graph_preserves_requires_grad() {
    let x = tensor_f32(&[1.0, 2.0, 3.0], (3,));
    x.set_requires_grad(true);
    let y = x.mul(&x).unwrap(); // non-leaf with an op
    assert!(y.requires_grad());
    assert!(y.op().is_some());

    let moved = Float::transfer(&y, &Cpu).unwrap();
    assert!(moved.requires_grad(), "trainability flag must survive the transfer");
    assert!(moved.op().is_none(), "graph must be severed (single-device Op<D>)");
    assert!(moved.is_leaf());
}

#[test]
fn transfer_copy_grad_flow_works_after_transfer() {
    // A transferred leaf must still accumulate gradients on the target device.
    let x = tensor_f32(&[2.0, 3.0], (2,));
    x.set_requires_grad(true);
    let moved = Float::transfer(&x, &Cpu).unwrap();
    assert!(moved.requires_grad());

    let y = moved.mul(&moved).unwrap();
    let grads = y.backward().unwrap();
    let gx = grads.get_by_id(moved.id()).unwrap();
    assert_close(&gx.to_vec().unwrap(), &[4.0, 6.0], 1e-5, 1e-5);
}

#[test]
fn transfer_meta_errors() {
    let meta = luma_tensor::Tensor::<Cpu>::phantom((2, 3), Cpu::default()).unwrap();
    assert!(meta.is_meta());
    let res = Float::transfer(&meta, &Cpu);
    assert!(matches!(res, Err(Error::MetaTensor)), "meta tensors cannot be copied");
}
