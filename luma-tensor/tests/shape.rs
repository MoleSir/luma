//! Tests for shape ops: reshape, transpose, contiguous, cat, broadcast_as.

mod common;
use common::*;
use luma_tensor::{Cpu, Tensor};

// ---- Reshape ----

#[test]
fn reshape_f32() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
    let r = t.reshape((3, 2)).unwrap();
    assert_eq!(r.dims(), &[3, 2]);
    let v = r.to_vec().unwrap();
    assert_close(&v, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-7, 1e-7);
}

#[test]
fn reshape_element_count_mismatch() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (2, 2));
    assert!(t.reshape((3, 2)).is_err());
}

// ---- Transpose ----

#[test]
fn transpose_f32() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
    let tr = t.transpose(0usize, 1usize).unwrap();
    assert_eq!(tr.dims(), &[3, 2]);
    let v = tr.to_vec().unwrap();
    // Row-major transpose: row0 [1,2,3] row1 [4,5,6] -> row0 [1,4] row1 [2,5] row2 [3,6]
    assert_close(&v, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-7, 1e-7);
}

#[test]
fn transpose_last() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
    let tr = t.transpose_last().unwrap(); // dims[-2], dims[-1] = (0,1)
    assert_eq!(tr.dims(), &[3, 2]);
}

// ---- Contiguous ----

#[test]
fn contiguous_after_transpose() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
    let tr = t.transpose(0usize, 1usize).unwrap();
    assert!(!tr.is_contiguous());
    let c = tr.contiguous().unwrap();
    assert!(c.is_contiguous());
    assert_close(&tr.to_vec().unwrap(), &c.to_vec().unwrap(), 1e-7, 1e-7);
}

#[test]
fn already_contiguous_is_noop() {
    let t = tensor_f32(&[1.0, 2.0, 3.0], (3,));
    assert!(t.is_contiguous());
    // contiguous() on a contiguous tensor should return a clone (same data)
    let c = t.contiguous().unwrap();
    assert_eq!(t.to_vec().unwrap(), c.to_vec().unwrap());
}

// ---- Cat ----

#[test]
fn cat_dim0_f32() {
    let a = tensor_f32(&[1.0, 2.0, 3.0], (1, 3));
    let b = tensor_f32(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0], (2, 3));
    let c = Tensor::<Cpu>::cat(&[&a, &b], 0usize).unwrap();
    assert_eq!(c.dims(), &[3, 3]);
    assert_close(
        &c.to_vec().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        1e-7,
        1e-7,
    );
}

#[test]
fn cat_empty() {
    let arrs: &[&Tensor<Cpu>] = &[];
    assert!(Tensor::<Cpu>::cat(arrs, 0usize).is_err());
}

// ---- Broadcast ----

#[test]
fn broadcast_as_f32() {
    let t = tensor_f32(&[1.0, 2.0, 3.0], (1, 3));
    let b = t.broadcast_as((2, 3)).unwrap();
    assert_eq!(b.dims(), &[2, 3]);
    assert_close(
        &b.to_vec().unwrap(),
        &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        1e-7,
        1e-7,
    );
}

#[test]
fn broadcast_incompatible() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
    assert!(t.broadcast_as((3, 2)).is_err());
}

// ---- Narrow ----

#[test]
fn narrow_dim0() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
    let n = t.narrow(0usize, 0, 1).unwrap();
    assert_eq!(n.dims(), &[1, 3]);
    assert_close(&n.to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-7, 1e-7);
}

// ---- Squeeze / Unsqueeze ----

#[test]
fn squeeze_dim1() {
    let t = tensor_f32(&[1.0, 2.0, 3.0], (1, 3)); // (1,3)
    let s = t.squeeze(0usize).unwrap(); // -> (3,)
    assert_eq!(s.dims(), &[3]);
    assert_close(&s.to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-7, 1e-7);
}

#[test]
fn unsqueeze() {
    let t = tensor_f32(&[1.0, 2.0, 3.0], (3,)); // (3,)
    let u = t.unsqueeze(0usize).unwrap(); // -> (1,3)
    assert_eq!(u.dims(), &[1, 3]);
    assert_close(&u.to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-7, 1e-7);
}

// ---- Flatten ----

#[test]
fn flatten_all() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (2, 2));
    let f = t.flatten_all().unwrap();
    assert_eq!(f.dims(), &[4]);
    assert_close(&f.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0], 1e-7, 1e-7);
}

#[test]
fn flatten_range() {
    let t = tensor_f32(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        (3, 2, 2),
    );
    let f = t.flatten(1usize, 2usize).unwrap(); // flatten dims 1..=2
    assert_eq!(f.dims(), &[3, 4]);
}

// ---- Permute ----

#[test]
fn permute_f32() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
    let p = t.permute([1usize, 0]).unwrap();
    assert_eq!(p.dims(), &[3, 2]);
    assert_close(&p.to_vec().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-7, 1e-7);
}

// ---- Stack / Split / Chunk / Repeat ----

#[test]
fn stack_f32() {
    let a = tensor_f32(&[1.0, 2.0, 3.0], (3,));
    let b = tensor_f32(&[4.0, 5.0, 6.0], (3,));
    let s = Tensor::<Cpu>::stack(&[&a, &b], 0usize).unwrap();
    assert_eq!(s.dims(), &[2, 3]);
    assert_close(&s.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-7, 1e-7);
}

#[test]
fn split_f32() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (4,));
    let parts = t.split(0usize).unwrap();
    assert_eq!(parts.len(), 4);
    assert_close(&parts[0].to_vec().unwrap(), &[1.0], 1e-7, 1e-7);
    assert_close(&parts[3].to_vec().unwrap(), &[4.0], 1e-7, 1e-7);
}

#[test]
fn chunk_f32() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0], (5,));
    let chunks = t.chunk(2, 0usize).unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].dims(), &[3]); // ceil(5/2) = 3
    assert_eq!(chunks[1].dims(), &[2]);
}

#[test]
fn repeat_dim_f32() {
    let t = tensor_f32(&[1.0, 2.0], (1, 2));
    let r = t.repeat_dim(0usize, 3).unwrap();
    assert_eq!(r.dims(), &[3, 2]);
    assert_close(&r.to_vec().unwrap(), &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0], 1e-7, 1e-7);
}
