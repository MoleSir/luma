//! Tests for indexing ops: index_select, gather, index_add, scatter_add, and fancy indexing (.i / s! / Indexer).

mod common;
use common::*;
use luma_tensor::{Cpu, IndexOp, Tensor, s, Slice};

// ---- Index Select ----

#[test]
fn index_select_dim0() {
    let data = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2)); // [[1,2],[3,4],[5,6]]
    let idx = tensor_i32(&[0, 2], (2,)); // select rows 0 and 2
    let result = data.index_select(&idx, 0usize).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    assert_close(&result.to_vec().unwrap(), &[1.0, 2.0, 5.0, 6.0], 1e-7, 1e-7);
}

#[test]
fn index_select_dim1() {
    let data = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2)); // [[1,2],[3,4],[5,6]]
    let idx = tensor_i32(&[0], (1,)); // select column 0
    let result = data.index_select(&idx, 1usize).unwrap();
    assert_eq!(result.dims(), &[3, 1]);
    assert_close(&result.to_vec().unwrap(), &[1.0, 3.0, 5.0], 1e-7, 1e-7);
}

// ---- Gather ----

#[test]
fn gather_dim1() {
    let data = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3)); // [[1,2,3],[4,5,6]]
    let idx = tensor_i32(&[0, 2, 0, 1], (2, 2)); // gather along dim=1
    let result = data.gather(&idx, 1usize).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    // row 0: idxs [0,2] → [1,3]; row 1: idxs [0,1] → [4,5]
    assert_close(&result.to_vec().unwrap(), &[1.0, 3.0, 4.0, 5.0], 1e-7, 1e-7);
}

// ---- Index Add ----

#[test]
fn index_add_f32() {
    let init = tensor_f32(&[0.0, 0.0, 0.0, 0.0], (4,));
    let idx = tensor_i32(&[0, 2], (2,));
    let src = tensor_f32(&[10.0, 20.0], (2,));
    let result = init.index_add(&idx, &src, 0usize).unwrap();
    assert_close(&result.to_vec().unwrap(), &[10.0, 0.0, 20.0, 0.0], 1e-7, 1e-7);
}

#[test]
fn index_add_2d() {
    let init = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2));
    let idx = tensor_i32(&[0, 2], (2,));
    let src = tensor_f32(&[10.0, 20.0, 30.0, 40.0], (2, 2));
    let result = init.index_add(&idx, &src, 0usize).unwrap();
    // add src[0,*] to init[0,*]: [1+10, 2+20] = [11, 22]
    // add src[1,*] to init[2,*]: [5+30, 6+40] = [35, 46]
    assert_close(&result.to_vec().unwrap(), &[11.0, 22.0, 3.0, 4.0, 35.0, 46.0], 1e-7, 1e-7);
}

// ---- Scatter Add ----

#[test]
fn scatter_add_f32() {
    let init = tensor_f32(&[0.0, 0.0, 0.0, 0.0], (4,));
    // scatter: index tensor maps src positions to output positions
    let idx = tensor_i32(&[1, 3], (2,)); // src[0] → out[1], src[1] → out[3]
    let src = tensor_f32(&[10.0, 20.0], (2,));
    let result = init.scatter_add(&idx, &src, 0usize).unwrap();
    assert_close(&result.to_vec().unwrap(), &[0.0, 10.0, 0.0, 20.0], 1e-7, 1e-7);
}

// ---- Fancy indexing: .i() with single index ----

#[test]
fn i_select_row() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2)); // [[1,2],[3,4],[5,6]]
    let row = t.i(1usize).unwrap();
    assert_eq!(row.dims(), &[2]);
    assert_close(&row.to_vec().unwrap(), &[3.0, 4.0], 1e-7, 1e-7);
}

#[test]
fn i_select_negative() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2));
    let row = t.i(luma_tensor::D::Minus1).unwrap(); // last row
    assert_close(&row.to_vec().unwrap(), &[5.0, 6.0], 1e-7, 1e-7);
}

#[test]
fn i_slice_range() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2));
    let sub = t.i(s!(1:3)).unwrap();
    assert_eq!(sub.dims(), &[2, 2]);
    assert_close(&sub.to_vec().unwrap(), &[3.0, 4.0, 5.0, 6.0], 1e-7, 1e-7);
}

#[test]
fn i_slice_full() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (2, 2));
    let sub = t.i(s!(:)).unwrap();
    assert_eq!(sub.dims(), t.dims());
    assert_close(&sub.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0], 1e-7, 1e-7);
}

#[test]
fn i_slice_with_step() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (6,));
    let sub = t.i(s!(::2)).unwrap(); // every 2nd element
    assert_eq!(sub.dims(), &[3]);
    assert_close(&sub.to_vec().unwrap(), &[1.0, 3.0, 5.0], 1e-7, 1e-7);
}

// ---- Fancy indexing: tuple of indexers ----

#[test]
fn i_tuple_select_slice() {
    let t = Tensor::<Cpu>::new(&[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]).unwrap(); // (3, 3)
    // select row 0, then columns 1:3 → shape (2,)
    let sub = t.i((0usize, s!(1:3))).unwrap();
    assert_eq!(sub.dims(), &[2]);
    assert_close(&sub.to_vec().unwrap(), &[2.0, 3.0], 1e-7, 1e-7);
}

#[test]
fn i_tuple_slice_slice() {
    let t = Tensor::<Cpu>::new(&[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]).unwrap(); // (3, 3)
    // rows 1:3, columns :2 → shape (2, 2)
    let sub = t.i((s!(1:3), s!(:2))).unwrap();
    assert_eq!(sub.dims(), &[2, 2]);
    assert_close(&sub.to_vec().unwrap(), &[4.0, 5.0, 7.0, 8.0], 1e-7, 1e-7);
}

// ---- Boolean mask indexing ----

#[test]
fn i_boolean_mask() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (4,));
    let mask = Tensor::<Cpu, luma_tensor::Bool>::new(&[true, false, true, false]).unwrap();
    let result = t.i(mask).unwrap();
    assert_eq!(result.dims(), &[2]);
    assert_close(&result.to_vec().unwrap(), &[1.0, 3.0], 1e-7, 1e-7);
}

#[test]
fn i_boolean_mask_2d() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2)); // [[1,2],[3,4],[5,6]]
    let mask = Tensor::<Cpu, luma_tensor::Bool>::new(&[true, false, true]).unwrap();
    let result = t.i(mask).unwrap(); // select rows 0 and 2
    assert_eq!(result.dims(), &[2, 2]);
    assert_close(&result.to_vec().unwrap(), &[1.0, 2.0, 5.0, 6.0], 1e-7, 1e-7);
}

// ---- get() convenience ----

#[test]
fn get_first_element() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0], (4,));
    let s = t.get(1).unwrap();
    assert_eq!(s.element_count(), 1);
    assert!((s.to_scalar().unwrap() - 2.0).abs() < 1e-7);
}

#[test]
fn get_row_2d() {
    let t = tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2));
    let row = t.get(0).unwrap();
    assert_eq!(row.dims(), &[2]);
    assert_close(&row.to_vec().unwrap(), &[1.0, 2.0], 1e-7, 1e-7);
}

// ---- Slice struct ----

#[test]
fn slice_resolve_none() {
    let s = Slice::new(0, None, 1);
    assert_eq!(s.resolve(10), (0, 10, 1));
}

#[test]
fn slice_resolve_negative() {
    let s = Slice::new(1, Some(-2), 1); // 1..8 for dim_size=10
    assert_eq!(s.resolve(10), (1, 8, 1));
}

#[test]
fn slice_display() {
    let s = Slice::new(1, Some(5), 1);
    assert_eq!(format!("{}", s), "1:5");
    let s2 = Slice::new(0, None, 1);
    assert_eq!(format!("{}", s2), "0:");
}
