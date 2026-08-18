use super::*;
use luma_tensor::Device;

#[allow(dead_code)]
pub fn test_index_select_dim0(device: &impl Device) {
    let data = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), device);
    let idx = tensor_i32_dev(&[0, 2], (2,), device);
    let result = data.index_select(&idx, 0usize).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    assert_close(&result.to_vec().unwrap(), &[1.0, 2.0, 5.0, 6.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_index_select_dim1(device: &impl Device) {
    let data = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), device);
    let idx = tensor_i32_dev(&[0], (1,), device);
    let result = data.index_select(&idx, 1usize).unwrap();
    assert_eq!(result.dims(), &[3, 1]);
    assert_close(&result.to_vec().unwrap(), &[1.0, 3.0, 5.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_gather_dim1(device: &impl Device) {
    let data = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    let idx = tensor_i32_dev(&[0, 2, 0, 1], (2, 2), device);
    let result = data.gather(&idx, 1usize).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    assert_close(&result.to_vec().unwrap(), &[1.0, 3.0, 4.0, 5.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_index_add_f32(device: &impl Device) {
    let init = tensor_f32_dev(&[0.0, 0.0, 0.0, 0.0], (4,), device);
    let idx = tensor_i32_dev(&[0, 2], (2,), device);
    let src = tensor_f32_dev(&[10.0, 20.0], (2,), device);
    let result = init.index_add(&idx, &src, 0usize).unwrap();
    assert_close(&result.to_vec().unwrap(), &[10.0, 0.0, 20.0, 0.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_scatter_add_f32(device: &impl Device) {
    let init = tensor_f32_dev(&[0.0, 0.0, 0.0, 0.0], (4,), device);
    let idx = tensor_i32_dev(&[1, 3], (2,), device);
    let src = tensor_f32_dev(&[10.0, 20.0], (2,), device);
    let result = init.scatter_add(&idx, &src, 0usize).unwrap();
    assert_close(&result.to_vec().unwrap(), &[0.0, 10.0, 0.0, 20.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_index_add_2d(device: &impl Device) {
    let init = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), device);
    let idx = tensor_i32_dev(&[0, 2], (2,), device);
    let src = tensor_f32_dev(&[10.0, 20.0, 30.0, 40.0], (2, 2), device);
    let result = init.index_add(&idx, &src, 0usize).unwrap();
    assert_close(&result.to_vec().unwrap(), &[11.0, 22.0, 3.0, 4.0, 35.0, 46.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_i_select_row(device: &impl Device) {
    use luma_tensor::IndexOp;
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), device);
    let row = t.i(1usize).unwrap();
    assert_eq!(row.dims(), &[2]);
    assert_close(&row.to_vec().unwrap(), &[3.0, 4.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_i_select_negative(device: &impl Device) {
    use luma_tensor::{D, IndexOp};
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), device);
    let row = t.i(D::Minus1).unwrap();
    assert_close(&row.to_vec().unwrap(), &[5.0, 6.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_i_slice_range(device: &impl Device) {
    use luma_tensor::{IndexOp, s};
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), device);
    let sub = t.i(s!(1:3)).unwrap();
    assert_eq!(sub.dims(), &[2, 2]);
    assert_close(&sub.to_vec().unwrap(), &[3.0, 4.0, 5.0, 6.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_i_slice_full(device: &impl Device) {
    use luma_tensor::{IndexOp, s};
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let sub = t.i(s!(:)).unwrap();
    assert_eq!(sub.dims(), t.dims());
    assert_close(&sub.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_i_slice_with_step(device: &impl Device) {
    use luma_tensor::{IndexOp, s};
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (6,), device);
    let sub = t.i(s!(::2)).unwrap();
    assert_eq!(sub.dims(), &[3]);
    assert_close(&sub.to_vec().unwrap(), &[1.0, 3.0, 5.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_i_tuple_select_slice(device: &impl Device) {
    use luma_tensor::{IndexOp, s};
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], (3, 3), device);
    let sub = t.i((0usize, s!(1:3))).unwrap();
    assert_eq!(sub.dims(), &[2]);
    assert_close(&sub.to_vec().unwrap(), &[2.0, 3.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_i_tuple_slice_slice(device: &impl Device) {
    use luma_tensor::{IndexOp, s};
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], (3, 3), device);
    let sub = t.i((s!(1:3), s!(:2))).unwrap();
    assert_eq!(sub.dims(), &[2, 2]);
    assert_close(&sub.to_vec().unwrap(), &[4.0, 5.0, 7.0, 8.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_i_boolean_mask(device: &impl Device) {
    use luma_tensor::IndexOp;
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (4,), device);
    let mask = tensor_bool_dev(&[true, false, true, false], (4,), device);
    let result = t.i(mask).unwrap();
    assert_eq!(result.dims(), &[2]);
    assert_close(&result.to_vec().unwrap(), &[1.0, 3.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_i_boolean_mask_2d(device: &impl Device) {
    use luma_tensor::IndexOp;
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), device);
    let mask = tensor_bool_dev(&[true, false, true], (3,), device);
    let result = t.i(mask).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    assert_close(&result.to_vec().unwrap(), &[1.0, 2.0, 5.0, 6.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_get_element(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (4,), device);
    let s = t.get(1).unwrap();
    assert_eq!(s.element_count(), 1);
    assert!((s.to_scalar().unwrap() - 2.0).abs() < 1e-5);
}

#[allow(dead_code)]
pub fn test_get_row_2d(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), device);
    let row = t.get(0).unwrap();
    assert_eq!(row.dims(), &[2]);
    assert_close(&row.to_vec().unwrap(), &[1.0, 2.0], 1e-5, 1e-5);
}
