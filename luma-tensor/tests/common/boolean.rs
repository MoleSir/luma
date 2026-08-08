use luma_tensor::Device;
use super::*;

#[allow(dead_code)]
pub fn test_bool_and(device: &impl Device) {
    let a = tensor_bool_dev(&[true, true, false, false], (4,), device);
    let b = tensor_bool_dev(&[true, false, true, false], (4,), device);
    let c = a.and(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![true, false, false, false]);
}

#[allow(dead_code)]
pub fn test_bool_or(device: &impl Device) {
    let a = tensor_bool_dev(&[true, true, false, false], (4,), device);
    let b = tensor_bool_dev(&[true, false, true, false], (4,), device);
    let c = a.or(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![true, true, true, false]);
}

#[allow(dead_code)]
pub fn test_bool_xor(device: &impl Device) {
    let a = tensor_bool_dev(&[true, true, false, false], (4,), device);
    let b = tensor_bool_dev(&[true, false, true, false], (4,), device);
    let c = a.xor(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![false, true, true, false]);
}

#[allow(dead_code)]
pub fn test_bool_not(device: &impl Device) {
    let a = tensor_bool_dev(&[true, false, true], (3,), device);
    let c = a.not().unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![false, true, false]);
}

#[allow(dead_code)]
pub fn test_pick_f32(device: &impl Device) {
    let mask = tensor_bool_dev(&[true, false, true, false], (4,), device);
    let on_true = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (4,), device);
    let on_false = tensor_f32_dev(&[10.0, 20.0, 30.0, 40.0], (4,), device);
    let result = mask.pick(&on_true, &on_false).unwrap();
    assert_close(&result.to_vec().unwrap(), &[1.0, 20.0, 3.0, 40.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_bool_all_all(device: &impl Device) {
    let t = tensor_bool_dev(&[true, true, true], (3,), device);
    assert!(t.all_all().unwrap());
    let f = tensor_bool_dev(&[true, false, true], (3,), device);
    assert!(!f.all_all().unwrap());
}

#[allow(dead_code)]
pub fn test_bool_any_all(device: &impl Device) {
    let t = tensor_bool_dev(&[false, false, false], (3,), device);
    assert!(!t.any_all().unwrap());
    let f = tensor_bool_dev(&[false, true, false], (3,), device);
    assert!(f.any_all().unwrap());
}

#[allow(dead_code)]
pub fn test_bool_true_count(device: &impl Device) {
    let t = tensor_bool_dev(&[true, false, true, true, false], (5,), device);
    assert_eq!(t.true_count().unwrap(), 3);
}

#[allow(dead_code)]
pub fn test_pick_scalar_true(device: &impl Device) {
    let mask = tensor_bool_dev(&[true, false, true], (3,), device);
    let on_false = tensor_f32_dev(&[10.0, 20.0, 30.0], (3,), device);
    let result = mask.pick_true(5.0, &on_false).unwrap();
    assert_close(&result.to_vec().unwrap(), &[5.0, 20.0, 5.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_pick_scalar_false(device: &impl Device) {
    let mask = tensor_bool_dev(&[true, false, true], (3,), device);
    let on_true = tensor_f32_dev(&[1.0, 2.0, 3.0], (3,), device);
    let result = mask.pick_false(&on_true, 0.0).unwrap();
    assert_close(&result.to_vec().unwrap(), &[1.0, 0.0, 3.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_allclose_exact(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0], (3,), device);
    let b = tensor_f32_dev(&[1.0, 2.0, 3.0], (3,), device);
    assert!(a.allclose(&b, 1e-5, 1e-5).unwrap());
}

#[allow(dead_code)]
pub fn test_allclose_false(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0], (2,), device);
    let b = tensor_f32_dev(&[1.0, 100.0], (2,), device);
    assert!(!a.allclose(&b, 1e-5, 1e-5).unwrap());
}

#[allow(dead_code)]
pub fn test_allclose_int(device: &impl Device) {
    let a = tensor_i32_dev(&[1, 2, 3], (3,), device);
    let b = tensor_i32_dev(&[1, 2, 3], (3,), device);
    assert!(a.allclose(&b).unwrap());
    let c = tensor_i32_dev(&[1, 2, 4], (3,), device);
    assert!(!a.allclose(&c).unwrap());
}

#[allow(dead_code)]
pub fn test_allclose_bool(device: &impl Device) {
    let a = tensor_bool_dev(&[true, false, true], (3,), device);
    let b = tensor_bool_dev(&[true, false, true], (3,), device);
    assert!(a.allclose(&b).unwrap());
    let c = tensor_bool_dev(&[true, true, true], (3,), device);
    assert!(!a.allclose(&c).unwrap());
}

#[allow(dead_code)]
pub fn test_bool_false_count(device: &impl Device) {
    let t = tensor_bool_dev(&[true, false, true], (3,), device);
    assert_eq!(t.false_count().unwrap(), 1);
}

#[allow(dead_code)]
pub fn test_pick_bool(device: &impl Device) {
    let mask = tensor_bool_dev(&[true, false, true, false], (4,), device);
    let on_true = tensor_bool_dev(&[true, true, false, false], (4,), device);
    let on_false = tensor_bool_dev(&[false, false, true, true], (4,), device);
    let result = mask.pick(&on_true, &on_false).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![true, false, false, true]);
}

#[allow(dead_code)]
pub fn test_pick_int(device: &impl Device) {
    let mask = tensor_bool_dev(&[true, false, true, false], (4,), device);
    let on_true = tensor_i32_dev(&[1, 2, 3, 4], (4,), device);
    let on_false = tensor_i32_dev(&[10, 20, 30, 40], (4,), device);
    let result = mask.pick(&on_true, &on_false).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![1i64, 20, 3, 40]);
}

#[allow(dead_code)]
pub fn test_pick_int_scalar_true(device: &impl Device) {
    let mask = tensor_bool_dev(&[true, false, true], (3,), device);
    let on_false = tensor_i32_dev(&[10, 20, 30], (3,), device);
    let result = mask.pick_true(5i64, &on_false).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![5i64, 20, 5]);
}

#[allow(dead_code)]
pub fn test_pick_int_scalar_false(device: &impl Device) {
    let mask = tensor_bool_dev(&[true, false, true], (3,), device);
    let on_true = tensor_i32_dev(&[1, 2, 3], (3,), device);
    let result = mask.pick_false(&on_true, 0i64).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![1i64, 0, 3]);
}

#[allow(dead_code)]
pub fn test_pick_bool_scalar_true(device: &impl Device) {
    let mask = tensor_bool_dev(&[true, false, true], (3,), device);
    let on_false = tensor_bool_dev(&[true, false, false], (3,), device);
    let result = mask.pick_true(true, &on_false).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![true, false, true]);
}

#[allow(dead_code)]
pub fn test_pick_bool_scalar_false(device: &impl Device) {
    let mask = tensor_bool_dev(&[true, false, true], (3,), device);
    let on_true = tensor_bool_dev(&[true, true, true], (3,), device);
    let result = mask.pick_false(&on_true, false).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![true, false, true]);
}
