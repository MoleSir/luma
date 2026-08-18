use super::*;
use luma_tensor::Device;
use luma_tensor::IndexOp;
use luma_tensor::dtype::{BoolDType, FloatDType, IntDType};
use luma_tensor::{Bool, Float, Int, Tensor};

#[allow(dead_code)]
pub fn test_zeros_like_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    let z = t.zeros_like().unwrap();
    assert_eq!(z.dims(), t.dims());
    assert!(z.to_vec().unwrap().iter().all(|&x| x == 0.0));
}

#[allow(dead_code)]
pub fn test_ones_like_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[0.0, 0.0, 0.0, 0.0], (2, 2), device);
    let o = t.ones_like().unwrap();
    assert_eq!(o.dims(), t.dims());
    assert!(o.to_vec().unwrap().iter().all(|&x| (x - 1.0).abs() < 1e-5));
}

#[allow(dead_code)]
pub fn test_from_slice_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    assert_eq!(t.dims(), &[2, 3]);
    assert_close(&t.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_rand_like_shape(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let r = t.rand_like(0.0, 1.0).unwrap();
    assert_eq!(r.dims(), t.dims());
}

#[allow(dead_code)]
pub fn test_randn_like_shape(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let r = t.randn_like(0.0, 1.0).unwrap();
    assert_eq!(r.dims(), t.dims());
}

#[allow(dead_code)]
pub fn test_full_scalar(device: &impl Device) {
    let t = tensor_f32_dev(&[42.0], (1,), device);
    let v = t.to_vec().unwrap();
    assert!((v[0] - 42.0).abs() < 1e-5);
}

// ---- from_bytes / to_bytes round-trip tests ----

#[allow(dead_code)]
pub fn test_bytes_roundtrip_f32(device: &impl Device) {
    let data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = tensor_f32_dev(&data, (2, 3), device);
    let bytes = t.to_bytes().unwrap();
    let t2 = <Tensor<_, Float>>::from_bytes(&bytes, (2, 3), (device, FloatDType::F32)).unwrap();
    assert_eq!(t2.dims(), &[2, 3]);
    assert_close(&t2.to_vec().unwrap(), &data, 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_bytes_roundtrip_f64(device: &impl Device) {
    let data = [1.0, 2.0, 3.0, 4.0];
    let t = tensor_f64_dev(&data, (2, 2), device);
    let bytes = t.to_bytes().unwrap();
    let t2: Tensor<_, Float> = <Tensor<_, Float>>::from_bytes(&bytes, (2, 2), (device, FloatDType::F64)).unwrap();
    assert_eq!(t2.dims(), &[2, 2]);
    assert_close(&t2.to_vec().unwrap(), &data, 1e-10, 1e-10);
}

#[allow(dead_code)]
pub fn test_bytes_roundtrip_i32(device: &impl Device) {
    let data = [10i64, 20, 30, 40, 50, 60];
    let t = tensor_i32_dev(&data, (3, 2), device);
    let bytes = t.to_bytes().unwrap();
    let t2 = <Tensor<_, Int>>::from_bytes(&bytes, (3, 2), (device, IntDType::I32)).unwrap();
    assert_eq!(t2.dims(), &[3, 2]);
    assert_eq!(t2.to_vec().unwrap(), data);
}

#[allow(dead_code)]
pub fn test_bytes_roundtrip_u8(device: &impl Device) {
    let data = [1i64, 2, 3, 4];
    let t = tensor_u8_dev(&data, (2, 2), device);
    let bytes = t.to_bytes().unwrap();
    let t2 = <Tensor<_, Int>>::from_bytes(&bytes, (2, 2), (device, IntDType::U8)).unwrap();
    assert_eq!(t2.dims(), &[2, 2]);
    assert_eq!(t2.to_vec().unwrap(), data);
}

#[allow(dead_code)]
pub fn test_bytes_roundtrip_bool(device: &impl Device) {
    let data = [true, false, true, false, false, true];
    let t = tensor_bool_dev(&data, (2, 3), device);
    let bytes = t.to_bytes().unwrap();
    let t2 = <Tensor<_, Bool>>::from_bytes(&bytes, (2, 3), (device, BoolDType::Bool)).unwrap();
    assert_eq!(t2.dims(), &[2, 3]);
    assert_eq!(t2.to_vec().unwrap(), data);
}

#[allow(dead_code)]
pub fn test_bytes_non_contiguous(device: &impl Device) {
    // Slice then to_bytes should still produce logical-order bytes.
    let data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let t = tensor_f32_dev(&data, (3, 3), device);
    let sliced = t.i((.., ..2)).unwrap(); // first 2 columns → non-contiguous
    let bytes = sliced.to_bytes().unwrap();
    // Round-trip: from_bytes with the sliced shape
    let t2 = <Tensor<_, Float>>::from_bytes(&bytes, (3, 2), (device, FloatDType::F32)).unwrap();
    assert_eq!(t2.dims(), &[3, 2]);
    assert_close(&t2.to_vec().unwrap(), &[1.0, 2.0, 4.0, 5.0, 7.0, 8.0], 1e-5, 1e-5);
}

// ---- DynTensor round-trip tests ----

#[allow(dead_code)]
pub fn test_dyn_tensor_roundtrip_float(device: &impl Device) {
    use luma_tensor::DynTensor;
    let data = [1.0f64, 2.0, 3.0, 4.0];
    let t = tensor_f32_dev(&data, (2, 2), device);
    let dt = DynTensor::Float(t);
    assert_eq!(dt.dtype(), luma_tensor::DType::F32);
    let bytes = dt.to_bytes().unwrap();
    let dt2 = DynTensor::from_bytes(&bytes, luma_tensor::DType::F32, (2, 2), device).unwrap();
    let recovered = dt2.into_float().unwrap();
    assert_close(&recovered.to_vec().unwrap(), &data, 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_dyn_tensor_roundtrip_int(device: &impl Device) {
    use luma_tensor::DynTensor;
    let data = [10i64, 20, 30, 40];
    let t = tensor_i32_dev(&data, (2, 2), device);
    let dt = DynTensor::Int(t);
    assert_eq!(dt.dtype(), luma_tensor::DType::I32);
    let bytes = dt.to_bytes().unwrap();
    let dt2 = DynTensor::from_bytes(&bytes, luma_tensor::DType::I32, (2, 2), device).unwrap();
    let recovered = dt2.into_int().unwrap();
    assert_eq!(recovered.to_vec().unwrap(), data);
}

#[allow(dead_code)]
pub fn test_dyn_tensor_roundtrip_bool(device: &impl Device) {
    use luma_tensor::DynTensor;
    let data = [true, false, true, false, false, true];
    let t = tensor_bool_dev(&data, (3, 2), device);
    let dt = DynTensor::Bool(t);
    assert_eq!(dt.dtype(), luma_tensor::DType::Bool);
    let bytes = dt.to_bytes().unwrap();
    let dt2 = DynTensor::from_bytes(&bytes, luma_tensor::DType::Bool, (3, 2), device).unwrap();
    let recovered = dt2.into_bool().unwrap();
    assert_eq!(recovered.to_vec().unwrap(), data);
}

#[allow(dead_code)]
pub fn test_dyn_tensor_accessors(device: &impl Device) {
    use luma_tensor::DynTensor;
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    let dt = DynTensor::Float(t);
    assert_eq!(dt.dims(), &[2, 3]);
    assert!(dt.as_float().is_some());
    assert!(dt.as_int().is_none());
    assert!(dt.as_bool().is_none());
    // device accessor just needs to return something
    let _dev = dt.device();
}
