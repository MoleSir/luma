use super::*;
use luma_tensor::Device;

#[allow(dead_code)]
pub fn test_cat_dim0_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0], (1, 3), device);
    let b = tensor_f32_dev(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0], (2, 3), device);
    let c = Tensor::cat(&[&a, &b], 0usize).unwrap();
    assert_eq!(c.dims(), &[3, 3]);
    assert_close(&c.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_contiguous_after_transpose(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    let tr = t.transpose(0usize, 1usize).unwrap();
    assert!(!tr.is_contiguous());
    let c = tr.contiguous().unwrap();
    assert!(c.is_contiguous());
    assert_close(&tr.to_vec().unwrap(), &c.to_vec().unwrap(), 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_reshape_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    let r = t.reshape((3, 2)).unwrap();
    assert_eq!(r.dims(), &[3, 2]);
    assert_close(&r.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_transpose_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    let tr = t.transpose(0usize, 1usize).unwrap();
    assert_eq!(tr.dims(), &[3, 2]);
    assert_close(&tr.to_vec().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_cat_empty() {
    let arrs: &[&luma_tensor::Tensor<luma_tensor::Cpu>] = &[];
    assert!(luma_tensor::Tensor::<luma_tensor::Cpu>::cat(arrs, 0usize).is_err());
}

#[allow(dead_code)]
pub fn test_broadcast_as_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0], (1, 3), device);
    let b = t.broadcast_as((2, 3)).unwrap();
    assert_eq!(b.dims(), &[2, 3]);
    assert_close(&b.to_vec().unwrap(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_narrow_dim0(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    let n = t.narrow(0usize, 0, 1).unwrap();
    assert_eq!(n.dims(), &[1, 3]);
    assert_close(&n.to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_squeeze_dim1(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0], (1, 3), device);
    let s = t.squeeze(0usize).unwrap();
    assert_eq!(s.dims(), &[3]);
    assert_close(&s.to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_unsqueeze(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0], (3,), device);
    let u = t.unsqueeze(0usize).unwrap();
    assert_eq!(u.dims(), &[1, 3]);
    assert_close(&u.to_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_flatten_all(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let f = t.flatten_all().unwrap();
    assert_eq!(f.dims(), &[4]);
    assert_close(&f.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_permute_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    let p = t.permute([1usize, 0]).unwrap();
    assert_eq!(p.dims(), &[3, 2]);
    assert_close(&p.to_vec().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_split_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (4,), device);
    let parts = t.split(0usize).unwrap();
    assert_eq!(parts.len(), 4);
    assert_close(&parts[0].to_vec().unwrap(), &[1.0], 1e-5, 1e-5);
    assert_close(&parts[3].to_vec().unwrap(), &[4.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_repeat_dim_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0], (1, 2), device);
    let r = t.repeat_dim(0usize, 3).unwrap();
    assert_eq!(r.dims(), &[3, 2]);
    assert_close(&r.to_vec().unwrap(), &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_transpose_last(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    let tr = t.transpose_last().unwrap();
    assert_eq!(tr.dims(), &[3, 2]);
}

#[allow(dead_code)]
pub fn test_already_contiguous_is_noop(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0], (3,), device);
    assert!(t.is_contiguous());
    let c = t.contiguous().unwrap();
    assert_eq!(t.to_vec().unwrap(), c.to_vec().unwrap());
}

#[allow(dead_code)]
pub fn test_flatten_range(device: &impl Device) {
    let vals: Vec<f64> = (1..=12).map(|x| x as f64).collect();
    let t = tensor_f32_dev(&vals, (3, 2, 2), device);
    let f = t.flatten(1usize, 2usize).unwrap();
    assert_eq!(f.dims(), &[3, 4]);
}

#[allow(dead_code)]
pub fn test_stack_f32(device: &impl Device) {
    let a = tensor_f32_dev(&[1.0, 2.0, 3.0], (3,), device);
    let b = tensor_f32_dev(&[4.0, 5.0, 6.0], (3,), device);
    let s = luma_tensor::Tensor::stack(&[&a, &b], 0usize).unwrap();
    assert_eq!(s.dims(), &[2, 3]);
    assert_close(&s.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_chunk_f32(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0], (5,), device);
    let chunks = t.chunk(2, 0usize).unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].dims(), &[3]);
    assert_eq!(chunks[1].dims(), &[2]);
}

// ---- copy_ / phantom ----

#[allow(dead_code)]
pub fn test_copy_float(device: &impl Device) {
    let mut t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let id_before = t.id();
    let src = tensor_f32_dev(&[5.0, 6.0, 7.0, 8.0], (2, 2), device);
    t.copy_(&src).unwrap();
    // data from src
    assert_close(&t.to_vec().unwrap(), &[5.0, 6.0, 7.0, 8.0], 1e-5, 1e-5);
    // id preserved
    assert_eq!(t.id(), id_before);
    // layout is now contiguous
    assert!(t.is_contiguous());
}

#[allow(dead_code)]
pub fn test_copy_shape_mismatch(device: &impl Device) {
    let mut t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let src = tensor_f32_dev(&[5.0, 6.0], (1, 2), device);
    assert!(t.copy_(&src).is_err());
}

#[allow(dead_code)]
pub fn test_phantom(device: &impl Device) {
    use luma_tensor::Float;
    use luma_tensor::dtype::FloatDType;
    let p = luma_tensor::Tensor::<_, Float>::phantom((2, 3), (device, FloatDType::F32)).unwrap();
    assert_eq!(p.dims(), &[2, 3]);
    assert!(p.is_meta()); // no storage allocated
}

#[allow(dead_code)]
pub fn test_phantom_then_copy(device: &impl Device) {
    use luma_tensor::Float;
    use luma_tensor::dtype::FloatDType;
    let mut p = luma_tensor::Tensor::<_, Float>::phantom((2, 2), (device, FloatDType::F32)).unwrap();
    assert!(p.is_meta());
    let src = tensor_f32_dev(&[9.0, 8.0, 7.0, 6.0], (2, 2), device);
    p.copy_(&src).unwrap();
    assert!(!p.is_meta()); // storage allocated by copy_
    assert_close(&p.to_vec().unwrap(), &[9.0, 8.0, 7.0, 6.0], 1e-5, 1e-5);
}

#[allow(dead_code)]
pub fn test_copy_preserves_requires_grad(device: &impl Device) {
    let mut t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (2, 2), device);
    let requires_grad_before = t.requires_grad();
    let src = tensor_f32_dev(&[5.0, 6.0, 7.0, 8.0], (2, 2), device);
    t.copy_(&src).unwrap();
    // copy_ should NOT change requires_grad (it just replaces data)
    assert_eq!(t.requires_grad(), requires_grad_before);
}
