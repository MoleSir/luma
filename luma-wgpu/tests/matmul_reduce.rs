//! End-to-end matmul/reduce checks against the shared device-generic suite.
//!
//! F32/F16 matmul and the f32/f16 reduce/arg-reduce ops are covered here.

use luma_cpu::Cpu;
use luma_tensor::tests::{assert_close, cross, grad, half_precision, matmul, nn, reduce};
use luma_tensor::{FloatDType, FloatOps, Layout, Shape, Tensor};
use luma_wgpu::Wgpu;

use std::sync::LazyLock;

static DEVICE: LazyLock<Option<Wgpu>> = LazyLock::new(|| match Wgpu::new(0) {
    Ok(device) => Some(device),
    Err(e) => {
        eprintln!("skip wgpu test: {e}");
        None
    }
});

fn device() -> Option<Wgpu> {
    DEVICE.clone()
}

#[test]
fn matmul_f32() {
    let Some(dev) = device() else { return };
    matmul::test_matmul_2x2(&dev);
    matmul::test_matmul_2x3_3x2(&dev);
    matmul::test_large_matmul_f32(&dev);
    nn::test_matmul_transposed_weight_small_f32(&dev);
    nn::test_matmul_add_bias_f32(&dev);
}

#[test]
fn matmul_backward_f32() {
    let Some(dev) = device() else { return };
    grad::test_grad_matmul(&dev);
}

#[test]
fn add_matmul_f32() {
    let Some(dev) = device() else { return };

    let lhs_l = Layout::contiguous(Shape::from((2, 3)));
    let rhs_l = Layout::contiguous(Shape::from((3, 2)));
    let lhs = Wgpu::f_from_f32(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &dev).unwrap();
    let rhs = Wgpu::f_from_f32(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &dev).unwrap();

    let dst_l = Layout::contiguous(Shape::from((2, 2)));
    let mut dst = Wgpu::f_from_f32(vec![1.0f32, 2.0, 3.0, 4.0], &dev).unwrap();
    Wgpu::f_add_matmul_(&mut dst, &dst_l, &lhs, &lhs_l, &rhs, &rhs_l).unwrap();

    let got = Wgpu::f_to_vec(&dst, &dst_l).unwrap();
    assert_close(&got, &[23.0, 30.0, 52.0, 68.0], 1e-5, 1e-5);
}

#[test]
fn matmul_batched_f32() {
    let Some(dev) = device() else { return };
    let cpu = Cpu::default();

    let a_data: Vec<f64> = (1..=12).map(f64::from).collect();
    let b_data: Vec<f64> = (1..=12).map(f64::from).collect();

    let a = Tensor::<Wgpu>::from_slice(&a_data, (2, 2, 3), (&dev, FloatDType::F32)).unwrap();
    let b = Tensor::<Wgpu>::from_slice(&b_data, (2, 3, 2), (&dev, FloatDType::F32)).unwrap();
    let ca = Tensor::<Cpu>::from_slice(&a_data, (2, 2, 3), (&cpu, FloatDType::F32)).unwrap();
    let cb = Tensor::<Cpu>::from_slice(&b_data, (2, 3, 2), (&cpu, FloatDType::F32)).unwrap();

    let got = a.matmul(&b).unwrap();
    let want = ca.matmul(&cb).unwrap();
    assert_eq!(got.dims(), &[2, 2, 2]);
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-5);
}

#[test]
fn matmul_equal_batch_product_f32() {
    let Some(dev) = device() else { return };
    let cpu = Cpu::default();

    let a_data: Vec<f64> = (1..=36).map(f64::from).collect();
    let b_data: Vec<f64> = (1..=36).map(f64::from).collect();

    // Different batch shapes, equal batch product: 2*3 == 1*6.
    let a = Tensor::<Wgpu>::from_slice(&a_data, (2, 3, 2, 3), (&dev, FloatDType::F32)).unwrap();
    let b = Tensor::<Wgpu>::from_slice(&b_data, (1, 6, 3, 2), (&dev, FloatDType::F32)).unwrap();
    let ca = Tensor::<Cpu>::from_slice(&a_data, (2, 3, 2, 3), (&cpu, FloatDType::F32)).unwrap();
    let cb = Tensor::<Cpu>::from_slice(&b_data, (1, 6, 3, 2), (&cpu, FloatDType::F32)).unwrap();

    let got = a.matmul(&b).unwrap();
    let want = ca.matmul(&cb).unwrap();
    assert_eq!(got.dims(), &[2, 3, 2, 2]);
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-5);
}

#[test]
fn matmul_non_contiguous_lhs_f32() {
    let Some(dev) = device() else { return };
    let cpu = Cpu::default();

    let a_data: Vec<f64> = (1..=6).map(f64::from).collect();
    let b_data: Vec<f64> = (1..=12).map(f64::from).collect();

    let a = Tensor::<Wgpu>::from_slice(&a_data, (3, 2), (&dev, FloatDType::F32)).unwrap().transpose(0, 1).unwrap();
    let b = Tensor::<Wgpu>::from_slice(&b_data, (3, 4), (&dev, FloatDType::F32)).unwrap();
    let ca = Tensor::<Cpu>::from_slice(&a_data, (3, 2), (&cpu, FloatDType::F32)).unwrap().transpose(0, 1).unwrap();
    let cb = Tensor::<Cpu>::from_slice(&b_data, (3, 4), (&cpu, FloatDType::F32)).unwrap();

    let got = a.matmul(&b).unwrap();
    let want = ca.matmul(&cb).unwrap();
    assert_eq!(got.dims(), &[2, 4]);
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-5);
}

#[test]
fn matmul_f16() {
    let Some(dev) = device() else { return };
    if !dev.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }
    half_precision::test_half_matmul(&dev, FloatDType::F16, 0.5);
}

#[test]
fn sum_f32() {
    let Some(dev) = device() else { return };
    reduce::test_sum_dim_f32(&dev);
    reduce::test_sum_keepdim_f32(&dev);
    reduce::test_sum_all_f32(&dev);
    reduce::test_sum_dims_f32(&dev);
    cross::test_transpose_sum(&dev);
    cross::test_slice_sum(&dev);
    cross::test_broadcast_sum(&dev);
    reduce::test_large_sum_f32(&dev);
}

#[test]
fn min_max_mean_prod_f32() {
    let Some(dev) = device() else { return };
    reduce::test_max_dim_f32(&dev);
    reduce::test_max_keepdim_f32(&dev);
    reduce::test_max_all_f32(&dev);
    reduce::test_min_dim_f32(&dev);
    reduce::test_min_all_f32(&dev);
    reduce::test_mean_dim_f32(&dev);
    reduce::test_mean_all_f32(&dev);
    reduce::test_prod_dim_f32(&dev);
    reduce::test_prod_all_f32(&dev);
    cross::test_transpose_max(&dev);
}

#[test]
fn reduce_f16_ops() {
    let Some(dev) = device() else { return };
    if !dev.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }

    let m = Tensor::<Wgpu>::from_slice(&[1.0f64, 2.0, 3.0, 4.0], (2, 2), (&dev, FloatDType::F16)).unwrap();
    assert_close(&m.sum(0usize).unwrap().to_vec().unwrap(), &[4.0, 6.0], 0.05, 0.05);
    assert_close(&m.sum_keepdim(1usize).unwrap().to_vec().unwrap(), &[3.0, 7.0], 0.05, 0.05);
    assert_close(&m.sum_all().unwrap().to_vec().unwrap(), &[10.0], 0.05, 0.05);
    assert_close(&m.min(1usize).unwrap().to_vec().unwrap(), &[1.0, 3.0], 0.05, 0.05);
    assert_close(&m.max(1usize).unwrap().to_vec().unwrap(), &[2.0, 4.0], 0.05, 0.05);
    assert_close(&m.mean_all().unwrap().to_vec().unwrap(), &[2.5], 0.05, 0.05);
    assert_close(&m.prod_all().unwrap().to_vec().unwrap(), &[24.0], 0.05, 0.05);
}

#[test]
fn reduce_f32() {
    let Some(dev) = device() else { return };
    reduce::test_sum_dim_f32(&dev);
    reduce::test_sum_keepdim_f32(&dev);
    reduce::test_sum_all_f32(&dev);
    reduce::test_sum_dims_f32(&dev);
    reduce::test_max_dim_f32(&dev);
    reduce::test_max_keepdim_f32(&dev);
    reduce::test_max_all_f32(&dev);
    reduce::test_min_dim_f32(&dev);
    reduce::test_min_all_f32(&dev);
    reduce::test_mean_dim_f32(&dev);
    reduce::test_mean_all_f32(&dev);
    reduce::test_prod_dim_f32(&dev);
    reduce::test_prod_all_f32(&dev);
    reduce::test_argmax_f32(&dev);
    reduce::test_argmin_f32(&dev);
    reduce::test_argmax_keepdim(&dev);
    cross::test_transpose_sum(&dev);
    cross::test_transpose_max(&dev);
    cross::test_slice_sum(&dev);
    cross::test_broadcast_sum(&dev);
}

#[test]
fn arg_reduce_f32() {
    let Some(dev) = device() else { return };

    reduce::test_argmax_f32(&dev);
    reduce::test_argmin_f32(&dev);
    reduce::test_argmax_keepdim(&dev);

    let x = Tensor::<Wgpu>::from_slice(&[3.0f64, 1.0, 3.0, 2.0, 3.0, 0.0], (2, 3), (&dev, FloatDType::F32)).unwrap();
    assert_eq!(x.argmax(1usize).unwrap().to_vec().unwrap(), vec![0, 1]);
    assert_eq!(x.argmin(1usize).unwrap().to_vec().unwrap(), vec![1, 2]);

    let t = x.transpose(0, 1).unwrap();
    assert_eq!(t.argmax(0usize).unwrap().to_vec().unwrap(), vec![0, 1]);

    let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let big = Tensor::<Wgpu>::from_slice(&data, (1000,), (&dev, FloatDType::F32)).unwrap();
    assert_eq!(big.argmax(0usize).unwrap().to_vec().unwrap(), vec![999]);
    assert_eq!(big.argmin(0usize).unwrap().to_vec().unwrap(), vec![0]);
}

#[test]
fn arg_reduce_f16() {
    let Some(dev) = device() else { return };
    if !dev.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }

    let x = Tensor::<Wgpu>::from_slice(&[1.0f64, 5.0, 3.0, 2.0, 4.0, 6.0], (2, 3), (&dev, FloatDType::F16)).unwrap();
    assert_eq!(x.argmax(1usize).unwrap().to_vec().unwrap(), vec![1, 2]);
    assert_eq!(x.argmin(1usize).unwrap().to_vec().unwrap(), vec![0, 0]);

    let ties = Tensor::<Wgpu>::from_slice(&[3.0f64, 1.0, 3.0, 2.0, 3.0, 0.0], (2, 3), (&dev, FloatDType::F16)).unwrap();
    assert_eq!(ties.argmax(1usize).unwrap().to_vec().unwrap(), vec![0, 1]);
    assert_eq!(ties.argmin(1usize).unwrap().to_vec().unwrap(), vec![1, 2]);
}

#[test]
fn reduce_f16() {
    let Some(dev) = device() else { return };
    if !dev.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }
    half_precision::test_half_reduce(&dev, FloatDType::F16, 0.05);
}
