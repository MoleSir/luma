use luma_cpu::Cpu;
use luma_tensor::tests::{assert_close, cross, half_precision, numeric};
use luma_tensor::{FloatDType, Tensor};
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
fn binary_strided_f32() {
    let Some(dev) = device() else { return };
    cross::test_transpose_add(&dev);
    cross::test_transpose_sub(&dev);
    cross::test_permute_add(&dev);
    numeric::test_broadcast_add_f32(&dev);
    numeric::test_broadcast_mul_f32(&dev);
    numeric::test_broadcast_eq_f32(&dev);
}

#[test]
fn unary_strided_f32() {
    let Some(dev) = device() else { return };
    let cpu = Cpu::default();

    let data: Vec<f64> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let x = Tensor::<Wgpu>::from_slice(&data, (3, 2), (&dev, FloatDType::F32)).unwrap().transpose(0, 1).unwrap();
    let cx = Tensor::<Cpu>::from_slice(&data, (3, 2), (&cpu, FloatDType::F32)).unwrap().transpose(0, 1).unwrap();

    let got = x.neg().unwrap();
    let want = cx.neg().unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-5);

    let got = x.relu().unwrap();
    let want = cx.relu().unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-5);
}

#[test]
fn cmp_strided_f32() {
    let Some(dev) = device() else { return };
    let cpu = Cpu::default();

    let a_data: Vec<f64> = vec![1.0, 5.0, 3.0, 2.0, 4.0, 6.0];
    let b_data: Vec<f64> = vec![2.0, 4.0, 1.0, 6.0, 5.0, 3.0];

    let a = Tensor::<Wgpu>::from_slice(&a_data, (2, 3), (&dev, FloatDType::F32)).unwrap().transpose(0, 1).unwrap();
    let b = Tensor::<Wgpu>::from_slice(&b_data, (2, 3), (&dev, FloatDType::F32)).unwrap().transpose(0, 1).unwrap();
    let ca = Tensor::<Cpu>::from_slice(&a_data, (2, 3), (&cpu, FloatDType::F32)).unwrap().transpose(0, 1).unwrap();
    let cb = Tensor::<Cpu>::from_slice(&b_data, (2, 3), (&cpu, FloatDType::F32)).unwrap().transpose(0, 1).unwrap();

    assert_eq!(a.eq(&b).unwrap().to_vec().unwrap(), ca.eq(&cb).unwrap().to_vec().unwrap());
    assert_eq!(a.lt(&b).unwrap().to_vec().unwrap(), ca.lt(&cb).unwrap().to_vec().unwrap());

    assert_eq!(a.lt_scalar(3.0).unwrap().to_vec().unwrap(), ca.lt_scalar(3.0).unwrap().to_vec().unwrap());
}

#[test]
fn binary_strided_f16() {
    let Some(dev) = device() else { return };
    if !dev.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }
    let cpu = Cpu::default();

    let a_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

    let a = Tensor::<Wgpu>::from_slice(&a_data, (2, 3), (&dev, FloatDType::F16)).unwrap().transpose(0, 1).unwrap();
    let b = Tensor::<Wgpu>::from_slice(&b_data, (2, 3), (&dev, FloatDType::F16)).unwrap().transpose(0, 1).unwrap();
    let ca = Tensor::<Cpu>::from_slice(&a_data, (2, 3), (&cpu, FloatDType::F16)).unwrap().transpose(0, 1).unwrap();
    let cb = Tensor::<Cpu>::from_slice(&b_data, (2, 3), (&cpu, FloatDType::F16)).unwrap().transpose(0, 1).unwrap();

    let got = a.add(&b).unwrap();
    let want = ca.add(&cb).unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 2e-2, 2e-2);
}

#[test]
fn binary_scalar_strided_f32() {
    let Some(dev) = device() else { return };
    let cpu = Cpu::default();

    numeric::test_add_scalar_f32(&dev);
    numeric::test_sub_scalar_f32(&dev);
    numeric::test_sub_scalar_lhs_f32(&dev);
    numeric::test_mul_scalar_f32(&dev);
    numeric::test_div_scalar_f32(&dev);
    numeric::test_div_scalar_lhs_f32(&dev);

    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = Tensor::<Wgpu>::from_slice(&data, (2, 3), (&dev, FloatDType::F32)).unwrap().transpose(0, 1).unwrap();
    let cx = Tensor::<Cpu>::from_slice(&data, (2, 3), (&cpu, FloatDType::F32)).unwrap().transpose(0, 1).unwrap();

    let got = x.add_scalar(10.0).unwrap();
    let want = cx.add_scalar(10.0).unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-5);

    let got = x.sub_scalar_lhs(10.0).unwrap();
    let want = cx.sub_scalar_lhs(10.0).unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-5);

    let got = x.div_scalar(2.0).unwrap();
    let want = cx.div_scalar(2.0).unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-5);
}

#[test]
fn unary_cmp_strided_f16() {
    let Some(dev) = device() else { return };
    if !dev.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }
    let cpu = Cpu::default();

    let data: Vec<f64> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let x = Tensor::<Wgpu>::from_slice(&data, (3, 2), (&dev, FloatDType::F16)).unwrap().transpose(0, 1).unwrap();
    let cx = Tensor::<Cpu>::from_slice(&data, (3, 2), (&cpu, FloatDType::F16)).unwrap().transpose(0, 1).unwrap();

    let got = x.neg().unwrap();
    let want = cx.neg().unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 2e-2, 2e-2);

    let got = x.lt_scalar(0.5).unwrap();
    let want = cx.lt_scalar(0.5).unwrap();
    assert_eq!(got.to_vec().unwrap(), want.to_vec().unwrap());
}

#[test]
fn contiguous_offset_f32() {
    let Some(dev) = device() else { return };
    let cpu = Cpu::default();

    let a_data: Vec<f64> = (1..=12).map(f64::from).collect();
    let b_data: Vec<f64> = (1..=8).map(|v| v as f64 * 10.0).collect();

    let a = Tensor::<Wgpu>::from_slice(&a_data, (3, 4), (&dev, FloatDType::F32)).unwrap().narrow(0, 1, 2).unwrap();
    let b = Tensor::<Wgpu>::from_slice(&b_data, (2, 4), (&dev, FloatDType::F32)).unwrap();
    let ca = Tensor::<Cpu>::from_slice(&a_data, (3, 4), (&cpu, FloatDType::F32)).unwrap().narrow(0, 1, 2).unwrap();
    let cb = Tensor::<Cpu>::from_slice(&b_data, (2, 4), (&cpu, FloatDType::F32)).unwrap();

    assert!(a.is_contiguous());
    assert_eq!(a.to_vec().unwrap(), ca.to_vec().unwrap());

    let got = a.add(&b).unwrap();
    let want = ca.add(&cb).unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-5);

    let got = a.neg().unwrap();
    let want = ca.neg().unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-5);

    // 64 f32 elements = 256 bytes, matching wgpu's default binding-offset alignment.
    let long_data: Vec<f64> = (0..200).map(f64::from).collect();
    let x = Tensor::<Wgpu>::from_slice(&long_data, (200,), (&dev, FloatDType::F32)).unwrap().narrow(0, 64, 10).unwrap();
    let cx = Tensor::<Cpu>::from_slice(&long_data, (200,), (&cpu, FloatDType::F32)).unwrap().narrow(0, 64, 10).unwrap();
    assert_close(&x.to_vec().unwrap(), &cx.to_vec().unwrap(), 1e-5, 1e-5);
}

#[test]
fn contiguous_offset_f16() {
    let Some(dev) = device() else { return };
    if !dev.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }
    let cpu = Cpu::default();

    let data: Vec<f64> = (1..=12).map(f64::from).collect();
    let x = Tensor::<Wgpu>::from_slice(&data, (3, 4), (&dev, FloatDType::F16)).unwrap().narrow(0, 1, 2).unwrap();
    let cx = Tensor::<Cpu>::from_slice(&data, (3, 4), (&cpu, FloatDType::F16)).unwrap().narrow(0, 1, 2).unwrap();

    assert!(x.is_contiguous());
    assert_close(&x.to_vec().unwrap(), &cx.to_vec().unwrap(), 2e-2, 2e-2);

    // Odd element start: byte offset 2 is not 4-byte aligned.
    let y = Tensor::<Wgpu>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], (5,), (&dev, FloatDType::F16)).unwrap().narrow(0, 1, 3).unwrap();
    let cy = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], (5,), (&cpu, FloatDType::F16)).unwrap().narrow(0, 1, 3).unwrap();
    assert_close(&y.to_vec().unwrap(), &cy.to_vec().unwrap(), 2e-2, 2e-2);

    let got = x.neg().unwrap();
    let want = cx.neg().unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 2e-2, 2e-2);
}

#[test]
fn binary_one_side_strided_f32() {
    let Some(dev) = device() else { return };
    let cpu = Cpu::default();

    let a_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

    let a = Tensor::<Wgpu>::from_slice(&a_data, (2, 3), (&dev, FloatDType::F32)).unwrap().transpose(0, 1).unwrap();
    let b = Tensor::<Wgpu>::from_slice(&b_data, (3, 2), (&dev, FloatDType::F32)).unwrap();
    let ca = Tensor::<Cpu>::from_slice(&a_data, (2, 3), (&cpu, FloatDType::F32)).unwrap().transpose(0, 1).unwrap();
    let cb = Tensor::<Cpu>::from_slice(&b_data, (3, 2), (&cpu, FloatDType::F32)).unwrap();

    let got = a.add(&b).unwrap();
    let want = ca.add(&cb).unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-5);

    let got = b.add(&a).unwrap();
    let want = cb.add(&ca).unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-5);
}

#[test]
fn inplace_strided_f32() {
    let Some(dev) = device() else { return };
    let cpu = Cpu::default();

    let x_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y_data: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

    let xb = Tensor::<Wgpu>::from_slice(&x_data, (2, 3), (&dev, FloatDType::F32)).unwrap();
    let x = xb.transpose(0, 1).unwrap();
    let y = Tensor::<Wgpu>::from_slice(&y_data, (3, 2), (&dev, FloatDType::F32)).unwrap();
    let cxb = Tensor::<Cpu>::from_slice(&x_data, (2, 3), (&cpu, FloatDType::F32)).unwrap();
    let cx = cxb.transpose(0, 1).unwrap();
    let cy = Tensor::<Cpu>::from_slice(&y_data, (3, 2), (&cpu, FloatDType::F32)).unwrap();

    x.add_(&y).unwrap();
    cx.add_(&cy).unwrap();
    assert_close(&xb.to_vec().unwrap(), &cxb.to_vec().unwrap(), 1e-5, 1e-5);

    x.add_scalar_(2.0).unwrap();
    cx.add_scalar_(2.0).unwrap();
    assert_close(&xb.to_vec().unwrap(), &cxb.to_vec().unwrap(), 1e-5, 1e-5);
}

#[test]
fn inplace_strided_f16() {
    let Some(dev) = device() else { return };
    if !dev.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }
    let cpu = Cpu::default();

    let x_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y_data: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

    let xb = Tensor::<Wgpu>::from_slice(&x_data, (2, 3), (&dev, FloatDType::F16)).unwrap();
    let x = xb.transpose(0, 1).unwrap();
    let y = Tensor::<Wgpu>::from_slice(&y_data, (3, 2), (&dev, FloatDType::F16)).unwrap();
    let cxb = Tensor::<Cpu>::from_slice(&x_data, (2, 3), (&cpu, FloatDType::F16)).unwrap();
    let cx = cxb.transpose(0, 1).unwrap();
    let cy = Tensor::<Cpu>::from_slice(&y_data, (3, 2), (&cpu, FloatDType::F16)).unwrap();

    x.add_(&y).unwrap();
    cx.add_(&cy).unwrap();
    assert_close(&xb.to_vec().unwrap(), &cxb.to_vec().unwrap(), 2e-2, 2e-2);

    x.add_scalar_(2.0).unwrap();
    cx.add_scalar_(2.0).unwrap();
    assert_close(&xb.to_vec().unwrap(), &cxb.to_vec().unwrap(), 2e-2, 2e-2);
}

#[test]
fn binary_scalar_strided_f16() {
    let Some(dev) = device() else { return };
    if !dev.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }
    let cpu = Cpu::default();

    half_precision::test_half_scalar(&dev, FloatDType::F16, 0.05);

    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = Tensor::<Wgpu>::from_slice(&data, (2, 3), (&dev, FloatDType::F16)).unwrap().transpose(0, 1).unwrap();
    let cx = Tensor::<Cpu>::from_slice(&data, (2, 3), (&cpu, FloatDType::F16)).unwrap().transpose(0, 1).unwrap();

    let got = x.add_scalar(10.0).unwrap();
    let want = cx.add_scalar(10.0).unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 2e-2, 2e-2);

    let got = x.sub_scalar_lhs(10.0).unwrap();
    let want = cx.sub_scalar_lhs(10.0).unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 2e-2, 2e-2);
}

#[test]
fn contiguous_f32() {
    let Some(dev) = device() else { return };
    cross::test_permute_contiguous_add(&dev);

    let x = Tensor::<Wgpu>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), (&dev, FloatDType::F32)).unwrap();
    let c = x.transpose(0, 1).unwrap().contiguous().unwrap();
    assert!(c.is_contiguous());
    assert_close(&c.to_vec().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-5, 1e-5);
}

#[test]
fn contiguous_f16() {
    let Some(dev) = device() else { return };
    if !dev.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }
    let cpu = Cpu::default();

    let data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = Tensor::<Wgpu>::from_slice(&data, (2, 3), (&dev, FloatDType::F16)).unwrap().transpose(0, 1).unwrap();
    let cx = Tensor::<Cpu>::from_slice(&data, (2, 3), (&cpu, FloatDType::F16)).unwrap().transpose(0, 1).unwrap();
    let got = x.contiguous().unwrap();
    let want = cx.contiguous().unwrap();
    assert!(got.is_contiguous());
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 2e-2, 2e-2);
}
