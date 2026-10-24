use luma_tensor::tests::{assert_close, cast, half_precision};
use luma_tensor::{FloatDType, IntDType, Tensor};
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
fn cast_shared_f32_int() {
    let Some(dev) = device() else { return };
    cast::test_cast_f32_to_i32(&dev);
    cast::test_cast_i32_to_f32(&dev);
    cast::test_cast_i32_to_u32(&dev);
}

#[test]
fn cast_shared_f16() {
    let Some(dev) = device() else { return };
    if !dev.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }
    half_precision::test_half_cast(&dev, FloatDType::F16, 0.05);
}

#[test]
fn cast_float_float() {
    let Some(dev) = device() else { return };
    if !dev.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }

    let x = Tensor::<Wgpu>::from_slice(&[1.5f64, -2.25, 0.0, 4.0], (4,), (&dev, FloatDType::F32)).unwrap();
    let h = x.cast(FloatDType::F16).unwrap();
    assert_eq!(h.dtype(), FloatDType::F16);
    assert_close(&h.to_vec().unwrap(), &[1.5, -2.25, 0.0, 4.0], 0.05, 0.05);

    let back = h.cast(FloatDType::F32).unwrap();
    assert_eq!(back.dtype(), FloatDType::F32);
    assert_close(&back.to_vec().unwrap(), &[1.5, -2.25, 0.0, 4.0], 0.05, 0.05);
}

#[test]
fn cast_float_int() {
    let Some(dev) = device() else { return };

    let f = Tensor::<Wgpu>::from_slice(&[1.7f64, 2.3, -3.8], (3,), (&dev, FloatDType::F32)).unwrap();
    assert_eq!(f.cast(IntDType::I32).unwrap().to_vec().unwrap(), vec![1, 2, -3]);
    assert_eq!(f.cast(IntDType::U32).unwrap().to_vec().unwrap(), vec![1, 2, 0]);

    let i = Tensor::<Wgpu, luma_tensor::Int>::from_slice(&[-2i64, 0, 7], (3,), (&dev, IntDType::I32)).unwrap();
    let fi = i.cast(FloatDType::F32).unwrap();
    assert_close(&fi.to_vec().unwrap(), &[-2.0, 0.0, 7.0], 1e-5, 1e-5);

    let u = Tensor::<Wgpu, luma_tensor::Int>::from_slice(&[2i64, 0, 7], (3,), (&dev, IntDType::U32)).unwrap();
    let fu = u.cast(FloatDType::F32).unwrap();
    assert_close(&fu.to_vec().unwrap(), &[2.0, 0.0, 7.0], 1e-5, 1e-5);
}

#[test]
fn cast_f16_int() {
    let Some(dev) = device() else { return };
    if !dev.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }

    let f = Tensor::<Wgpu>::from_slice(&[1.5f64, -2.25, 3.0], (3,), (&dev, FloatDType::F16)).unwrap();
    assert_eq!(f.cast(IntDType::I32).unwrap().to_vec().unwrap(), vec![1, -2, 3]);
    assert_eq!(f.cast(IntDType::U32).unwrap().to_vec().unwrap(), vec![1, 0, 3]);

    let i = Tensor::<Wgpu, luma_tensor::Int>::from_slice(&[-2i64, 0, 7], (3,), (&dev, IntDType::I32)).unwrap();
    let fi = i.cast(FloatDType::F16).unwrap();
    assert_eq!(fi.dtype(), FloatDType::F16);
    assert_close(&fi.to_vec().unwrap(), &[-2.0, 0.0, 7.0], 0.05, 0.05);

    let u = Tensor::<Wgpu, luma_tensor::Int>::from_slice(&[2i64, 0, 7], (3,), (&dev, IntDType::U32)).unwrap();
    let fu = u.cast(FloatDType::F16).unwrap();
    assert_close(&fu.to_vec().unwrap(), &[2.0, 0.0, 7.0], 0.05, 0.05);
}

#[test]
fn cast_bool_targets() {
    let Some(dev) = device() else { return };
    let x = Tensor::<Wgpu, luma_tensor::Bool>::from_slice(&[true, false, true], (3,), (&dev, luma_tensor::BoolDType::Bool)).unwrap();

    assert_eq!(x.cast(IntDType::I32).unwrap().to_vec().unwrap(), vec![1, 0, 1]);
    assert_eq!(x.cast(IntDType::U32).unwrap().to_vec().unwrap(), vec![1, 0, 1]);
    assert_close(&x.cast(FloatDType::F32).unwrap().to_vec().unwrap(), &[1.0, 0.0, 1.0], 1e-5, 1e-5);
    assert_eq!(x.cast(luma_tensor::BoolDType::Bool).unwrap().to_vec().unwrap(), vec![true, false, true]);
}

#[test]
fn cast_strided() {
    let Some(dev) = device() else { return };

    let x = Tensor::<Wgpu>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), (&dev, FloatDType::F32))
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let y = x.cast(IntDType::I32).unwrap();
    assert_eq!(y.to_vec().unwrap(), vec![1, 4, 2, 5, 3, 6]);
}

#[test]
fn cast_float_int_edges() {
    let Some(dev) = device() else { return };

    let x = Tensor::<Wgpu>::from_slice(&[f64::NAN, 1e20, -1e20, 2147483648.0, -2147483649.0], (5,), (&dev, FloatDType::F32)).unwrap();
    assert_eq!(
        x.cast(IntDType::I32).unwrap().to_vec().unwrap(),
        vec![0, i32::MAX as i64, i32::MIN as i64, i32::MAX as i64, i32::MIN as i64]
    );

    let y = Tensor::<Wgpu>::from_slice(&[f64::NAN, -1.0, 0.0, 1e20], (4,), (&dev, FloatDType::F32)).unwrap();
    assert_eq!(y.cast(IntDType::U32).unwrap().to_vec().unwrap(), vec![0, 0, 0, u32::MAX as i64]);
}

#[test]
fn cast_unsupported() {
    let Some(dev) = device() else { return };

    let x = Tensor::<Wgpu>::from_slice(&[1.0f64], (1,), (&dev, FloatDType::F32)).unwrap();
    assert!(x.cast(FloatDType::F64).is_err());
    assert!(x.cast(FloatDType::BF16).is_err());
    assert!(x.cast(IntDType::U8).is_err());

    let i = Tensor::<Wgpu, luma_tensor::Int>::from_slice(&[1i64], (1,), (&dev, IntDType::I32)).unwrap();
    assert!(i.cast(FloatDType::F64).is_err());
    assert!(i.cast(IntDType::U8).is_err());
}
