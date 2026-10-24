use luma_tensor::tests::construct;
use luma_tensor::{FloatDType, FloatOps, Tensor};
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
fn construct_basic_f32() {
    let Some(dev) = device() else { return };
    construct::test_zeros_like_f32(&dev);
    construct::test_ones_like_f32(&dev);
    construct::test_from_slice_f32(&dev);
    construct::test_full_scalar(&dev);
    construct::test_rand_like_shape(&dev);
    construct::test_randn_like_shape(&dev);
}

#[test]
fn rand_uniform_f32() {
    let Some(dev) = device() else { return };

    let a = Tensor::<Wgpu>::rand(-2.0, 3.0, (4096,), (&dev, FloatDType::F32)).unwrap();
    let av = a.to_vec().unwrap();
    assert_eq!(av.len(), 4096);
    assert!(av.iter().all(|&v| (-2.0..=3.0).contains(&v)));
    assert!(av.windows(2).any(|w| w[0] != w[1]));

    let b = Tensor::<Wgpu>::rand(-2.0, 3.0, (4096,), (&dev, FloatDType::F32)).unwrap();
    assert_ne!(av, b.to_vec().unwrap());

    let z = Tensor::<Wgpu>::rand(0.0, 1.0, (0,), (&dev, FloatDType::F32)).unwrap();
    assert_eq!(z.element_count(), 0);
}

#[test]
fn rand_normal_f32() {
    let Some(dev) = device() else { return };

    let mean = 1.5;
    let std = 2.0;
    let a = Tensor::<Wgpu>::randn(mean, std, (8192,), (&dev, FloatDType::F32)).unwrap();
    let v = a.to_vec().unwrap();
    let got_mean = v.iter().sum::<f64>() / v.len() as f64;
    let got_var = v.iter().map(|x| (x - got_mean).powi(2)).sum::<f64>() / v.len() as f64;
    let got_std = got_var.sqrt();

    assert!((got_mean - mean).abs() < 0.1, "mean {got_mean}");
    assert!((got_std - std).abs() < 0.1, "std {got_std}");
}

#[test]
fn rand_f16() {
    let Some(dev) = device() else { return };
    if !dev.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }

    let a = Tensor::<Wgpu>::rand(0.0, 1.0, (256,), (&dev, FloatDType::F16)).unwrap();
    assert_eq!(a.dtype(), FloatDType::F16);
    assert!(a.to_vec().unwrap().iter().all(|&v| (0.0..=1.0).contains(&v)));

    let b = Tensor::<Wgpu>::randn(0.0, 1.0, (256,), (&dev, FloatDType::F16)).unwrap();
    assert_eq!(b.dtype(), FloatDType::F16);
}

#[test]
fn rand_invalid_parameters() {
    let Some(dev) = device() else { return };

    assert!(Tensor::<Wgpu>::rand(1.0, 1.0, (4,), (&dev, FloatDType::F32)).is_err());
    assert!(Tensor::<Wgpu>::randn(0.0, -1.0, (4,), (&dev, FloatDType::F32)).is_err());
}

#[test]
fn device_helpers() {
    let Some(dev) = device() else { return };

    let x = Wgpu::f_from_f32(vec![1.0f32, 2.0, 3.0, 4.0], &dev).unwrap();
    assert_eq!(x.len(), 4);
    dev.synchronize().unwrap();
}
