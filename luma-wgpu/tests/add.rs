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
fn add_f32() {
    let Some(dev) = device() else { return };

    let a = Tensor::<Wgpu>::from_slice(&[1.0f64, 2.0, 3.0, 4.0], (2, 2), (&dev, FloatDType::F32)).unwrap();
    let b = Tensor::<Wgpu>::from_slice(&[10.0f64, 20.0, 30.0, 40.0], (2, 2), (&dev, FloatDType::F32)).unwrap();

    let c = a.add(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn add_f16() {
    let Some(dev) = device() else { return };

    if !dev.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }

    let a = Tensor::<Wgpu>::from_slice(&[1.0f64, 2.0, 3.0, 4.0], (2, 2), (&dev, FloatDType::F16)).unwrap();
    let b = Tensor::<Wgpu>::from_slice(&[10.0f64, 20.0, 30.0, 40.0], (2, 2), (&dev, FloatDType::F16)).unwrap();

    let c = a.add(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn add_f16_odd_len() {
    let Some(dev) = device() else { return };

    if !dev.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }

    // Odd element count: the backing buffer is padded to 4 bytes, so this
    // exercises upload/readback padding for f16.
    let a = Tensor::<Wgpu>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], (5,), (&dev, FloatDType::F16)).unwrap();
    let b = Tensor::<Wgpu>::from_slice(&[10.0f64, 20.0, 30.0, 40.0, 50.0], (5,), (&dev, FloatDType::F16)).unwrap();

    let c = a.add(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![11.0, 22.0, 33.0, 44.0, 55.0]);
}
