use luma_tensor::tests::{boolean, cast, construct};
use luma_tensor::{Bool, BoolDType, FloatDType, IntDType, Tensor};
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
fn bool_construct() {
    let Some(dev) = device() else { return };

    let f = Tensor::<Wgpu, Bool>::falses((2, 3), (&dev, BoolDType::Bool)).unwrap();
    assert_eq!(f.to_vec().unwrap(), vec![false; 6]);

    let t = Tensor::<Wgpu, Bool>::trues((2, 3), (&dev, BoolDType::Bool)).unwrap();
    assert_eq!(t.to_vec().unwrap(), vec![true; 6]);

    let data = [true, false, true, false, true, false];
    let x = Tensor::<Wgpu, Bool>::from_slice(&data, (2, 3), (&dev, BoolDType::Bool)).unwrap();
    assert_eq!(x.to_vec().unwrap(), data);
    assert_eq!(x.to_bytes().unwrap().to_vec(), vec![1, 0, 1, 0, 1, 0]);
}

#[test]
fn bool_logical() {
    let Some(dev) = device() else { return };
    boolean::test_bool_and(&dev);
    boolean::test_bool_or(&dev);
    boolean::test_bool_xor(&dev);
    boolean::test_bool_not(&dev);
}

#[test]
fn bool_pick() {
    let Some(dev) = device() else { return };
    boolean::test_pick_f32(&dev);
    boolean::test_pick_scalar_true(&dev);
    boolean::test_pick_scalar_false(&dev);
    boolean::test_pick_bool(&dev);
    boolean::test_pick_int(&dev);
    boolean::test_pick_int_scalar_true(&dev);
    boolean::test_pick_int_scalar_false(&dev);
    boolean::test_pick_bool_scalar_true(&dev);
    boolean::test_pick_bool_scalar_false(&dev);
}

#[test]
fn bool_reduce() {
    let Some(dev) = device() else { return };
    boolean::test_bool_all_all(&dev);
    boolean::test_bool_any_all(&dev);
    boolean::test_bool_true_count(&dev);
    boolean::test_bool_false_count(&dev);
    boolean::test_allclose_bool(&dev);
}

#[test]
fn bool_cast() {
    let Some(dev) = device() else { return };
    cast::test_cast_f32_to_bool(&dev);
    cast::test_cast_i32_to_bool(&dev);
    cast::test_cast_bool_to_f32(&dev);
    cast::test_cast_bool_to_i32(&dev);
    cast::test_cast_bool_to_bool(&dev);

    let x = Tensor::<Wgpu, Bool>::from_slice(&[true, false, true], (3,), (&dev, BoolDType::Bool)).unwrap();
    let u = x.cast(IntDType::U32).unwrap();
    let uv: Vec<i64> = u.to_vec().unwrap();
    assert_eq!(uv, vec![1, 0, 1]);
}

#[test]
fn bool_cast_pick_f16() {
    let Some(dev) = device() else { return };
    if !dev.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }

    let f16_input = Tensor::<Wgpu>::from_slice(&[0.0f64, 1.0, -2.0, 0.0], (4,), (&dev, FloatDType::F16)).unwrap();
    let f16_bool = f16_input.cast_bool(BoolDType::Bool).unwrap();
    assert_eq!(f16_bool.to_vec().unwrap(), vec![false, true, true, false]);

    let x = Tensor::<Wgpu, Bool>::from_slice(&[true, false, true], (3,), (&dev, BoolDType::Bool)).unwrap();
    let f = x.cast(FloatDType::F16).unwrap();
    let fv = f.to_vec().unwrap();
    assert_eq!(fv.len(), 3);
    assert!(fv[0] > 0.9 && fv[1].abs() < 0.01 && fv[2] > 0.9);

    let mask = Tensor::<Wgpu, Bool>::from_slice(&[true, false, true, false], (4,), (&dev, BoolDType::Bool)).unwrap();
    let on_true = Tensor::<Wgpu>::from_slice(&[1.0f64, 2.0, 3.0, 4.0], (4,), (&dev, FloatDType::F16)).unwrap();
    let on_false = Tensor::<Wgpu>::from_slice(&[10.0f64, 20.0, 30.0, 40.0], (4,), (&dev, FloatDType::F16)).unwrap();
    let picked = mask.pick(&on_true, &on_false).unwrap();
    assert_eq!(picked.dtype(), FloatDType::F16);
    let pv = picked.to_vec().unwrap();
    assert!((pv[0] - 1.0).abs() < 0.01);
    assert!((pv[1] - 20.0).abs() < 0.05);
    assert!((pv[2] - 3.0).abs() < 0.01);
    assert!((pv[3] - 40.0).abs() < 0.1);

    let scalar = mask.pick_true(5.0f64, &on_false).unwrap();
    let sv = scalar.to_vec().unwrap();
    assert!(sv[0] > 4.9 && sv[0] < 5.1);
    assert!((sv[1] - 20.0).abs() < 0.05);
    assert!(sv[2] > 4.9 && sv[2] < 5.1);
    assert!((sv[3] - 40.0).abs() < 0.1);
}

#[test]
fn bool_bytes_roundtrip() {
    let Some(dev) = device() else { return };
    construct::test_bytes_roundtrip_bool(&dev);
}

#[test]
fn bool_strided() {
    let Some(dev) = device() else { return };

    let x = Tensor::<Wgpu, Bool>::from_slice(&[true, false, true, false, true, false], (2, 3), (&dev, BoolDType::Bool)).unwrap();
    let t = x.transpose(0, 1).unwrap();
    assert_eq!(t.true_count().unwrap(), 3);
    assert_eq!(t.contiguous().unwrap().to_vec().unwrap(), vec![true, false, false, true, true, false]);
    assert_eq!(t.not().unwrap().to_vec().unwrap(), vec![false, true, true, false, false, true]);

    let other = Tensor::<Wgpu, Bool>::from_slice(&[true, false, false, true, true, false], (3, 2), (&dev, BoolDType::Bool)).unwrap();
    assert!(t.allclose(&other).unwrap());

    let mask = Tensor::<Wgpu, Bool>::from_slice(&[true, false, true, false, true, false], (2, 3), (&dev, BoolDType::Bool))
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let on_true = Tensor::<Wgpu>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), (&dev, FloatDType::F32))
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let on_false = Tensor::<Wgpu>::from_slice(&[10.0f64, 20.0, 30.0, 40.0, 50.0, 60.0], (2, 3), (&dev, FloatDType::F32))
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let picked = mask.pick(&on_true, &on_false).unwrap();
    assert_eq!(picked.to_vec().unwrap(), vec![1.0, 40.0, 20.0, 5.0, 3.0, 60.0]);
}

#[test]
fn allclose_float_int() {
    let Some(dev) = device() else { return };
    boolean::test_allclose_exact(&dev);
    boolean::test_allclose_false(&dev);
    boolean::test_allclose_int(&dev);

    let a = Tensor::<Wgpu>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), (&dev, FloatDType::F32))
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let b = Tensor::<Wgpu>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), (&dev, FloatDType::F32))
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    assert!(a.allclose(&b, 1e-5, 1e-5).unwrap());

    if dev.supports_f16() {
        let a = Tensor::<Wgpu>::from_slice(&[1.0f64, 2.0, 3.0], (3,), (&dev, FloatDType::F16)).unwrap();
        let b = Tensor::<Wgpu>::from_slice(&[1.0f64, 2.0, 3.0], (3,), (&dev, FloatDType::F16)).unwrap();
        let c = Tensor::<Wgpu>::from_slice(&[1.0f64, 2.0, 4.0], (3,), (&dev, FloatDType::F16)).unwrap();
        assert!(a.allclose(&b, 1e-2, 1e-2).unwrap());
        assert!(!a.allclose(&c, 1e-2, 1e-2).unwrap());
    }
}
