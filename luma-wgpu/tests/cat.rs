use luma_cpu::Cpu;
use luma_tensor::tests::{assert_close, shape, tensor_f32_dev};
use luma_tensor::{Bool, BoolDType, FloatDType, Int, IntDType, Shape, Tensor};
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

fn i32_tensor(dev: &Wgpu, data: &[i64], shape: impl Into<Shape>) -> Tensor<Wgpu, Int> {
    Tensor::<Wgpu, Int>::from_slice(data, shape, (dev, IntDType::I32)).unwrap()
}

fn u32_tensor(dev: &Wgpu, data: &[i64], shape: impl Into<Shape>) -> Tensor<Wgpu, Int> {
    Tensor::<Wgpu, Int>::from_slice(data, shape, (dev, IntDType::U32)).unwrap()
}

fn f16_tensor(dev: &Wgpu, data: &[f64], shape: impl Into<Shape>) -> Tensor<Wgpu> {
    Tensor::<Wgpu>::from_slice(data, shape, (dev, FloatDType::F16)).unwrap()
}

fn cpu_f16_tensor(cpu: &Cpu, data: &[f64], shape: impl Into<Shape>) -> Tensor<Cpu> {
    Tensor::<Cpu>::from_slice(data, shape, (cpu, FloatDType::F16)).unwrap()
}

#[test]
fn cat_dim0_f32_shared() {
    let Some(dev) = device() else { return };
    shape::test_cat_dim0_f32(&dev);
}

#[test]
fn cat_f32_dim1_dim2() {
    let Some(dev) = device() else { return };
    let cpu = Cpu::default();

    let a_data = [1.0, 2.0, 3.0, 4.0];
    let b_data = [10.0, 20.0];
    let a = tensor_f32_dev(&a_data, (2, 2), &dev);
    let b = tensor_f32_dev(&b_data, (2, 1), &dev);
    let ca = tensor_f32_dev(&a_data, (2, 2), &cpu);
    let cb = tensor_f32_dev(&b_data, (2, 1), &cpu);
    let got = Tensor::cat(&[&a, &b], 1usize).unwrap();
    let want = Tensor::cat(&[&ca, &cb], 1usize).unwrap();
    assert_eq!(got.dims(), &[2, 3]);
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-5);

    let a_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = [10.0, 20.0, 30.0, 40.0];
    let a = tensor_f32_dev(&a_data, (1, 2, 3), &dev);
    let b = tensor_f32_dev(&b_data, (1, 2, 2), &dev);
    let ca = tensor_f32_dev(&a_data, (1, 2, 3), &cpu);
    let cb = tensor_f32_dev(&b_data, (1, 2, 2), &cpu);
    let got = Tensor::cat(&[&a, &b], 2usize).unwrap();
    let want = Tensor::cat(&[&ca, &cb], 2usize).unwrap();
    assert_eq!(got.dims(), &[1, 2, 5]);
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-5);
}

#[test]
fn cat_f16() {
    let Some(dev) = device() else { return };
    if !dev.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }
    let cpu = Cpu::default();

    let a_data = [1.0, 2.0, 3.0, 4.0];
    let b_data = [10.0, 20.0];
    let a = f16_tensor(&dev, &a_data, (2, 2));
    let b = f16_tensor(&dev, &b_data, (2, 1));
    let ca = cpu_f16_tensor(&cpu, &a_data, (2, 2));
    let cb = cpu_f16_tensor(&cpu, &b_data, (2, 1));
    let got = Tensor::cat(&[&a, &b], 1usize).unwrap();
    let want = Tensor::cat(&[&ca, &cb], 1usize).unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 2e-2, 2e-2);
}

#[test]
fn cat_i32_u32() {
    let Some(dev) = device() else { return };

    let a = i32_tensor(&dev, &[1, 2, 3, 4], (2, 2));
    let b = i32_tensor(&dev, &[10, 20], (2, 1));
    let got = Tensor::cat(&[&a, &b], 1usize).unwrap();
    assert_eq!(got.to_vec().unwrap(), vec![1, 2, 10, 3, 4, 20]);

    let a = u32_tensor(&dev, &[1, 2, 3, 4], (2, 2));
    let b = u32_tensor(&dev, &[10, 20], (2, 1));
    let got = Tensor::cat(&[&a, &b], 1usize).unwrap();
    assert_eq!(got.to_vec().unwrap(), vec![1, 2, 10, 3, 4, 20]);
}

#[test]
fn cat_bool() {
    let Some(dev) = device() else { return };

    let a = Tensor::<Wgpu, Bool>::from_slice(&[true, false, true, false], (2, 2), (&dev, BoolDType::Bool)).unwrap();
    let b = Tensor::<Wgpu, Bool>::from_slice(&[true, true], (2, 1), (&dev, BoolDType::Bool)).unwrap();
    let got = Tensor::cat(&[&a, &b], 1usize).unwrap();
    assert_eq!(got.dims(), &[2, 3]);
    assert_eq!(got.to_vec().unwrap(), vec![true, false, true, true, false, true]);
}

#[test]
fn cat_strided_sources() {
    let Some(dev) = device() else { return };
    let cpu = Cpu::default();

    let a_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = [10.0, 20.0];
    let a = tensor_f32_dev(&a_data, (2, 3), &dev).transpose(0, 1).unwrap();
    let b = tensor_f32_dev(&b_data, (1, 2), &dev);
    let ca = tensor_f32_dev(&a_data, (2, 3), &cpu).transpose(0, 1).unwrap();
    let cb = tensor_f32_dev(&b_data, (1, 2), &cpu);

    let got = Tensor::cat(&[&a, &b], 0usize).unwrap();
    let want = Tensor::cat(&[&ca, &cb], 0usize).unwrap();
    assert_eq!(got.dims(), &[4, 2]);
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-5);
}

#[test]
fn cat_three_sources() {
    let Some(dev) = device() else { return };

    let a = tensor_f32_dev(&[1.0, 2.0], (1, 2), &dev);
    let b = tensor_f32_dev(&[3.0, 4.0, 5.0, 6.0], (2, 2), &dev);
    let c = tensor_f32_dev(&[7.0, 8.0], (1, 2), &dev);
    let got = Tensor::cat(&[&a, &b, &c], 0usize).unwrap();
    assert_eq!(got.dims(), &[4, 2]);
    assert_eq!(got.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}
