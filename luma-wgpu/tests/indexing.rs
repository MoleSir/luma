use luma_cpu::Cpu;
use luma_tensor::tests::{assert_close, indexing, tensor_f32_dev, tensor_i32_dev};
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
fn index_select_gather_f32_shared() {
    let Some(dev) = device() else { return };
    indexing::test_index_select_dim0(&dev);
    indexing::test_index_select_dim1(&dev);
    indexing::test_gather_dim1(&dev);
}

#[test]
fn index_add_scatter_add_f32_shared() {
    let Some(dev) = device() else { return };
    indexing::test_index_add_f32(&dev);
    indexing::test_scatter_add_f32(&dev);
    indexing::test_index_add_2d(&dev);
}

#[test]
fn index_select_gather_f16() {
    let Some(dev) = device() else { return };
    if !dev.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }
    let cpu = Cpu::default();
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let x = f16_tensor(&dev, &data, (3, 2));
    let cx = cpu_f16_tensor(&cpu, &data, (3, 2));
    let idx = i32_tensor(&dev, &[0, 2], (2,));
    let cidx = tensor_i32_dev(&[0, 2], (2,), &cpu);

    let got = x.index_select(&idx, 0usize).unwrap();
    let want = cx.index_select(&cidx, 0usize).unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 2e-2, 2e-2);

    let gidx = i32_tensor(&dev, &[0, 2, 1, 0], (2, 2));
    let cgidx = tensor_i32_dev(&[0, 2, 1, 0], (2, 2), &cpu);
    let got = x.gather(&gidx, 0usize).unwrap();
    let want = cx.gather(&cgidx, 0usize).unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 2e-2, 2e-2);
}

#[test]
fn index_select_gather_i32_u32() {
    let Some(dev) = device() else { return };

    let idx = i32_tensor(&dev, &[0, 2], (2,));
    let x = i32_tensor(&dev, &[1, 2, 3, 4, 5, 6], (3, 2));
    assert_eq!(x.index_select(&idx, 0usize).unwrap().to_vec().unwrap(), vec![1, 2, 5, 6]);

    let gidx = i32_tensor(&dev, &[0, 1, 1, 0], (2, 2));
    assert_eq!(x.gather(&gidx, 1usize).unwrap().to_vec().unwrap(), vec![1, 2, 4, 3]);

    let uidx = u32_tensor(&dev, &[0, 2], (2,));
    let ux = u32_tensor(&dev, &[1, 2, 3, 4, 5, 6], (3, 2));
    assert_eq!(ux.index_select(&uidx, 0usize).unwrap().to_vec().unwrap(), vec![1, 2, 5, 6]);
    let ugidx = u32_tensor(&dev, &[0, 1, 1, 0], (2, 2));
    assert_eq!(ux.gather(&ugidx, 1usize).unwrap().to_vec().unwrap(), vec![1, 2, 4, 3]);
}

#[test]
fn index_select_gather_bool() {
    let Some(dev) = device() else { return };

    let x = Tensor::<Wgpu, Bool>::from_slice(&[true, false, true, false, true, false], (3, 2), (&dev, BoolDType::Bool)).unwrap();
    let idx = i32_tensor(&dev, &[0, 2], (2,));
    assert_eq!(x.index_select(&idx, 0usize).unwrap().to_vec().unwrap(), vec![true, false, true, false]);

    let gidx = i32_tensor(&dev, &[0, 1, 1, 0], (2, 2));
    assert_eq!(x.gather(&gidx, 1usize).unwrap().to_vec().unwrap(), vec![true, false, false, true]);
}

#[test]
fn index_add_scatter_add_i32_u32() {
    let Some(dev) = device() else { return };

    let init = i32_tensor(&dev, &[0, 0, 0, 0], (4,));
    let idx = i32_tensor(&dev, &[0, 2], (2,));
    let src = i32_tensor(&dev, &[10, 20], (2,));
    assert_eq!(init.index_add(&idx, &src, 0usize).unwrap().to_vec().unwrap(), vec![10, 0, 20, 0]);
    assert_eq!(init.scatter_add(&idx, &src, 0usize).unwrap().to_vec().unwrap(), vec![10, 0, 20, 0]);

    let uinit = u32_tensor(&dev, &[0, 0, 0, 0], (4,));
    let uidx = u32_tensor(&dev, &[0, 2], (2,));
    let usrc = u32_tensor(&dev, &[10, 20], (2,));
    assert_eq!(uinit.index_add(&uidx, &usrc, 0usize).unwrap().to_vec().unwrap(), vec![10, 0, 20, 0]);
    assert_eq!(uinit.scatter_add(&uidx, &usrc, 0usize).unwrap().to_vec().unwrap(), vec![10, 0, 20, 0]);
}

#[test]
fn index_add_scatter_add_f16() {
    let Some(dev) = device() else { return };
    if !dev.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }
    let cpu = Cpu::default();

    let init = f16_tensor(&dev, &[1.0, 2.0, 3.0, 4.0], (4,));
    let cinit = cpu_f16_tensor(&cpu, &[1.0, 2.0, 3.0, 4.0], (4,));
    let idx = i32_tensor(&dev, &[1, 1, 3], (3,));
    let cidx = tensor_i32_dev(&[1, 1, 3], (3,), &cpu);
    let src = f16_tensor(&dev, &[10.0, 20.0, 30.0], (3,));
    let csrc = cpu_f16_tensor(&cpu, &[10.0, 20.0, 30.0], (3,));

    let got = init.index_add(&idx, &src, 0usize).unwrap();
    let want = cinit.index_add(&cidx, &csrc, 0usize).unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 2e-2, 2e-2);

    let got = init.scatter_add(&idx, &src, 0usize).unwrap();
    let want = cinit.scatter_add(&cidx, &csrc, 0usize).unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 2e-2, 2e-2);
}

#[test]
fn index_add_scatter_add_duplicate_indices() {
    let Some(dev) = device() else { return };

    let init = tensor_f32_dev(&[0.0, 0.0, 0.0, 0.0], (4,), &dev);
    let idx = i32_tensor(&dev, &[1, 1, 1], (3,));
    let src = tensor_f32_dev(&[1.0, 2.0, 3.0], (3,), &dev);
    assert_eq!(init.index_add(&idx, &src, 0usize).unwrap().to_vec().unwrap(), vec![0.0, 6.0, 0.0, 0.0]);

    let sidx = i32_tensor(&dev, &[1, 1], (2,));
    let ssrc = tensor_f32_dev(&[10.0, 20.0], (2,), &dev);
    assert_eq!(init.scatter_add(&sidx, &ssrc, 0usize).unwrap().to_vec().unwrap(), vec![0.0, 30.0, 0.0, 0.0]);
}

#[test]
fn indexing_pad_sentinel() {
    let Some(dev) = device() else { return };
    let pad = i32::MAX as i64;

    let x = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), &dev);
    let idx = i32_tensor(&dev, &[0, pad], (2,));
    assert_eq!(x.index_select(&idx, 0usize).unwrap().to_vec().unwrap(), vec![1.0, 2.0, 0.0, 0.0]);

    let gx = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &dev);
    let gidx = i32_tensor(&dev, &[0, pad, pad, 2], (2, 2));
    assert_eq!(gx.gather(&gidx, 1usize).unwrap().to_vec().unwrap(), vec![1.0, 0.0, 0.0, 6.0]);

    let init = tensor_f32_dev(&[0.0, 0.0, 0.0, 0.0], (4,), &dev);
    let aidx = i32_tensor(&dev, &[1, pad, 2], (3,));
    let src = tensor_f32_dev(&[10.0, 20.0, 30.0], (3,), &dev);
    assert_eq!(init.index_add(&aidx, &src, 0usize).unwrap().to_vec().unwrap(), vec![0.0, 10.0, 30.0, 0.0]);
    assert_eq!(init.scatter_add(&aidx, &src, 0usize).unwrap().to_vec().unwrap(), vec![0.0, 10.0, 30.0, 0.0]);

    let upad = u32::MAX as i64;
    let uidx = u32_tensor(&dev, &[0, upad], (2,));
    assert_eq!(x.index_select(&uidx, 0usize).unwrap().to_vec().unwrap(), vec![1.0, 2.0, 0.0, 0.0]);
}

#[test]
fn indexing_strided_layouts() {
    let Some(dev) = device() else { return };
    let cpu = Cpu::default();

    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = tensor_f32_dev(&data, (2, 3), &dev).transpose(0, 1).unwrap();
    let cx = tensor_f32_dev(&data, (2, 3), &cpu).transpose(0, 1).unwrap().contiguous().unwrap();
    let idx = i32_tensor(&dev, &[2, 0], (2,));
    let cidx = tensor_i32_dev(&[2, 0], (2,), &cpu);

    let got = x.index_select(&idx, 0usize).unwrap();
    let want = cx.index_select(&cidx, 0usize).unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-5);

    let idx_data = [0, 2, 0, 1];
    let gidx = i32_tensor(&dev, &idx_data, (2, 2)).transpose(0, 1).unwrap();
    let cgidx = tensor_i32_dev(&idx_data, (2, 2), &cpu).transpose(0, 1).unwrap().contiguous().unwrap();
    let got = x.gather(&gidx, 0usize).unwrap();
    let want = cx.gather(&cgidx, 0usize).unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-5);

    let init = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), &dev);
    let cinit = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), &cpu);
    let src_data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let src = tensor_f32_dev(&src_data, (2, 3), &dev).transpose(0, 1).unwrap();
    let csrc = tensor_f32_dev(&src_data, (2, 3), &cpu).transpose(0, 1).unwrap().contiguous().unwrap();
    let add_idx = i32_tensor(&dev, &[0, 2, 1], (3,));
    let cadd_idx = tensor_i32_dev(&[0, 2, 1], (3,), &cpu);

    let got = init.index_add(&add_idx, &src, 0usize).unwrap();
    let want = cinit.index_add(&cadd_idx, &csrc, 0usize).unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-5);

    let scatter_idx = i32_tensor(&dev, &[0, 2, 1, 1, 0, 2], (2, 3)).transpose(0, 1).unwrap();
    let cscatter_idx = tensor_i32_dev(&[0, 2, 1, 1, 0, 2], (2, 3), &cpu).transpose(0, 1).unwrap().contiguous().unwrap();
    let got = init.scatter_add(&scatter_idx, &src, 0usize).unwrap();
    let want = cinit.scatter_add(&cscatter_idx, &csrc, 0usize).unwrap();
    assert_close(&got.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-5);
}
