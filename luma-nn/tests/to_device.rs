//! Tests for `ToDevice` — moving modules (and everything they hold) between
//! devices.
//! Run with: cargo test -p luma-nn --test to_device
//! Cuda-gated tests: cargo test -p luma-nn --features cuda --test to_device

use std::marker::PhantomData;

use luma_macros::Module as ModuleDerive;
use luma_nn::{BatchNorm1d, Buffer, Dropout, Linear, ToDevice};
#[cfg(feature = "cuda")]
use luma_tensor::Cuda;
use luma_tensor::dtype::{BoolDType, FloatDType};
use luma_tensor::{Bool, Cpu, Device, Int, Tensor};

// ---- fixtures: modules exercising every field category ----------------------

/// Sub-module, `Option`-wrapped and `Vec`-wrapped child modules.
#[derive(ModuleDerive)]
struct Block<D: Device> {
    linear: Linear<D>,
    extra: Option<Linear<D>>,
    layers: Vec<Linear<D>>,
}

/// Real modules + skipped config fields of different types.
#[derive(ModuleDerive)]
struct Net<D: Device> {
    block: Block<D>,
    bn: BatchNorm1d<D>,
    drop: Dropout<D>,

    #[module(skip)]
    label: String,
    #[module(skip)]
    steps: usize,
}

/// Buffers of all three kinds + a skipped float config.
#[derive(ModuleDerive)]
struct FlagHolder<D: Device> {
    mean: Buffer<D>,
    count: Buffer<D, Int>,
    seen: Buffer<D, Bool>,

    #[module(skip)]
    momentum: f64,
}

/// Enum module: one transferred variant, one skipped (cloned payload) variant.
#[derive(ModuleDerive)]
enum Act<D: Device> {
    Relu(ReluMod<D>),
    #[module(skip)]
    Config(usize),
}

#[derive(ModuleDerive)]
struct ReluMod<D: Device> {
    #[module(skip)]
    _marker: PhantomData<D>,
}

fn linear(in_f: usize, out_f: usize) -> Linear<Cpu> {
    Linear::new(in_f, out_f, true, Cpu::default()).unwrap()
}

fn block() -> Block<Cpu> {
    Block { linear: linear(3, 4), extra: Some(linear(4, 2)), layers: vec![linear(2, 5), linear(5, 1)] }
}

fn net() -> Net<Cpu> {
    Net {
        block: block(),
        bn: BatchNorm1d::new(4, Cpu::default()).unwrap(),
        drop: Dropout::new(0.5),
        label: "net-1".into(),
        steps: 42,
    }
}

fn flags() -> FlagHolder<Cpu> {
    FlagHolder {
        mean: Buffer::<Cpu>::new(Tensor::zeros(&[4], Cpu::default()).unwrap()),
        count: Buffer::<Cpu, Int>::new(Tensor::<Cpu, Int>::zeros(&[1], Cpu::default()).unwrap()),
        seen: Buffer::<Cpu, Bool>::new(Tensor::<Cpu, Bool>::from_slice(&[true], (1,), BoolDType::Bool).unwrap()),
        momentum: 0.9,
    }
}

// ---- tests -------------------------------------------------------------------

#[test]
fn linear_transfer_same_device() {
    let m = linear(3, 4);
    let moved: Linear<Cpu> = m.to_device(&Cpu).unwrap();

    // skipped config copied verbatim
    assert_eq!(moved.in_features, 3);
    assert_eq!(moved.out_features, 4);
    // same-device transfer shares the underlying parameter tensors (no copy)
    assert_eq!(moved.weight.id(), m.weight.id());
    assert_eq!(moved.bias.as_ref().unwrap().id(), m.bias.as_ref().unwrap().id());
    // parameters stay trainable
    assert!(moved.weight.requires_grad());
    assert!(moved.bias.as_ref().unwrap().requires_grad());

    // forward gives identical results
    let x = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), FloatDType::F32).unwrap();
    let a = m.forward(&x).unwrap().to_vec().unwrap();
    let b = moved.forward(&x).unwrap().to_vec().unwrap();
    for (av, bv) in a.iter().zip(&b) {
        assert!((av - bv).abs() < 1e-6);
    }
}

#[test]
fn nested_transfer_copies_all_field_categories() {
    let m = net();
    let moved: Net<Cpu> = m.to_device(&Cpu).unwrap();

    assert_eq!(moved.label, "net-1");
    assert_eq!(moved.steps, 42);
    assert_eq!(moved.drop.p, 0.5);
    assert_eq!(moved.bn.num_features, 4);
    assert_eq!(moved.bn.eps, m.bn.eps);
    assert_eq!(moved.bn.momentum, m.bn.momentum);
    assert_eq!(moved.block.layers.len(), 2);
    assert_eq!(moved.block.layers[0].in_features, 2);
    assert_eq!(moved.block.layers[1].out_features, 1);

    // parameters / buffers shared (same-device no-op at tensor level)
    assert_eq!(moved.block.linear.weight.id(), m.block.linear.weight.id());
    assert_eq!(moved.block.extra.as_ref().unwrap().weight.id(), m.block.extra.as_ref().unwrap().weight.id());
    assert_eq!(moved.block.layers[1].bias.as_ref().unwrap().id(), m.block.layers[1].bias.as_ref().unwrap().id());
    assert_eq!(moved.bn.gamma.id(), m.bn.gamma.id());
    assert_eq!(moved.bn.running_mean.id(), m.bn.running_mean.id());
    assert_eq!(moved.bn.running_var.id(), m.bn.running_var.id());
}

#[test]
fn buffer_kinds_transfer() {
    let m = flags();
    let moved: FlagHolder<Cpu> = m.to_device(&Cpu).unwrap();

    assert_eq!(moved.momentum, 0.9);
    assert_eq!(moved.mean.id(), m.mean.id());
    assert_eq!(moved.count.id(), m.count.id());
    assert_eq!(moved.seen.id(), m.seen.id());
    assert_eq!(moved.seen.to_vec().unwrap(), vec![true]);
    // buffers stay non-trainable
    assert!(!moved.mean.requires_grad());
}

#[test]
fn enum_module_transfers() {
    let a = Act::<Cpu>::Relu(ReluMod { _marker: PhantomData });
    let moved: Act<Cpu> = a.to_device(&Cpu).unwrap();
    assert!(matches!(moved, Act::Relu(_)));

    let c = Act::<Cpu>::Config(7);
    let moved_c: Act<Cpu> = c.to_device(&Cpu).unwrap();
    assert!(matches!(moved_c, Act::Config(7)));
}

#[test]
fn requires_grad_state_survives_transfer() {
    let m = linear(3, 2);
    // A param switched off (e.g. eval-style) must keep its flag.
    m.weight.set_requires_grad(false);
    let moved: Linear<Cpu> = m.to_device(&Cpu).unwrap();
    assert!(!moved.weight.requires_grad());

    m.weight.set_requires_grad(true);
    let moved2: Linear<Cpu> = m.to_device(&Cpu).unwrap();
    assert!(moved2.weight.requires_grad());
}

// ---- cuda roundtrip ----------------------------------------------------------

#[cfg(feature = "cuda")]
#[test]
fn linear_cpu_cuda_roundtrip() {
    let dev = Cuda::new(0).expect("cuda device 0");
    let m = linear(3, 4);
    let gpu: Linear<Cuda> = m.to_device(&dev).unwrap();

    assert_eq!(gpu.in_features, 3);
    assert!(gpu.weight.requires_grad(), "parameters must stay trainable on the GPU");

    // forward agrees with the CPU model
    let x = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), FloatDType::F32).unwrap();
    let xg = x.to_device(&dev).unwrap();
    let y_cpu = m.forward(&x).unwrap().to_vec().unwrap();
    let y_gpu = gpu.forward(&xg).unwrap().cpu().unwrap().to_vec().unwrap();
    for (a, b) in y_cpu.iter().zip(&y_gpu) {
        assert!((a - b).abs() < 1e-4);
    }

    // roundtrip back to CPU reproduces the original weights
    let back: Linear<Cpu> = gpu.to_device(&Cpu).unwrap();
    assert_ne!(back.weight.id(), m.weight.id(), "cross-device transfer must copy");
    let w1 = m.weight.to_vec().unwrap();
    let w2 = back.weight.to_vec().unwrap();
    for (a, b) in w1.iter().zip(&w2) {
        assert!((a - b).abs() < 1e-4);
    }
}

// ---- cuda roundtrips for the full fixture set ---------------------------------

#[cfg(feature = "cuda")]
#[test]
fn nested_cuda_roundtrip() {
    let dev = Cuda::new(0).expect("cuda device 0");
    let m = net();
    let gpu: Net<Cuda> = m.to_device(&dev).unwrap();

    // skipped config copied verbatim
    assert_eq!(gpu.label, "net-1");
    assert_eq!(gpu.steps, 42);
    assert_eq!(gpu.drop.p, 0.5);
    assert_eq!(gpu.bn.num_features, 4);
    assert_eq!(gpu.block.layers.len(), 2);

    // parameters are deep copies on the GPU, not the same handle
    assert_ne!(gpu.block.linear.weight.id(), m.block.linear.weight.id(), "cross-device transfer must copy");
    assert_ne!(gpu.bn.running_mean.id(), m.bn.running_mean.id());
    assert!(gpu.block.linear.weight.requires_grad(), "params stay trainable on the GPU");

    // roundtrip back reproduces every parameter and buffer
    let back: Net<Cpu> = gpu.to_device(&Cpu).unwrap();
    assert_eq!(back.label, "net-1");
    assert_eq!(back.block.layers.len(), 2);

    let w1 = m.block.linear.weight.to_vec().unwrap();
    let w2 = back.block.linear.weight.to_vec().unwrap();
    for (a, b) in w1.iter().zip(&w2) {
        assert!((a - b).abs() < 1e-4);
    }

    let r1 = m.bn.running_mean.to_vec().unwrap();
    let r2 = back.bn.running_mean.to_vec().unwrap();
    for (a, b) in r1.iter().zip(&r2) {
        assert!((a - b).abs() < 1e-4);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn buffer_kinds_cuda_roundtrip() {
    let dev = Cuda::new(0).expect("cuda device 0");
    let m = flags();
    let gpu: FlagHolder<Cuda> = m.to_device(&dev).unwrap();

    assert_eq!(gpu.momentum, 0.9);
    // all three buffer kinds hold their values on-device
    assert_eq!(gpu.mean.to_vec().unwrap(), m.mean.to_vec().unwrap());
    assert_eq!(gpu.count.to_vec().unwrap(), m.count.to_vec().unwrap());
    assert_eq!(gpu.seen.to_vec().unwrap(), vec![true]);
    // buffers stay non-trainable on the GPU
    assert!(!gpu.mean.requires_grad());

    let back: FlagHolder<Cpu> = gpu.to_device(&Cpu).unwrap();
    assert_eq!(back.mean.to_vec().unwrap(), m.mean.to_vec().unwrap());
    assert_eq!(back.count.to_vec().unwrap(), m.count.to_vec().unwrap());
    assert_eq!(back.seen.to_vec().unwrap(), vec![true]);
}

#[cfg(feature = "cuda")]
#[test]
fn enum_module_cuda_roundtrip() {
    let dev = Cuda::new(0).expect("cuda device 0");

    let a = Act::<Cpu>::Relu(ReluMod { _marker: PhantomData });
    let gpu: Act<Cuda> = a.to_device(&dev).unwrap();
    assert!(matches!(gpu, Act::Relu(_)));
    let back: Act<Cpu> = gpu.to_device(&Cpu).unwrap();
    assert!(matches!(back, Act::Relu(_)));

    let c = Act::<Cpu>::Config(7);
    let gpu_c: Act<Cuda> = c.to_device(&dev).unwrap();
    assert!(matches!(gpu_c, Act::Config(7)));
}

#[cfg(feature = "cuda")]
#[test]
fn requires_grad_state_survives_cross_device() {
    let dev = Cuda::new(0).expect("cuda device 0");
    let m = linear(3, 2);
    // A param switched off must keep its flag through a real cross-device copy.
    m.weight.set_requires_grad(false);

    let gpu: Linear<Cuda> = m.to_device(&dev).unwrap();
    assert!(!gpu.weight.requires_grad(), "disabled grad must stay disabled on the GPU");

    let back: Linear<Cpu> = gpu.to_device(&Cpu).unwrap();
    assert!(!back.weight.requires_grad(), "disabled grad must survive the roundtrip");
}
