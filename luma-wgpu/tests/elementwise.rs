use luma_cpu::Cpu;
use luma_tensor::{Bool, FloatDType, Result, Tensor};
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

fn assert_close(got: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(got.len(), expected.len(), "length mismatch: {} vs {}", got.len(), expected.len());
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        let diff = (g - e).abs();
        assert!(diff <= tol, "mismatch at {i}: got {g}, expected {e} (diff {diff:.3e})");
    }
}

fn check_unary<F, G>(device: &Wgpu, dtype: FloatDType, data: &[f64], tol: f64, cpu_op: F, gpu_op: G)
where
    F: Fn(&Tensor<Cpu>) -> Result<Tensor<Cpu>>,
    G: Fn(&Tensor<Wgpu>) -> Result<Tensor<Wgpu>>,
{
    let cpu = Tensor::<Cpu>::from_slice(data, (data.len(),), (&Cpu::default(), dtype)).unwrap();
    let gpu = Tensor::<Wgpu>::from_slice(data, (data.len(),), (device, dtype)).unwrap();

    let expected = cpu_op(&cpu).unwrap().to_vec().unwrap();
    let got = gpu_op(&gpu).unwrap().to_vec().unwrap();
    assert_close(&got, &expected, tol);
}

fn check_unary_inplace<F, G>(device: &Wgpu, dtype: FloatDType, data: &[f64], tol: f64, cpu_op: F, gpu_op: G)
where
    F: Fn(&Tensor<Cpu>) -> Result<()>,
    G: Fn(&Tensor<Wgpu>) -> Result<()>,
{
    let cpu = Tensor::<Cpu>::from_slice(data, (data.len(),), (&Cpu::default(), dtype)).unwrap();
    let gpu = Tensor::<Wgpu>::from_slice(data, (data.len(),), (device, dtype)).unwrap();

    cpu_op(&cpu).unwrap();
    gpu_op(&gpu).unwrap();
    assert_close(&gpu.to_vec().unwrap(), &cpu.to_vec().unwrap(), tol);
}

fn check_cmp<F, G>(device: &Wgpu, dtype: FloatDType, a: &[f64], b: &[f64], cpu_op: F, gpu_op: G)
where
    F: Fn(&Tensor<Cpu>, &Tensor<Cpu>) -> Result<Tensor<Cpu, Bool>>,
    G: Fn(&Tensor<Wgpu>, &Tensor<Wgpu>) -> Result<Tensor<Wgpu, Bool>>,
{
    let ca = Tensor::<Cpu>::from_slice(a, (a.len(),), (&Cpu::default(), dtype)).unwrap();
    let cb = Tensor::<Cpu>::from_slice(b, (b.len(),), (&Cpu::default(), dtype)).unwrap();
    let ga = Tensor::<Wgpu>::from_slice(a, (a.len(),), (device, dtype)).unwrap();
    let gb = Tensor::<Wgpu>::from_slice(b, (b.len(),), (device, dtype)).unwrap();

    assert_eq!(cpu_op(&ca, &cb).unwrap().to_vec().unwrap(), gpu_op(&ga, &gb).unwrap().to_vec().unwrap());
}

fn check_cmp_scalar<F, G>(device: &Wgpu, dtype: FloatDType, a: &[f64], scalar: f64, cpu_op: F, gpu_op: G)
where
    F: Fn(&Tensor<Cpu>, f64) -> Result<Tensor<Cpu, Bool>>,
    G: Fn(&Tensor<Wgpu>, f64) -> Result<Tensor<Wgpu, Bool>>,
{
    let ca = Tensor::<Cpu>::from_slice(a, (a.len(),), (&Cpu::default(), dtype)).unwrap();
    let ga = Tensor::<Wgpu>::from_slice(a, (a.len(),), (device, dtype)).unwrap();

    assert_eq!(cpu_op(&ca, scalar).unwrap().to_vec().unwrap(), gpu_op(&ga, scalar).unwrap().to_vec().unwrap());
}

#[test]
fn unary_f32() {
    let Some(device) = device() else { return };
    let data = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let tol = 1e-5;

    check_unary(&device, FloatDType::F32, &data, tol, |t| t.neg(), |t| t.neg());
    check_unary(&device, FloatDType::F32, &data, tol, |t| t.abs(), |t| t.abs());
    check_unary(&device, FloatDType::F32, &data, tol, |t| t.sign(), |t| t.sign());
    check_unary(&device, FloatDType::F32, &data, tol, |t| t.affine(2.5, -0.5), |t| t.affine(2.5, -0.5));
    check_unary(&device, FloatDType::F32, &data, tol, |t| t.pow(2.0), |t| t.pow(2.0));
    check_unary(&device, FloatDType::F32, &data, tol, |t| t.clamp(Some(-1.0), Some(1.0)), |t| t.clamp(Some(-1.0), Some(1.0)));
    check_unary(&device, FloatDType::F32, &data, tol, |t| t.clamp(Some(0.0), None), |t| t.clamp(Some(0.0), None));
    check_unary(&device, FloatDType::F32, &data, tol, |t| t.clamp(None, Some(0.0)), |t| t.clamp(None, Some(0.0)));
}

#[test]
fn float_unary_f32() {
    let Some(device) = device() else { return };
    let data = [0.25, 0.5, 1.0, 2.0, 4.0];
    let tol = 1e-4;

    macro_rules! check {
        ($method:ident) => {
            check_unary(&device, FloatDType::F32, &data, tol, |t| t.$method(), |t| t.$method());
        };
    }

    check!(exp);
    check!(ln);
    check!(sin);
    check!(cos);
    check!(tanh);
    check!(sqr);
    check!(sqrt);
    check!(recip);
    check!(gelu);
    check!(gelu_erf);
    check!(erf);
    check!(relu);
    check!(silu);
    check!(sigmoid);
    check!(floor);
    check!(ceil);
    check!(round);

    let negative = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    check_unary(&device, FloatDType::F32, &negative, tol, |t| t.leaky_relu(0.1), |t| t.leaky_relu(0.1));
}

#[test]
fn cmp_f32() {
    let Some(device) = device() else { return };
    let a = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let b = [-1.0, 0.0, 0.0, 1.0, 3.0];

    check_cmp(&device, FloatDType::F32, &a, &b, |x, y| x.eq(y), |x, y| x.eq(y));
    check_cmp(&device, FloatDType::F32, &a, &b, |x, y| x.ne(y), |x, y| x.ne(y));
    check_cmp(&device, FloatDType::F32, &a, &b, |x, y| x.lt(y), |x, y| x.lt(y));
    check_cmp(&device, FloatDType::F32, &a, &b, |x, y| x.le(y), |x, y| x.le(y));
    check_cmp(&device, FloatDType::F32, &a, &b, |x, y| x.gt(y), |x, y| x.gt(y));
    check_cmp(&device, FloatDType::F32, &a, &b, |x, y| x.ge(y), |x, y| x.ge(y));

    check_cmp_scalar(&device, FloatDType::F32, &a, 0.0, |x, s| x.eq_scalar(s), |x, s| x.eq_scalar(s));
    check_cmp_scalar(&device, FloatDType::F32, &a, 0.0, |x, s| x.lt_scalar(s), |x, s| x.lt_scalar(s));
    check_cmp_scalar(&device, FloatDType::F32, &a, 0.0, |x, s| x.ge_scalar(s), |x, s| x.ge_scalar(s));
}

#[test]
fn unary_inplace_f32() {
    let Some(device) = device() else { return };
    let data = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let tol = 1e-5;

    check_unary_inplace(&device, FloatDType::F32, &data, tol, |t| t.neg_().map(|_| ()), |t| t.neg_().map(|_| ()));
    check_unary_inplace(&device, FloatDType::F32, &data, tol, |t| t.abs_().map(|_| ()), |t| t.abs_().map(|_| ()));
    check_unary_inplace(&device, FloatDType::F32, &data, tol, |t| t.sign_().map(|_| ()), |t| t.sign_().map(|_| ()));
    check_unary_inplace(&device, FloatDType::F32, &data, tol, |t| t.affine_(2.5, -0.5).map(|_| ()), |t| t.affine_(2.5, -0.5).map(|_| ()));
    check_unary_inplace(&device, FloatDType::F32, &data, tol, |t| t.pow_(2.0).map(|_| ()), |t| t.pow_(2.0).map(|_| ()));
    check_unary_inplace(
        &device,
        FloatDType::F32,
        &data,
        tol,
        |t| t.clamp_(Some(-1.0), Some(1.0)).map(|_| ()),
        |t| t.clamp_(Some(-1.0), Some(1.0)).map(|_| ()),
    );
}

#[test]
fn unary_inplace_strided_f32() {
    let Some(device) = device() else { return };

    let gpu = Tensor::<Wgpu>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), (&device, FloatDType::F32)).unwrap();
    let view = gpu.transpose(0, 1).unwrap();
    view.neg_().unwrap();
    assert_close(&gpu.to_vec().unwrap(), &[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], 1e-5);
}

#[test]
fn float_unary_inplace_f32() {
    let Some(device) = device() else { return };
    let data = [0.25, 0.5, 1.0, 2.0, 4.0];
    let tol = 1e-4;

    check_unary_inplace(&device, FloatDType::F32, &data, tol, |t| t.exp_(), |t| t.exp_());
    check_unary_inplace(&device, FloatDType::F32, &data, tol, |t| t.ln_(), |t| t.ln_());
    check_unary_inplace(&device, FloatDType::F32, &data, tol, |t| t.sqrt_(), |t| t.sqrt_());
    check_unary_inplace(&device, FloatDType::F32, &data, tol, |t| t.relu_(), |t| t.relu_());

    let negative = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    check_unary_inplace(&device, FloatDType::F32, &negative, tol, |t| t.leaky_relu_(0.1), |t| t.leaky_relu_(0.1));
}

#[test]
fn float_unary_inplace_strided_f32() {
    let Some(device) = device() else { return };

    let gpu = Tensor::<Wgpu>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), (&device, FloatDType::F32)).unwrap();
    let view = gpu.transpose(0, 1).unwrap();
    view.sqr_().unwrap();
    assert_close(&gpu.to_vec().unwrap(), &[1.0, 4.0, 9.0, 16.0, 25.0, 36.0], 1e-5);
}

#[test]
fn unary_f16() {
    let Some(device) = device() else { return };
    if !device.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }

    let data = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let tol = 2e-2;

    check_unary(&device, FloatDType::F16, &data, tol, |t| t.neg(), |t| t.neg());
    check_unary(&device, FloatDType::F16, &data, tol, |t| t.abs(), |t| t.abs());
    check_unary(&device, FloatDType::F16, &data, tol, |t| t.sign(), |t| t.sign());
    check_unary(&device, FloatDType::F16, &data, tol, |t| t.affine(2.5, -0.5), |t| t.affine(2.5, -0.5));
    check_unary(&device, FloatDType::F16, &data, tol, |t| t.pow(2.0), |t| t.pow(2.0));
    check_unary(&device, FloatDType::F16, &data, tol, |t| t.clamp(Some(-1.0), Some(1.0)), |t| t.clamp(Some(-1.0), Some(1.0)));
}

#[test]
fn float_unary_f16() {
    let Some(device) = device() else { return };
    if !device.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }

    let data = [0.25, 0.5, 1.0, 2.0, 4.0];
    let tol = 2e-2;

    macro_rules! check {
        ($method:ident) => {
            check_unary(&device, FloatDType::F16, &data, tol, |t| t.$method(), |t| t.$method());
        };
    }

    check!(exp);
    check!(ln);
    check!(sin);
    check!(cos);
    check!(tanh);
    check!(sqr);
    check!(sqrt);
    check!(recip);
    check!(gelu);
    check!(gelu_erf);
    check!(erf);
    check!(relu);
    check!(silu);
    check!(sigmoid);
    check!(floor);
    check!(ceil);
    check!(round);

    let negative = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    check_unary(&device, FloatDType::F16, &negative, tol, |t| t.leaky_relu(0.1), |t| t.leaky_relu(0.1));
}

#[test]
fn unary_inplace_f16() {
    let Some(device) = device() else { return };
    if !device.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }

    let data = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let tol = 2e-2;
    check_unary_inplace(&device, FloatDType::F16, &data, tol, |t| t.neg_().map(|_| ()), |t| t.neg_().map(|_| ()));
    check_unary_inplace(&device, FloatDType::F16, &data, tol, |t| t.abs_().map(|_| ()), |t| t.abs_().map(|_| ()));
}

#[test]
fn float_unary_inplace_f16() {
    let Some(device) = device() else { return };
    if !device.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }

    let data = [0.25, 0.5, 1.0, 2.0, 4.0];
    let tol = 2e-2;
    check_unary_inplace(&device, FloatDType::F16, &data, tol, |t| t.exp_(), |t| t.exp_());
    check_unary_inplace(&device, FloatDType::F16, &data, tol, |t| t.sqrt_(), |t| t.sqrt_());
}

#[test]
fn cmp_f16() {
    let Some(device) = device() else { return };
    if !device.supports_f16() {
        eprintln!("skip wgpu f16 test: shader-f16 not supported");
        return;
    }

    let a = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let b = [-1.0, 0.0, 0.0, 1.0, 3.0];

    check_cmp(&device, FloatDType::F16, &a, &b, |x, y| x.eq(y), |x, y| x.eq(y));
    check_cmp(&device, FloatDType::F16, &a, &b, |x, y| x.ne(y), |x, y| x.ne(y));
    check_cmp(&device, FloatDType::F16, &a, &b, |x, y| x.lt(y), |x, y| x.lt(y));
    check_cmp(&device, FloatDType::F16, &a, &b, |x, y| x.le(y), |x, y| x.le(y));
    check_cmp(&device, FloatDType::F16, &a, &b, |x, y| x.gt(y), |x, y| x.gt(y));
    check_cmp(&device, FloatDType::F16, &a, &b, |x, y| x.ge(y), |x, y| x.ge(y));

    check_cmp_scalar(&device, FloatDType::F16, &a, 0.0, |x, s| x.eq_scalar(s), |x, s| x.eq_scalar(s));
    check_cmp_scalar(&device, FloatDType::F16, &a, 0.0, |x, s| x.lt_scalar(s), |x, s| x.lt_scalar(s));
    check_cmp_scalar(&device, FloatDType::F16, &a, 0.0, |x, s| x.ge_scalar(s), |x, s| x.ge_scalar(s));
}
