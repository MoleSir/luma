//! Smoke tests for the newer `luma-nn` modules.

use luma_cpu::Cpu;
use luma_nn::{
    AdaptiveAvgPool2d, AvgPool2d, AvgPool2dConfig, Conv2d, Conv2dConfig, GroupNorm, GroupNormConfig, Init, MaxPool2d, MultiheadAttention,
    MultiheadAttentionConfig,
};
use luma_tensor::{FloatDType, Tensor};

fn opts(cpu: &Cpu) -> (&Cpu, FloatDType) {
    (cpu, FloatDType::F32)
}

#[test]
fn conv2d_module_forward() {
    let cpu = Cpu::default();
    let config = Conv2dConfig { weight_init: Init::Ones, bias: false, ..Conv2dConfig::new(1, 1, (2, 2)) };
    let conv = Conv2d::<Cpu>::from_config(&config, opts(&cpu)).unwrap();

    let x = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], (1, 1, 3, 3), opts(&cpu))
        .unwrap();
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    let v = y.to_vec().unwrap();
    for (got, want) in v.iter().zip([12.0, 16.0, 24.0, 28.0]) {
        assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
    }
}

#[test]
fn group_norm_module_forward() {
    let cpu = Cpu::default();
    let config = GroupNormConfig { affine: false, ..GroupNormConfig::new(2, 4) };
    let norm = GroupNorm::<Cpu>::from_config(&config, opts(&cpu)).unwrap();

    let data: Vec<f64> = (1..=24).map(|v| v as f64).collect();
    let x = Tensor::<Cpu>::from_slice(&data, (2, 4, 3), opts(&cpu)).unwrap();
    let y = norm.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 4, 3]);

    let expected = luma_nn::functional::group_norm(&x, 2, None, None, 1e-5).unwrap();
    assert!(y.allclose(&expected, 1e-6, 1e-6).unwrap());
}

#[test]
fn multihead_attention_module_forward() {
    let cpu = Cpu::default();
    let config = MultiheadAttentionConfig { bias: false, causal: false, ..MultiheadAttentionConfig::new(4, 2) };
    let attn = MultiheadAttention::<Cpu>::from_config(&config, opts(&cpu)).unwrap();

    let data: Vec<f64> = (0..(2 * 3 * 4)).map(|i| ((i % 7) as f64 - 3.0) / 7.0).collect();
    let x = Tensor::<Cpu>::from_slice(&data, (2, 3, 4), opts(&cpu)).unwrap();
    let y = attn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 3, 4]);
    assert!(y.to_vec().unwrap().iter().all(|v| v.is_finite()));

    let y2 = attn.forward_with_start_pos(&x, 0).unwrap();
    assert!(y.allclose(&y2, 1e-6, 1e-6).unwrap());
}

#[test]
fn max_pool2d_module_forward() {
    let cpu = Cpu::default();
    let pool = MaxPool2d::<Cpu>::new((2, 2));
    let data: Vec<f64> = (1..=16).map(|v| v as f64).collect();
    let x = Tensor::<Cpu>::from_slice(&data, (1, 1, 4, 4), opts(&cpu)).unwrap();
    let y = pool.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    let v = y.to_vec().unwrap();
    for (got, want) in v.iter().zip([6.0, 8.0, 14.0, 16.0]) {
        assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
    }
}

#[test]
fn avg_pool2d_module_forward() {
    let cpu = Cpu::default();
    let config = AvgPool2dConfig { stride: (1, 1), padding: (1, 1), ..AvgPool2dConfig::new((2, 2)) };
    let pool = AvgPool2d::<Cpu>::from_config(&config);
    let x = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], (1, 1, 3, 3), opts(&cpu)).unwrap();
    let y = pool.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4, 4]);
    let expected = luma_nn::functional::avg_pool2d(&x, (2, 2), (1, 1), (1, 1), false, true).unwrap();
    assert!(y.allclose(&expected, 1e-6, 1e-6).unwrap());
}

#[test]
fn adaptive_avg_pool2d_module_forward() {
    let cpu = Cpu::default();
    let pool = AdaptiveAvgPool2d::<Cpu>::new((1, 1));
    let data: Vec<f64> = (1..=16).map(|v| v as f64).collect();
    let x = Tensor::<Cpu>::from_slice(&data, (1, 1, 4, 4), opts(&cpu)).unwrap();
    let y = pool.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 1, 1]);
    assert!((y.to_vec().unwrap()[0] - 8.5).abs() < 1e-6);
}
