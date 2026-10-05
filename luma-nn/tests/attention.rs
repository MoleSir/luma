//! Tests for `scaled_dot_product_attention`.
//!
//! - `Math` path is checked against a plain f64 reference on CPU.
//! - `Flash` must error on non-CUDA devices.
//! - With `--features cuda`, `Flash` must agree with `Math` on GPU.

use luma_cpu::Cpu;
use luma_nn::NnError;
use luma_nn::functional::{AttentionConfig, scaled_dot_product_attention};
use luma_tensor::dtype::FloatDType;
use luma_tensor::Tensor;

#[derive(Clone, Copy)]
struct Case {
    batch: usize,
    sq: usize,
    skv: usize,
    hq: usize,
    hkv: usize,
    d: usize,
    causal: bool,
    start_pos: usize,
}

fn data(n: usize) -> Vec<f64> {
    (0..n).map(|i| (((i * 7 + 3) % 17) as f64) / 17.0 - 0.5).collect()
}

#[allow(clippy::too_many_arguments)]
fn reference(q: &[f64], k: &[f64], v: &[f64], case: &Case) -> Vec<f64> {
    let Case { batch, sq, skv, hq, hkv, d, causal, start_pos } = *case;
    let group = hq / hkv;
    let scale = 1.0 / (d as f64).sqrt();
    let mut out = vec![0.0f64; batch * sq * hq * d];

    for b in 0..batch {
        for h in 0..hq {
            let kh = h / group;
            for i in 0..sq {
                let g = start_pos + i;
                let q_off = ((b * sq + i) * hq + h) * d;
                let vis: Vec<usize> = (0..skv).filter(|&j| !causal || j <= g).collect();

                let mut scores = Vec::with_capacity(vis.len());
                let mut max = f64::NEG_INFINITY;
                for &j in &vis {
                    let k_off = ((b * skv + j) * hkv + kh) * d;
                    let mut s = 0.0;
                    for x in 0..d {
                        s += q[q_off + x] * k[k_off + x];
                    }
                    s *= scale;
                    max = max.max(s);
                    scores.push(s);
                }

                let mut denom = 0.0;
                let mut acc = vec![0.0f64; d];
                for (idx, &j) in vis.iter().enumerate() {
                    let p = (scores[idx] - max).exp();
                    denom += p;
                    let v_off = ((b * skv + j) * hkv + kh) * d;
                    for x in 0..d {
                        acc[x] += p * v[v_off + x];
                    }
                }
                for x in 0..d {
                    out[q_off + x] = acc[x] / denom;
                }
            }
        }
    }
    out
}

fn assert_close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol, "idx {i}: got {x}, want {y}");
    }
}

fn math_case(case: &Case) -> (Vec<f64>, Vec<f64>) {
    let Case { batch, sq, skv, hq, hkv, d, .. } = *case;
    let qd = data(batch * sq * hq * d);
    let kd = data(batch * skv * hkv * d);
    let vd = data(batch * skv * hkv * d);
    let cpu = Cpu::default();

    let q = Tensor::<Cpu>::from_slice(&qd, (batch, sq, hq, d), (&cpu, FloatDType::F32)).unwrap();
    let k = Tensor::<Cpu>::from_slice(&kd, (batch, skv, hkv, d), (&cpu, FloatDType::F32)).unwrap();
    let v = Tensor::<Cpu>::from_slice(&vd, (batch, skv, hkv, d), (&cpu, FloatDType::F32)).unwrap();

    let cfg = AttentionConfig::math().causal(case.causal).start_pos(case.start_pos);
    let got = scaled_dot_product_attention(&q, &k, &v, &cfg).unwrap().to_vec().unwrap();
    let want = reference(&qd, &kd, &vd, case);
    (got, want)
}

const BASE: Case =
    Case { batch: 2, sq: 3, skv: 5, hq: 2, hkv: 1, d: 4, causal: true, start_pos: 2 };

#[test]
fn math_causal_gqa_start_pos() {
    let (got, want) = math_case(&BASE);
    assert_close(&got, &want, 1e-4);
}

#[test]
fn math_non_causal_mha() {
    let case = Case { causal: false, hkv: 2, start_pos: 0, ..BASE };
    let (got, want) = math_case(&case);
    assert_close(&got, &want, 1e-4);
}

#[test]
fn math_causal_start_pos_zero() {
    let case = Case { batch: 1, sq: 5, skv: 5, hq: 2, hkv: 2, d: 8, causal: true, start_pos: 0 };
    let (got, want) = math_case(&case);
    assert_close(&got, &want, 1e-4);
}

#[test]
fn math_head_size_32_gqa() {
    let case = Case { batch: 1, sq: 4, skv: 4, hq: 4, hkv: 1, d: 32, causal: true, start_pos: 0 };
    let (got, want) = math_case(&case);
    assert_close(&got, &want, 1e-4);
}

#[test]
fn math_gradients_flow() {
    let (batch, sq, skv, hq, hkv, d) = (1, 3, 3, 2, 1, 4);
    let cpu = Cpu::default();
    let qd = data(batch * sq * hq * d);
    let kd = data(batch * skv * hkv * d);
    let vd = data(batch * skv * hkv * d);
    let q = Tensor::<Cpu>::from_slice(&qd, (batch, sq, hq, d), (&cpu, FloatDType::F32)).unwrap();
    let k = Tensor::<Cpu>::from_slice(&kd, (batch, skv, hkv, d), (&cpu, FloatDType::F32)).unwrap();
    let v = Tensor::<Cpu>::from_slice(&vd, (batch, skv, hkv, d), (&cpu, FloatDType::F32)).unwrap();
    q.set_requires_grad(true);
    k.set_requires_grad(true);
    v.set_requires_grad(true);

    let out = scaled_dot_product_attention(&q, &k, &v, &AttentionConfig::math()).unwrap();
    assert!(out.requires_grad());
    let grads = out.sum_all().unwrap().backward().unwrap();
    assert!(grads.get(&q).is_some());
    assert!(grads.get(&k).is_some());
    assert!(grads.get(&v).is_some());
}

#[test]
fn flash_on_cpu_errors() {
    let case = BASE;
    let (batch, sq, skv, hq, hkv, d) = (case.batch, case.sq, case.skv, case.hq, case.hkv, case.d);
    let cpu = Cpu::default();
    let qd = data(batch * sq * hq * d);
    let kd = data(batch * skv * hkv * d);
    let vd = data(batch * skv * hkv * d);
    let q = Tensor::<Cpu>::from_slice(&qd, (batch, sq, hq, d), (&cpu, FloatDType::F32)).unwrap();
    let k = Tensor::<Cpu>::from_slice(&kd, (batch, skv, hkv, d), (&cpu, FloatDType::F32)).unwrap();
    let v = Tensor::<Cpu>::from_slice(&vd, (batch, skv, hkv, d), (&cpu, FloatDType::F32)).unwrap();

    let err = scaled_dot_product_attention(&q, &k, &v, &AttentionConfig::flash()).err().expect("expected error");
    assert!(matches!(
        err,
        NnError::FlashAttentionRequiresCuda | NnError::FlashAttentionUnsupported(_)
    ));
}

#[test]
fn math_rejects_bad_gqa() {
    let cpu = Cpu::default();
    let (batch, sq, skv, hq, hkv, d) = (1, 2, 2, 3, 2, 4);
    let q = Tensor::<Cpu>::from_slice(&data(batch * sq * hq * d), (batch, sq, hq, d), (&cpu, FloatDType::F32)).unwrap();
    let k = Tensor::<Cpu>::from_slice(&data(batch * skv * hkv * d), (batch, skv, hkv, d), (&cpu, FloatDType::F32)).unwrap();
    let v = k.clone();
    assert!(scaled_dot_product_attention(&q, &k, &v, &AttentionConfig::math()).is_err());
}

// ---------------------------------------------------------------------------
// CUDA: Flash must agree with Math, and enforce its restrictions.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use luma_cuda::Cuda;

    fn run(dev: &Cuda, case: &Case) -> (Vec<f64>, Vec<f64>) {
        let Case { batch, sq, skv, hq, hkv, d, causal, start_pos } = *case;
        let qd = data(batch * sq * hq * d);
        let kd = data(batch * skv * hkv * d);
        let vd = data(batch * skv * hkv * d);

        let q = Tensor::<Cuda>::from_slice(&qd, (batch, sq, hq, d), (dev, FloatDType::F32)).unwrap();
        let k = Tensor::<Cuda>::from_slice(&kd, (batch, skv, hkv, d), (dev, FloatDType::F32)).unwrap();
        let v = Tensor::<Cuda>::from_slice(&vd, (batch, skv, hkv, d), (dev, FloatDType::F32)).unwrap();

        let math = scaled_dot_product_attention(
            &q, &k, &v,
            &AttentionConfig::math().causal(causal).start_pos(start_pos),
        )
        .unwrap()
        .to_vec()
        .unwrap();

        let flash = scaled_dot_product_attention(
            &q, &k, &v,
            &AttentionConfig::flash().causal(causal).start_pos(start_pos),
        )
        .unwrap()
        .to_vec()
        .unwrap();

        // flash and math must agree, and both must match the reference
        assert_close(&flash, &math, 3e-3);
        let want = reference(&qd, &kd, &vd, case);
        (flash, want)
    }

    #[test]
    fn flash_matches_reference() {
        let dev = Cuda::new(0).unwrap();
        let cases = [
            Case { batch: 1, sq: 4, skv: 4, hq: 2, hkv: 2, d: 32, causal: true, start_pos: 0 },
            Case { batch: 2, sq: 3, skv: 5, hq: 2, hkv: 1, d: 64, causal: true, start_pos: 2 },
            Case { batch: 1, sq: 5, skv: 5, hq: 4, hkv: 1, d: 128, causal: true, start_pos: 0 },
            Case { batch: 1, sq: 4, skv: 6, hq: 4, hkv: 2, d: 32, causal: true, start_pos: 2 },
        ];
        for case in cases {
            let (flash, want) = run(&dev, &case);
            assert_close(&flash, &want, 3e-3);
        }
    }

    #[test]
    fn flash_requires_grad_errors() {
        let dev = Cuda::new(0).unwrap();
        let (batch, sq, skv, hq, hkv, d) = (1, 4, 4, 2, 2, 32);
        let q = Tensor::<Cuda>::from_slice(&data(batch * sq * hq * d), (batch, sq, hq, d), (&dev, FloatDType::F32)).unwrap();
        let k = Tensor::<Cuda>::from_slice(&data(batch * skv * hkv * d), (batch, skv, hkv, d), (&dev, FloatDType::F32)).unwrap();
        let v = Tensor::<Cuda>::from_slice(&data(batch * skv * hkv * d), (batch, skv, hkv, d), (&dev, FloatDType::F32)).unwrap();
        q.set_requires_grad(true);
        let err = scaled_dot_product_attention(&q, &k, &v, &AttentionConfig::flash()).err().expect("expected error");
        assert!(matches!(err, NnError::FlashAttentionUnsupported(_)));
    }

    #[test]
    fn flash_rejects_bad_head_size() {
        let dev = Cuda::new(0).unwrap();
        let (batch, sq, skv, hq, hkv, d) = (1, 2, 2, 2, 2, 48);
        let q = Tensor::<Cuda>::from_slice(&data(batch * sq * hq * d), (batch, sq, hq, d), (&dev, FloatDType::F32)).unwrap();
        let k = Tensor::<Cuda>::from_slice(&data(batch * skv * hkv * d), (batch, skv, hkv, d), (&dev, FloatDType::F32)).unwrap();
        let v = k.clone();
        let err = scaled_dot_product_attention(&q, &k, &v, &AttentionConfig::flash()).err().expect("expected error");
        assert!(matches!(err, NnError::FlashAttentionUnsupported(_)));
    }
}
