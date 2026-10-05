//! Tensor-level API tests for the python-`flash_attn`-aligned entry points.

use luma_cuda::Cuda;
use luma_flash_attn::{flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache};
use luma_tensor::Tensor;
use luma_tensor::dtype::FloatDType;
use luma_tensor::tensor::IntTensor;

// ---------------------------------------------------------------------------
// Deterministic data + reference implementations (f64).
// ---------------------------------------------------------------------------

fn data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let x = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
            ((x >> 40) as f64 / (1u64 << 24) as f64) * 2.0 - 1.0
        })
        .collect()
}

/// Batched attention reference. `q` is `(b, sq, hq, d)`, `k/v` `(b, skv, hkv, d)`.
/// Query token `i` is at global position `start_pos + i` and sees `j <= that`.
#[allow(clippy::too_many_arguments)]
fn ref_attention(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    b: usize,
    sq: usize,
    skv: usize,
    hq: usize,
    hkv: usize,
    d: usize,
    scale: f64,
    start_pos: usize,
) -> Vec<f64> {
    let group = hq / hkv;
    let mut out = vec![0.0f64; b * sq * hq * d];
    for bi in 0..b {
        for h in 0..hq {
            let kh = h / group;
            for i in 0..sq {
                let g = start_pos + i;
                let q_off = ((bi * sq + i) * hq + h) * d;

                let mut scores = Vec::with_capacity(skv);
                let mut max = f64::NEG_INFINITY;
                for j in 0..skv {
                    if j > g {
                        continue;
                    }
                    let k_off = ((bi * skv + j) * hkv + kh) * d;
                    let mut dot = 0.0;
                    for x in 0..d {
                        dot += q[q_off + x] * k[k_off + x];
                    }
                    let s = dot * scale;
                    max = max.max(s);
                    scores.push((j, s));
                }
                if scores.is_empty() {
                    continue;
                }
                let mut denom = 0.0;
                let mut acc = vec![0.0f64; d];
                for &(j, s) in &scores {
                    let p = (s - max).exp();
                    denom += p;
                    let v_off = ((bi * skv + j) * hkv + kh) * d;
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

fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let mut worst = 0.0f64;
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        let diff = (a - b).abs();
        worst = worst.max(diff);
        let t = tol + tol * b.abs();
        assert!(diff <= t, "{what}: idx {i} got {a} want {b} diff {diff} tol {t}");
    }
    assert!(worst.is_finite(), "{what}: non-finite output");
}

fn f32_tensor(dev: &Cuda, values: &[f64], shape: impl Into<luma_tensor::Shape>) -> Tensor<Cuda> {
    Tensor::<Cuda>::from_slice(values, shape, (dev, FloatDType::F32)).unwrap()
}

fn tensor_dtype(dev: &Cuda, values: &[f64], shape: impl Into<luma_tensor::Shape>, dt: FloatDType) -> Tensor<Cuda> {
    Tensor::<Cuda>::from_slice(values, shape, (dev, dt)).unwrap()
}

// ---------------------------------------------------------------------------
// flash_attn_func
// ---------------------------------------------------------------------------

#[test]
fn func_matches_reference() {
    let dev = Cuda::new(0).unwrap();
    let cases = [
        (1usize, 4usize, 4usize, 2usize, 2usize, 32usize),
        (2, 3, 5, 2, 1, 64),
        (1, 5, 5, 4, 1, 128),
        (2, 33, 33, 8, 2, 64),
    ];
    for (ci, (b, sq, skv, hq, hkv, d)) in cases.into_iter().enumerate() {
        let (qd, kd, vd) = (
            data(b * sq * hq * d, ci as u64 + 1),
            data(b * skv * hkv * d, ci as u64 + 10),
            data(b * skv * hkv * d, ci as u64 + 20),
        );
        let q = f32_tensor(&dev, &qd, (b, sq, hq, d));
        let k = f32_tensor(&dev, &kd, (b, skv, hkv, d));
        let v = f32_tensor(&dev, &vd, (b, skv, hkv, d));

        let out = flash_attn_func(&q, &k, &v, None, true).unwrap();
        assert_eq!(out.dims(), &[b, sq, hq, d]);
        let scale = 1.0 / (d as f64).sqrt();
        let want = ref_attention(&qd, &kd, &vd, b, sq, skv, hq, hkv, d, scale, 0);
        assert_close(&out.to_vec().unwrap(), &want, 3e-3, "flash_attn_func");
    }
}

#[test]
fn func_custom_scale() {
    let dev = Cuda::new(0).unwrap();
    let (b, sq, skv, hq, hkv, d) = (1usize, 6usize, 6usize, 4usize, 2usize, 64usize);
    let (qd, kd, vd) = (data(b * sq * hq * d, 31), data(b * skv * hkv * d, 32), data(b * skv * hkv * d, 33));
    let q = f32_tensor(&dev, &qd, (b, sq, hq, d));
    let k = f32_tensor(&dev, &kd, (b, skv, hkv, d));
    let v = f32_tensor(&dev, &vd, (b, skv, hkv, d));

    let scale = 0.7;
    let out = flash_attn_func(&q, &k, &v, Some(scale), true).unwrap();
    let want = ref_attention(&qd, &kd, &vd, b, sq, skv, hq, hkv, d, scale, 0);
    assert_close(&out.to_vec().unwrap(), &want, 3e-3, "flash_attn_func custom scale");
}

#[test]
fn func_non_causal_errors() {
    let dev = Cuda::new(0).unwrap();
    let q = f32_tensor(&dev, &data(2 * 4 * 2 * 32, 1), (2, 4, 2, 32));
    let k = f32_tensor(&dev, &data(2 * 4 * 32, 2), (2, 4, 1, 32));
    let v = f32_tensor(&dev, &data(2 * 4 * 32, 3), (2, 4, 1, 32));
    assert!(flash_attn_func(&q, &k, &v, None, false).is_err());
}

#[test]
fn func_rejects_f64() {
    let dev = Cuda::new(0).unwrap();
    let q = Tensor::<Cuda>::from_slice(&data(2 * 4 * 2 * 32, 1), (2, 4, 2, 32), (&dev, FloatDType::F64)).unwrap();
    let k = Tensor::<Cuda>::from_slice(&data(2 * 4 * 32, 2), (2, 4, 1, 32), (&dev, FloatDType::F64)).unwrap();
    let v = Tensor::<Cuda>::from_slice(&data(2 * 4 * 32, 3), (2, 4, 1, 32), (&dev, FloatDType::F64)).unwrap();
    assert!(flash_attn_func(&q, &k, &v, None, true).is_err());
}

#[test]
fn func_backward_not_implemented() {
    let dev = Cuda::new(0).unwrap();
    let (b, sq, hq, hkv, d) = (1usize, 4usize, 2usize, 1usize, 32usize);
    let q = f32_tensor(&dev, &data(b * sq * hq * d, 1), (b, sq, hq, d));
    let k = f32_tensor(&dev, &data(b * sq * hkv * d, 2), (b, sq, hkv, d));
    let v = f32_tensor(&dev, &data(b * sq * hkv * d, 3), (b, sq, hkv, d));
    q.set_requires_grad(true);
    let out = flash_attn_func(&q, &k, &v, None, true).unwrap();
    assert!(out.requires_grad());
    assert!(out.sum_all().unwrap().backward().is_err());
}

// ---------------------------------------------------------------------------
// flash_attn_varlen_func
// ---------------------------------------------------------------------------

#[test]
fn varlen_matches_reference() {
    let dev = Cuda::new(0).unwrap();
    // Two packed sequences, q_len == k_len, GQA 4:2, d=32.
    let (hq, hkv, d) = (4usize, 2usize, 32usize);
    let lens = [3usize, 5usize];
    let total: usize = lens.iter().sum();
    let mut cu = vec![0i32];
    for &l in &lens {
        cu.push(cu.last().unwrap() + l as i32);
    }

    let qd = data(total * hq * d, 41);
    let kd = data(total * hkv * d, 42);
    let vd = data(total * hkv * d, 43);
    let q = f32_tensor(&dev, &qd, (total, hq, d));
    let k = f32_tensor(&dev, &kd, (total, hkv, d));
    let v = f32_tensor(&dev, &vd, (total, hkv, d));
    let cu_t = IntTensor::<Cuda>::from_vec_i32(cu.clone(), (cu.len(),), &dev).unwrap();

    let out = flash_attn_varlen_func(&q, &k, &v, &cu_t, &cu_t, 5, 5, None, true).unwrap();
    assert_eq!(out.dims(), &[total, hq, d]);
    let got = out.to_vec().unwrap();

    let scale = 1.0 / (d as f64).sqrt();
    let mut want = vec![0.0f64; total * hq * d];
    let mut off = 0usize;
    for &l in &lens {
        let qs = qd[off * hq * d..(off + l) * hq * d].to_vec();
        let ks = kd[off * hkv * d..(off + l) * hkv * d].to_vec();
        let vs = vd[off * hkv * d..(off + l) * hkv * d].to_vec();
        let r = ref_attention(&qs, &ks, &vs, 1, l, l, hq, hkv, d, scale, 0);
        want[off * hq * d..(off + l) * hq * d].copy_from_slice(&r);
        off += l;
    }
    assert_close(&got, &want, 3e-3, "flash_attn_varlen_func");
}

#[test]
fn varlen_prefix_kv_matches_reference() {
    let dev = Cuda::new(0).unwrap();
    // seq0: q_len=3, k_len=5 (2-token prefix); seq1: q_len=4, k_len=4.
    let (hq, hkv, d) = (2usize, 2usize, 64usize);
    let q_lens = [3usize, 4usize];
    let k_lens = [5usize, 4usize];
    let total_q: usize = q_lens.iter().sum();
    let total_k: usize = k_lens.iter().sum();
    let cu_q = [0i32, 3, 7];
    let cu_k = [0i32, 5, 9];

    let qd = data(total_q * hq * d, 51);
    let kd = data(total_k * hkv * d, 52);
    let vd = data(total_k * hkv * d, 53);
    let q = f32_tensor(&dev, &qd, (total_q, hq, d));
    let k = f32_tensor(&dev, &kd, (total_k, hkv, d));
    let v = f32_tensor(&dev, &vd, (total_k, hkv, d));
    let cu_q_t = IntTensor::<Cuda>::from_vec_i32(cu_q.to_vec(), (3,), &dev).unwrap();
    let cu_k_t = IntTensor::<Cuda>::from_vec_i32(cu_k.to_vec(), (3,), &dev).unwrap();

    let out = flash_attn_varlen_func(&q, &k, &v, &cu_q_t, &cu_k_t, 4, 5, None, true).unwrap();
    let got = out.to_vec().unwrap();

    let scale = 1.0 / (d as f64).sqrt();
    let mut want = vec![0.0f64; total_q * hq * d];
    let (mut q_off, mut k_off, mut o_off) = (0usize, 0usize, 0usize);
    for s in 0..q_lens.len() {
        let ql = q_lens[s];
        let kl = k_lens[s];
        let qs = qd[q_off * hq * d..(q_off + ql) * hq * d].to_vec();
        let ks = kd[k_off * hkv * d..(k_off + kl) * hkv * d].to_vec();
        let vs = vd[k_off * hkv * d..(k_off + kl) * hkv * d].to_vec();
        let r = ref_attention(&qs, &ks, &vs, 1, ql, kl, hq, hkv, d, scale, kl - ql);
        want[o_off * hq * d..(o_off + ql) * hq * d].copy_from_slice(&r);
        q_off += ql;
        k_off += kl;
        o_off += ql;
    }
    assert_close(&got, &want, 3e-3, "flash_attn_varlen_func prefix");
}

// ---------------------------------------------------------------------------
// flash_attn_with_kvcache
// ---------------------------------------------------------------------------

#[test]
fn kvcache_matches_reference() {
    let dev = Cuda::new(0).unwrap();
    let (batch, block_size, blocks_per_seq) = (2usize, 4usize, 2usize);
    let num_blocks = batch * blocks_per_seq;
    let (hq, hkv, d) = (4usize, 2usize, 64usize);
    let cache_seqlens = [5i32, 7i32];

    let mut block_table = Vec::with_capacity(batch * blocks_per_seq);
    for s in 0..batch {
        for blk in 0..blocks_per_seq {
            block_table.push((s * blocks_per_seq + blk) as i32);
        }
    }

    let qd = data(batch * hq * d, 61);
    let kd = data(num_blocks * block_size * hkv * d, 62);
    let vd = data(num_blocks * block_size * hkv * d, 63);
    let q = f32_tensor(&dev, &qd, (batch, 1, hq, d));
    let k_cache = f32_tensor(&dev, &kd, (num_blocks, block_size, hkv, d));
    let v_cache = f32_tensor(&dev, &vd, (num_blocks, block_size, hkv, d));
    let lens = IntTensor::<Cuda>::from_vec_i32(cache_seqlens.to_vec(), (batch,), &dev).unwrap();
    let bt = IntTensor::<Cuda>::from_vec_i32(block_table.clone(), (batch, blocks_per_seq), &dev).unwrap();

    let out = flash_attn_with_kvcache(&q, &k_cache, &v_cache, &lens, &bt, None).unwrap();
    assert_eq!(out.dims(), &[batch, 1, hq, d]);
    let got = out.to_vec().unwrap();

    let scale = 1.0 / (d as f64).sqrt();
    let group = hq / hkv;
    let mut want = vec![0.0f64; batch * hq * d];
    for s in 0..batch {
        let l = cache_seqlens[s] as usize;
        for h in 0..hq {
            let kh = h / group;
            let q_off = (s * hq + h) * d;
            let mut scores = Vec::with_capacity(l);
            let mut max = f64::NEG_INFINITY;
            for j in 0..l {
                let phys = block_table[s * blocks_per_seq + j / block_size] as usize;
                let slot = phys * block_size + (j % block_size);
                let k_off = (slot * hkv + kh) * d;
                let mut dot = 0.0;
                for x in 0..d {
                    dot += qd[q_off + x] * kd[k_off + x];
                }
                let sc = dot * scale;
                max = max.max(sc);
                scores.push((slot, sc));
            }
            let mut denom = 0.0;
            let mut acc = vec![0.0f64; d];
            for &(slot, sc) in &scores {
                let p = (sc - max).exp();
                denom += p;
                let v_off = (slot * hkv + kh) * d;
                for x in 0..d {
                    acc[x] += p * vd[v_off + x];
                }
            }
            for x in 0..d {
                want[(s * hq + h) * d + x] = acc[x] / denom;
            }
        }
    }
    assert_close(&got, &want, 3e-3, "flash_attn_with_kvcache");
}

// ---------------------------------------------------------------------------
// f16 / bf16
// ---------------------------------------------------------------------------

const HALF_CASES: [(FloatDType, f64); 2] = [(FloatDType::F16, 1e-2), (FloatDType::BF16, 5e-2)];

#[test]
fn func_matches_reference_half() {
    let dev = Cuda::new(0).unwrap();
    let shapes = [
        (1usize, 4usize, 4usize, 2usize, 2usize, 32usize),
        (2, 3, 5, 2, 1, 64),
        (1, 5, 5, 4, 1, 128),
    ];
    for (dt, tol) in HALF_CASES {
        for (ci, (b, sq, skv, hq, hkv, d)) in shapes.into_iter().enumerate() {
            let (qd, kd, vd) = (
                data(b * sq * hq * d, ci as u64 + 71),
                data(b * skv * hkv * d, ci as u64 + 81),
                data(b * skv * hkv * d, ci as u64 + 91),
            );
            let q = tensor_dtype(&dev, &qd, (b, sq, hq, d), dt);
            let k = tensor_dtype(&dev, &kd, (b, skv, hkv, d), dt);
            let v = tensor_dtype(&dev, &vd, (b, skv, hkv, d), dt);

            let out = flash_attn_func(&q, &k, &v, None, true).unwrap();
            assert_eq!(out.dtype(), dt, "output dtype not preserved ({dt:?})");
            assert_eq!(out.dims(), &[b, sq, hq, d]);
            let scale = 1.0 / (d as f64).sqrt();
            let want = ref_attention(&qd, &kd, &vd, b, sq, skv, hq, hkv, d, scale, 0);
            assert_close(&out.to_vec().unwrap(), &want, tol, &format!("flash_attn_func {dt:?}"));
        }
    }
}

#[test]
fn varlen_matches_reference_half() {
    let dev = Cuda::new(0).unwrap();
    let (hq, hkv, d) = (4usize, 2usize, 32usize);
    let lens = [3usize, 5usize];
    let total: usize = lens.iter().sum();
    let mut cu = vec![0i32];
    for &l in &lens {
        cu.push(cu.last().unwrap() + l as i32);
    }
    let (qd, kd, vd) = (data(total * hq * d, 141), data(total * hkv * d, 142), data(total * hkv * d, 143));
    let cu_t = IntTensor::<Cuda>::from_vec_i32(cu.clone(), (cu.len(),), &dev).unwrap();

    let scale = 1.0 / (d as f64).sqrt();
    let mut want = vec![0.0f64; total * hq * d];
    let mut off = 0usize;
    for &l in &lens {
        let qs = qd[off * hq * d..(off + l) * hq * d].to_vec();
        let ks = kd[off * hkv * d..(off + l) * hkv * d].to_vec();
        let vs = vd[off * hkv * d..(off + l) * hkv * d].to_vec();
        let r = ref_attention(&qs, &ks, &vs, 1, l, l, hq, hkv, d, scale, 0);
        want[off * hq * d..(off + l) * hq * d].copy_from_slice(&r);
        off += l;
    }

    for (dt, tol) in HALF_CASES {
        let q = tensor_dtype(&dev, &qd, (total, hq, d), dt);
        let k = tensor_dtype(&dev, &kd, (total, hkv, d), dt);
        let v = tensor_dtype(&dev, &vd, (total, hkv, d), dt);
        let out = flash_attn_varlen_func(&q, &k, &v, &cu_t, &cu_t, 5, 5, None, true).unwrap();
        assert_eq!(out.dtype(), dt, "output dtype not preserved ({dt:?})");
        assert_close(&out.to_vec().unwrap(), &want, tol, &format!("flash_attn_varlen_func {dt:?}"));
    }
}

#[test]
fn kvcache_matches_reference_half() {
    let dev = Cuda::new(0).unwrap();
    let (batch, block_size, blocks_per_seq) = (2usize, 4usize, 2usize);
    let num_blocks = batch * blocks_per_seq;
    let (hq, hkv, d) = (4usize, 2usize, 64usize);
    let cache_seqlens = [5i32, 7i32];

    let mut block_table = Vec::with_capacity(batch * blocks_per_seq);
    for s in 0..batch {
        for blk in 0..blocks_per_seq {
            block_table.push((s * blocks_per_seq + blk) as i32);
        }
    }

    let qd = data(batch * hq * d, 161);
    let kd = data(num_blocks * block_size * hkv * d, 162);
    let vd = data(num_blocks * block_size * hkv * d, 163);
    let lens = IntTensor::<Cuda>::from_vec_i32(cache_seqlens.to_vec(), (batch,), &dev).unwrap();
    let bt = IntTensor::<Cuda>::from_vec_i32(block_table.clone(), (batch, blocks_per_seq), &dev).unwrap();

    let scale = 1.0 / (d as f64).sqrt();
    let group = hq / hkv;
    let mut want = vec![0.0f64; batch * hq * d];
    for s in 0..batch {
        let l = cache_seqlens[s] as usize;
        for h in 0..hq {
            let kh = h / group;
            let q_off = (s * hq + h) * d;
            let mut scores = Vec::with_capacity(l);
            let mut max = f64::NEG_INFINITY;
            for j in 0..l {
                let phys = block_table[s * blocks_per_seq + j / block_size] as usize;
                let slot = phys * block_size + (j % block_size);
                let k_off = (slot * hkv + kh) * d;
                let mut dot = 0.0;
                for x in 0..d {
                    dot += qd[q_off + x] * kd[k_off + x];
                }
                let sc = dot * scale;
                max = max.max(sc);
                scores.push((slot, sc));
            }
            let mut denom = 0.0;
            let mut acc = vec![0.0f64; d];
            for &(slot, sc) in &scores {
                let p = (sc - max).exp();
                denom += p;
                let v_off = (slot * hkv + kh) * d;
                for x in 0..d {
                    acc[x] += p * vd[v_off + x];
                }
            }
            for x in 0..d {
                want[(s * hq + h) * d + x] = acc[x] / denom;
            }
        }
    }

    for (dt, tol) in HALF_CASES {
        let q = tensor_dtype(&dev, &qd, (batch, 1, hq, d), dt);
        let k_cache = tensor_dtype(&dev, &kd, (num_blocks, block_size, hkv, d), dt);
        let v_cache = tensor_dtype(&dev, &vd, (num_blocks, block_size, hkv, d), dt);
        let out = flash_attn_with_kvcache(&q, &k_cache, &v_cache, &lens, &bt, None).unwrap();
        assert_eq!(out.dtype(), dt, "output dtype not preserved ({dt:?})");
        assert_close(&out.to_vec().unwrap(), &want, tol, &format!("flash_attn_with_kvcache {dt:?}"));
    }
}
