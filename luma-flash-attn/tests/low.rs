use luma_flash_attn::{FlashAttnError, low::flash_attn_f32};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use std::sync::Arc;

fn default_scale(head_size: usize) -> f32 {
    1.0 / (head_size as f32).sqrt()
}

#[derive(Clone, Copy, PartialEq)]
enum MaskKind {
    None,
    Random,
    AllOnes,
}

#[derive(Clone, Copy)]
struct Case {
    batch: usize,
    sq: usize,
    skv: usize,
    hq: usize,
    hkv: usize,
    d: usize,
    start_pos: i32,
    mask: MaskKind,
}

impl Case {
    fn mask_len(&self) -> usize {
        self.batch * self.skv
    }

    /// 生成 mask：1 = 有效，0 = padding
    fn build_mask(&self, rng: &mut Rng) -> Vec<u8> {
        match self.mask {
            MaskKind::None => Vec::new(),
            MaskKind::AllOnes => vec![1u8; self.mask_len()],
            MaskKind::Random => (0..self.mask_len())
                .map(|_| if rng.next_f32() > -0.2 { 1 } else { 0 })
                .collect(),
        }
    }
}

fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    case: &Case,
    mask: Option<&[u8]>,
) -> Vec<f32> {
    let Case { batch, sq, skv, hq, hkv, d, start_pos, .. } = *case;
    let group = hq / hkv;
    let scale = 1.0 / (d as f64).sqrt();
    let mut out = vec![0.0f32; batch * sq * hq * d];

    for b in 0..batch {
        for h in 0..hq {
            let kh = h / group;
            for i in 0..sq {
                let g = start_pos + i as i32;

                // 该行 Q 是否有效：有 mask 时既要位置在范围内，又要 mask 为 1
                let q_valid = match mask {
                    Some(m) => g >= 0 && (g as usize) < skv && m[b * skv + g as usize] != 0,
                    None => true,
                };

                let q_off = ((b * sq + i) * hq + h) * d;

                let mut scores: Vec<(usize, f64)> = Vec::new();
                let mut m = f64::NEG_INFINITY;
                for j in 0..skv {
                    let mut visible = match mask {
                        Some(m) => m[b * skv + j] != 0,
                        None => true,
                    };
                    // causal：K 索引 j 必须 <= Q 的全局位置
                    if visible {
                        visible = g >= j as i32;
                    }
                    if !visible {
                        continue;
                    }
                    let k_off = ((b * skv + j) * hkv + kh) * d;
                    let mut s = 0.0f64;
                    for x in 0..d {
                        s += q[q_off + x] as f64 * k[k_off + x] as f64;
                    }
                    s *= scale;
                    m = m.max(s);
                    scores.push((j, s));
                }

                // 无效行、或整行被 mask/causal 屏蔽 → 输出 0
                if !q_valid || scores.is_empty() {
                    continue;
                }

                let mut denom = 0.0f64;
                let mut acc = vec![0.0f64; d];
                for &(j, s) in &scores {
                    let p = (s - m).exp();
                    denom += p;
                    let v_off = ((b * skv + j) * hkv + kh) * d;
                    for x in 0..d {
                        acc[x] += p * v[v_off + x] as f64;
                    }
                }
                for x in 0..d {
                    out[q_off + x] = (acc[x] / denom) as f32;
                }
            }
        }
    }
    out
}

fn context() -> Arc<CudaContext> {
    CudaContext::new(0).expect("需要可用的 CUDA device 0 才能运行 luma-flash-attn 测试")
}

fn assert_close(got: &[f32], want: &[f32], case: &Case, seed: u64) {
    assert_eq!(got.len(), want.len(), "长度不一致");
    let mut worst = 0.0f32;
    for (idx, (&a, &b)) in got.iter().zip(want).enumerate() {
        let diff = (a - b).abs();
        // f32 + --use_fast_math 的 __expf 引入一定误差，取宽容一些的阈值
        let tol = 3e-3 + 3e-3 * b.abs();
        worst = worst.max(diff);
        assert!(
            diff <= tol,
            "seed={seed} case(batch={} sq={} skv={} hq={} hkv={} d={} sp={} mask={:?}) \
             idx={idx} got={a} want={b} diff={diff} tol={tol}",
            case.batch, case.sq, case.skv, case.hq, case.hkv, case.d, case.start_pos, case.mask as u8
        );
    }
    assert!(worst.is_finite(), "输出出现 NaN/Inf：worst={worst}");
}

fn run_case(seed: u64, case: &Case) -> Vec<f32> {
    let ctx = context();
    let stream = ctx.default_stream();

    let q_sz = case.batch * case.sq * case.hq * case.d;
    let kv_sz = case.batch * case.skv * case.hkv * case.d;

    let mut rng = Rng::new(seed);
    let q_h: Vec<f32> = (0..q_sz).map(|_| rng.next_f32()).collect();
    let k_h: Vec<f32> = (0..kv_sz).map(|_| rng.next_f32()).collect();
    let v_h: Vec<f32> = (0..kv_sz).map(|_| rng.next_f32()).collect();
    let mask_h = case.build_mask(&mut rng);
    let mask = match case.mask {
        MaskKind::None => None,
        _ => Some(stream.clone_htod(&mask_h).unwrap()),
    };

    let q = stream.clone_htod(&q_h).unwrap();
    let k = stream.clone_htod(&k_h).unwrap();
    let v = stream.clone_htod(&v_h).unwrap();
    let mut o = stream.alloc_zeros::<f32>(q_sz).unwrap();

    flash_attn_f32(
        &q,
        &k,
        &v,
        &mut o,
        case.batch as i32,
        case.sq as i32,
        case.skv as i32,
        case.hq as i32,
        case.hkv as i32,
        case.d as i32,
        case.start_pos,
        default_scale(case.d),
        mask.as_ref(),
        &stream,
    )
    .unwrap_or_else(|e| panic!("flash_attn_f32 失败: {e}"));

    stream.synchronize().unwrap();
    let got = stream.clone_dtoh(&o).unwrap();

    let want = reference(
        &q_h,
        &k_h,
        &v_h,
        case,
        if case.mask == MaskKind::None { None } else { Some(&mask_h) },
    );
    assert_close(&got, &want, case, seed);

    got
}

#[allow(clippy::too_many_arguments)]
fn case(batch: usize, sq: usize, skv: usize, hq: usize, hkv: usize, d: usize, start_pos: i32, mask: MaskKind) -> Case {
    Case { batch, sq, skv, hq, hkv, d, start_pos, mask }
}

#[test]
fn mha_head32_exact_tiles() {
    // seq 恰好是 Br/Bc=64 的整数倍
    run_case(1, &case(2, 64, 64, 4, 4, 32, 0, MaskKind::None));
}

#[test]
fn mha_head64_unaligned_seq() {
    // 非 64 对齐，测试边界加载与越界保护
    run_case(2, &case(1, 100, 100, 4, 4, 64, 0, MaskKind::None));
}

#[test]
fn mha_head32_uneven_seq() {
    run_case(3, &case(3, 129, 200, 6, 6, 32, 0, MaskKind::None));
}

#[test]
fn gqa_head128() {
    // GQA：8 个 Q head 共享 2 个 KV head，head_size=128（需 >48KB 动态 smem opt-in）
    run_case(4, &case(2, 70, 70, 8, 2, 128, 0, MaskKind::None));
}

#[test]
fn causal_zero_start_pos() {
    // 第 0 行只能看到第 0 个 K，避免 NaN
    run_case(5, &case(1, 32, 32, 1, 1, 64, 0, MaskKind::None));
}

#[test]
fn mask_random_padding() {
    run_case(6, &case(2, 128, 128, 4, 4, 32, 0, MaskKind::Random));
}

#[test]
fn mask_all_ones_equals_no_mask() {
    let seed = 7;
    let base = case(1, 64, 64, 2, 2, 64, 0, MaskKind::None);
    let masked = case(1, 64, 64, 2, 2, 64, 0, MaskKind::AllOnes);
    let a = run_case(seed, &base);
    let b = run_case(seed, &masked);
    assert_close(&b, &a, &masked, seed);
}

#[test]
fn kv_cache_start_pos_offset() {
    // 生成阶段：已有 start_pos 个 KV，本步只算 sq 个新 Q
    run_case(8, &case(1, 40, 72, 8, 2, 64, 32, MaskKind::None));
}

#[test]
fn start_pos_unaligned_to_block() {
    run_case(9, &case(2, 50, 120, 4, 4, 32, 70, MaskKind::None));
}

#[test]
fn gqa_mask_with_start_pos() {
    run_case(10, &case(2, 33, 100, 8, 4, 128, 67, MaskKind::Random));
}

#[test]
fn single_token_decode() {
    // 典型 decode：q_seq_len = 1
    run_case(11, &case(1, 1, 96, 4, 4, 64, 95, MaskKind::None));
}

#[allow(clippy::type_complexity)]
fn tiny_inputs() -> (Arc<CudaContext>, Arc<CudaStream>, CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>) {
    let ctx = context();
    let stream = ctx.default_stream();
    let q = stream.alloc_zeros::<f32>(1).unwrap();
    let k = stream.alloc_zeros::<f32>(1).unwrap();
    let v = stream.alloc_zeros::<f32>(1).unwrap();
    let o = stream.alloc_zeros::<f32>(1).unwrap();
    (ctx, stream, q, k, v, o)
}

#[test]
fn err_invalid_head_size() {
    let (_ctx, stream, q, k, v, mut o) = tiny_inputs();
    let e = flash_attn_f32(&q, &k, &v, &mut o, 1, 1, 1, 1, 1, 48, 0, default_scale(48), None, &stream).unwrap_err();
    assert!(matches!(e, FlashAttnError::InvalidHeadSize(_)), "got {e:?}");
}

#[test]
fn err_non_multiple_heads() {
    let (_ctx, stream, q, k, v, mut o) = tiny_inputs();
    let e = flash_attn_f32(&q, &k, &v, &mut o, 1, 1, 1, 3, 2, 32, 0, default_scale(32), None, &stream).unwrap_err();
    assert!(matches!(e, FlashAttnError::InvalidShape(_)), "got {e:?}");
}

#[test]
fn err_negative_start_pos() {
    let (_ctx, stream, q, k, v, mut o) = tiny_inputs();
    let e = flash_attn_f32(&q, &k, &v, &mut o, 1, 1, 1, 1, 1, 32, -1, default_scale(32), None, &stream).unwrap_err();
    assert!(matches!(e, FlashAttnError::InvalidStartPos(_)), "got {e:?}");
}

#[test]
fn err_zero_shape() {
    let (_ctx, stream, q, k, v, mut o) = tiny_inputs();
    let e = flash_attn_f32(&q, &k, &v, &mut o, 0, 1, 1, 1, 1, 32, 0, default_scale(32), None, &stream).unwrap_err();
    assert!(matches!(e, FlashAttnError::InvalidShape(_)), "got {e:?}");
}

#[test]
fn err_buffer_too_small() {
    // 参数合法但 GPU 缓冲区过小，必须在调用 kernel 前被拦截
    let (_ctx, stream, q, k, v, mut o) = tiny_inputs();
    let e = flash_attn_f32(&q, &k, &v, &mut o, 1, 64, 64, 4, 4, 32, 0, default_scale(32), None, &stream).unwrap_err();
    assert!(matches!(e, FlashAttnError::BufferTooSmall(_)), "got {e:?}");
}

#[test]
fn err_mask_buffer_too_small() {
    let ctx = context();
    let stream = ctx.default_stream();
    let mask = stream.alloc_zeros::<u8>(1).unwrap();
    // 形状合法且 q/k/v/o 足够大，但 mask 过小
    let n = 64 * 4 * 32;
    let q = stream.alloc_zeros::<f32>(n).unwrap();
    let k = stream.alloc_zeros::<f32>(n).unwrap();
    let v = stream.alloc_zeros::<f32>(n).unwrap();
    let mut o = stream.alloc_zeros::<f32>(n).unwrap();
    let e = flash_attn_f32(&q, &k, &v, &mut o, 1, 64, 64, 4, 4, 32, 0, default_scale(32), Some(&mask), &stream).unwrap_err();
    assert!(matches!(e, FlashAttnError::BufferTooSmall(_)), "got {e:?}");
}

#[test]
fn err_null_ptr_from_c_abi() {
    // 安全封装无法传空指针，直接测 C ABI 的返回码与错误串
    let ctx = context();
    let stream = ctx.default_stream();
    let code = unsafe {
        luma_flash_attn::ffi::flash_attn_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            1,
            1,
            1,
            1,
            1,
            32,
            0,
            1.0,
            std::ptr::null(),
            stream.cu_stream() as *mut core::ffi::c_void,
        )
    };
    assert_eq!(code, luma_flash_attn::ffi::FLASH_ATTN_ERR_NULL_PTR);
    let err = FlashAttnError::from_status(code).unwrap_err();
    assert!(matches!(err, FlashAttnError::NullPtr(_)), "got {err:?}");
    // 出错时 last_error 应给出可读信息
    let msg = unsafe { core::ffi::CStr::from_ptr(luma_flash_attn::ffi::flash_attn_last_error()) };
    assert!(!msg.to_string_lossy().is_empty(), "last_error 为空");
}

// ---------------------------------------------------------------------------
// 确定性 PRNG（xorshift64*），避免给本 crate 引入 rand 依赖
// ---------------------------------------------------------------------------
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// 均匀分布于 [-1, 1)
    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // 24 位
        (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}