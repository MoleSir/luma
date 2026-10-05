# luma-flash-attn

Flash Attention CUDA kernels for luma (f32/f16/bf16, causal, GQA, forward only).

## Entry points

- `flash_attn_func` — batched attention, `q/k/v` of shape `(batch, seq, heads, head_size)`.
- `flash_attn_varlen_func` — packed variable-length attention.
- `flash_attn_with_kvcache` — attention over a paged KV cache (decode when `seqlen_q == 1`, prefill when `> 1`).

## Usage

```rust
use luma_cuda::Cuda;
use luma_flash_attn::flash_attn_func;
use luma_tensor::dtype::FloatDType;
use luma_tensor::Tensor;

let dev = Cuda::new(0).unwrap();
let (b, s, h, d) = (2, 8, 4, 64);
let nums = vec![0.1f64; b * s * h * d];

let q = Tensor::<Cuda>::from_slice(&nums, (b, s, h, d), (&dev, FloatDType::F32)).unwrap();
let k = q.clone();
let v = q.clone();

let out = flash_attn_func(&q, &k, &v, None, true, 0).unwrap();
assert_eq!(out.dims(), &[b, s, h, d]);
```

## Notes

- `head_size` must be 32/64/128; `q_heads` must be a multiple of `kv_heads` (GQA).
- Causal only, and there is no backward pass (inference only).
- Each entry point documents prefill / decode usage in its rustdoc; `start_pos`
  (on `flash_attn_func`) and `block_table` (on `flash_attn_varlen_func`) let you
  run decode over a concatenated or paged KV cache.
- Built with `nvcc`; without it the crate falls back to a stub that errors at runtime.

## License

MIT — see [LICENSE](LICENSE).
