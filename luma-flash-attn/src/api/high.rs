use std::any::Any;

use half::{bf16, f16};
use luma_cuda::{Cuda, CudaFloatSlice, CudaFloatStorage, CudaIntSlice};
use luma_tensor::tensor::IntTensor;
use luma_tensor::{CustomOp3, CustomOpError, Device, Float, Shape, Tensor};

use crate::api::low as lowlevel;
use crate::error::FlashAttnError;

// ---------------------------------------------------------------------------
// Public, CUDA-only, python-`flash_attn`-aligned entry points.
// ---------------------------------------------------------------------------

/// `flash_attn.flash_attn_func`: batched scaled dot-product attention.
///
/// - `q`: `(batch, q_seq_len,  q_num_heads,  head_size)`
/// - `k`: `(batch, kv_seq_len, kv_num_heads, head_size)`
/// - `v`: same shape as `k`
/// - returns: `(batch, q_seq_len, q_num_heads, head_size)`
///
/// `head_size` must be 32/64/128 and `q_num_heads` a multiple of
/// `kv_num_heads` (GQA). The kernel is causal-only for now, so `causal` must be
/// `true`; `softmax_scale=None` means `1 / sqrt(head_size)`. Supported dtypes
/// are f32/f16/bf16.
pub fn flash_attn_func(
    q: &Tensor<Cuda, Float>,
    k: &Tensor<Cuda, Float>,
    v: &Tensor<Cuda, Float>,
    softmax_scale: Option<f64>,
    causal: bool,
) -> Result<Tensor<Cuda, Float>, FlashAttnError> {
    if !causal {
        return Err(FlashAttnError::Unsupported(
            "flash_attn_func: non-causal is not supported yet (kernels are causal-only)".into(),
        ));
    }
    Ok(q.custom_op3(k, v, Box::new(FlashAttnFuncOp::new(softmax_scale, causal)))?)
}

/// `flash_attn.flash_attn_varlen_func`: packed variable-length attention.
///
/// - `q`: `(total_q,  q_num_heads,  head_size)`
/// - `k`: `(total_kv, kv_num_heads, head_size)`
/// - `v`: same shape as `k`
/// - `cu_seqlens_q` / `cu_seqlens_k`: `(batch + 1,)` `i32`, monotonic offsets
/// - returns: `(total_q, q_num_heads, head_size)`
///
/// Causal is bottom-right aligned (a query token sees the last key of its seq).
/// Supported dtypes are f32/f16/bf16.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_varlen_func(
    q: &Tensor<Cuda, Float>,
    k: &Tensor<Cuda, Float>,
    v: &Tensor<Cuda, Float>,
    cu_seqlens_q: &IntTensor<Cuda>,
    cu_seqlens_k: &IntTensor<Cuda>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: Option<f64>,
    causal: bool,
) -> Result<Tensor<Cuda, Float>, FlashAttnError> {
    if !causal {
        return Err(FlashAttnError::Unsupported(
            "flash_attn_varlen_func: non-causal is not supported yet (kernels are causal-only)".into(),
        ));
    }
    let op = FlashAttnVarlenOp {
        cu_seqlens_q: cu_seqlens_q.clone(),
        cu_seqlens_k: cu_seqlens_k.clone(),
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
    };
    Ok(q.custom_op3(k, v, Box::new(op))?)
}

/// `flash_attn.flash_attn_with_kvcache`: single-token decode over a paged KV cache.
///
/// - `q`: `(batch, 1, q_num_heads, head_size)`
/// - `k_cache` / `v_cache`: `(num_blocks, block_size, kv_num_heads, head_size)`
/// - `cache_seqlens`: `(batch,)` `i32` (valid cached length per sequence)
/// - `block_table`: `(batch, max_blocks)` `i32` (logical block -> physical block)
/// - returns: `(batch, 1, q_num_heads, head_size)`
///
/// Only `seqlen_q == 1` and a paged cache are supported in this version.
/// Supported dtypes are f32/f16/bf16.
pub fn flash_attn_with_kvcache(
    q: &Tensor<Cuda, Float>,
    k_cache: &Tensor<Cuda, Float>,
    v_cache: &Tensor<Cuda, Float>,
    cache_seqlens: &IntTensor<Cuda>,
    block_table: &IntTensor<Cuda>,
    softmax_scale: Option<f64>,
) -> Result<Tensor<Cuda, Float>, FlashAttnError> {
    let op = FlashAttnKvcacheOp {
        cache_seqlens: cache_seqlens.clone(),
        block_table: block_table.clone(),
        softmax_scale,
    };
    Ok(q.custom_op3(k_cache, v_cache, Box::new(op))?)
}

// ---------------------------------------------------------------------------
// CustomOp3 implementations (forward only; backward not implemented).
// ---------------------------------------------------------------------------

/// Shared op used by both [`flash_attn_func`] and `luma-nn`'s generic
/// `scaled_dot_product_attention` (which calls it via `custom_op3`).
pub struct FlashAttnFuncOp {
    softmax_scale: Option<f64>,
    causal: bool,
    start_pos: usize,
}

impl FlashAttnFuncOp {
    pub fn new(softmax_scale: Option<f64>, causal: bool) -> Self {
        Self { softmax_scale, causal, start_pos: 0 }
    }

    /// Like [`FlashAttnFuncOp::new`], but with an explicit causal offset (used by
    /// `luma-nn` for KV-cache decoding). The python-aligned [`flash_attn_func`]
    /// always uses `start_pos == 0`.
    pub fn with_start_pos(softmax_scale: Option<f64>, causal: bool, start_pos: usize) -> Self {
        Self { softmax_scale, causal, start_pos }
    }
}

impl<D: Device> CustomOp3<D> for FlashAttnFuncOp {
    fn name(&self) -> String {
        "flash_attn_func".to_string()
    }

    fn forward(
        &self,
        q: &Tensor<D, Float>,
        k: &Tensor<D, Float>,
        v: &Tensor<D, Float>,
    ) -> Result<(D::FloatStorage, Shape), CustomOpError> {
        let q = as_cuda_float(q)
            .ok_or_else(|| CustomOpError::msg("flash_attn_func: only CUDA tensors are supported"))?;
        let k = as_cuda_float(k)
            .ok_or_else(|| CustomOpError::msg("flash_attn_func: only CUDA tensors are supported"))?;
        let v = as_cuda_float(v)
            .ok_or_else(|| CustomOpError::msg("flash_attn_func: only CUDA tensors are supported"))?;

        if !self.causal {
            return Err(CustomOpError::msg("flash_attn_func: non-causal is not supported yet"));
        }

        let (batch, q_seq_len, q_num_heads, head_size) = q.dims4()?;
        let (k_batch, kv_seq_len, kv_num_heads, k_head_size) = k.dims4()?;
        if k_batch != batch {
            return Err(CustomOpError::msg("flash_attn_func: batch mismatch between q and k"));
        }
        if k_head_size != head_size {
            return Err(CustomOpError::msg("flash_attn_func: q/k head_size mismatch"));
        }
        if k.shape() != v.shape() {
            return Err(CustomOpError::msg("flash_attn_func: k/v shape mismatch"));
        }
        if kv_num_heads == 0 || q_num_heads % kv_num_heads != 0 {
            return Err(CustomOpError::msg(
                "flash_attn_func: q_num_heads must be a multiple of kv_num_heads",
            ));
        }
        check_head_size("flash_attn_func", head_size)?;
        let scale = self.softmax_scale.unwrap_or_else(|| 1.0 / (head_size as f64).sqrt()) as f32;

        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let device = q.device().clone();
        let stream = device.stream();
        let n = batch * q_seq_len * q_num_heads * head_size;

        let q_g = q.storage_read()?;
        let k_g = k.storage_read()?;
        let v_g = v.storage_read()?;

        let o_slice = match (&q_g.slice, &k_g.slice, &v_g.slice) {
            (CudaFloatSlice::F32(qs), CudaFloatSlice::F32(ks), CudaFloatSlice::F32(vs)) => {
                let mut o = alloc_f32(&device, n, "flash_attn_func")?;
                lowlevel::flash_attn::<f32>(
                    qs, ks, vs, &mut o, batch as i32, q_seq_len as i32, kv_seq_len as i32,
                    q_num_heads as i32, kv_num_heads as i32, head_size as i32,
                    self.start_pos as i32, scale, None, &stream,
                )?;
                CudaFloatSlice::F32(o)
            }
            (CudaFloatSlice::F16(qs), CudaFloatSlice::F16(ks), CudaFloatSlice::F16(vs)) => {
                let mut o = alloc_f16(&device, n, "flash_attn_func")?;
                lowlevel::flash_attn::<f16>(
                    qs, ks, vs, &mut o, batch as i32, q_seq_len as i32, kv_seq_len as i32,
                    q_num_heads as i32, kv_num_heads as i32, head_size as i32,
                    self.start_pos as i32, scale, None, &stream,
                )?;
                CudaFloatSlice::F16(o)
            }
            (CudaFloatSlice::BF16(qs), CudaFloatSlice::BF16(ks), CudaFloatSlice::BF16(vs)) => {
                let mut o = alloc_bf16(&device, n, "flash_attn_func")?;
                lowlevel::flash_attn::<bf16>(
                    qs, ks, vs, &mut o, batch as i32, q_seq_len as i32, kv_seq_len as i32,
                    q_num_heads as i32, kv_num_heads as i32, head_size as i32,
                    self.start_pos as i32, scale, None, &stream,
                )?;
                CudaFloatSlice::BF16(o)
            }
            _ => return Err(CustomOpError::msg("flash_attn_func: q/k/v must share dtype f32/f16/bf16")),
        };

        let storage = CudaFloatStorage { slice: o_slice, device };
        let shape = Shape::from(vec![batch, q_seq_len, q_num_heads, head_size]);
        Ok((erase_float_storage::<D>(storage)?, shape))
    }

    fn backward(
        &self,
        _arg1: &Tensor<D, Float>,
        _arg2: &Tensor<D, Float>,
        _arg3: &Tensor<D, Float>,
        _ret: &Tensor<D, Float>,
        _ret_grad: &Tensor<D, Float>,
    ) -> Result<(Tensor<D, Float>, Tensor<D, Float>, Tensor<D, Float>), CustomOpError> {
        Err(CustomOpError::msg("flash_attn_func: backward is not implemented"))
    }
}

pub struct FlashAttnVarlenOp {
    cu_seqlens_q: IntTensor<Cuda>,
    cu_seqlens_k: IntTensor<Cuda>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: Option<f64>,
    causal: bool,
}

impl CustomOp3<Cuda> for FlashAttnVarlenOp {
    fn name(&self) -> String {
        "flash_attn_varlen_func".to_string()
    }

    fn forward(
        &self,
        q: &Tensor<Cuda, Float>,
        k: &Tensor<Cuda, Float>,
        v: &Tensor<Cuda, Float>,
    ) -> Result<(CudaFloatStorage, Shape), CustomOpError> {
        if !self.causal {
            return Err(CustomOpError::msg("flash_attn_varlen_func: non-causal is not supported yet"));
        }

        let (total_q, q_heads, head_size) = q.dims3()?;
        let (_total_kv, kv_heads, k_head_size) = k.dims3()?;
        if k_head_size != head_size {
            return Err(CustomOpError::msg("flash_attn_varlen_func: q/k head_size mismatch"));
        }
        if k.shape() != v.shape() {
            return Err(CustomOpError::msg("flash_attn_varlen_func: k/v shape mismatch"));
        }
        if kv_heads == 0 || q_heads % kv_heads != 0 {
            return Err(CustomOpError::msg(
                "flash_attn_varlen_func: q_heads must be a multiple of kv_heads",
            ));
        }
        check_head_size("flash_attn_varlen_func", head_size)?;
        if self.max_seqlen_q == 0 || self.max_seqlen_k == 0 {
            return Err(CustomOpError::msg("flash_attn_varlen_func: max_seqlen_q/k must be > 0"));
        }

        let cu_q_len = self.cu_seqlens_q.shape().element_count();
        let cu_k_len = self.cu_seqlens_k.shape().element_count();
        if cu_q_len != cu_k_len || cu_q_len == 0 {
            return Err(CustomOpError::msg(
                "flash_attn_varlen_func: cu_seqlens_q/k must have the same non-zero length",
            ));
        }
        let num_seqs = cu_q_len - 1;
        let scale = self.softmax_scale.unwrap_or_else(|| 1.0 / (head_size as f64).sqrt()) as f32;

        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let device = q.device().clone();
        let stream = device.stream();
        let n = total_q * q_heads * head_size;

        let q_g = q.storage_read()?;
        let k_g = k.storage_read()?;
        let v_g = v.storage_read()?;
        let cu_q_g = self.cu_seqlens_q.storage_read()?;
        let cu_k_g = self.cu_seqlens_k.storage_read()?;
        let (CudaIntSlice::I32(cu_q_slice), CudaIntSlice::I32(cu_k_slice)) =
            (&cu_q_g.slice, &cu_k_g.slice)
        else {
            return Err(CustomOpError::msg("flash_attn_varlen_func: cu_seqlens must be i32"));
        };

        let o_slice = match (&q_g.slice, &k_g.slice, &v_g.slice) {
            (CudaFloatSlice::F32(qs), CudaFloatSlice::F32(ks), CudaFloatSlice::F32(vs)) => {
                let mut o = alloc_f32(&device, n, "flash_attn_varlen_func")?;
                lowlevel::flash_attn_varlen::<f32>(
                    qs, ks, vs, &mut o, cu_q_slice, cu_k_slice,
                    num_seqs as i32, self.max_seqlen_q as i32, self.max_seqlen_k as i32,
                    q_heads as i32, kv_heads as i32, head_size as i32, scale, &stream,
                )?;
                CudaFloatSlice::F32(o)
            }
            (CudaFloatSlice::F16(qs), CudaFloatSlice::F16(ks), CudaFloatSlice::F16(vs)) => {
                let mut o = alloc_f16(&device, n, "flash_attn_varlen_func")?;
                lowlevel::flash_attn_varlen::<f16>(
                    qs, ks, vs, &mut o, cu_q_slice, cu_k_slice,
                    num_seqs as i32, self.max_seqlen_q as i32, self.max_seqlen_k as i32,
                    q_heads as i32, kv_heads as i32, head_size as i32, scale, &stream,
                )?;
                CudaFloatSlice::F16(o)
            }
            (CudaFloatSlice::BF16(qs), CudaFloatSlice::BF16(ks), CudaFloatSlice::BF16(vs)) => {
                let mut o = alloc_bf16(&device, n, "flash_attn_varlen_func")?;
                lowlevel::flash_attn_varlen::<bf16>(
                    qs, ks, vs, &mut o, cu_q_slice, cu_k_slice,
                    num_seqs as i32, self.max_seqlen_q as i32, self.max_seqlen_k as i32,
                    q_heads as i32, kv_heads as i32, head_size as i32, scale, &stream,
                )?;
                CudaFloatSlice::BF16(o)
            }
            _ => return Err(CustomOpError::msg("flash_attn_varlen_func: q/k/v must share dtype f32/f16/bf16")),
        };

        let storage = CudaFloatStorage { slice: o_slice, device };
        let shape = Shape::from(vec![total_q, q_heads, head_size]);
        Ok((storage, shape))
    }

    fn backward(
        &self,
        _arg1: &Tensor<Cuda, Float>,
        _arg2: &Tensor<Cuda, Float>,
        _arg3: &Tensor<Cuda, Float>,
        _ret: &Tensor<Cuda, Float>,
        _ret_grad: &Tensor<Cuda, Float>,
    ) -> Result<(Tensor<Cuda, Float>, Tensor<Cuda, Float>, Tensor<Cuda, Float>), CustomOpError> {
        Err(CustomOpError::msg("flash_attn_varlen_func: backward is not implemented"))
    }
}

pub struct FlashAttnKvcacheOp {
    cache_seqlens: IntTensor<Cuda>,
    block_table: IntTensor<Cuda>,
    softmax_scale: Option<f64>,
}

impl CustomOp3<Cuda> for FlashAttnKvcacheOp {
    fn name(&self) -> String {
        "flash_attn_with_kvcache".to_string()
    }

    fn forward(
        &self,
        q: &Tensor<Cuda, Float>,
        k_cache: &Tensor<Cuda, Float>,
        v_cache: &Tensor<Cuda, Float>,
    ) -> Result<(CudaFloatStorage, Shape), CustomOpError> {
        let (batch, seqlen_q, q_heads, head_size) = q.dims4()?;
        if seqlen_q != 1 {
            return Err(CustomOpError::msg(
                "flash_attn_with_kvcache: only seqlen_q == 1 is supported",
            ));
        }
        let (_num_blocks, block_size, kv_heads, k_head_size) = k_cache.dims4()?;
        if k_head_size != head_size {
            return Err(CustomOpError::msg("flash_attn_with_kvcache: q/k_cache head_size mismatch"));
        }
        if k_cache.shape() != v_cache.shape() {
            return Err(CustomOpError::msg("flash_attn_with_kvcache: k_cache/v_cache shape mismatch"));
        }
        if kv_heads == 0 || q_heads % kv_heads != 0 {
            return Err(CustomOpError::msg(
                "flash_attn_with_kvcache: q_heads must be a multiple of kv_heads",
            ));
        }
        check_head_size("flash_attn_with_kvcache", head_size)?;
        let (bt_rows, max_blocks) = self.block_table.dims2()?;
        if bt_rows != batch {
            return Err(CustomOpError::msg("flash_attn_with_kvcache: block_table batch mismatch"));
        }
        if self.cache_seqlens.shape().element_count() != batch {
            return Err(CustomOpError::msg(
                "flash_attn_with_kvcache: cache_seqlens length must equal batch",
            ));
        }
        let scale = self.softmax_scale.unwrap_or_else(|| 1.0 / (head_size as f64).sqrt()) as f32;

        let q = q.contiguous()?;
        let k_cache = k_cache.contiguous()?;
        let v_cache = v_cache.contiguous()?;

        let device = q.device().clone();
        let stream = device.stream();
        let n = batch * q_heads * head_size;

        let q_g = q.storage_read()?;
        let k_g = k_cache.storage_read()?;
        let v_g = v_cache.storage_read()?;
        let lens_g = self.cache_seqlens.storage_read()?;
        let bt_g = self.block_table.storage_read()?;
        let (CudaIntSlice::I32(lens_slice), CudaIntSlice::I32(bt_slice)) =
            (&lens_g.slice, &bt_g.slice)
        else {
            return Err(CustomOpError::msg(
                "flash_attn_with_kvcache: cache_seqlens/block_table must be i32",
            ));
        };

        let o_slice = match (&q_g.slice, &k_g.slice, &v_g.slice) {
            (CudaFloatSlice::F32(qs), CudaFloatSlice::F32(ks), CudaFloatSlice::F32(vs)) => {
                let mut o = alloc_f32(&device, n, "flash_attn_with_kvcache")?;
                lowlevel::flash_attn_with_kvcache::<f32>(
                    qs, ks, vs, &mut o, bt_slice, lens_slice,
                    batch as i32, q_heads as i32, kv_heads as i32, head_size as i32,
                    block_size as i32, max_blocks as i32, scale, &stream,
                )?;
                CudaFloatSlice::F32(o)
            }
            (CudaFloatSlice::F16(qs), CudaFloatSlice::F16(ks), CudaFloatSlice::F16(vs)) => {
                let mut o = alloc_f16(&device, n, "flash_attn_with_kvcache")?;
                lowlevel::flash_attn_with_kvcache::<f16>(
                    qs, ks, vs, &mut o, bt_slice, lens_slice,
                    batch as i32, q_heads as i32, kv_heads as i32, head_size as i32,
                    block_size as i32, max_blocks as i32, scale, &stream,
                )?;
                CudaFloatSlice::F16(o)
            }
            (CudaFloatSlice::BF16(qs), CudaFloatSlice::BF16(ks), CudaFloatSlice::BF16(vs)) => {
                let mut o = alloc_bf16(&device, n, "flash_attn_with_kvcache")?;
                lowlevel::flash_attn_with_kvcache::<bf16>(
                    qs, ks, vs, &mut o, bt_slice, lens_slice,
                    batch as i32, q_heads as i32, kv_heads as i32, head_size as i32,
                    block_size as i32, max_blocks as i32, scale, &stream,
                )?;
                CudaFloatSlice::BF16(o)
            }
            _ => return Err(CustomOpError::msg("flash_attn_with_kvcache: q/k/v must share dtype f32/f16/bf16")),
        };

        let storage = CudaFloatStorage { slice: o_slice, device };
        let shape = Shape::from(vec![batch, 1, q_heads, head_size]);
        Ok((storage, shape))
    }

    fn backward(
        &self,
        _arg1: &Tensor<Cuda, Float>,
        _arg2: &Tensor<Cuda, Float>,
        _arg3: &Tensor<Cuda, Float>,
        _ret: &Tensor<Cuda, Float>,
        _ret_grad: &Tensor<Cuda, Float>,
    ) -> Result<(Tensor<Cuda, Float>, Tensor<Cuda, Float>, Tensor<Cuda, Float>), CustomOpError> {
        Err(CustomOpError::msg("flash_attn_with_kvcache: backward is not implemented"))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn check_head_size(op: &str, head_size: usize) -> Result<(), CustomOpError> {
    if matches!(head_size, 32 | 64 | 128) {
        Ok(())
    } else {
        Err(CustomOpError::msg(format!("{op}: head_size must be 32/64/128, got {head_size}")))
    }
}

fn alloc_f32(device: &Cuda, n: usize, op: &str) -> Result<cudarc::driver::CudaSlice<f32>, CustomOpError> {
    device.alloc::<f32>(n).map_err(|e| CustomOpError::msg(format!("{op} alloc: {e}")))
}

fn alloc_f16(device: &Cuda, n: usize, op: &str) -> Result<cudarc::driver::CudaSlice<f16>, CustomOpError> {
    device.alloc::<f16>(n).map_err(|e| CustomOpError::msg(format!("{op} alloc: {e}")))
}

fn alloc_bf16(device: &Cuda, n: usize, op: &str) -> Result<cudarc::driver::CudaSlice<bf16>, CustomOpError> {
    device.alloc::<bf16>(n).map_err(|e| CustomOpError::msg(format!("{op} alloc: {e}")))
}

fn as_cuda_float<D: Device>(t: &Tensor<D, Float>) -> Option<&Tensor<Cuda, Float>> {
    (t as &dyn Any).downcast_ref::<Tensor<Cuda, Float>>()
}

fn erase_float_storage<D: Device>(storage: CudaFloatStorage) -> Result<D::FloatStorage, CustomOpError> {
    let boxed: Box<dyn Any> = Box::new(storage);
    boxed
        .downcast::<D::FloatStorage>()
        .map(|b| *b)
        .map_err(|_| CustomOpError::msg("flash_attn: internal device mismatch"))
}
