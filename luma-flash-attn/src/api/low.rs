use core::ffi::c_void;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DeviceRepr};
use half::{bf16, f16};
use luma_tensor::FloatDType;

use crate::api::ffi;
use crate::error::FlashAttnError;

/// Storage types supported by the flash-attention kernels.
///
/// Compute always runs in f32; the type parameter only selects the device
/// buffer / C-ABI entry point (f32, f16 or bf16).
pub trait FlashDtype: DeviceRepr + Copy + 'static {
    const DTYPE: FloatDType;

    /// # Safety
    /// All pointers must reference valid device buffers usable on `stream`.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn ffi_batch(
        q: *const Self,
        k: *const Self,
        v: *const Self,
        o: *mut Self,
        batch: i32,
        q_seq_len: i32,
        kv_seq_len: i32,
        q_num_heads: i32,
        kv_num_heads: i32,
        head_size: i32,
        start_pos: i32,
        scale: f32,
        attn_mask: *const u8,
        stream: *mut c_void,
    ) -> i32;

    /// # Safety
    /// All pointers must reference valid device buffers usable on `stream`.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn ffi_varlen(
        q: *const Self,
        k: *const Self,
        v: *const Self,
        o: *mut Self,
        cu_seqlens_q: *const i32,
        cu_seqlens_k: *const i32,
        block_tables: *const i32,
        num_seqs: i32,
        max_seqlen_q: i32,
        max_seqlen_k: i32,
        q_heads: i32,
        kv_heads: i32,
        head_size: i32,
        block_size: i32,
        max_blocks: i32,
        k_paged: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;

    /// # Safety
    /// All pointers must reference valid device buffers usable on `stream`.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn ffi_kvcache(
        q: *const Self,
        k_cache: *const Self,
        v_cache: *const Self,
        o: *mut Self,
        block_tables: *const i32,
        context_lens: *const i32,
        num_seqs: i32,
        q_seq_len: i32,
        q_heads: i32,
        kv_heads: i32,
        head_size: i32,
        block_size: i32,
        max_blocks: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
}

macro_rules! impl_flash_dtype {
    ($T:ty, $dtype:path, $batch:ident, $varlen:ident, $kvcache:ident) => {
        impl FlashDtype for $T {
            const DTYPE: FloatDType = $dtype;

            unsafe fn ffi_batch(
                q: *const Self, k: *const Self, v: *const Self, o: *mut Self,
                batch: i32, q_seq_len: i32, kv_seq_len: i32, q_num_heads: i32, kv_num_heads: i32,
                head_size: i32, start_pos: i32, scale: f32, attn_mask: *const u8, stream: *mut c_void,
            ) -> i32 {
                unsafe {
                    ffi::$batch(q, k, v, o, batch, q_seq_len, kv_seq_len, q_num_heads, kv_num_heads,
                                head_size, start_pos, scale, attn_mask, stream)
                }
            }

            unsafe fn ffi_varlen(
                q: *const Self, k: *const Self, v: *const Self, o: *mut Self,
                cu_seqlens_q: *const i32, cu_seqlens_k: *const i32, block_tables: *const i32,
                num_seqs: i32, max_seqlen_q: i32, max_seqlen_k: i32,
                q_heads: i32, kv_heads: i32, head_size: i32,
                block_size: i32, max_blocks: i32, k_paged: i32, scale: f32, stream: *mut c_void,
            ) -> i32 {
                unsafe {
                    ffi::$varlen(q, k, v, o, cu_seqlens_q, cu_seqlens_k, block_tables,
                                 num_seqs, max_seqlen_q, max_seqlen_k, q_heads, kv_heads, head_size,
                                 block_size, max_blocks, k_paged, scale, stream)
                }
            }

            unsafe fn ffi_kvcache(
                q: *const Self, k_cache: *const Self, v_cache: *const Self, o: *mut Self,
                block_tables: *const i32, context_lens: *const i32,
                num_seqs: i32, q_seq_len: i32, q_heads: i32, kv_heads: i32, head_size: i32,
                block_size: i32, max_blocks: i32, scale: f32, stream: *mut c_void,
            ) -> i32 {
                unsafe {
                    ffi::$kvcache(q, k_cache, v_cache, o, block_tables, context_lens,
                                  num_seqs, q_seq_len, q_heads, kv_heads, head_size,
                                  block_size, max_blocks, scale, stream)
                }
            }
        }
    };
}

impl_flash_dtype!(f32, FloatDType::F32, flash_attn_f32, flash_attn_varlen_f32, flash_attn_with_kvcache_f32);
impl_flash_dtype!(f16, FloatDType::F16, flash_attn_f16, flash_attn_varlen_f16, flash_attn_with_kvcache_f16);
impl_flash_dtype!(bf16, FloatDType::BF16, flash_attn_bf16, flash_attn_varlen_bf16, flash_attn_with_kvcache_bf16);

/// Low-level batched flash attention forward (causal + optional padding mask, GQA).
///
/// Operates directly on device buffers. Layouts (contiguous):
/// `q/o` are `(batch, q_seq_len, q_num_heads, head_size)`, `k/v` are
/// `(batch, kv_seq_len, kv_num_heads, head_size)`, `attn_mask` is
/// `(batch, kv_seq_len)`.
///
/// The caller must `stream.synchronize()` (or use `clone_dtoh`) before reading
/// the result to observe execution-time errors.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn<T: FlashDtype>(
    q: &impl DevicePtr<T>,
    k: &impl DevicePtr<T>,
    v: &impl DevicePtr<T>,
    o: &mut CudaSlice<T>,
    batch: i32,
    q_seq_len: i32,
    kv_seq_len: i32,
    q_num_heads: i32,
    kv_num_heads: i32,
    head_size: i32,
    start_pos: i32,
    scale: f32,
    attn_mask: Option<&CudaSlice<u8>>,
    stream: &CudaStream,
) -> Result<(), FlashAttnError> {
    validate(q, k, v, &*o, batch, q_seq_len, kv_seq_len, q_num_heads, kv_num_heads, head_size, start_pos, attn_mask)?;

    let (q_ptr, _q_guard) = q.device_ptr(stream);
    let (k_ptr, _k_guard) = k.device_ptr(stream);
    let (v_ptr, _v_guard) = v.device_ptr(stream);
    let (o_ptr, _o_guard) = o.device_ptr(stream);

    let (mask_ptr, _mask_guard) = match attn_mask {
        Some(m) => {
            let (p, guard) = m.device_ptr(stream);
            (p as *const u8, Some(guard))
        }
        None => (std::ptr::null(), None),
    };

    let code = unsafe {
        T::ffi_batch(
            q_ptr as *const T,
            k_ptr as *const T,
            v_ptr as *const T,
            o_ptr as *mut T,
            batch,
            q_seq_len,
            kv_seq_len,
            q_num_heads,
            kv_num_heads,
            head_size,
            start_pos,
            scale,
            mask_ptr,
            stream.cu_stream() as *mut c_void,
        )
    };

    FlashAttnError::from_status(code)
}

/// `flash_attn::<f32>` convenience wrapper, retaining the historical name.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_f32(
    q: &impl DevicePtr<f32>,
    k: &impl DevicePtr<f32>,
    v: &impl DevicePtr<f32>,
    o: &mut CudaSlice<f32>,
    batch: i32,
    q_seq_len: i32,
    kv_seq_len: i32,
    q_num_heads: i32,
    kv_num_heads: i32,
    head_size: i32,
    start_pos: i32,
    scale: f32,
    attn_mask: Option<&CudaSlice<u8>>,
    stream: &CudaStream,
) -> Result<(), FlashAttnError> {
    flash_attn(q, k, v, o, batch, q_seq_len, kv_seq_len, q_num_heads, kv_num_heads, head_size, start_pos, scale, attn_mask, stream)
}

/// Low-level packed variable-length flash attention (causal, bottom-right aligned).
///
/// `q/o` are `(total_q, q_heads, head_size)` and `cu_seqlens_*` are
/// `(num_seqs + 1,)`.
///
/// - `block_tables == None`: contiguous K/V, `k/v` are
///   `(total_kv, kv_heads, head_size)`.
/// - `block_tables == Some(t)`: paged K/V, `k/v` are a cache
///   `(num_blocks, block_size, kv_heads, head_size)` addressed by `t`
///   `(num_seqs, max_blocks)`; `max_blocks` is inferred as `t.len() / num_seqs`.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_varlen<T: FlashDtype>(
    q: &impl DevicePtr<T>,
    k: &impl DevicePtr<T>,
    v: &impl DevicePtr<T>,
    o: &mut CudaSlice<T>,
    cu_seqlens_q: &impl DevicePtr<i32>,
    cu_seqlens_k: &impl DevicePtr<i32>,
    num_seqs: i32,
    max_seqlen_q: i32,
    max_seqlen_k: i32,
    q_heads: i32,
    kv_heads: i32,
    head_size: i32,
    scale: f32,
    block_tables: Option<&impl DevicePtr<i32>>,
    block_size: i32,
    stream: &CudaStream,
) -> Result<(), FlashAttnError> {
    let (q_ptr, _q_guard) = q.device_ptr(stream);
    let (k_ptr, _k_guard) = k.device_ptr(stream);
    let (v_ptr, _v_guard) = v.device_ptr(stream);
    let (o_ptr, _o_guard) = o.device_ptr(stream);
    let (cu_q_ptr, _cu_q_guard) = cu_seqlens_q.device_ptr(stream);
    let (cu_k_ptr, _cu_k_guard) = cu_seqlens_k.device_ptr(stream);

    let mut _bt_guard = None;
    let bt_ptr = match block_tables {
        Some(bt) => {
            let (p, g) = bt.device_ptr(stream);
            _bt_guard = Some(g);
            p as *const i32
        }
        None => std::ptr::null(),
    };
    let k_paged = if block_tables.is_some() { 1 } else { 0 };
    let max_blocks = match block_tables {
        Some(bt) if num_seqs > 0 => (bt.len() / num_seqs as usize) as i32,
        _ => 0,
    };

    let code = unsafe {
        T::ffi_varlen(
            q_ptr as *const T,
            k_ptr as *const T,
            v_ptr as *const T,
            o_ptr as *mut T,
            cu_q_ptr as *const i32,
            cu_k_ptr as *const i32,
            bt_ptr,
            num_seqs,
            max_seqlen_q,
            max_seqlen_k,
            q_heads,
            kv_heads,
            head_size,
            block_size,
            max_blocks,
            k_paged,
            scale,
            stream.cu_stream() as *mut c_void,
        )
    };

    FlashAttnError::from_status(code)
}

/// Low-level attention over a paged KV cache.
///
/// `q`/`o` are `(num_seqs, q_seq_len, q_heads, head_size)`; `q_seq_len == 1`
/// selects the decode kernel, `> 1` the prefill kernel. `k_cache`/`v_cache` are
/// `(num_blocks, block_size, kv_heads, head_size)`, `context_lens` is
/// `(num_seqs,)` = total cached length (including the q tokens, which must
/// already be present in the cache). Causal is bottom-right aligned.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_with_kvcache<T: FlashDtype>(
    q: &impl DevicePtr<T>,
    k_cache: &impl DevicePtr<T>,
    v_cache: &impl DevicePtr<T>,
    o: &mut CudaSlice<T>,
    block_tables: &impl DevicePtr<i32>,
    context_lens: &impl DevicePtr<i32>,
    num_seqs: i32,
    q_seq_len: i32,
    q_heads: i32,
    kv_heads: i32,
    head_size: i32,
    block_size: i32,
    max_blocks: i32,
    scale: f32,
    stream: &CudaStream,
) -> Result<(), FlashAttnError> {
    if q_seq_len <= 0 {
        return Err(FlashAttnError::InvalidShape(format!("q_seq_len={q_seq_len} must be > 0")));
    }
    let q_need = num_seqs as u64 * q_seq_len as u64 * q_heads as u64 * head_size as u64;
    check_len("q", q.len(), q_need)?;
    check_len("o", o.len(), q_need)?;

    let (q_ptr, _q_guard) = q.device_ptr(stream);
    let (k_ptr, _k_guard) = k_cache.device_ptr(stream);
    let (v_ptr, _v_guard) = v_cache.device_ptr(stream);
    let (o_ptr, _o_guard) = o.device_ptr(stream);
    let (bt_ptr, _bt_guard) = block_tables.device_ptr(stream);
    let (lens_ptr, _lens_guard) = context_lens.device_ptr(stream);

    let code = unsafe {
        T::ffi_kvcache(
            q_ptr as *const T,
            k_ptr as *const T,
            v_ptr as *const T,
            o_ptr as *mut T,
            bt_ptr as *const i32,
            lens_ptr as *const i32,
            num_seqs,
            q_seq_len,
            q_heads,
            kv_heads,
            head_size,
            block_size,
            max_blocks,
            scale,
            stream.cu_stream() as *mut c_void,
        )
    };

    FlashAttnError::from_status(code)
}

#[allow(clippy::too_many_arguments)]
fn validate<T: FlashDtype>(
    q: &impl DevicePtr<T>,
    k: &impl DevicePtr<T>,
    v: &impl DevicePtr<T>,
    o: &impl DevicePtr<T>,
    batch: i32,
    q_seq_len: i32,
    kv_seq_len: i32,
    q_num_heads: i32,
    kv_num_heads: i32,
    head_size: i32,
    start_pos: i32,
    attn_mask: Option<&CudaSlice<u8>>,
) -> Result<(), FlashAttnError> {
    if batch <= 0 || q_seq_len <= 0 || kv_seq_len <= 0 || q_num_heads <= 0 || kv_num_heads <= 0 {
        return Err(FlashAttnError::InvalidShape(format!(
            "batch={batch} q_seq_len={q_seq_len} kv_seq_len={kv_seq_len} \
             q_num_heads={q_num_heads} kv_num_heads={kv_num_heads} must be positive"
        )));
    }
    if q_num_heads % kv_num_heads != 0 {
        return Err(FlashAttnError::InvalidShape(format!(
            "q_num_heads={q_num_heads} must be a multiple of kv_num_heads={kv_num_heads} (GQA)"
        )));
    }
    if start_pos < 0 {
        return Err(FlashAttnError::InvalidStartPos(format!("start_pos={start_pos}")));
    }
    if !matches!(head_size, 32 | 64 | 128) {
        return Err(FlashAttnError::InvalidHeadSize(format!(
            "head_size={head_size} (only 32/64/128)"
        )));
    }

    let d = head_size as u64;
    let q_need = batch as u64 * q_seq_len as u64 * q_num_heads as u64 * d;
    let kv_need = batch as u64 * kv_seq_len as u64 * kv_num_heads as u64 * d;
    let mask_need = batch as u64 * kv_seq_len as u64;

    check_len("q", q.len(), q_need)?;
    check_len("k", k.len(), kv_need)?;
    check_len("v", v.len(), kv_need)?;
    check_len("o", o.len(), q_need)?;
    if let Some(m) = attn_mask {
        check_len("attn_mask", m.len(), mask_need)?;
    }
    Ok(())
}

fn check_len(name: &str, actual: usize, need: u64) -> Result<(), FlashAttnError> {
    if actual as u64 >= need {
        return Ok(());
    }
    Err(FlashAttnError::BufferTooSmall(format!(
        "{name}: need at least {need} elements, got {actual}"
    )))
}
