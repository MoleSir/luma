#![allow(non_camel_case_types)]
use core::ffi::{c_char, c_void};

use half::{bf16, f16};

pub const FLASH_ATTN_OK: i32 = 0;
pub const FLASH_ATTN_ERR_NULL_PTR: i32 = 1;
pub const FLASH_ATTN_ERR_INVALID_SHAPE: i32 = 2;
pub const FLASH_ATTN_ERR_INVALID_HEAD_SIZE: i32 = 3;
pub const FLASH_ATTN_ERR_INVALID_START_POS: i32 = 4;
pub const FLASH_ATTN_ERR_CUDA: i32 = 5;

unsafe extern "C" {
    /// Batched causal flash attention (f32).
    pub unsafe fn flash_attn_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        o: *mut f32,
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

    /// Batched causal flash attention (f16).
    pub unsafe fn flash_attn_f16(
        q: *const f16,
        k: *const f16,
        v: *const f16,
        o: *mut f16,
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

    /// Batched causal flash attention (bf16).
    pub unsafe fn flash_attn_bf16(
        q: *const bf16,
        k: *const bf16,
        v: *const bf16,
        o: *mut bf16,
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

    /// Packed variable-length causal flash attention (f32, optional paged prefix).
    pub unsafe fn flash_attn_varlen_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        o: *mut f32,
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

    /// Packed variable-length causal flash attention (f16).
    pub unsafe fn flash_attn_varlen_f16(
        q: *const f16,
        k: *const f16,
        v: *const f16,
        o: *mut f16,
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

    /// Packed variable-length causal flash attention (bf16).
    pub unsafe fn flash_attn_varlen_bf16(
        q: *const bf16,
        k: *const bf16,
        v: *const bf16,
        o: *mut bf16,
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

    /// Paged KV-cache attention (decode when `q_seq_len == 1`, prefill when `> 1`) (f32).
    pub unsafe fn flash_attn_with_kvcache_f32(
        q: *const f32,
        k_cache: *const f32,
        v_cache: *const f32,
        o: *mut f32,
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

    /// Paged KV-cache attention (decode when `q_seq_len == 1`, prefill when `> 1`) (f16).
    pub unsafe fn flash_attn_with_kvcache_f16(
        q: *const f16,
        k_cache: *const f16,
        v_cache: *const f16,
        o: *mut f16,
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

    /// Paged KV-cache attention (decode when `q_seq_len == 1`, prefill when `> 1`) (bf16).
    pub unsafe fn flash_attn_with_kvcache_bf16(
        q: *const bf16,
        k_cache: *const bf16,
        v_cache: *const bf16,
        o: *mut bf16,
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

    /// Thread-local error string shared by all kernels.
    pub unsafe fn flash_attn_last_error() -> *const c_char;
}
