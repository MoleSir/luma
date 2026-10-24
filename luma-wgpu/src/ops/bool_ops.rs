use crate::launch::BoolOp;
use crate::storage::readback_bytes;
use crate::{Wgpu, WgpuBoolStorage, WgpuFloatStorage, WgpuIntStorage};
use luma_tensor::{BoolDType, BoolOps, FloatDType, IntDType, Layout, ReduceOp, Result, Shape};
use std::borrow::Cow;

fn decode_bool_at(bytes: &[u8], offset: usize) -> bool {
    u32::from_le_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]]) != 0
}

fn upload_words(device: &Wgpu, words: &[u32]) -> Result<WgpuBoolStorage> {
    let buffer = device.alloc_storage_output(std::mem::size_of_val(words))?;
    if !words.is_empty() {
        device.queue().write_buffer(&buffer, 0, bytemuck::cast_slice(words));
    }
    Ok(WgpuBoolStorage::new(buffer, device))
}

impl BoolOps<Wgpu> for Wgpu {
    fn b_falses(shape: &Shape, device: &Wgpu, _dtype: BoolDType) -> Result<WgpuBoolStorage> {
        let n = shape.element_count();
        let buffer = device.alloc_storage_output(n * std::mem::size_of::<u32>())?;
        Ok(WgpuBoolStorage::new(buffer, device))
    }

    fn b_trues(shape: &Shape, device: &Wgpu, _dtype: BoolDType) -> Result<WgpuBoolStorage> {
        upload_words(device, &vec![1u32; shape.element_count()])
    }

    fn b_from_bool<'a>(data: impl Into<Cow<'a, [bool]>>, device: &Wgpu) -> Result<WgpuBoolStorage> {
        let data = data.into();
        let words: Vec<u32> = data.iter().map(|&v| u32::from(v)).collect();
        upload_words(device, &words)
    }

    fn b_from_bytes<'a>(bytes: impl Into<Cow<'a, [u8]>>, shape: &Shape, device: &Wgpu, _dtype: BoolDType) -> Result<WgpuBoolStorage> {
        let bytes = bytes.into();
        let expected = shape.element_count();
        if bytes.len() != expected {
            return Err(luma_tensor::Error::ElementSizeMismatch { expected, got: bytes.len(), op: "wgpu b_from_bytes" });
        }
        let words: Vec<u32> = bytes.iter().map(|&v| u32::from(v != 0)).collect();
        upload_words(device, &words)
    }

    fn b_contiguous(x: &WgpuBoolStorage, layout: &Layout) -> Result<WgpuBoolStorage> {
        crate::launch::launch_copy(x, layout)
    }

    fn b_cast_float(x: &WgpuBoolStorage, layout: &Layout, to: FloatDType) -> Result<WgpuFloatStorage> {
        crate::launch::launch_cast_float(x, layout, to)
    }

    fn b_cast_int(x: &WgpuBoolStorage, layout: &Layout, to: IntDType) -> Result<WgpuIntStorage> {
        crate::launch::launch_cast_int(x, layout, to)
    }

    fn b_cast_bool(x: &WgpuBoolStorage, layout: &Layout, _to: BoolDType) -> Result<WgpuBoolStorage> {
        crate::launch::launch_copy(x, layout)
    }

    fn b_index_select(
        x: &WgpuBoolStorage,
        x_l: &Layout,
        idx: &WgpuIntStorage,
        idx_l: &Layout,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<WgpuBoolStorage> {
        crate::launch::launch_index_select(x, x_l, idx, idx_l, dim, out_shape)
    }

    fn b_gather(
        x: &WgpuBoolStorage,
        x_l: &Layout,
        idx: &WgpuIntStorage,
        idx_l: &Layout,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<WgpuBoolStorage> {
        crate::launch::launch_gather(x, x_l, idx, idx_l, dim, out_shape)
    }

    fn b_to_vec(x: &WgpuBoolStorage, layout: &Layout) -> Result<Vec<bool>> {
        let n = layout.shape().element_count();
        if n == 0 {
            return Ok(Vec::new());
        }

        if layout.is_contiguous() {
            let bytes = readback_bytes(&x.device, &x.buffer, layout.start_offset() * 4, n * 4)?;
            Ok((0..n).map(|i| decode_bool_at(&bytes, i * 4)).collect())
        } else {
            let bytes = readback_bytes(&x.device, &x.buffer, 0, x.buffer.size() as usize)?;
            Ok(layout.storage_indices().map(|i| decode_bool_at(&bytes, i * 4)).collect())
        }
    }

    fn b_to_bytes<'a>(x: &'a WgpuBoolStorage, layout: &Layout) -> Result<Cow<'a, [u8]>> {
        let n = layout.shape().element_count();
        if n == 0 {
            return Ok(Cow::Owned(Vec::new()));
        }

        if layout.is_contiguous() {
            let bytes = readback_bytes(&x.device, &x.buffer, layout.start_offset() * 4, n * 4)?;
            Ok(Cow::Owned((0..n).map(|i| u8::from(decode_bool_at(&bytes, i * 4))).collect()))
        } else {
            let bytes = readback_bytes(&x.device, &x.buffer, 0, x.buffer.size() as usize)?;
            Ok(Cow::Owned(layout.storage_indices().map(|i| u8::from(decode_bool_at(&bytes, i * 4))).collect()))
        }
    }

    fn b_and(lhs: &WgpuBoolStorage, lhs_l: &Layout, rhs: &WgpuBoolStorage, rhs_l: &Layout) -> Result<WgpuBoolStorage> {
        crate::launch::launch_bool_binary(lhs, lhs_l, rhs, rhs_l, BoolOp::And)
    }

    fn b_or(lhs: &WgpuBoolStorage, lhs_l: &Layout, rhs: &WgpuBoolStorage, rhs_l: &Layout) -> Result<WgpuBoolStorage> {
        crate::launch::launch_bool_binary(lhs, lhs_l, rhs, rhs_l, BoolOp::Or)
    }

    fn b_xor(lhs: &WgpuBoolStorage, lhs_l: &Layout, rhs: &WgpuBoolStorage, rhs_l: &Layout) -> Result<WgpuBoolStorage> {
        crate::launch::launch_bool_binary(lhs, lhs_l, rhs, rhs_l, BoolOp::Xor)
    }

    fn b_not(x: &WgpuBoolStorage, layout: &Layout) -> Result<WgpuBoolStorage> {
        crate::launch::launch_bool_not(x, layout)
    }

    fn b_reduce_all(x: &WgpuBoolStorage, layout: &Layout, dims: &[usize], keepdim: bool, out_shape: &Shape) -> Result<WgpuBoolStorage> {
        crate::launch::launch_reduce(x, layout, dims, keepdim, ReduceOp::Min, out_shape)
    }

    fn b_reduce_any(x: &WgpuBoolStorage, layout: &Layout, dims: &[usize], keepdim: bool, out_shape: &Shape) -> Result<WgpuBoolStorage> {
        crate::launch::launch_reduce(x, layout, dims, keepdim, ReduceOp::Max, out_shape)
    }

    fn b_true_count(x: &WgpuBoolStorage, layout: &Layout) -> Result<usize> {
        let dims: Vec<usize> = (0..layout.shape().rank()).collect();
        let out = crate::launch::launch_reduce(x, layout, &dims, false, ReduceOp::Sum, &Shape::from(()))?;
        let bytes = readback_bytes(&x.device, &out.buffer, 0, 4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize)
    }

    fn b_cat(srcs: &[(&WgpuBoolStorage, &Layout)], dim: usize, out_shape: &Shape) -> Result<WgpuBoolStorage> {
        crate::launch::launch_cat(srcs, dim, out_shape)
    }

    fn b_pick(
        mask: &WgpuBoolStorage,
        mask_l: &Layout,
        on_true: &WgpuBoolStorage,
        true_l: &Layout,
        on_false: &WgpuBoolStorage,
        false_l: &Layout,
    ) -> Result<WgpuBoolStorage> {
        crate::launch::launch_pick(mask, mask_l, on_true, true_l, on_false, false_l)
    }

    fn b_pick_true(
        mask: &WgpuBoolStorage,
        mask_l: &Layout,
        value: bool,
        on_false: &WgpuBoolStorage,
        false_l: &Layout,
    ) -> Result<WgpuBoolStorage> {
        crate::launch::launch_pick_scalar(mask, mask_l, on_false, false_l, u32::from(value), true)
    }

    fn b_pick_false(
        mask: &WgpuBoolStorage,
        mask_l: &Layout,
        on_true: &WgpuBoolStorage,
        true_l: &Layout,
        value: bool,
    ) -> Result<WgpuBoolStorage> {
        crate::launch::launch_pick_scalar(mask, mask_l, on_true, true_l, u32::from(value), false)
    }

    fn b_allclose(a: &WgpuBoolStorage, a_l: &Layout, b: &WgpuBoolStorage, b_l: &Layout) -> Result<bool> {
        if a_l.shape().element_count() != b_l.shape().element_count() {
            return Ok(false);
        }
        if a_l.shape().element_count() == 0 {
            return Ok(true);
        }

        let a_bytes = readback_bytes(&a.device, &a.buffer, 0, a.buffer.size() as usize)?;
        let b_bytes = readback_bytes(&b.device, &b.buffer, 0, b.buffer.size() as usize)?;
        let a_words: Vec<u32> = a_bytes.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
        let b_words: Vec<u32> = b_bytes.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();

        Ok(a_l.storage_indices().zip(b_l.storage_indices()).all(|(ai, bi)| (a_words[ai] != 0) == (b_words[bi] != 0)))
    }
}
