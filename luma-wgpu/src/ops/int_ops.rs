use crate::storage::{WgpuStorage, readback_bytes};
use crate::{Wgpu, WgpuBoolStorage, WgpuError, WgpuFloatStorage, WgpuIntStorage};
use luma_tensor::{BinaryOp, BoolDType, CmpOp, FloatDType, IntDType, IntOps, Layout, ReduceOp, Result, Shape, UnaryOp};
use std::borrow::Cow;

impl IntOps<Wgpu> for Wgpu {
    fn i_zeros(shape: &Shape, device: &Wgpu, dtype: IntDType) -> Result<WgpuIntStorage> {
        Self::i_full(shape, 0, device, dtype)
    }

    fn i_ones(shape: &Shape, device: &Wgpu, dtype: IntDType) -> Result<WgpuIntStorage> {
        Self::i_full(shape, 1, device, dtype)
    }

    fn i_full(shape: &Shape, value: i64, device: &Wgpu, dtype: IntDType) -> Result<WgpuIntStorage> {
        let n = shape.element_count();
        match dtype {
            IntDType::I32 => {
                let data: Vec<i32> = vec![value as i32; n];
                WgpuIntStorage::upload_bytes(device, dtype, bytemuck::cast_slice(&data))
            }
            IntDType::U32 => {
                let data: Vec<u32> = vec![value as u32; n];
                WgpuIntStorage::upload_bytes(device, dtype, bytemuck::cast_slice(&data))
            }
            IntDType::U8 => Err(WgpuError::UnsupportedDType(dtype.into(), "wgpu i_full").into()),
        }
    }

    fn i_from_i64<'a>(data: impl Into<Cow<'a, [i64]>>, device: &Wgpu) -> Result<WgpuIntStorage> {
        let data = data.into();
        let values: Vec<i32> = data.iter().map(|&v| v as i32).collect();
        WgpuIntStorage::upload_bytes(device, IntDType::I32, bytemuck::cast_slice(&values))
    }

    fn i_from_i32<'a>(data: impl Into<Cow<'a, [i32]>>, device: &Wgpu) -> Result<WgpuIntStorage> {
        let data = data.into();
        WgpuIntStorage::upload_bytes(device, IntDType::I32, bytemuck::cast_slice(data.as_ref()))
    }

    fn i_from_u32<'a>(data: impl Into<Cow<'a, [u32]>>, device: &Wgpu) -> Result<WgpuIntStorage> {
        let data = data.into();
        WgpuIntStorage::upload_bytes(device, IntDType::U32, bytemuck::cast_slice(data.as_ref()))
    }

    fn i_from_u8<'a>(_data: impl Into<Cow<'a, [u8]>>, _device: &Wgpu) -> Result<WgpuIntStorage> {
        Err(WgpuError::UnsupportedDType(luma_tensor::DType::U8, "wgpu i_from_u8").into())
    }

    fn i_from_bytes<'a>(bytes: impl Into<Cow<'a, [u8]>>, shape: &Shape, device: &Wgpu, dtype: IntDType) -> Result<WgpuIntStorage> {
        let bytes = bytes.into();
        let Some(elem_bytes) = (match dtype {
            IntDType::I32 | IntDType::U32 => Some(4),
            IntDType::U8 => None,
        }) else {
            return Err(WgpuError::UnsupportedDType(dtype.into(), "wgpu i_from_bytes").into());
        };
        let expected = shape.element_count() * elem_bytes;
        if bytes.len() != expected {
            return Err(luma_tensor::Error::ElementSizeMismatch { expected, got: bytes.len(), op: "wgpu i_from_bytes" });
        }
        WgpuIntStorage::upload_bytes(device, dtype, bytes.as_ref())
    }

    fn i_arange(start: i64, end: i64, step: i64, device: &Wgpu, dtype: IntDType) -> Result<(WgpuIntStorage, usize)> {
        if step == 0 {
            return Err(luma_tensor::Error::Msg("arange step cannot be 0".into()));
        }

        let mut data = Vec::new();
        let mut v = start;
        if step > 0 {
            while v < end {
                data.push(v);
                v += step;
            }
        } else {
            while v > end {
                data.push(v);
                v += step;
            }
        }

        let n = data.len();
        let storage = match dtype {
            IntDType::I32 => {
                let values: Vec<i32> = data.iter().map(|&x| x as i32).collect();
                WgpuIntStorage::upload_bytes(device, dtype, bytemuck::cast_slice(&values))?
            }
            IntDType::U32 => {
                let values: Vec<u32> = data.iter().map(|&x| x as u32).collect();
                WgpuIntStorage::upload_bytes(device, dtype, bytemuck::cast_slice(&values))?
            }
            IntDType::U8 => {
                return Err(WgpuError::UnsupportedDType(luma_tensor::DType::U8, "wgpu i_arange").into());
            }
        };

        Ok((storage, n))
    }

    fn i_contiguous(x: &WgpuIntStorage, layout: &Layout) -> Result<WgpuIntStorage> {
        crate::launch::launch_copy(x, layout)
    }

    fn i_cast_float(x: &WgpuIntStorage, layout: &Layout, to: FloatDType) -> Result<WgpuFloatStorage> {
        crate::launch::launch_cast_float(x, layout, to)
    }

    fn i_cast_int(x: &WgpuIntStorage, layout: &Layout, to: IntDType) -> Result<WgpuIntStorage> {
        crate::launch::launch_cast_int(x, layout, to)
    }

    fn i_cast_bool(x: &WgpuIntStorage, layout: &Layout, _to: BoolDType) -> Result<WgpuBoolStorage> {
        let bits = int_scalar_bits(x.dtype().as_int(), 0, "wgpu i_cast_bool")?;
        crate::launch::launch_cmp_scalar::<WgpuIntStorage>(x, layout, bits, CmpOp::Ne)
    }

    fn i_to_vec(x: &WgpuIntStorage, layout: &Layout) -> Result<Vec<i64>> {
        let n = layout.shape().element_count();
        if n == 0 {
            return Ok(Vec::new());
        }
        let dtype = x.dtype();
        if layout.is_contiguous() {
            let bytes = readback_bytes(&x.device, x.wgpu_buffer(), layout.start_offset() * 4, n * 4)?;
            Ok((0..n).map(|i| decode_int_at(&bytes, i * 4, dtype)).collect())
        } else {
            let bytes = readback_bytes(&x.device, x.wgpu_buffer(), 0, x.wgpu_buffer().size() as usize)?;
            Ok(layout.storage_indices().map(|i| decode_int_at(&bytes, i * 4, dtype)).collect())
        }
    }

    fn i_to_bytes<'a>(x: &'a WgpuIntStorage, layout: &Layout) -> Result<Cow<'a, [u8]>> {
        let n = layout.shape().element_count();
        if n == 0 {
            return Ok(Cow::Owned(Vec::new()));
        }
        if layout.is_contiguous() {
            Ok(Cow::Owned(readback_bytes(&x.device, x.wgpu_buffer(), layout.start_offset() * 4, n * 4)?))
        } else {
            let all = readback_bytes(&x.device, x.wgpu_buffer(), 0, x.wgpu_buffer().size() as usize)?;
            let mut out = Vec::with_capacity(n * 4);
            for i in layout.storage_indices() {
                out.extend_from_slice(&all[i * 4..i * 4 + 4]);
            }
            Ok(Cow::Owned(out))
        }
    }

    fn i_binary(lhs: &WgpuIntStorage, lhs_l: &Layout, rhs: &WgpuIntStorage, rhs_l: &Layout, op: BinaryOp) -> Result<WgpuIntStorage> {
        crate::launch::launch_binary::<WgpuIntStorage>(lhs, lhs_l, rhs, rhs_l, op)
    }

    fn i_binary_(dst: &mut WgpuIntStorage, dst_l: &Layout, src: &WgpuIntStorage, src_l: &Layout, op: BinaryOp) -> Result<()> {
        crate::launch::launch_binary_inplace::<WgpuIntStorage>(dst, dst_l, src, src_l, op)
    }

    fn i_binary_scalar(lhs: &WgpuIntStorage, lhs_l: &Layout, rhs: i64, op: BinaryOp) -> Result<WgpuIntStorage> {
        let bits = int_scalar_bits(lhs.dtype().as_int(), rhs, "wgpu i_binary_scalar")?;
        crate::launch::launch_binary_scalar::<WgpuIntStorage>(lhs, lhs_l, bits, false, op)
    }

    fn i_binary_scalar_(dst: &mut WgpuIntStorage, dst_l: &Layout, rhs: i64, op: BinaryOp) -> Result<()> {
        let bits = int_scalar_bits(dst.dtype().as_int(), rhs, "wgpu i_binary_scalar_")?;
        crate::launch::launch_binary_scalar_inplace::<WgpuIntStorage>(dst, dst_l, bits, op)
    }

    fn i_binary_scalar_lhs(scalar: i64, rhs: &WgpuIntStorage, rhs_l: &Layout, op: BinaryOp) -> Result<WgpuIntStorage> {
        let bits = int_scalar_bits(rhs.dtype().as_int(), scalar, "wgpu i_binary_scalar_lhs")?;
        crate::launch::launch_binary_scalar::<WgpuIntStorage>(rhs, rhs_l, bits, true, op)
    }

    fn i_unary(x: &WgpuIntStorage, layout: &Layout, op: UnaryOp<i64>) -> Result<WgpuIntStorage> {
        let (arg0, arg1, arg0_opt, arg1_opt) = super::unary_op_args(op);
        let dtype = x.dtype().as_int();
        let arg0_bits = int_scalar_bits(dtype, arg0, "wgpu i_unary")?;
        let arg1_bits = int_scalar_bits(dtype, arg1, "wgpu i_unary")?;
        crate::launch::launch_unary::<WgpuIntStorage>(x, layout, super::unary_op_code(op), arg0_bits, arg1_bits, arg0_opt, arg1_opt)
    }

    fn i_unary_(dst: &mut WgpuIntStorage, dst_l: &Layout, op: UnaryOp<i64>) -> Result<()> {
        let (arg0, arg1, arg0_opt, arg1_opt) = super::unary_op_args(op);
        let dtype = dst.dtype().as_int();
        let arg0_bits = int_scalar_bits(dtype, arg0, "wgpu i_unary_")?;
        let arg1_bits = int_scalar_bits(dtype, arg1, "wgpu i_unary_")?;
        crate::launch::launch_unary_inplace::<WgpuIntStorage>(
            dst,
            dst_l,
            super::unary_op_code(op),
            arg0_bits,
            arg1_bits,
            arg0_opt,
            arg1_opt,
        )
    }

    fn i_matmul(lhs: &WgpuIntStorage, lhs_l: &Layout, rhs: &WgpuIntStorage, rhs_l: &Layout, out_shape: &Shape) -> Result<WgpuIntStorage> {
        crate::launch::launch_matmul::<WgpuIntStorage>(lhs, lhs_l, rhs, rhs_l, out_shape)
    }

    fn i_cmp(lhs: &WgpuIntStorage, lhs_l: &Layout, rhs: &WgpuIntStorage, rhs_l: &Layout, op: CmpOp) -> Result<WgpuBoolStorage> {
        crate::launch::launch_cmp::<WgpuIntStorage>(lhs, lhs_l, rhs, rhs_l, op)
    }

    fn i_cmp_scalar(lhs: &WgpuIntStorage, lhs_l: &Layout, rhs: i64, op: CmpOp) -> Result<WgpuBoolStorage> {
        let bits = int_scalar_bits(lhs.dtype().as_int(), rhs, "wgpu i_cmp_scalar")?;
        crate::launch::launch_cmp_scalar::<WgpuIntStorage>(lhs, lhs_l, bits, op)
    }

    fn i_reduce(
        x: &WgpuIntStorage,
        layout: &Layout,
        dims: &[usize],
        keepdim: bool,
        op: ReduceOp,
        out_shape: &Shape,
    ) -> Result<WgpuIntStorage> {
        if matches!(op, ReduceOp::Mean) {
            return Err(luma_tensor::Error::Msg("Mean reduce not supported for int types".into()));
        }
        crate::launch::launch_reduce(x, layout, dims, keepdim, op, out_shape)
    }

    fn i_arg_reduce(
        x: &WgpuIntStorage,
        layout: &Layout,
        dim: usize,
        keepdim: bool,
        take_max: bool,
        out_shape: &Shape,
    ) -> Result<WgpuIntStorage> {
        crate::launch::launch_arg_reduce(x, layout, dim, keepdim, take_max, out_shape)
    }

    fn i_index_select(
        x: &WgpuIntStorage,
        x_l: &Layout,
        idx: &WgpuIntStorage,
        idx_l: &Layout,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<WgpuIntStorage> {
        crate::launch::launch_index_select(x, x_l, idx, idx_l, dim, out_shape)
    }

    fn i_gather(
        x: &WgpuIntStorage,
        x_l: &Layout,
        idx: &WgpuIntStorage,
        idx_l: &Layout,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<WgpuIntStorage> {
        crate::launch::launch_gather(x, x_l, idx, idx_l, dim, out_shape)
    }

    fn i_index_add(
        init: &WgpuIntStorage,
        init_l: &Layout,
        idx: &WgpuIntStorage,
        idx_l: &Layout,
        src: &WgpuIntStorage,
        src_l: &Layout,
        dim: usize,
    ) -> Result<WgpuIntStorage> {
        crate::launch::launch_index_add(init, init_l, idx, idx_l, src, src_l, dim)
    }

    fn i_scatter_add(
        init: &WgpuIntStorage,
        init_l: &Layout,
        idx: &WgpuIntStorage,
        idx_l: &Layout,
        src: &WgpuIntStorage,
        src_l: &Layout,
        dim: usize,
    ) -> Result<WgpuIntStorage> {
        crate::launch::launch_scatter_add(init, init_l, idx, idx_l, src, src_l, dim)
    }

    fn i_cat(srcs: &[(&WgpuIntStorage, &Layout)], dim: usize, out_shape: &Shape) -> Result<WgpuIntStorage> {
        crate::launch::launch_cat(srcs, dim, out_shape)
    }

    fn i_pick(
        mask: &WgpuBoolStorage,
        mask_l: &Layout,
        on_true: &WgpuIntStorage,
        true_l: &Layout,
        on_false: &WgpuIntStorage,
        false_l: &Layout,
    ) -> Result<WgpuIntStorage> {
        crate::launch::launch_pick(mask, mask_l, on_true, true_l, on_false, false_l)
    }

    fn i_pick_true(
        mask: &WgpuBoolStorage,
        mask_l: &Layout,
        value: i64,
        on_false: &WgpuIntStorage,
        false_l: &Layout,
    ) -> Result<WgpuIntStorage> {
        let bits = int_scalar_bits(on_false.dtype().as_int(), value, "wgpu i_pick_true")?;
        crate::launch::launch_pick_scalar(mask, mask_l, on_false, false_l, bits, true)
    }

    fn i_pick_false(
        mask: &WgpuBoolStorage,
        mask_l: &Layout,
        on_true: &WgpuIntStorage,
        true_l: &Layout,
        value: i64,
    ) -> Result<WgpuIntStorage> {
        let bits = int_scalar_bits(on_true.dtype().as_int(), value, "wgpu i_pick_false")?;
        crate::launch::launch_pick_scalar(mask, mask_l, on_true, true_l, bits, false)
    }

    fn i_allclose(a: &WgpuIntStorage, a_l: &Layout, b: &WgpuIntStorage, b_l: &Layout) -> Result<bool> {
        if a.dtype() != b.dtype() {
            return Err(luma_tensor::Error::DTypeMismatch { lhs: a.dtype(), rhs: b.dtype(), op: "wgpu i_allclose" });
        }
        if a_l.shape().element_count() != b_l.shape().element_count() {
            return Ok(false);
        }
        if a_l.shape().element_count() == 0 {
            return Ok(true);
        }

        let a_bytes = readback_bytes(&a.device, a.wgpu_buffer(), 0, a.wgpu_buffer().size() as usize)?;
        let b_bytes = readback_bytes(&b.device, b.wgpu_buffer(), 0, b.wgpu_buffer().size() as usize)?;
        let same = match a.dtype() {
            luma_tensor::DType::I32 => a_l.storage_indices().zip(b_l.storage_indices()).all(|(ai, bi)| {
                let av = i32::from_le_bytes([a_bytes[ai * 4], a_bytes[ai * 4 + 1], a_bytes[ai * 4 + 2], a_bytes[ai * 4 + 3]]);
                let bv = i32::from_le_bytes([b_bytes[bi * 4], b_bytes[bi * 4 + 1], b_bytes[bi * 4 + 2], b_bytes[bi * 4 + 3]]);
                av == bv
            }),
            luma_tensor::DType::U32 => a_l.storage_indices().zip(b_l.storage_indices()).all(|(ai, bi)| {
                let av = u32::from_le_bytes([a_bytes[ai * 4], a_bytes[ai * 4 + 1], a_bytes[ai * 4 + 2], a_bytes[ai * 4 + 3]]);
                let bv = u32::from_le_bytes([b_bytes[bi * 4], b_bytes[bi * 4 + 1], b_bytes[bi * 4 + 2], b_bytes[bi * 4 + 3]]);
                av == bv
            }),
            _ => unreachable!("unsupported wgpu int dtype"),
        };
        Ok(same)
    }
}

fn decode_int_at(bytes: &[u8], offset: usize, dtype: luma_tensor::DType) -> i64 {
    match dtype {
        luma_tensor::DType::I32 => i32::from_le_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]]) as i64,
        luma_tensor::DType::U32 => u32::from_le_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]]) as i64,
        _ => unreachable!("unsupported wgpu int dtype"),
    }
}

fn int_scalar_bits(dtype: IntDType, value: i64, op: &'static str) -> Result<u32> {
    match dtype {
        IntDType::I32 => Ok((value as i32) as u32),
        IntDType::U32 => Ok(value as u32),
        IntDType::U8 => Err(WgpuError::UnsupportedDType(dtype.into(), op).into()),
    }
}
