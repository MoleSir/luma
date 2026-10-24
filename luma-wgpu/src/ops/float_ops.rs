use crate::storage::readback_bytes;
use crate::{Wgpu, WgpuBoolStorage, WgpuError, WgpuFloatBuffer, WgpuFloatStorage, WgpuIntStorage};
use luma_tensor::{
    BinaryOp, BoolDType, CmpOp, DType, FloatDType, FloatOps, FloatUnaryOp, IntDType, Layout, ReduceOp, Result, Shape, UnaryOp,
};
use std::borrow::Cow;

impl FloatOps<Wgpu> for Wgpu {
    fn f_zeros(shape: &Shape, device: &Wgpu, dtype: FloatDType) -> Result<WgpuFloatStorage> {
        let Some(elem_bytes) = float_elem_bytes(dtype) else {
            return Err(WgpuError::UnsupportedDType(dtype.into(), "wgpu f_zeros").into());
        };
        if dtype == FloatDType::F16 && !device.supports_f16() {
            return Err(WgpuError::UnsupportedFeature("shader-f16").into());
        }

        let bytes = shape.element_count() * elem_bytes;
        let buffer = device.alloc_storage_output(bytes)?;
        Ok(WgpuFloatStorage::new(buffer, dtype, device))
    }

    fn f_ones(shape: &Shape, device: &Wgpu, dtype: FloatDType) -> Result<WgpuFloatStorage> {
        Self::f_full(shape, 1.0, device, dtype)
    }

    fn f_full(shape: &Shape, value: f64, device: &Wgpu, dtype: FloatDType) -> Result<WgpuFloatStorage> {
        let n = shape.element_count();
        match dtype {
            FloatDType::F32 => Self::f_from_f32(vec![value as f32; n], device),
            FloatDType::F16 => Self::f_from_f16(vec![half::f16::from_f64(value); n], device),
            FloatDType::F64 | FloatDType::BF16 => Err(WgpuError::UnsupportedDType(dtype.into(), "wgpu f_full").into()),
        }
    }

    fn f_from_f64<'a>(_data: impl Into<Cow<'a, [f64]>>, _device: &Wgpu) -> Result<WgpuFloatStorage> {
        Err(WgpuError::UnsupportedDType(DType::F64, "wgpu f_from_f64").into())
    }

    fn f_from_f32<'a>(data: impl Into<Cow<'a, [f32]>>, device: &Wgpu) -> Result<WgpuFloatStorage> {
        let data = data.into();
        WgpuFloatStorage::upload_bytes(device, FloatDType::F32, bytemuck::cast_slice(data.as_ref()))
    }

    fn f_from_f16<'a>(data: impl Into<Cow<'a, [half::f16]>>, device: &Wgpu) -> Result<WgpuFloatStorage> {
        let data = data.into();
        WgpuFloatStorage::upload_bytes(device, FloatDType::F16, bytemuck::cast_slice(data.as_ref()))
    }

    fn f_from_bf16<'a>(_data: impl Into<Cow<'a, [half::bf16]>>, _device: &Wgpu) -> Result<WgpuFloatStorage> {
        Err(WgpuError::UnsupportedDType(DType::BF16, "wgpu f_from_bf16").into())
    }

    fn f_from_bytes<'a>(bytes: impl Into<Cow<'a, [u8]>>, shape: &Shape, device: &Wgpu, dtype: FloatDType) -> Result<WgpuFloatStorage> {
        let bytes = bytes.into();
        let Some(elem_bytes) = float_elem_bytes(dtype) else {
            return Err(WgpuError::UnsupportedDType(dtype.into(), "wgpu f_from_bytes").into());
        };
        let expected = shape.element_count() * elem_bytes;
        if bytes.len() != expected {
            return Err(luma_tensor::Error::ElementSizeMismatch { expected, got: bytes.len(), op: "wgpu f_from_bytes" });
        }
        WgpuFloatStorage::upload_bytes(device, dtype, bytes.as_ref())
    }

    fn f_rand_uniform(shape: &Shape, lo: f64, hi: f64, device: &Wgpu, dtype: FloatDType) -> Result<WgpuFloatStorage> {
        crate::launch::launch_random_uniform(device, shape, lo, hi, dtype)
    }

    fn f_rand_normal(shape: &Shape, mean: f64, std: f64, device: &Wgpu, dtype: FloatDType) -> Result<WgpuFloatStorage> {
        crate::launch::launch_random_normal(device, shape, mean, std, dtype)
    }

    fn f_contiguous(x: &WgpuFloatStorage, layout: &Layout) -> Result<WgpuFloatStorage> {
        crate::launch::launch_copy(x, layout)
    }

    fn f_cast_float(x: &WgpuFloatStorage, layout: &Layout, to: FloatDType) -> Result<WgpuFloatStorage> {
        crate::launch::launch_cast_float(x, layout, to)
    }

    fn f_cast_int(x: &WgpuFloatStorage, layout: &Layout, to: IntDType) -> Result<WgpuIntStorage> {
        crate::launch::launch_cast_int(x, layout, to)
    }

    fn f_cast_bool(x: &WgpuFloatStorage, layout: &Layout, _to: BoolDType) -> Result<WgpuBoolStorage> {
        let (_, dtype) = float_buffer(x, "wgpu f_cast_bool")?;
        let bits = float_scalar_bits(dtype, 0.0, "wgpu f_cast_bool")?;
        crate::launch::launch_cmp_scalar::<WgpuFloatStorage>(x, layout, bits, CmpOp::Ne)
    }

    fn f_to_vec(x: &WgpuFloatStorage, layout: &Layout) -> Result<Vec<f64>> {
        let (buffer, dtype) = float_buffer(x, "wgpu f_to_vec")?;
        let elem_bytes = float_elem_bytes(dtype).expect("wgpu float storage has a supported dtype");
        let n = layout.shape().element_count();
        if n == 0 {
            return Ok(Vec::new());
        }

        if layout.is_contiguous() {
            let bytes = readback_bytes(&x.device, buffer, layout.start_offset() * elem_bytes, n * elem_bytes)?;
            Ok((0..n).map(|i| decode_float_at(&bytes, i * elem_bytes, dtype)).collect())
        } else {
            let bytes = readback_bytes(&x.device, buffer, 0, buffer.size() as usize)?;
            Ok(layout.storage_indices().map(|i| decode_float_at(&bytes, i * elem_bytes, dtype)).collect())
        }
    }

    fn f_to_bytes<'a>(x: &'a WgpuFloatStorage, layout: &Layout) -> Result<Cow<'a, [u8]>> {
        let (buffer, dtype) = float_buffer(x, "wgpu f_to_bytes")?;
        let elem_bytes = float_elem_bytes(dtype).expect("wgpu float storage has a supported dtype");
        let n = layout.shape().element_count();
        if n == 0 {
            return Ok(Cow::Owned(Vec::new()));
        }

        if layout.is_contiguous() {
            Ok(Cow::Owned(readback_bytes(&x.device, buffer, layout.start_offset() * elem_bytes, n * elem_bytes)?))
        } else {
            let all = readback_bytes(&x.device, buffer, 0, buffer.size() as usize)?;
            let mut out = Vec::with_capacity(n * elem_bytes);
            for i in layout.storage_indices() {
                let start = i * elem_bytes;
                out.extend_from_slice(&all[start..start + elem_bytes]);
            }
            Ok(Cow::Owned(out))
        }
    }

    fn f_binary(lhs: &WgpuFloatStorage, lhs_l: &Layout, rhs: &WgpuFloatStorage, rhs_l: &Layout, op: BinaryOp) -> Result<WgpuFloatStorage> {
        crate::launch::launch_binary::<WgpuFloatStorage>(lhs, lhs_l, rhs, rhs_l, op)
    }

    fn f_binary_(dst: &mut WgpuFloatStorage, dst_l: &Layout, src: &WgpuFloatStorage, src_l: &Layout, op: BinaryOp) -> Result<()> {
        crate::launch::launch_binary_inplace::<WgpuFloatStorage>(dst, dst_l, src, src_l, op)
    }

    fn f_binary_scalar(lhs: &WgpuFloatStorage, lhs_l: &Layout, rhs: f64, op: BinaryOp) -> Result<WgpuFloatStorage> {
        let (_, dtype) = float_buffer(lhs, "wgpu f_binary_scalar")?;
        let bits = float_scalar_bits(dtype, rhs, "wgpu f_binary_scalar")?;
        crate::launch::launch_binary_scalar::<WgpuFloatStorage>(lhs, lhs_l, bits, false, op)
    }

    fn f_binary_scalar_(dst: &mut WgpuFloatStorage, dst_l: &Layout, rhs: f64, op: BinaryOp) -> Result<()> {
        let (_, dtype) = float_buffer(dst, "wgpu f_binary_scalar_")?;
        let bits = float_scalar_bits(dtype, rhs, "wgpu f_binary_scalar_")?;
        crate::launch::launch_binary_scalar_inplace::<WgpuFloatStorage>(dst, dst_l, bits, op)
    }

    fn f_binary_scalar_lhs(scalar: f64, rhs: &WgpuFloatStorage, rhs_l: &Layout, op: BinaryOp) -> Result<WgpuFloatStorage> {
        let (_, dtype) = float_buffer(rhs, "wgpu f_binary_scalar_lhs")?;
        let bits = float_scalar_bits(dtype, scalar, "wgpu f_binary_scalar_lhs")?;
        crate::launch::launch_binary_scalar::<WgpuFloatStorage>(rhs, rhs_l, bits, true, op)
    }

    fn f_cmp(lhs: &WgpuFloatStorage, lhs_l: &Layout, rhs: &WgpuFloatStorage, rhs_l: &Layout, op: CmpOp) -> Result<WgpuBoolStorage> {
        crate::launch::launch_cmp::<WgpuFloatStorage>(lhs, lhs_l, rhs, rhs_l, op)
    }

    fn f_cmp_scalar(lhs: &WgpuFloatStorage, lhs_l: &Layout, rhs: f64, op: CmpOp) -> Result<WgpuBoolStorage> {
        let (_, dtype) = float_buffer(lhs, "wgpu f_cmp_scalar")?;
        let bits = float_scalar_bits(dtype, rhs, "wgpu f_cmp_scalar")?;
        crate::launch::launch_cmp_scalar::<WgpuFloatStorage>(lhs, lhs_l, bits, op)
    }

    fn f_unary(x: &WgpuFloatStorage, layout: &Layout, op: UnaryOp<f64>) -> Result<WgpuFloatStorage> {
        let (_, dtype) = float_buffer(x, "wgpu f_unary")?;
        let (arg0, arg1, arg0_opt, arg1_opt) = super::unary_op_args(op);
        let arg0_bits = float_scalar_bits(dtype, arg0, "wgpu f_unary")?;
        let arg1_bits = float_scalar_bits(dtype, arg1, "wgpu f_unary")?;
        crate::launch::launch_unary::<WgpuFloatStorage>(x, layout, super::unary_op_code(op), arg0_bits, arg1_bits, arg0_opt, arg1_opt)
    }

    fn f_unary_(dst: &mut WgpuFloatStorage, dst_l: &Layout, op: UnaryOp<f64>) -> Result<()> {
        let (_, dtype) = float_buffer(dst, "wgpu f_unary_")?;
        let (arg0, arg1, arg0_opt, arg1_opt) = super::unary_op_args(op);
        let arg0_bits = float_scalar_bits(dtype, arg0, "wgpu f_unary_")?;
        let arg1_bits = float_scalar_bits(dtype, arg1, "wgpu f_unary_")?;
        crate::launch::launch_unary_inplace::<WgpuFloatStorage>(
            dst,
            dst_l,
            super::unary_op_code(op),
            arg0_bits,
            arg1_bits,
            arg0_opt,
            arg1_opt,
        )
    }

    fn f_float_unary(x: &WgpuFloatStorage, layout: &Layout, op: FloatUnaryOp) -> Result<WgpuFloatStorage> {
        crate::launch::launch_float_unary(x, layout, op)
    }

    fn f_float_unary_(dst: &mut WgpuFloatStorage, dst_l: &Layout, op: FloatUnaryOp) -> Result<()> {
        crate::launch::launch_float_unary_inplace(dst, dst_l, op)
    }

    fn f_reduce(
        x: &WgpuFloatStorage,
        layout: &Layout,
        dims: &[usize],
        keepdim: bool,
        op: ReduceOp,
        out_shape: &Shape,
    ) -> Result<WgpuFloatStorage> {
        crate::launch::launch_reduce(x, layout, dims, keepdim, op, out_shape)
    }

    fn f_arg_reduce(
        x: &WgpuFloatStorage,
        layout: &Layout,
        dim: usize,
        keepdim: bool,
        take_max: bool,
        out_shape: &Shape,
    ) -> Result<WgpuIntStorage> {
        crate::launch::launch_arg_reduce(x, layout, dim, keepdim, take_max, out_shape)
    }

    fn f_matmul(
        lhs: &WgpuFloatStorage,
        lhs_l: &Layout,
        rhs: &WgpuFloatStorage,
        rhs_l: &Layout,
        out_shape: &Shape,
    ) -> Result<WgpuFloatStorage> {
        crate::launch::launch_matmul::<WgpuFloatStorage>(lhs, lhs_l, rhs, rhs_l, out_shape)
    }

    fn f_add_matmul_(
        dst: &mut WgpuFloatStorage,
        dst_l: &Layout,
        lhs: &WgpuFloatStorage,
        lhs_l: &Layout,
        rhs: &WgpuFloatStorage,
        rhs_l: &Layout,
    ) -> Result<()> {
        crate::launch::launch_add_matmul_::<WgpuFloatStorage>(dst, dst_l, lhs, lhs_l, rhs, rhs_l)
    }

    fn f_index_select(
        x: &WgpuFloatStorage,
        x_l: &Layout,
        idx: &WgpuIntStorage,
        idx_l: &Layout,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<WgpuFloatStorage> {
        crate::launch::launch_index_select(x, x_l, idx, idx_l, dim, out_shape)
    }

    fn f_gather(
        x: &WgpuFloatStorage,
        x_l: &Layout,
        idx: &WgpuIntStorage,
        idx_l: &Layout,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<WgpuFloatStorage> {
        crate::launch::launch_gather(x, x_l, idx, idx_l, dim, out_shape)
    }

    fn f_index_add(
        init: &WgpuFloatStorage,
        init_l: &Layout,
        idx: &WgpuIntStorage,
        idx_l: &Layout,
        src: &WgpuFloatStorage,
        src_l: &Layout,
        dim: usize,
    ) -> Result<WgpuFloatStorage> {
        crate::launch::launch_index_add(init, init_l, idx, idx_l, src, src_l, dim)
    }

    fn f_scatter_add(
        init: &WgpuFloatStorage,
        init_l: &Layout,
        idx: &WgpuIntStorage,
        idx_l: &Layout,
        src: &WgpuFloatStorage,
        src_l: &Layout,
        dim: usize,
    ) -> Result<WgpuFloatStorage> {
        crate::launch::launch_scatter_add(init, init_l, idx, idx_l, src, src_l, dim)
    }

    fn f_cat(srcs: &[(&WgpuFloatStorage, &Layout)], dim: usize, out_shape: &Shape) -> Result<WgpuFloatStorage> {
        crate::launch::launch_cat(srcs, dim, out_shape)
    }

    fn f_pick(
        mask: &WgpuBoolStorage,
        mask_l: &Layout,
        on_true: &WgpuFloatStorage,
        true_l: &Layout,
        on_false: &WgpuFloatStorage,
        false_l: &Layout,
    ) -> Result<WgpuFloatStorage> {
        crate::launch::launch_pick(mask, mask_l, on_true, true_l, on_false, false_l)
    }

    fn f_pick_true(
        mask: &WgpuBoolStorage,
        mask_l: &Layout,
        value: f64,
        on_false: &WgpuFloatStorage,
        false_l: &Layout,
    ) -> Result<WgpuFloatStorage> {
        let (_, dtype) = float_buffer(on_false, "wgpu f_pick_true")?;
        let bits = float_scalar_bits(dtype, value, "wgpu f_pick_true")?;
        crate::launch::launch_pick_scalar(mask, mask_l, on_false, false_l, bits, true)
    }

    fn f_pick_false(
        mask: &WgpuBoolStorage,
        mask_l: &Layout,
        on_true: &WgpuFloatStorage,
        true_l: &Layout,
        value: f64,
    ) -> Result<WgpuFloatStorage> {
        let (_, dtype) = float_buffer(on_true, "wgpu f_pick_false")?;
        let bits = float_scalar_bits(dtype, value, "wgpu f_pick_false")?;
        crate::launch::launch_pick_scalar(mask, mask_l, on_true, true_l, bits, false)
    }

    fn f_allclose(a: &WgpuFloatStorage, a_l: &Layout, b: &WgpuFloatStorage, b_l: &Layout, rtol: f64, atol: f64) -> Result<bool> {
        let (a_buf, a_dtype) = float_buffer(a, "wgpu f_allclose")?;
        let (b_buf, b_dtype) = float_buffer(b, "wgpu f_allclose")?;
        if a_dtype != b_dtype {
            return Err(luma_tensor::Error::DTypeMismatch { lhs: a_dtype.into(), rhs: b_dtype.into(), op: "wgpu f_allclose" });
        }
        if a_l.shape().element_count() != b_l.shape().element_count() {
            return Ok(false);
        }
        if a_l.shape().element_count() == 0 {
            return Ok(true);
        }

        let a_bytes = readback_bytes(&a.device, a_buf, 0, a_buf.size() as usize)?;
        let b_bytes = readback_bytes(&b.device, b_buf, 0, b_buf.size() as usize)?;
        let rtol = rtol as f32;
        let atol = atol as f32;
        match a_dtype {
            FloatDType::F32 => {
                let av: Vec<f32> = a_bytes.chunks_exact(4).map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]])).collect();
                let bv: Vec<f32> = b_bytes.chunks_exact(4).map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]])).collect();
                Ok(a_l.storage_indices().zip(b_l.storage_indices()).all(|(ai, bi)| {
                    let x = av[ai];
                    let y = bv[bi];
                    (x - y).abs() <= atol + rtol * y.abs()
                }))
            }
            FloatDType::F16 => {
                let av: Vec<f32> =
                    a_bytes.chunks_exact(2).map(|v| half::f16::from_bits(u16::from_le_bytes([v[0], v[1]])).to_f32()).collect();
                let bv: Vec<f32> =
                    b_bytes.chunks_exact(2).map(|v| half::f16::from_bits(u16::from_le_bytes([v[0], v[1]])).to_f32()).collect();
                Ok(a_l.storage_indices().zip(b_l.storage_indices()).all(|(ai, bi)| {
                    let x = av[ai];
                    let y = bv[bi];
                    (x - y).abs() <= atol + rtol * y.abs()
                }))
            }
            FloatDType::F64 | FloatDType::BF16 => unreachable!("unsupported wgpu float dtype"),
        }
    }
}

fn decode_float_at(bytes: &[u8], offset: usize, dtype: FloatDType) -> f64 {
    match dtype {
        FloatDType::F32 => f32::from_le_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]]) as f64,
        FloatDType::F16 => half::f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f64(),
        FloatDType::F64 | FloatDType::BF16 => unreachable!("unsupported wgpu float dtype"),
    }
}

pub(crate) fn float_elem_bytes(dtype: FloatDType) -> Option<usize> {
    match dtype {
        FloatDType::F32 => Some(4),
        FloatDType::F16 => Some(2),
        FloatDType::F64 | FloatDType::BF16 => None,
    }
}

pub(crate) fn float_buffer<'a>(storage: &'a WgpuFloatStorage, _op: &'static str) -> Result<(&'a wgpu::Buffer, FloatDType)> {
    let (buffer, dtype) = match &storage.buffer {
        WgpuFloatBuffer::F32(buffer) => (buffer, FloatDType::F32),
        WgpuFloatBuffer::F16(buffer) => (buffer, FloatDType::F16),
    };
    Ok((buffer, dtype))
}

fn float_scalar_bits(dtype: FloatDType, value: f64, op: &'static str) -> Result<u32> {
    match dtype {
        FloatDType::F32 | FloatDType::F16 => Ok((value as f32).to_bits()),
        FloatDType::F64 | FloatDType::BF16 => Err(WgpuError::UnsupportedDType(dtype.into(), op).into()),
    }
}
