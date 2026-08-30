//! `impl BoolOps for Cpu`. Bool storage is a plain `Vec<bool>`, so no per-dtype
//! dispatch is needed; kernels operate on it directly.

use std::borrow::Cow;

use super::kernels::{elementwise as ew, reduce, shape as shape_k};
use super::{Cpu, CpuBoolStorage, CpuFloatStorage, CpuIntStorage};
use crate::device::cpu::int_ids_as_usize;
use crate::device::cpu::kernels::indexing;
use crate::dtype::{BoolDType, FloatDType, IntDType};
use crate::{BoolOps, Device, Error, Layout, Result, Shape};

impl BoolOps<Cpu> for Cpu {
    fn b_falses(shape: &Shape, _device: &Cpu, _dtype: BoolDType) -> Result<<Cpu as Device>::BoolStorage> {
        Ok(CpuBoolStorage(vec![false; shape.element_count()]))
    }

    fn b_trues(shape: &Shape, _device: &Cpu, _dtype: BoolDType) -> Result<<Cpu as Device>::BoolStorage> {
        Ok(CpuBoolStorage(vec![true; shape.element_count()]))
    }

    fn b_from_bool<'a>(data: impl Into<Cow<'a, [bool]>>, _device: &Cpu) -> Result<<Cpu as Device>::BoolStorage> {
        let data = data.into();
        Ok(match data {
            Cow::Owned(v) => CpuBoolStorage(v),
            Cow::Borrowed(s) => CpuBoolStorage(s.to_vec()),
        })
    }

    fn b_from_bytes<'a>(
        bytes: impl Into<Cow<'a, [u8]>>,
        _shape: &Shape,
        _device: &Cpu,
        _dtype: BoolDType,
    ) -> Result<<Cpu as Device>::BoolStorage> {
        let bytes = bytes.into();
        Ok(CpuBoolStorage(bytes.iter().map(|&x| x != 0).collect()))
    }

    fn b_contiguous(x: &<Cpu as Device>::BoolStorage, l: &Layout) -> Result<<Cpu as Device>::BoolStorage> {
        Ok(CpuBoolStorage(super::kernels::iter::gather(&x.0, l)))
    }

    fn b_cast_float(x: &CpuBoolStorage, layout: &Layout, to: FloatDType) -> Result<CpuFloatStorage> {
        let s = match to {
            FloatDType::F32 => CpuFloatStorage::F32(layout.storage_indices().map(|i| if x.0[i] { 1.0 } else { 0.0 }).collect()),
            FloatDType::F64 => CpuFloatStorage::F64(layout.storage_indices().map(|i| if x.0[i] { 1.0 } else { 0.0 }).collect()),
        };
        Ok(s)
    }

    fn b_cast_int(x: &CpuBoolStorage, layout: &Layout, to: IntDType) -> Result<CpuIntStorage> {
        let s = match to {
            IntDType::I32 => CpuIntStorage::I32(layout.storage_indices().map(|i| x.0[i] as i32).collect()),
            IntDType::U32 => CpuIntStorage::U32(layout.storage_indices().map(|i| x.0[i] as u32).collect()),
            IntDType::U8 => CpuIntStorage::U8(layout.storage_indices().map(|i| x.0[i] as u8).collect()),
        };
        Ok(s)
    }

    fn b_cast_bool(x: &CpuBoolStorage, layout: &Layout, _to: BoolDType) -> Result<CpuBoolStorage> {
        Ok(CpuBoolStorage(layout.storage_indices().map(|i| x.0[i]).collect()))
    }

    fn b_index_select(
        x: &CpuBoolStorage,
        x_l: &Layout,
        idx: &CpuIntStorage,
        idx_l: &Layout,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<CpuBoolStorage> {
        let ids = int_ids_as_usize(idx, idx_l);
        let (v, dims) = indexing::index_select(&x.0, x_l, &ids, idx_l, dim)?;
        debug_assert_eq!(&dims, out_shape.dims(), "cpu b_index_select shape must match the layer");
        Ok(CpuBoolStorage(v))
    }

    fn b_gather(
        x: &CpuBoolStorage,
        x_l: &Layout,
        idx: &CpuIntStorage,
        idx_l: &Layout,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<CpuBoolStorage> {
        let ids = int_ids_as_usize(idx, idx_l);
        let (v, dims) = indexing::gather(&x.0, x_l, &ids, idx_l, dim)?;
        debug_assert_eq!(&dims, out_shape.dims(), "cpu b_gather shape must match the layer");
        Ok(CpuBoolStorage(v))
    }

    fn b_to_vec(x: &<Cpu as Device>::BoolStorage, layout: &Layout) -> Result<Vec<bool>> {
        Ok(layout.storage_indices().map(|i| x.0[i]).collect())
    }

    fn b_to_bytes<'a>(x: &'a <Cpu as Device>::BoolStorage, layout: &Layout) -> Result<Cow<'a, [u8]>> {
        // bool is not Pod, so bytemuck doesn't work — convert manually.
        if layout.is_contiguous() {
            let bytes: Vec<u8> = x.0.iter().map(|&b| b as u8).collect();
            Ok(Cow::Owned(bytes))
        } else {
            let contig = Self::b_contiguous(x, layout)?;
            let bytes: Vec<u8> = contig.0.iter().map(|&b| b as u8).collect();
            Ok(Cow::Owned(bytes))
        }
    }

    fn b_and(
        lhs: &<Cpu as Device>::BoolStorage,
        lhs_l: &Layout,
        rhs: &<Cpu as Device>::BoolStorage,
        rhs_l: &Layout,
    ) -> Result<<Cpu as Device>::BoolStorage> {
        Ok(CpuBoolStorage(ew::binary(&lhs.0, lhs_l, &rhs.0, rhs_l, |a, b| a & b)))
    }

    fn b_or(
        lhs: &<Cpu as Device>::BoolStorage,
        lhs_l: &Layout,
        rhs: &<Cpu as Device>::BoolStorage,
        rhs_l: &Layout,
    ) -> Result<<Cpu as Device>::BoolStorage> {
        Ok(CpuBoolStorage(ew::binary(&lhs.0, lhs_l, &rhs.0, rhs_l, |a, b| a | b)))
    }

    fn b_xor(
        lhs: &<Cpu as Device>::BoolStorage,
        lhs_l: &Layout,
        rhs: &<Cpu as Device>::BoolStorage,
        rhs_l: &Layout,
    ) -> Result<<Cpu as Device>::BoolStorage> {
        Ok(CpuBoolStorage(ew::binary(&lhs.0, lhs_l, &rhs.0, rhs_l, |a, b| a ^ b)))
    }

    fn b_not(x: &<Cpu as Device>::BoolStorage, l: &Layout) -> Result<<Cpu as Device>::BoolStorage> {
        Ok(CpuBoolStorage(ew::unary(&x.0, l, |v| !v)))
    }

    fn b_reduce_all(
        x: &<Cpu as Device>::BoolStorage,
        l: &Layout,
        dims: &[usize],
        keepdim: bool,
        out_shape: &Shape,
    ) -> Result<<Cpu as Device>::BoolStorage> {
        // reduce as u8 with min (all true == min == 1), then back to bool.
        let as_u8: Vec<u8> = super::kernels::iter::gather(&x.0, l).iter().map(|&b| b as u8).collect();
        let contig = Layout::contiguous(l.shape().clone());
        let (v, shape) = reduce::reduce_dims(&as_u8, &contig, dims, keepdim, reduce::Reducer::Min)?;
        debug_assert_eq!(shape.dims(), out_shape.dims(), "cpu b_reduce_all shape must match the layer");
        Ok(CpuBoolStorage(v.into_iter().map(|u| u != 0).collect()))
    }

    fn b_reduce_any(
        x: &<Cpu as Device>::BoolStorage,
        l: &Layout,
        dims: &[usize],
        keepdim: bool,
        out_shape: &Shape,
    ) -> Result<<Cpu as Device>::BoolStorage> {
        let as_u8: Vec<u8> = super::kernels::iter::gather(&x.0, l).iter().map(|&b| b as u8).collect();
        let contig = Layout::contiguous(l.shape().clone());
        let (v, shape) = reduce::reduce_dims(&as_u8, &contig, dims, keepdim, reduce::Reducer::Max)?;
        debug_assert_eq!(shape.dims(), out_shape.dims(), "cpu b_reduce_any shape must match the layer");
        Ok(CpuBoolStorage(v.into_iter().map(|u| u != 0).collect()))
    }

    fn b_true_count(x: &<Cpu as Device>::BoolStorage, l: &Layout) -> Result<usize> {
        Ok(l.storage_indices().filter(|&i| x.0[i]).count())
    }

    fn b_cat(srcs: &[(&<Cpu as Device>::BoolStorage, &Layout)], dim: usize, out_shape: &Shape) -> Result<<Cpu as Device>::BoolStorage> {
        if srcs.is_empty() {
            return Err(Error::OpRequiresAtLeastOneTensor { op: "cat" });
        }
        let views: Vec<(&[bool], &Layout)> = srcs.iter().map(|(s, l)| (s.0.as_slice(), *l)).collect();
        let (v, shape) = shape_k::cat(&views, dim)?;
        debug_assert_eq!(shape.dims(), out_shape.dims(), "cpu b_cat shape must match the layer");
        Ok(CpuBoolStorage(v))
    }

    fn b_pick(
        mask: &<Cpu as Device>::BoolStorage,
        mask_l: &Layout,
        on_true: &<Cpu as Device>::BoolStorage,
        true_l: &Layout,
        on_false: &<Cpu as Device>::BoolStorage,
        false_l: &Layout,
    ) -> Result<<Cpu as Device>::BoolStorage> {
        let m: Vec<bool> = mask_l.storage_indices().map(|i| mask.0[i]).collect();
        let tv: Vec<bool> = true_l.storage_indices().map(|i| on_true.0[i]).collect();
        let fv: Vec<bool> = false_l.storage_indices().map(|i| on_false.0[i]).collect();
        Ok(CpuBoolStorage(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { fv[i] }).collect()))
    }

    fn b_pick_true(
        mask: &<Cpu as Device>::BoolStorage,
        mask_l: &Layout,
        value: bool,
        on_false: &<Cpu as Device>::BoolStorage,
        false_l: &Layout,
    ) -> Result<<Cpu as Device>::BoolStorage> {
        let m: Vec<bool> = mask_l.storage_indices().map(|i| mask.0[i]).collect();
        let fv: Vec<bool> = false_l.storage_indices().map(|i| on_false.0[i]).collect();
        Ok(CpuBoolStorage(m.iter().enumerate().map(|(i, &c)| if c { value } else { fv[i] }).collect()))
    }

    fn b_pick_false(
        mask: &<Cpu as Device>::BoolStorage,
        mask_l: &Layout,
        on_true: &<Cpu as Device>::BoolStorage,
        true_l: &Layout,
        value: bool,
    ) -> Result<<Cpu as Device>::BoolStorage> {
        let m: Vec<bool> = mask_l.storage_indices().map(|i| mask.0[i]).collect();
        let tv: Vec<bool> = true_l.storage_indices().map(|i| on_true.0[i]).collect();
        Ok(CpuBoolStorage(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { value }).collect()))
    }

    fn b_allclose(a: &CpuBoolStorage, a_l: &Layout, b: &CpuBoolStorage, b_l: &Layout) -> Result<bool> {
        Ok(a_l.storage_indices().zip(b_l.storage_indices()).all(|(ai, bi)| a.0[ai] == b.0[bi]))
    }
}
