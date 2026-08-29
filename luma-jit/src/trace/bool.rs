//! `BoolOps<Trace>`: record bool ops into the graph.

use std::borrow::Cow;

use luma_tensor::dtype::{BoolDType, FloatDType, IntDType};
use luma_tensor::{BoolOps, DType, Layout, Result, Shape, ViewOp};

use super::{Trace, TraceBoolStorage, TraceFloatStorage, TraceIntStorage};
use super::{readback_unsupported, reduce_out_shape};
use crate::graph::{NodeOp, Scalar, ValueId};

impl BoolOps<Trace> for Trace {
    fn b_falses(shape: &Shape, device: &Trace, dtype: BoolDType) -> Result<TraceBoolStorage> {
        Ok(device.bool_leaf(dtype, shape))
    }
    fn b_trues(shape: &Shape, device: &Trace, dtype: BoolDType) -> Result<TraceBoolStorage> {
        Ok(device.bool_leaf(dtype, shape))
    }
    fn b_from_bool<'a>(_data: impl Into<Cow<'a, [bool]>>, device: &Trace) -> Result<TraceBoolStorage> {
        Ok(device.bool_leaf(BoolDType::Bool, &Shape::from(())))
    }
    fn b_from_bytes<'a>(bytes: impl Into<Cow<'a, [u8]>>, shape: &Shape, device: &Trace, dtype: BoolDType) -> Result<TraceBoolStorage> {
        Ok(device.bool_const(dtype, shape, bytes.into().into_owned()))
    }

    fn b_contiguous(x: &TraceBoolStorage, _layout: &Layout) -> Result<TraceBoolStorage> {
        Ok(x.clone())
    }
    fn b_cast_float(x: &TraceBoolStorage, layout: &Layout, to: FloatDType) -> Result<TraceFloatStorage> {
        let id = x.device.emit(NodeOp::Cast(to.into()), vec![x.value], to.into(), layout.shape().clone())?;
        Ok(TraceFloatStorage { value: id, dtype: to, device: x.device.clone() })
    }
    fn b_cast_int(x: &TraceBoolStorage, layout: &Layout, to: IntDType) -> Result<TraceIntStorage> {
        let id = x.device.emit(NodeOp::Cast(to.into()), vec![x.value], to.into(), layout.shape().clone())?;
        Ok(TraceIntStorage { value: id, dtype: to, device: x.device.clone() })
    }
    fn b_cast_bool(x: &TraceBoolStorage, layout: &Layout, to: BoolDType) -> Result<TraceBoolStorage> {
        let id = x.device.emit(NodeOp::Cast(to.into()), vec![x.value], to.into(), layout.shape().clone())?;
        Ok(TraceBoolStorage { value: id, dtype: to, device: x.device.clone() })
    }

    fn b_index_select(x: &TraceBoolStorage, x_l: &Layout, idx: &TraceIntStorage, idx_l: &Layout, dim: usize) -> Result<(TraceBoolStorage, Shape)> {
        let mut dims = x_l.shape().dims().to_vec();
        dims[dim] = idx_l.element_count();
        let out_shape = Shape::from(dims);
        let id = x.device.emit(NodeOp::IndexSelect(dim), vec![x.value, idx.value], x.dtype.into(), out_shape.clone())?;
        Ok((TraceBoolStorage { value: id, dtype: x.dtype, device: x.device.clone() }, out_shape))
    }

    fn b_gather(x: &TraceBoolStorage, _x_l: &Layout, idx: &TraceIntStorage, idx_l: &Layout, dim: usize) -> Result<(TraceBoolStorage, Shape)> {
        let out_shape = idx_l.shape().clone();
        let id = x.device.emit(NodeOp::Gather(dim), vec![x.value, idx.value], x.dtype.into(), out_shape.clone())?;
        Ok((TraceBoolStorage { value: id, dtype: x.dtype, device: x.device.clone() }, out_shape))
    }

    fn b_to_vec(_x: &TraceBoolStorage, _layout: &Layout) -> Result<Vec<bool>> {
        readback_unsupported("to_vec")
    }
    fn b_to_bytes<'a>(_x: &'a TraceBoolStorage, _layout: &Layout) -> Result<Cow<'a, [u8]>> {
        readback_unsupported("to_bytes")
    }

    fn b_and(lhs: &TraceBoolStorage, lhs_l: &Layout, rhs: &TraceBoolStorage, _rhs_l: &Layout) -> Result<TraceBoolStorage> {
        let id = lhs.device.emit(NodeOp::And, vec![lhs.value, rhs.value], DType::Bool, lhs_l.shape().clone())?;
        Ok(TraceBoolStorage { value: id, dtype: BoolDType::Bool, device: lhs.device.clone() })
    }
    fn b_or(lhs: &TraceBoolStorage, lhs_l: &Layout, rhs: &TraceBoolStorage, _rhs_l: &Layout) -> Result<TraceBoolStorage> {
        let id = lhs.device.emit(NodeOp::Or, vec![lhs.value, rhs.value], DType::Bool, lhs_l.shape().clone())?;
        Ok(TraceBoolStorage { value: id, dtype: BoolDType::Bool, device: lhs.device.clone() })
    }
    fn b_xor(lhs: &TraceBoolStorage, lhs_l: &Layout, rhs: &TraceBoolStorage, _rhs_l: &Layout) -> Result<TraceBoolStorage> {
        let id = lhs.device.emit(NodeOp::Xor, vec![lhs.value, rhs.value], DType::Bool, lhs_l.shape().clone())?;
        Ok(TraceBoolStorage { value: id, dtype: BoolDType::Bool, device: lhs.device.clone() })
    }
    fn b_not(x: &TraceBoolStorage, layout: &Layout) -> Result<TraceBoolStorage> {
        let id = x.device.emit(NodeOp::Not, vec![x.value], DType::Bool, layout.shape().clone())?;
        Ok(TraceBoolStorage { value: id, dtype: BoolDType::Bool, device: x.device.clone() })
    }
    fn b_reduce_all(x: &TraceBoolStorage, layout: &Layout, dims: &[usize], keepdim: bool) -> Result<(TraceBoolStorage, Shape)> {
        let out_shape = reduce_out_shape(layout.shape(), dims, keepdim);
        let id = x.device.emit(NodeOp::ReduceAll(dims.to_vec()), vec![x.value], DType::Bool, out_shape.clone())?;
        Ok((TraceBoolStorage { value: id, dtype: BoolDType::Bool, device: x.device.clone() }, out_shape))
    }
    fn b_reduce_any(x: &TraceBoolStorage, layout: &Layout, dims: &[usize], keepdim: bool) -> Result<(TraceBoolStorage, Shape)> {
        let out_shape = reduce_out_shape(layout.shape(), dims, keepdim);
        let id = x.device.emit(NodeOp::ReduceAny(dims.to_vec()), vec![x.value], DType::Bool, out_shape.clone())?;
        Ok((TraceBoolStorage { value: id, dtype: BoolDType::Bool, device: x.device.clone() }, out_shape))
    }
    fn b_true_count(_x: &TraceBoolStorage, _layout: &Layout) -> Result<usize> {
        readback_unsupported("true_count")
    }
    fn b_cat(srcs: &[(&TraceBoolStorage, &Layout)], dim: usize) -> Result<(TraceBoolStorage, Shape)> {
        let first = &srcs[0];
        let mut dims = first.1.shape().dims().to_vec();
        dims[dim] = srcs.iter().map(|(_, l)| l.dims()[dim]).sum();
        let out_shape = Shape::from(dims);
        let inputs: Vec<ValueId> = srcs.iter().map(|(s, _)| s.value).collect();
        let id = first.0.device.emit(NodeOp::Cat(dim), inputs, DType::Bool, out_shape.clone())?;
        Ok((TraceBoolStorage { value: id, dtype: BoolDType::Bool, device: first.0.device.clone() }, out_shape))
    }
    fn b_view(src: &TraceBoolStorage, _src_l: &Layout, dst_l: &Layout, view: ViewOp) -> Result<Option<TraceBoolStorage>> {
        let out = src.device.emit_view(src.value, dst_l, view);
        Ok(Some(TraceBoolStorage { value: out, dtype: BoolDType::Bool, device: src.device.clone() }))
    }
    fn b_pick(
        mask: &TraceBoolStorage,
        _mask_l: &Layout,
        on_true: &TraceBoolStorage,
        true_l: &Layout,
        on_false: &TraceBoolStorage,
        _false_l: &Layout,
    ) -> Result<TraceBoolStorage> {
        let id = on_true
            .device
            .emit(NodeOp::Pick, vec![mask.value, on_true.value, on_false.value], DType::Bool, true_l.shape().clone())?;
        Ok(TraceBoolStorage { value: id, dtype: BoolDType::Bool, device: on_true.device.clone() })
    }
    fn b_pick_true(
        mask: &TraceBoolStorage,
        _mask_l: &Layout,
        value: bool,
        on_false: &TraceBoolStorage,
        false_l: &Layout,
    ) -> Result<TraceBoolStorage> {
        let id = on_false.device.emit(
            NodeOp::PickTrue(Scalar::Bool(value)),
            vec![mask.value, on_false.value],
            DType::Bool,
            false_l.shape().clone(),
        )?;
        Ok(TraceBoolStorage { value: id, dtype: BoolDType::Bool, device: on_false.device.clone() })
    }
    fn b_pick_false(
        mask: &TraceBoolStorage,
        _mask_l: &Layout,
        on_true: &TraceBoolStorage,
        true_l: &Layout,
        value: bool,
    ) -> Result<TraceBoolStorage> {
        let id = on_true.device.emit(
            NodeOp::PickFalse(Scalar::Bool(value)),
            vec![mask.value, on_true.value],
            DType::Bool,
            true_l.shape().clone(),
        )?;
        Ok(TraceBoolStorage { value: id, dtype: BoolDType::Bool, device: on_true.device.clone() })
    }
    fn b_allclose(_a: &TraceBoolStorage, _a_l: &Layout, _b: &TraceBoolStorage, _b_l: &Layout) -> Result<bool> {
        readback_unsupported("allclose")
    }
}
