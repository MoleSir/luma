//! `IntOps<Trace>`: record int ops into the graph.

use std::borrow::Cow;

use luma_tensor::dtype::{BoolDType, FloatDType, IntDType};
use luma_tensor::{BinaryOp, CmpOp, DType, IntOps, Layout, ReduceOp, Result, Shape, UnaryOp, ViewOp};

use super::{Trace, TraceBoolStorage, TraceFloatStorage, TraceIntStorage};
use super::{arange_len, inplace_unsupported, readback_unsupported};
use crate::graph::{NodeOp, Scalar, ValueId};

impl IntOps<Trace> for Trace {
    fn i_zeros(shape: &Shape, device: &Trace, dtype: IntDType) -> Result<TraceIntStorage> {
        Ok(device.int_leaf(dtype, shape))
    }
    fn i_ones(shape: &Shape, device: &Trace, dtype: IntDType) -> Result<TraceIntStorage> {
        Ok(device.int_leaf(dtype, shape))
    }
    fn i_full(shape: &Shape, _value: i64, device: &Trace, dtype: IntDType) -> Result<TraceIntStorage> {
        Ok(device.int_leaf(dtype, shape))
    }
    fn i_from_i64<'a>(_data: impl Into<Cow<'a, [i64]>>, device: &Trace) -> Result<TraceIntStorage> {
        Ok(device.int_leaf(IntDType::I32, &Shape::from(())))
    }
    fn i_from_i32<'a>(_data: impl Into<Cow<'a, [i32]>>, device: &Trace) -> Result<TraceIntStorage> {
        Ok(device.int_leaf(IntDType::I32, &Shape::from(())))
    }
    fn i_from_u32<'a>(_data: impl Into<Cow<'a, [u32]>>, device: &Trace) -> Result<TraceIntStorage> {
        Ok(device.int_leaf(IntDType::U32, &Shape::from(())))
    }
    fn i_from_u8<'a>(_data: impl Into<Cow<'a, [u8]>>, device: &Trace) -> Result<TraceIntStorage> {
        Ok(device.int_leaf(IntDType::U8, &Shape::from(())))
    }
    fn i_from_bytes<'a>(bytes: impl Into<Cow<'a, [u8]>>, shape: &Shape, device: &Trace, dtype: IntDType) -> Result<TraceIntStorage> {
        Ok(device.int_const(dtype, shape, bytes.into().into_owned()))
    }
    fn i_arange(start: i64, end: i64, step: i64, device: &Trace, dtype: IntDType) -> Result<(TraceIntStorage, usize)> {
        let len = arange_len(start, end, step);
        let id = device.emit(NodeOp::Arange(start, end, step), vec![], dtype.into(), Shape::from((len,)))?;
        Ok((TraceIntStorage { value: id, dtype, device: device.clone() }, len))
    }

    fn i_contiguous(x: &TraceIntStorage, _layout: &Layout) -> Result<TraceIntStorage> {
        Ok(x.clone())
    }
    fn i_cast_float(x: &TraceIntStorage, layout: &Layout, to: FloatDType) -> Result<TraceFloatStorage> {
        let id = x.device.emit(NodeOp::Cast(to.into()), vec![x.value], to.into(), layout.shape().clone())?;
        Ok(TraceFloatStorage { value: id, dtype: to, device: x.device.clone() })
    }
    fn i_cast_int(x: &TraceIntStorage, layout: &Layout, to: IntDType) -> Result<TraceIntStorage> {
        let id = x.device.emit(NodeOp::Cast(to.into()), vec![x.value], to.into(), layout.shape().clone())?;
        Ok(TraceIntStorage { value: id, dtype: to, device: x.device.clone() })
    }
    fn i_cast_bool(x: &TraceIntStorage, layout: &Layout, to: BoolDType) -> Result<TraceBoolStorage> {
        let id = x.device.emit(NodeOp::Cast(to.into()), vec![x.value], to.into(), layout.shape().clone())?;
        Ok(TraceBoolStorage { value: id, dtype: to, device: x.device.clone() })
    }
    fn i_to_vec(_x: &TraceIntStorage, _layout: &Layout) -> Result<Vec<i64>> {
        readback_unsupported("to_vec")
    }
    fn i_to_bytes<'a>(_x: &'a TraceIntStorage, _layout: &Layout) -> Result<Cow<'a, [u8]>> {
        readback_unsupported("to_bytes")
    }

    fn i_binary(lhs: &TraceIntStorage, lhs_l: &Layout, rhs: &TraceIntStorage, _rhs_l: &Layout, op: BinaryOp) -> Result<TraceIntStorage> {
        let id = lhs.device.emit(NodeOp::Binary(op), vec![lhs.value, rhs.value], lhs.dtype.into(), lhs_l.shape().clone())?;
        Ok(TraceIntStorage { value: id, dtype: lhs.dtype, device: lhs.device.clone() })
    }
    fn i_binary_(_dst: &mut TraceIntStorage, _dst_l: &Layout, _src: &TraceIntStorage, _src_l: &Layout, _op: BinaryOp) -> Result<()> {
        inplace_unsupported("i_binary_")
    }
    fn i_binary_scalar(lhs: &TraceIntStorage, lhs_l: &Layout, rhs: i64, op: BinaryOp) -> Result<TraceIntStorage> {
        let id =
            lhs.device
                .emit(NodeOp::BinaryScalarRhs(Scalar::I64(rhs), op), vec![lhs.value], lhs.dtype.into(), lhs_l.shape().clone())?;
        Ok(TraceIntStorage { value: id, dtype: lhs.dtype, device: lhs.device.clone() })
    }
    fn i_binary_scalar_(_dst: &mut TraceIntStorage, _dst_l: &Layout, _rhs: i64, _op: BinaryOp) -> Result<()> {
        inplace_unsupported("i_binary_scalar_")
    }
    fn i_binary_scalar_lhs(scalar: i64, rhs: &TraceIntStorage, rhs_l: &Layout, op: BinaryOp) -> Result<TraceIntStorage> {
        let id =
            rhs.device
                .emit(NodeOp::BinaryScalarLhs(Scalar::I64(scalar), op), vec![rhs.value], rhs.dtype.into(), rhs_l.shape().clone())?;
        Ok(TraceIntStorage { value: id, dtype: rhs.dtype, device: rhs.device.clone() })
    }
    fn i_unary(x: &TraceIntStorage, layout: &Layout, op: UnaryOp<i64>) -> Result<TraceIntStorage> {
        let id = x.device.emit(NodeOp::UnaryI(op), vec![x.value], x.dtype.into(), layout.shape().clone())?;
        Ok(TraceIntStorage { value: id, dtype: x.dtype, device: x.device.clone() })
    }
    fn i_unary_(_dst: &mut TraceIntStorage, _dst_l: &Layout, _op: UnaryOp<i64>) -> Result<()> {
        inplace_unsupported("i_unary_")
    }
    fn i_matmul(
        lhs: &TraceIntStorage,
        _lhs_l: &Layout,
        rhs: &TraceIntStorage,
        _rhs_l: &Layout,
        out_shape: &Shape,
    ) -> Result<TraceIntStorage> {
        let id = lhs.device.emit(NodeOp::Matmul, vec![lhs.value, rhs.value], lhs.dtype.into(), out_shape.clone())?;
        Ok(TraceIntStorage { value: id, dtype: lhs.dtype, device: lhs.device.clone() })
    }
    fn i_cmp(lhs: &TraceIntStorage, lhs_l: &Layout, rhs: &TraceIntStorage, _rhs_l: &Layout, op: CmpOp) -> Result<TraceBoolStorage> {
        let id = lhs.device.emit(NodeOp::Cmp(op), vec![lhs.value, rhs.value], DType::Bool, lhs_l.shape().clone())?;
        Ok(TraceBoolStorage { value: id, dtype: BoolDType::Bool, device: lhs.device.clone() })
    }
    fn i_cmp_scalar(lhs: &TraceIntStorage, lhs_l: &Layout, rhs: i64, op: CmpOp) -> Result<TraceBoolStorage> {
        let id = lhs.device.emit(NodeOp::CmpScalar(Scalar::I64(rhs), op), vec![lhs.value], DType::Bool, lhs_l.shape().clone())?;
        Ok(TraceBoolStorage { value: id, dtype: BoolDType::Bool, device: lhs.device.clone() })
    }
    fn i_reduce(
        x: &TraceIntStorage,
        _layout: &Layout,
        dims: &[usize],
        _keepdim: bool,
        op: ReduceOp,
        out_shape: &Shape,
    ) -> Result<TraceIntStorage> {
        let id = x.device.emit(NodeOp::Reduce(op, dims.to_vec()), vec![x.value], x.dtype.into(), out_shape.clone())?;
        Ok(TraceIntStorage { value: id, dtype: x.dtype, device: x.device.clone() })
    }
    fn i_arg_reduce(
        x: &TraceIntStorage,
        _layout: &Layout,
        dim: usize,
        _keepdim: bool,
        take_max: bool,
        out_shape: &Shape,
    ) -> Result<TraceIntStorage> {
        let id = x.device.emit(NodeOp::ArgReduce(dim, take_max), vec![x.value], DType::I32, out_shape.clone())?;
        Ok(TraceIntStorage { value: id, dtype: IntDType::I32, device: x.device.clone() })
    }
    fn i_index_select(
        x: &TraceIntStorage,
        _x_l: &Layout,
        idx: &TraceIntStorage,
        _idx_l: &Layout,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<TraceIntStorage> {
        let id = x.device.emit(NodeOp::IndexSelect(dim), vec![x.value, idx.value], x.dtype.into(), out_shape.clone())?;
        Ok(TraceIntStorage { value: id, dtype: x.dtype, device: x.device.clone() })
    }
    fn i_gather(
        x: &TraceIntStorage,
        _x_l: &Layout,
        idx: &TraceIntStorage,
        _idx_l: &Layout,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<TraceIntStorage> {
        let id = x.device.emit(NodeOp::Gather(dim), vec![x.value, idx.value], x.dtype.into(), out_shape.clone())?;
        Ok(TraceIntStorage { value: id, dtype: x.dtype, device: x.device.clone() })
    }
    fn i_index_add(
        init: &TraceIntStorage,
        init_l: &Layout,
        idx: &TraceIntStorage,
        _idx_l: &Layout,
        src: &TraceIntStorage,
        _src_l: &Layout,
        dim: usize,
    ) -> Result<TraceIntStorage> {
        let id =
            init.device
                .emit(NodeOp::IndexAdd(dim), vec![init.value, idx.value, src.value], init.dtype.into(), init_l.shape().clone())?;
        Ok(TraceIntStorage { value: id, dtype: init.dtype, device: init.device.clone() })
    }
    fn i_scatter_add(
        init: &TraceIntStorage,
        init_l: &Layout,
        idx: &TraceIntStorage,
        _idx_l: &Layout,
        src: &TraceIntStorage,
        _src_l: &Layout,
        dim: usize,
    ) -> Result<TraceIntStorage> {
        let id =
            init.device
                .emit(NodeOp::ScatterAdd(dim), vec![init.value, idx.value, src.value], init.dtype.into(), init_l.shape().clone())?;
        Ok(TraceIntStorage { value: id, dtype: init.dtype, device: init.device.clone() })
    }
    fn i_cat(srcs: &[(&TraceIntStorage, &Layout)], dim: usize, out_shape: &Shape) -> Result<TraceIntStorage> {
        let first = &srcs[0];
        let inputs: Vec<ValueId> = srcs.iter().map(|(s, _)| s.value).collect();
        let id = first.0.device.emit(NodeOp::Cat(dim), inputs, first.0.dtype.into(), out_shape.clone())?;
        Ok(TraceIntStorage { value: id, dtype: first.0.dtype, device: first.0.device.clone() })
    }
    fn i_view(src: &TraceIntStorage, _src_l: &Layout, dst_l: &Layout, view: ViewOp) -> Result<Option<TraceIntStorage>> {
        let out = src.device.emit_view(src.value, dst_l, view);
        Ok(Some(TraceIntStorage { value: out, dtype: src.dtype, device: src.device.clone() }))
    }
    fn i_pick(
        mask: &TraceBoolStorage,
        _mask_l: &Layout,
        on_true: &TraceIntStorage,
        true_l: &Layout,
        on_false: &TraceIntStorage,
        _false_l: &Layout,
    ) -> Result<TraceIntStorage> {
        let id = on_true.device.emit(
            NodeOp::Pick,
            vec![mask.value, on_true.value, on_false.value],
            on_true.dtype.into(),
            true_l.shape().clone(),
        )?;
        Ok(TraceIntStorage { value: id, dtype: on_true.dtype, device: on_true.device.clone() })
    }
    fn i_pick_true(
        mask: &TraceBoolStorage,
        _mask_l: &Layout,
        value: i64,
        on_false: &TraceIntStorage,
        false_l: &Layout,
    ) -> Result<TraceIntStorage> {
        let id = on_false.device.emit(
            NodeOp::PickTrue(Scalar::I64(value)),
            vec![mask.value, on_false.value],
            on_false.dtype.into(),
            false_l.shape().clone(),
        )?;
        Ok(TraceIntStorage { value: id, dtype: on_false.dtype, device: on_false.device.clone() })
    }
    fn i_pick_false(
        mask: &TraceBoolStorage,
        _mask_l: &Layout,
        on_true: &TraceIntStorage,
        true_l: &Layout,
        value: i64,
    ) -> Result<TraceIntStorage> {
        let id = on_true.device.emit(
            NodeOp::PickFalse(Scalar::I64(value)),
            vec![mask.value, on_true.value],
            on_true.dtype.into(),
            true_l.shape().clone(),
        )?;
        Ok(TraceIntStorage { value: id, dtype: on_true.dtype, device: on_true.device.clone() })
    }
    fn i_allclose(_a: &TraceIntStorage, _a_l: &Layout, _b: &TraceIntStorage, _b_l: &Layout) -> Result<bool> {
        readback_unsupported("allclose")
    }
}
