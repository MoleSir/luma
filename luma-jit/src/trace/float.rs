//! `FloatOps<Trace>`: record float ops into the graph.

use std::borrow::Cow;

use luma_tensor::dtype::{BoolDType, FloatDType, IntDType};
use luma_tensor::{BinaryOp, CmpOp, DType, FloatOps, FloatUnaryOp, Layout, ReduceOp, Result, Shape, UnaryOp, ViewOp};

use super::{Trace, TraceBoolStorage, TraceFloatStorage, TraceIntStorage};
use super::{inplace_unsupported, matmul_out_shape, readback_unsupported, reduce_out_shape};
use crate::graph::{NodeOp, Scalar, ValueId};

impl FloatOps<Trace> for Trace {
    // ---- construction ----
    fn f_zeros(shape: &Shape, device: &Trace, dtype: FloatDType) -> Result<TraceFloatStorage> {
        Ok(device.float_leaf(dtype, shape))
    }
    fn f_ones(shape: &Shape, device: &Trace, dtype: FloatDType) -> Result<TraceFloatStorage> {
        Ok(device.float_leaf(dtype, shape))
    }
    fn f_full(shape: &Shape, _value: f64, device: &Trace, dtype: FloatDType) -> Result<TraceFloatStorage> {
        Ok(device.float_leaf(dtype, shape))
    }
    fn f_from_f64<'a>(_data: impl Into<Cow<'a, [f64]>>, device: &Trace) -> Result<TraceFloatStorage> {
        // Shape is not known at this seam (it is attached later by `from_storage`),
        // so the leaf is recorded as a scalar. Construction methods that carry a
        // shape (`full`/`zeros`/`ones`) are preferred when tracing.
        Ok(device.float_leaf(FloatDType::F64, &Shape::from(())))
    }
    fn f_from_f32<'a>(_data: impl Into<Cow<'a, [f32]>>, device: &Trace) -> Result<TraceFloatStorage> {
        Ok(device.float_leaf(FloatDType::F32, &Shape::from(())))
    }
    fn f_from_bytes<'a>(bytes: impl Into<Cow<'a, [u8]>>, shape: &Shape, device: &Trace, dtype: FloatDType) -> Result<TraceFloatStorage> {
        // Captured data becomes a constant leaf in the graph (state saving).
        Ok(device.float_const(dtype, shape, bytes.into().into_owned()))
    }
    fn f_rand_uniform(shape: &Shape, _lo: f64, _hi: f64, device: &Trace, dtype: FloatDType) -> Result<TraceFloatStorage> {
        Ok(device.float_leaf(dtype, shape))
    }
    fn f_rand_normal(shape: &Shape, _mean: f64, _std: f64, device: &Trace, dtype: FloatDType) -> Result<TraceFloatStorage> {
        Ok(device.float_leaf(dtype, shape))
    }

    // ---- materialization / read-back ----
    fn f_contiguous(x: &TraceFloatStorage, _layout: &Layout) -> Result<TraceFloatStorage> {
        // A symbolic value has no physical layout; contiguous is the identity.
        Ok(x.clone())
    }
    fn f_cast_float(x: &TraceFloatStorage, layout: &Layout, to: FloatDType) -> Result<TraceFloatStorage> {
        let id = x.device.emit(NodeOp::Cast(to.into()), vec![x.value], to.into(), layout.shape().clone())?;
        Ok(TraceFloatStorage { value: id, dtype: to, device: x.device.clone() })
    }
    fn f_cast_int(x: &TraceFloatStorage, layout: &Layout, to: IntDType) -> Result<TraceIntStorage> {
        let id = x.device.emit(NodeOp::Cast(to.into()), vec![x.value], to.into(), layout.shape().clone())?;
        Ok(TraceIntStorage { value: id, dtype: to, device: x.device.clone() })
    }
    fn f_cast_bool(x: &TraceFloatStorage, layout: &Layout, to: BoolDType) -> Result<TraceBoolStorage> {
        let id = x.device.emit(NodeOp::Cast(to.into()), vec![x.value], to.into(), layout.shape().clone())?;
        Ok(TraceBoolStorage { value: id, dtype: to, device: x.device.clone() })
    }
    fn f_to_vec(_x: &TraceFloatStorage, _layout: &Layout) -> Result<Vec<f64>> {
        readback_unsupported("to_vec")
    }
    fn f_to_bytes<'a>(_x: &'a TraceFloatStorage, _layout: &Layout) -> Result<Cow<'a, [u8]>> {
        readback_unsupported("to_bytes")
    }

    // ---- binary ----
    fn f_binary(
        lhs: &TraceFloatStorage,
        lhs_l: &Layout,
        rhs: &TraceFloatStorage,
        _rhs_l: &Layout,
        op: BinaryOp,
    ) -> Result<TraceFloatStorage> {
        let id = lhs.device.emit(NodeOp::Binary(op), vec![lhs.value, rhs.value], lhs.dtype.into(), lhs_l.shape().clone())?;
        Ok(TraceFloatStorage { value: id, dtype: lhs.dtype, device: lhs.device.clone() })
    }
    fn f_binary_(_dst: &mut TraceFloatStorage, _dst_l: &Layout, _src: &TraceFloatStorage, _src_l: &Layout, _op: BinaryOp) -> Result<()> {
        inplace_unsupported("binary_")
    }
    fn f_binary_scalar(lhs: &TraceFloatStorage, lhs_l: &Layout, rhs: f64, op: BinaryOp) -> Result<TraceFloatStorage> {
        let id =
            lhs.device
                .emit(NodeOp::BinaryScalarRhs(Scalar::F64(rhs), op), vec![lhs.value], lhs.dtype.into(), lhs_l.shape().clone())?;
        Ok(TraceFloatStorage { value: id, dtype: lhs.dtype, device: lhs.device.clone() })
    }
    fn f_binary_scalar_(_dst: &mut TraceFloatStorage, _dst_l: &Layout, _rhs: f64, _op: BinaryOp) -> Result<()> {
        inplace_unsupported("binary_scalar_")
    }
    fn f_binary_scalar_lhs(scalar: f64, rhs: &TraceFloatStorage, rhs_l: &Layout, op: BinaryOp) -> Result<TraceFloatStorage> {
        let id =
            rhs.device
                .emit(NodeOp::BinaryScalarLhs(Scalar::F64(scalar), op), vec![rhs.value], rhs.dtype.into(), rhs_l.shape().clone())?;
        Ok(TraceFloatStorage { value: id, dtype: rhs.dtype, device: rhs.device.clone() })
    }

    // ---- comparison ----
    fn f_cmp(lhs: &TraceFloatStorage, lhs_l: &Layout, rhs: &TraceFloatStorage, _rhs_l: &Layout, op: CmpOp) -> Result<TraceBoolStorage> {
        let id = lhs.device.emit(NodeOp::Cmp(op), vec![lhs.value, rhs.value], DType::Bool, lhs_l.shape().clone())?;
        Ok(TraceBoolStorage { value: id, dtype: BoolDType::Bool, device: lhs.device.clone() })
    }
    fn f_cmp_scalar(lhs: &TraceFloatStorage, lhs_l: &Layout, rhs: f64, op: CmpOp) -> Result<TraceBoolStorage> {
        let id = lhs.device.emit(NodeOp::CmpScalar(Scalar::F64(rhs), op), vec![lhs.value], DType::Bool, lhs_l.shape().clone())?;
        Ok(TraceBoolStorage { value: id, dtype: BoolDType::Bool, device: lhs.device.clone() })
    }

    // ---- unary ----
    fn f_unary(x: &TraceFloatStorage, layout: &Layout, op: UnaryOp<f64>) -> Result<TraceFloatStorage> {
        let id = x.device.emit(NodeOp::Unary(op), vec![x.value], x.dtype.into(), layout.shape().clone())?;
        Ok(TraceFloatStorage { value: id, dtype: x.dtype, device: x.device.clone() })
    }
    fn f_unary_(_dst: &mut TraceFloatStorage, _dst_l: &Layout, _op: UnaryOp<f64>) -> Result<()> {
        inplace_unsupported("unary_")
    }
    fn f_float_unary(x: &TraceFloatStorage, layout: &Layout, op: FloatUnaryOp) -> Result<TraceFloatStorage> {
        let id = x.device.emit(NodeOp::FloatUnary(op), vec![x.value], x.dtype.into(), layout.shape().clone())?;
        Ok(TraceFloatStorage { value: id, dtype: x.dtype, device: x.device.clone() })
    }
    fn f_float_unary_(_dst: &mut TraceFloatStorage, _dst_l: &Layout, _op: FloatUnaryOp) -> Result<()> {
        inplace_unsupported("float_unary_")
    }

    // ---- reduction ----
    fn f_reduce(x: &TraceFloatStorage, layout: &Layout, dims: &[usize], keepdim: bool, op: ReduceOp) -> Result<(TraceFloatStorage, Shape)> {
        let out_shape = reduce_out_shape(layout.shape(), dims, keepdim);
        let id = x.device.emit(NodeOp::Reduce(op, dims.to_vec()), vec![x.value], x.dtype.into(), out_shape.clone())?;
        Ok((TraceFloatStorage { value: id, dtype: x.dtype, device: x.device.clone() }, out_shape))
    }
    fn f_arg_reduce(x: &TraceFloatStorage, layout: &Layout, dim: usize, keepdim: bool, take_max: bool) -> Result<(TraceIntStorage, Shape)> {
        let out_shape = reduce_out_shape(layout.shape(), &[dim], keepdim);
        let id = x.device.emit(NodeOp::ArgReduce(dim, take_max), vec![x.value], DType::I32, out_shape.clone())?;
        Ok((TraceIntStorage { value: id, dtype: IntDType::I32, device: x.device.clone() }, out_shape))
    }

    // ---- matmul ----
    fn f_matmul(lhs: &TraceFloatStorage, lhs_l: &Layout, rhs: &TraceFloatStorage, rhs_l: &Layout) -> Result<(TraceFloatStorage, Shape)> {
        let out_shape = matmul_out_shape(lhs_l.shape(), rhs_l.shape())?;
        let id = lhs.device.emit(NodeOp::Matmul, vec![lhs.value, rhs.value], lhs.dtype.into(), out_shape.clone())?;
        Ok((TraceFloatStorage { value: id, dtype: lhs.dtype, device: lhs.device.clone() }, out_shape))
    }
    fn f_add_matmul_(
        _dst: &mut TraceFloatStorage,
        _dst_l: &Layout,
        _lhs: &TraceFloatStorage,
        _lhs_l: &Layout,
        _rhs: &TraceFloatStorage,
        _rhs_l: &Layout,
    ) -> Result<()> {
        inplace_unsupported("add_matmul_")
    }

    // ---- indexing ----
    fn f_index_select(
        x: &TraceFloatStorage,
        x_l: &Layout,
        idx: &TraceIntStorage,
        idx_l: &Layout,
        dim: usize,
    ) -> Result<(TraceFloatStorage, Shape)> {
        let mut dims = x_l.shape().dims().to_vec();
        dims[dim] = idx_l.element_count();
        let out_shape = Shape::from(dims);
        let id = x.device.emit(NodeOp::IndexSelect(dim), vec![x.value, idx.value], x.dtype.into(), out_shape.clone())?;
        Ok((TraceFloatStorage { value: id, dtype: x.dtype, device: x.device.clone() }, out_shape))
    }
    fn f_gather(
        x: &TraceFloatStorage,
        _x_l: &Layout,
        idx: &TraceIntStorage,
        idx_l: &Layout,
        dim: usize,
    ) -> Result<(TraceFloatStorage, Shape)> {
        let out_shape = idx_l.shape().clone();
        let id = x.device.emit(NodeOp::Gather(dim), vec![x.value, idx.value], x.dtype.into(), out_shape.clone())?;
        Ok((TraceFloatStorage { value: id, dtype: x.dtype, device: x.device.clone() }, out_shape))
    }
    fn f_index_add(
        init: &TraceFloatStorage,
        init_l: &Layout,
        idx: &TraceIntStorage,
        _idx_l: &Layout,
        src: &TraceFloatStorage,
        _src_l: &Layout,
        dim: usize,
    ) -> Result<TraceFloatStorage> {
        let id =
            init.device
                .emit(NodeOp::IndexAdd(dim), vec![init.value, idx.value, src.value], init.dtype.into(), init_l.shape().clone())?;
        Ok(TraceFloatStorage { value: id, dtype: init.dtype, device: init.device.clone() })
    }
    fn f_scatter_add(
        init: &TraceFloatStorage,
        init_l: &Layout,
        idx: &TraceIntStorage,
        _idx_l: &Layout,
        src: &TraceFloatStorage,
        _src_l: &Layout,
        dim: usize,
    ) -> Result<TraceFloatStorage> {
        let id =
            init.device
                .emit(NodeOp::ScatterAdd(dim), vec![init.value, idx.value, src.value], init.dtype.into(), init_l.shape().clone())?;
        Ok(TraceFloatStorage { value: id, dtype: init.dtype, device: init.device.clone() })
    }

    // ---- shape ----
    fn f_cat(srcs: &[(&TraceFloatStorage, &Layout)], dim: usize) -> Result<(TraceFloatStorage, Shape)> {
        let first = &srcs[0];
        let mut dims = first.1.shape().dims().to_vec();
        dims[dim] = srcs.iter().map(|(_, l)| l.dims()[dim]).sum();
        let out_shape = Shape::from(dims);
        let inputs: Vec<ValueId> = srcs.iter().map(|(s, _)| s.value).collect();
        let id = first.0.device.emit(NodeOp::Cat(dim), inputs, first.0.dtype.into(), out_shape.clone())?;
        Ok((TraceFloatStorage { value: id, dtype: first.0.dtype, device: first.0.device.clone() }, out_shape))
    }

    fn f_view(src: &TraceFloatStorage, _src_l: &Layout, dst_l: &Layout, view: ViewOp) -> Result<Option<TraceFloatStorage>> {
        let out = src.device.emit_view(src.value, dst_l, view);
        Ok(Some(TraceFloatStorage { value: out, dtype: src.dtype, device: src.device.clone() }))
    }

    // ---- nn ----
    fn f_softmax(x: &TraceFloatStorage, layout: &Layout, dim: usize) -> Result<TraceFloatStorage> {
        let id = x.device.emit(NodeOp::Softmax(dim), vec![x.value], x.dtype.into(), layout.shape().clone())?;
        Ok(TraceFloatStorage { value: id, dtype: x.dtype, device: x.device.clone() })
    }
    fn f_rms_norm(
        x: &TraceFloatStorage,
        x_l: &Layout,
        weight: &TraceFloatStorage,
        _weight_l: &Layout,
        eps: f64,
    ) -> Result<TraceFloatStorage> {
        let id = x.device.emit(NodeOp::RmsNorm(eps), vec![x.value, weight.value], x.dtype.into(), x_l.shape().clone())?;
        Ok(TraceFloatStorage { value: id, dtype: x.dtype, device: x.device.clone() })
    }

    // ---- pick ----
    fn f_pick(
        mask: &TraceBoolStorage,
        _mask_l: &Layout,
        on_true: &TraceFloatStorage,
        true_l: &Layout,
        on_false: &TraceFloatStorage,
        _false_l: &Layout,
    ) -> Result<TraceFloatStorage> {
        let id = on_true.device.emit(
            NodeOp::Pick,
            vec![mask.value, on_true.value, on_false.value],
            on_true.dtype.into(),
            true_l.shape().clone(),
        )?;
        Ok(TraceFloatStorage { value: id, dtype: on_true.dtype, device: on_true.device.clone() })
    }
    fn f_pick_true(
        mask: &TraceBoolStorage,
        _mask_l: &Layout,
        value: f64,
        on_false: &TraceFloatStorage,
        false_l: &Layout,
    ) -> Result<TraceFloatStorage> {
        let id = on_false.device.emit(
            NodeOp::PickTrue(Scalar::F64(value)),
            vec![mask.value, on_false.value],
            on_false.dtype.into(),
            false_l.shape().clone(),
        )?;
        Ok(TraceFloatStorage { value: id, dtype: on_false.dtype, device: on_false.device.clone() })
    }
    fn f_pick_false(
        mask: &TraceBoolStorage,
        _mask_l: &Layout,
        on_true: &TraceFloatStorage,
        true_l: &Layout,
        value: f64,
    ) -> Result<TraceFloatStorage> {
        let id = on_true.device.emit(
            NodeOp::PickFalse(Scalar::F64(value)),
            vec![mask.value, on_true.value],
            on_true.dtype.into(),
            true_l.shape().clone(),
        )?;
        Ok(TraceFloatStorage { value: id, dtype: on_true.dtype, device: on_true.device.clone() })
    }

    // ---- allclose ----
    fn f_allclose(_a: &TraceFloatStorage, _a_l: &Layout, _b: &TraceFloatStorage, _b_l: &Layout, _rtol: f64, _atol: f64) -> Result<bool> {
        readback_unsupported("allclose")
    }
}
