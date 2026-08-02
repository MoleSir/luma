//! `impl FloatOps for Cpu`: dispatch `CpuFloatStorage` variants to generic kernels.

use rand::rng;
use rand_distr::{Distribution, Normal, Uniform};

use super::kernels::{elementwise as ew, indexing, matmul, nn, reduce};
use super::{Cpu, CpuBoolStorage, CpuFloatStorage, CpuIntStorage, int_ids_as_usize, usize_to_int_storage};
use crate::dtype::{BoolDType, FloatDType, IntDType};
use crate::{
    BinaryOp, CmpOp, DType, Device, Error, FloatOps, Layout, Result, Shape, UnaryOp, dispatch_float, dispatch_float_raw, dispatch_float2,
    dispatch_float2_raw,
};

/// Build a float storage of `dtype`, filling `n` elements via `f32`/`f64` closures.
fn build(n: usize, dtype: FloatDType, f32v: impl Fn() -> f32, f64v: impl Fn() -> f64) -> CpuFloatStorage {
    match dtype {
        FloatDType::F32 => CpuFloatStorage::F32((0..n).map(|_| f32v()).collect()),
        FloatDType::F64 => CpuFloatStorage::F64((0..n).map(|_| f64v()).collect()),
    }
}

impl FloatOps<Cpu> for Cpu {
    fn f_zeros(shape: &Shape, _device: &Cpu, dtype: FloatDType) -> Result<<Cpu as Device>::FloatStorage> {
        Ok(build(shape.element_count(), dtype, || 0.0, || 0.0))
    }

    fn f_ones(shape: &Shape, _device: &Cpu, dtype: FloatDType) -> Result<<Cpu as Device>::FloatStorage> {
        Ok(build(shape.element_count(), dtype, || 1.0, || 1.0))
    }

    fn f_full(shape: &Shape, value: f64, _device: &Cpu, dtype: FloatDType) -> Result<<Cpu as Device>::FloatStorage> {
        Ok(build(shape.element_count(), dtype, || value as f32, || value))
    }

    fn f_from_f64(data: &[f64], dtype: FloatDType) -> Result<<Cpu as Device>::FloatStorage> {
        Ok(match dtype {
            FloatDType::F32 => CpuFloatStorage::F32(data.iter().map(|&v| v as f32).collect()),
            FloatDType::F64 => CpuFloatStorage::F64(data.to_vec()),
        })
    }

    fn f_rand_uniform(shape: &Shape, lo: f64, hi: f64, _device: &Cpu, dtype: FloatDType) -> Result<<Cpu as Device>::FloatStorage> {
        let n = shape.element_count();
        let mut r = rng();
        let s = match dtype {
            FloatDType::F64 => {
                let u = Uniform::new(lo, hi).map_err(|e| Error::Rand(e.to_string()))?;
                CpuFloatStorage::F64((0..n).map(|_| u.sample(&mut r)).collect())
            }
            FloatDType::F32 => {
                let u = Uniform::new(lo as f32, hi as f32).map_err(|e| Error::Rand(e.to_string()))?;
                CpuFloatStorage::F32((0..n).map(|_| u.sample(&mut r)).collect())
            }
        };
        Ok(s)
    }

    fn f_rand_normal(shape: &Shape, mean: f64, std: f64, _device: &Cpu, dtype: FloatDType) -> Result<<Cpu as Device>::FloatStorage> {
        let n = shape.element_count();
        let mut r = rng();
        let s = match dtype {
            FloatDType::F64 => {
                let d = Normal::new(mean, std).map_err(|e| Error::Rand(e.to_string()))?;
                CpuFloatStorage::F64((0..n).map(|_| d.sample(&mut r)).collect())
            }
            FloatDType::F32 => {
                let d = Normal::new(mean as f32, std as f32).map_err(|e| Error::Rand(e.to_string()))?;
                CpuFloatStorage::F32((0..n).map(|_| d.sample(&mut r)).collect())
            }
        };
        Ok(s)
    }

    fn f_contiguous(x: &<Cpu as Device>::FloatStorage, l: &Layout) -> Result<<Cpu as Device>::FloatStorage> {
        Ok(dispatch_float!(x, |d| super::kernels::iter::gather(d, l)))
    }

    fn f_cast_float(x: &CpuFloatStorage, layout: &Layout, to: FloatDType) -> Result<CpuFloatStorage> {
        let s = match to {
            FloatDType::F32 => CpuFloatStorage::F32(dispatch_float_raw!(x, |d| layout.storage_indices().map(|i| d[i] as f32).collect())),
            FloatDType::F64 => CpuFloatStorage::F64(dispatch_float_raw!(x, |d| layout.storage_indices().map(|i| d[i] as f64).collect())),
        };
        Ok(s)
    }

    fn f_cast_int(x: &CpuFloatStorage, layout: &Layout, to: IntDType) -> Result<CpuIntStorage> {
        let s = match to {
            IntDType::I32 => CpuIntStorage::I32(dispatch_float_raw!(x, |d| layout.storage_indices().map(|i| d[i] as i32).collect())),
            IntDType::U32 => CpuIntStorage::U32(dispatch_float_raw!(x, |d| layout.storage_indices().map(|i| d[i] as u32).collect())),
            IntDType::U8 => CpuIntStorage::U8(dispatch_float_raw!(x, |d| layout.storage_indices().map(|i| d[i] as u8).collect())),
        };
        Ok(s)
    }

    fn f_cast_bool(x: &CpuFloatStorage, layout: &Layout, _to: BoolDType) -> Result<CpuBoolStorage> {
        Ok(CpuBoolStorage(dispatch_float_raw!(x, |d| layout.storage_indices().map(|i| d[i] != 0.).collect())))
    }

    fn f_to_vec(x: &<Cpu as Device>::FloatStorage, layout: &Layout) -> Result<Vec<f64>> {
        Ok(match x {
            CpuFloatStorage::F32(d) => layout.storage_indices().map(|i| d[i] as f64).collect(),
            CpuFloatStorage::F64(d) => layout.storage_indices().map(|i| d[i]).collect(),
        })
    }

    fn f_binary(
        lhs: &<Cpu as Device>::FloatStorage,
        lhs_l: &Layout,
        rhs: &<Cpu as Device>::FloatStorage,
        rhs_l: &Layout,
        op: BinaryOp,
    ) -> Result<<Cpu as Device>::FloatStorage> {
        dispatch_float2!(lhs, rhs, "binary", |a, b| ew::num_binary(a, lhs_l, b, rhs_l, op))
    }

    fn f_binary_scalar(
        lhs: &<Cpu as Device>::FloatStorage,
        lhs_l: &Layout,
        rhs: f64,
        op: BinaryOp,
    ) -> Result<<Cpu as Device>::FloatStorage> {
        Ok(match lhs {
            CpuFloatStorage::F32(d) => CpuFloatStorage::F32(ew::num_binary_scalar(d, lhs_l, rhs as f32, op)),
            CpuFloatStorage::F64(d) => CpuFloatStorage::F64(ew::num_binary_scalar(d, lhs_l, rhs, op)),
        })
    }

    fn f_unary(x: &<Cpu as Device>::FloatStorage, l: &Layout, op: UnaryOp) -> Result<<Cpu as Device>::FloatStorage> {
        Ok(dispatch_float!(x, |d| ew::float_unary(d, l, op)))
    }

    fn f_cmp(
        lhs: &<Cpu as Device>::FloatStorage,
        lhs_l: &Layout,
        rhs: &<Cpu as Device>::FloatStorage,
        rhs_l: &Layout,
        op: CmpOp,
    ) -> Result<<Cpu as Device>::BoolStorage> {
        let v = dispatch_float2_raw!(lhs, rhs, "cmp", |a, b| ew::num_cmp(a, lhs_l, b, rhs_l, op))?;
        Ok(CpuBoolStorage(v))
    }

    fn f_reduce(
        x: &<Cpu as Device>::FloatStorage,
        l: &Layout,
        dims: &[usize],
        keepdim: bool,
        op: crate::ReduceOp,
    ) -> Result<(<Cpu as Device>::FloatStorage, Shape)> {
        let reducer = reduce::Reducer::from(op);
        match x {
            CpuFloatStorage::F32(d) => {
                let (v, s) = reduce::reduce_dims(d, l, dims, keepdim, reducer)?;
                Ok((CpuFloatStorage::F32(v), s))
            }
            CpuFloatStorage::F64(d) => {
                let (v, s) = reduce::reduce_dims(d, l, dims, keepdim, reducer)?;
                Ok((CpuFloatStorage::F64(v), s))
            }
        }
    }

    fn f_arg_reduce(
        x: &<Cpu as Device>::FloatStorage,
        l: &Layout,
        dim: usize,
        keepdim: bool,
        take_max: bool,
    ) -> Result<(<Cpu as Device>::IntStorage, Shape)> {
        let (idx, shape) = dispatch_float_raw!(x, |d| reduce::arg_reduce(d, l, dim, keepdim, take_max))?;
        Ok((usize_to_int_storage(&idx, DType::U32), shape))
    }

    fn f_matmul(
        lhs: &<Cpu as Device>::FloatStorage,
        lhs_l: &Layout,
        rhs: &<Cpu as Device>::FloatStorage,
        rhs_l: &Layout,
    ) -> Result<(<Cpu as Device>::FloatStorage, Shape)> {
        match (lhs, rhs) {
            (CpuFloatStorage::F32(a), CpuFloatStorage::F32(b)) => {
                let (v, s) = matmul::matmul(a, lhs_l, b, rhs_l)?;
                Ok((CpuFloatStorage::F32(v), s))
            }
            (CpuFloatStorage::F64(a), CpuFloatStorage::F64(b)) => {
                let (v, s) = matmul::matmul(a, lhs_l, b, rhs_l)?;
                Ok((CpuFloatStorage::F64(v), s))
            }
            (l, r) => Err(Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: "matmul" }),
        }
    }

    fn f_add_matmul_(
        dst: &mut <Cpu as Device>::FloatStorage,
        dst_l: &Layout,
        lhs: &<Cpu as Device>::FloatStorage,
        lhs_l: &Layout,
        rhs: &<Cpu as Device>::FloatStorage,
        rhs_l: &Layout,
    ) -> Result<()> {
        // dst += lhs @ rhs  (fused, no temporary product buffer)
        match (dst, lhs, rhs) {
            (CpuFloatStorage::F32(d), CpuFloatStorage::F32(l), CpuFloatStorage::F32(r)) => {
                matmul::add_matmul(d, dst_l, l, lhs_l, r, rhs_l)
            }
            (CpuFloatStorage::F64(d), CpuFloatStorage::F64(l), CpuFloatStorage::F64(r)) => {
                matmul::add_matmul(d, dst_l, l, lhs_l, r, rhs_l)
            }
            (_d, l, r) => Err(Error::DTypeMismatch {
                lhs: l.dtype(),
                rhs: r.dtype(),
                op: "f_add_matmul_",
            }),
        }
    }

    fn f_binary_(
        dst: &mut <Cpu as Device>::FloatStorage,
        dst_l: &Layout,
        src: &<Cpu as Device>::FloatStorage,
        src_l: &Layout,
        op: BinaryOp,
    ) -> Result<()> {
        match (dst, src) {
            (CpuFloatStorage::F32(d), CpuFloatStorage::F32(s)) => {
                ew::binary_(d, dst_l, s, src_l, binary_fn_f32(op));
                Ok(())
            }
            (CpuFloatStorage::F64(d), CpuFloatStorage::F64(s)) => {
                ew::binary_(d, dst_l, s, src_l, binary_fn_f64(op));
                Ok(())
            }
            (d, s) => Err(Error::DTypeMismatch { lhs: d.dtype(), rhs: s.dtype(), op: "in-place binary" }),
        }
    }

    fn f_index_select(
        x: &<Cpu as Device>::FloatStorage,
        x_l: &Layout,
        idx: &<Cpu as Device>::IntStorage,
        idx_l: &Layout,
        dim: usize,
    ) -> Result<(<Cpu as Device>::FloatStorage, Shape)> {
        let ids = int_ids_as_usize(idx, idx_l);
        match x {
            CpuFloatStorage::F32(d) => {
                let (v, dims) = indexing::index_select(d, x_l, &ids, idx_l, dim)?;
                Ok((CpuFloatStorage::F32(v), Shape::from(dims)))
            }
            CpuFloatStorage::F64(d) => {
                let (v, dims) = indexing::index_select(d, x_l, &ids, idx_l, dim)?;
                Ok((CpuFloatStorage::F64(v), Shape::from(dims)))
            }
        }
    }

    fn f_gather(
        x: &<Cpu as Device>::FloatStorage,
        x_l: &Layout,
        idx: &<Cpu as Device>::IntStorage,
        idx_l: &Layout,
        dim: usize,
    ) -> Result<(<Cpu as Device>::FloatStorage, Shape)> {
        let ids = int_ids_as_usize(idx, idx_l);
        match x {
            CpuFloatStorage::F32(d) => {
                let (v, dims) = indexing::gather(d, x_l, &ids, idx_l, dim)?;
                Ok((CpuFloatStorage::F32(v), Shape::from(dims)))
            }
            CpuFloatStorage::F64(d) => {
                let (v, dims) = indexing::gather(d, x_l, &ids, idx_l, dim)?;
                Ok((CpuFloatStorage::F64(v), Shape::from(dims)))
            }
        }
    }

    fn f_index_add(
        init: &<Cpu as Device>::FloatStorage,
        init_l: &Layout,
        idx: &<Cpu as Device>::IntStorage,
        idx_l: &Layout,
        src: &<Cpu as Device>::FloatStorage,
        _src_l: &Layout,
        dim: usize,
    ) -> Result<<Cpu as Device>::FloatStorage> {
        let ids = int_ids_as_usize(idx, idx_l);
        dispatch_float2!(init, src, "index-add", |a, b| indexing::index_add(a, init_l, &ids, idx_l, b, dim)?)
    }

    fn f_scatter_add(
        init: &<Cpu as Device>::FloatStorage,
        init_l: &Layout,
        idx: &<Cpu as Device>::IntStorage,
        idx_l: &Layout,
        src: &<Cpu as Device>::FloatStorage,
        _src_l: &Layout,
        dim: usize,
    ) -> Result<<Cpu as Device>::FloatStorage> {
        let ids = int_ids_as_usize(idx, idx_l);
        dispatch_float2!(init, src, "scatter-add", |a, b| indexing::scatter_add(a, init_l, &ids, idx_l, b, dim)?)
    }

    fn f_cat(srcs: &[(&<Cpu as Device>::FloatStorage, &Layout)], dim: usize) -> Result<(<Cpu as Device>::FloatStorage, Shape)> {
        if srcs.is_empty() {
            return Err(Error::OpRequiresAtLeastOneTensor { op: "cat" });
        }
        // all must share dtype (checked against the first)
        let dt = srcs[0].0.dtype();
        for (s, _) in srcs {
            if s.dtype() != dt {
                return Err(Error::DTypeMismatch { lhs: dt, rhs: s.dtype(), op: "cat" });
            }
        }
        match dt {
            DType::F32 => {
                let views: Vec<(&[f32], &Layout)> = srcs.iter().map(|(s, l)| (as_f32(s), *l)).collect();
                let (v, shape) = super::kernels::shape::cat(&views, dim)?;
                Ok((CpuFloatStorage::F32(v), shape))
            }
            _ => {
                let views: Vec<(&[f64], &Layout)> = srcs.iter().map(|(s, l)| (as_f64(s), *l)).collect();
                let (v, shape) = super::kernels::shape::cat(&views, dim)?;
                Ok((CpuFloatStorage::F64(v), shape))
            }
        }
    }

    fn f_softmax(x: &<Cpu as Device>::FloatStorage, l: &Layout, dim: usize) -> Result<<Cpu as Device>::FloatStorage> {
        match x {
            CpuFloatStorage::F32(d) => Ok(CpuFloatStorage::F32(nn::softmax(d, l, dim)?)),
            CpuFloatStorage::F64(d) => Ok(CpuFloatStorage::F64(nn::softmax(d, l, dim)?)),
        }
    }

    fn f_rms_norm(
        x: &<Cpu as Device>::FloatStorage,
        x_l: &Layout,
        weight: &<Cpu as Device>::FloatStorage,
        weight_l: &Layout,
        eps: f64,
    ) -> Result<<Cpu as Device>::FloatStorage> {
        match (x, weight) {
            (CpuFloatStorage::F32(d), CpuFloatStorage::F32(w)) => Ok(CpuFloatStorage::F32(nn::rms_norm(d, x_l, w, weight_l, eps as f32)?)),
            (CpuFloatStorage::F64(d), CpuFloatStorage::F64(w)) => Ok(CpuFloatStorage::F64(nn::rms_norm(d, x_l, w, weight_l, eps)?)),
            (l, r) => Err(Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: "rms_norm" }),
        }
    }

    fn f_if_else(
        mask: &<Cpu as Device>::BoolStorage,
        mask_l: &Layout,
        on_true: &<Cpu as Device>::FloatStorage,
        true_l: &Layout,
        on_false: &<Cpu as Device>::FloatStorage,
        false_l: &Layout,
    ) -> Result<<Cpu as Device>::FloatStorage> {
        let m: Vec<bool> = mask_l.storage_indices().map(|i| mask.0[i]).collect();
        match (on_true, on_false) {
            (CpuFloatStorage::F32(t), CpuFloatStorage::F32(f)) => {
                let tv = super::kernels::iter::gather(t, true_l);
                let fv = super::kernels::iter::gather(f, false_l);
                Ok(CpuFloatStorage::F32(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { fv[i] }).collect()))
            }
            (CpuFloatStorage::F64(t), CpuFloatStorage::F64(f)) => {
                let tv = super::kernels::iter::gather(t, true_l);
                let fv = super::kernels::iter::gather(f, false_l);
                Ok(CpuFloatStorage::F64(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { fv[i] }).collect()))
            }
            (l, r) => Err(Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: "if_else" }),
        }
    }
}

fn as_f32(s: &CpuFloatStorage) -> &[f32] {
    match s {
        CpuFloatStorage::F32(d) => d,
        _ => unreachable!("dtype checked by caller"),
    }
}

fn as_f64(s: &CpuFloatStorage) -> &[f64] {
    match s {
        CpuFloatStorage::F64(d) => d,
        _ => unreachable!("dtype checked by caller"),
    }
}

fn binary_fn_f32(op: BinaryOp) -> fn(f32, f32) -> f32 {
    use super::kernels::element::CpuNum;
    match op {
        BinaryOp::Add => |a, b| a + b,
        BinaryOp::Sub => |a, b| a - b,
        BinaryOp::Mul => |a, b| a * b,
        BinaryOp::Div => |a, b| a / b,
        BinaryOp::Maximum => CpuNum::maximum,
        BinaryOp::Minimum => CpuNum::minimum,
    }
}

fn binary_fn_f64(op: BinaryOp) -> fn(f64, f64) -> f64 {
    use super::kernels::element::CpuNum;
    match op {
        BinaryOp::Add => |a, b| a + b,
        BinaryOp::Sub => |a, b| a - b,
        BinaryOp::Mul => |a, b| a * b,
        BinaryOp::Div => |a, b| a / b,
        BinaryOp::Maximum => CpuNum::maximum,
        BinaryOp::Minimum => CpuNum::minimum,
    }
}
