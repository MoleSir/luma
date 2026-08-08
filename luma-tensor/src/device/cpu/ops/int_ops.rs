//! `impl IntOps for Cpu`: dispatch `CpuIntStorage` variants to generic kernels.

use super::kernels::{elementwise as ew, indexing, matmul, reduce};
use super::{Cpu, CpuBoolStorage, CpuFloatStorage, CpuIntStorage, int_ids_as_usize};
use crate::dtype::{BoolDType, FloatDType, IntDType};
use crate::{
    BinaryOp, CmpOp, DType, Device, Error, IntOps, Layout, Result, Shape, dispatch_int, dispatch_int_raw, dispatch_int2, dispatch_int2_raw,
};

fn build_int(n: usize, dtype: IntDType, v: i64) -> CpuIntStorage {
    match dtype {
        IntDType::I32 => CpuIntStorage::I32(vec![v as i32; n]),
        IntDType::U32 => CpuIntStorage::U32(vec![v as u32; n]),
        IntDType::U8 => CpuIntStorage::U8(vec![v as u8; n]),
    }
}

impl IntOps<Cpu> for Cpu {
    fn i_zeros(shape: &Shape, _device: &Cpu, dtype: IntDType) -> Result<<Cpu as Device>::IntStorage> {
        Ok(build_int(shape.element_count(), dtype, 0))
    }

    fn i_ones(shape: &Shape, _device: &Cpu, dtype: IntDType) -> Result<<Cpu as Device>::IntStorage> {
        Ok(build_int(shape.element_count(), dtype, 1))
    }

    fn i_full(shape: &Shape, value: i64, _device: &Cpu, dtype: IntDType) -> Result<<Cpu as Device>::IntStorage> {
        Ok(build_int(shape.element_count(), dtype, value))
    }

    fn i_from_i64(data: &[i64], _device: &Cpu, dtype: IntDType) -> Result<<Cpu as Device>::IntStorage> {
        Ok(match dtype {
            IntDType::I32 => CpuIntStorage::I32(data.iter().map(|&v| v as i32).collect()),
            IntDType::U32 => CpuIntStorage::U32(data.iter().map(|&v| v as u32).collect()),
            IntDType::U8 => CpuIntStorage::U8(data.iter().map(|&v| v as u8).collect()),
        })
    }

    fn i_arange(start: i64, end: i64, step: i64, _device: &Cpu, dtype: IntDType) -> Result<(<Cpu as Device>::IntStorage, usize)> {
        if step == 0 {
            return Err(Error::Msg("arange step cannot be 0".into()));
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
        let storage = Cpu::i_from_i64(&data, &Cpu, dtype)?;
        Ok((storage, n))
    }

    fn i_contiguous(x: &<Cpu as Device>::IntStorage, l: &Layout) -> Result<<Cpu as Device>::IntStorage> {
        Ok(dispatch_int!(x, |d| super::kernels::iter::gather(d, l)))
    }

    fn i_cast_float(x: &CpuIntStorage, layout: &Layout, to: FloatDType) -> Result<CpuFloatStorage> {
        let s = match to {
            FloatDType::F32 => CpuFloatStorage::F32(dispatch_int_raw!(x, |d| layout.storage_indices().map(|i| d[i] as f32).collect())),
            FloatDType::F64 => CpuFloatStorage::F64(dispatch_int_raw!(x, |d| layout.storage_indices().map(|i| d[i] as f64).collect())),
        };
        Ok(s)
    }

    fn i_cast_int(x: &CpuIntStorage, layout: &Layout, to: IntDType) -> Result<CpuIntStorage> {
        let s = match to {
            IntDType::I32 => CpuIntStorage::I32(dispatch_int_raw!(x, |d| layout.storage_indices().map(|i| d[i] as i32).collect())),
            IntDType::U32 => CpuIntStorage::U32(dispatch_int_raw!(x, |d| layout.storage_indices().map(|i| d[i] as u32).collect())),
            IntDType::U8 => CpuIntStorage::U8(dispatch_int_raw!(x, |d| layout.storage_indices().map(|i| d[i] as u8).collect())),
        };
        Ok(s)
    }

    fn i_cast_bool(x: &CpuIntStorage, layout: &Layout, _to: BoolDType) -> Result<CpuBoolStorage> {
        Ok(CpuBoolStorage(dispatch_int_raw!(x, |d| layout.storage_indices().map(|i| d[i] != 0).collect())))
    }

    fn i_to_vec(x: &<Cpu as Device>::IntStorage, layout: &Layout) -> Result<Vec<i64>> {
        Ok(match x {
            CpuIntStorage::I32(d) => layout.storage_indices().map(|i| d[i] as i64).collect(),
            CpuIntStorage::U32(d) => layout.storage_indices().map(|i| d[i] as i64).collect(),
            CpuIntStorage::U8(d) => layout.storage_indices().map(|i| d[i] as i64).collect(),
        })
    }

    fn i_binary(
        lhs: &<Cpu as Device>::IntStorage,
        lhs_l: &Layout,
        rhs: &<Cpu as Device>::IntStorage,
        rhs_l: &Layout,
        op: BinaryOp,
    ) -> Result<<Cpu as Device>::IntStorage> {
        dispatch_int2!(lhs, rhs, "int binary", |a, b| ew::num_binary(a, lhs_l, b, rhs_l, op))
    }

    fn i_binary_scalar(lhs: &<Cpu as Device>::IntStorage, lhs_l: &Layout, rhs: i64, op: BinaryOp) -> Result<<Cpu as Device>::IntStorage> {
        Ok(match lhs {
            CpuIntStorage::I32(d) => CpuIntStorage::I32(ew::num_binary_scalar(d, lhs_l, rhs as i32, op)),
            CpuIntStorage::U32(d) => CpuIntStorage::U32(ew::num_binary_scalar(d, lhs_l, rhs as u32, op)),
            CpuIntStorage::U8(d) => CpuIntStorage::U8(ew::num_binary_scalar(d, lhs_l, rhs as u8, op)),
        })
    }

    fn i_neg(x: &<Cpu as Device>::IntStorage, l: &Layout) -> Result<<Cpu as Device>::IntStorage> {
        use super::kernels::element::CpuNum;
        fn neg<T: CpuNum>(v: T) -> T {
            T::ZERO - v
        }
        Ok(dispatch_int!(x, |d| ew::unary(d, l, neg)))
    }

    fn i_abs(x: &<Cpu as Device>::IntStorage, l: &Layout) -> Result<<Cpu as Device>::IntStorage> {
        use super::kernels::element::CpuNum;
        Ok(dispatch_int!(x, |d| ew::unary(d, l, CpuNum::abs)))
    }

    fn i_sign(x: &<Cpu as Device>::IntStorage, l: &Layout) -> Result<<Cpu as Device>::IntStorage> {
        use super::kernels::element::CpuNum;
        Ok(dispatch_int!(x, |d| ew::unary(d, l, CpuNum::signum)))
    }

    fn i_affine(x: &<Cpu as Device>::IntStorage, l: &Layout, mul: i64, add: i64) -> Result<<Cpu as Device>::IntStorage> {
        let (mul, add) = (mul, add);
        Ok(match x {
            CpuIntStorage::I32(d) => CpuIntStorage::I32(ew::unary(d, l, |v| (v as i64 * mul + add) as i32)),
            CpuIntStorage::U32(d) => CpuIntStorage::U32(ew::unary(d, l, |v| (v as u64 * mul as u64 + add as u64) as u32)),
            CpuIntStorage::U8(d) => CpuIntStorage::U8(ew::unary(d, l, |v| (v as i64 * mul + add) as u8)),
        })
    }

    fn i_pow(x: &<Cpu as Device>::IntStorage, l: &Layout, exp: i64) -> Result<<Cpu as Device>::IntStorage> {
        Ok(match x {
            CpuIntStorage::I32(d) => CpuIntStorage::I32(ew::unary(d, l, |v| (v as i64).pow(exp as u32) as i32)),
            CpuIntStorage::U32(d) => CpuIntStorage::U32(ew::unary(d, l, |v| (v as u64).pow(exp as u32) as u32)),
            CpuIntStorage::U8(d) => CpuIntStorage::U8(ew::unary(d, l, |v| (v as u64).pow(exp as u32) as u8)),
        })
    }

    fn i_clamp(x: &<Cpu as Device>::IntStorage, l: &Layout, min: Option<i64>, max: Option<i64>) -> Result<<Cpu as Device>::IntStorage> {
        Ok(match x {
            CpuIntStorage::I32(d) => {
                let lo = min.map(|v| v as i32);
                let hi = max.map(|v| v as i32);
                CpuIntStorage::I32(ew::unary(d, l, |v| {
                    let mut val = v;
                    if let Some(lo) = lo { val = val.max(lo); }
                    if let Some(hi) = hi { val = val.min(hi); }
                    val
                }))
            }
            CpuIntStorage::U32(d) => {
                let lo = min.map(|v| v.max(0) as u32);
                let hi = max.map(|v| v as u32);
                CpuIntStorage::U32(ew::unary(d, l, |v| {
                    let mut val = v;
                    if let Some(lo) = lo { val = val.max(lo); }
                    if let Some(hi) = hi { val = val.min(hi); }
                    val
                }))
            }
            CpuIntStorage::U8(d) => {
                let lo = min.map(|v| v.max(0) as u8);
                let hi = max.map(|v| v as u8);
                CpuIntStorage::U8(ew::unary(d, l, |v| {
                    let mut val = v;
                    if let Some(lo) = lo { val = val.max(lo); }
                    if let Some(hi) = hi { val = val.min(hi); }
                    val
                }))
            }
        })
    }

    fn i_matmul(
        lhs: &<Cpu as Device>::IntStorage,
        lhs_l: &Layout,
        rhs: &<Cpu as Device>::IntStorage,
        rhs_l: &Layout,
    ) -> Result<(<Cpu as Device>::IntStorage, Shape)> {
        let result = match (lhs, rhs) {
            (CpuIntStorage::I32(a), CpuIntStorage::I32(b)) => {
                let (vec, shape) = matmul::matmul(a, lhs_l, b, rhs_l)?;
                (CpuIntStorage::I32(vec), shape)
            }
            (CpuIntStorage::U32(a), CpuIntStorage::U32(b)) => {
                let (vec, shape) = matmul::matmul(a, lhs_l, b, rhs_l)?;
                (CpuIntStorage::U32(vec), shape)
            }
            (CpuIntStorage::U8(a), CpuIntStorage::U8(b)) => {
                let (vec, shape) = matmul::matmul(a, lhs_l, b, rhs_l)?;
                (CpuIntStorage::U8(vec), shape)
            }
            (l, r) => return Err(Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: "int matmul" }),
        };
        Ok(result)
    }

    fn i_cmp(
        lhs: &<Cpu as Device>::IntStorage,
        lhs_l: &Layout,
        rhs: &<Cpu as Device>::IntStorage,
        rhs_l: &Layout,
        op: CmpOp,
    ) -> Result<<Cpu as Device>::BoolStorage> {
        let v = dispatch_int2_raw!(lhs, rhs, "int cmp", |a, b| ew::num_cmp(a, lhs_l, b, rhs_l, op))?;
        Ok(CpuBoolStorage(v))
    }

    fn i_reduce(
        x: &<Cpu as Device>::IntStorage,
        l: &Layout,
        dims: &[usize],
        keepdim: bool,
        op: crate::ReduceOp,
    ) -> Result<(<Cpu as Device>::IntStorage, Shape)> {
        let reducer = reduce::Reducer::from(op);
        match x {
            CpuIntStorage::I32(d) => {
                let (v, s) = reduce::reduce_dims(d, l, dims, keepdim, reducer)?;
                Ok((CpuIntStorage::I32(v), s))
            }
            CpuIntStorage::U32(d) => {
                let (v, s) = reduce::reduce_dims(d, l, dims, keepdim, reducer)?;
                Ok((CpuIntStorage::U32(v), s))
            }
            CpuIntStorage::U8(d) => {
                let (v, s) = reduce::reduce_dims(d, l, dims, keepdim, reducer)?;
                Ok((CpuIntStorage::U8(v), s))
            }
        }
    }

    fn i_index_select(
        x: &<Cpu as Device>::IntStorage,
        x_l: &Layout,
        idx: &<Cpu as Device>::IntStorage,
        idx_l: &Layout,
        dim: usize,
    ) -> Result<(<Cpu as Device>::IntStorage, Shape)> {
        let ids = int_ids_as_usize(idx, idx_l);
        match x {
            CpuIntStorage::I32(d) => {
                let (v, dims) = indexing::index_select(d, x_l, &ids, idx_l, dim)?;
                Ok((CpuIntStorage::I32(v), Shape::from(dims)))
            }
            CpuIntStorage::U32(d) => {
                let (v, dims) = indexing::index_select(d, x_l, &ids, idx_l, dim)?;
                Ok((CpuIntStorage::U32(v), Shape::from(dims)))
            }
            CpuIntStorage::U8(d) => {
                let (v, dims) = indexing::index_select(d, x_l, &ids, idx_l, dim)?;
                Ok((CpuIntStorage::U8(v), Shape::from(dims)))
            }
        }
    }

    fn i_gather(
        x: &<Cpu as Device>::IntStorage,
        x_l: &Layout,
        idx: &<Cpu as Device>::IntStorage,
        idx_l: &Layout,
        dim: usize,
    ) -> Result<(<Cpu as Device>::IntStorage, Shape)> {
        let ids = int_ids_as_usize(idx, idx_l);
        match x {
            CpuIntStorage::I32(d) => {
                let (v, dims) = indexing::gather(d, x_l, &ids, idx_l, dim)?;
                Ok((CpuIntStorage::I32(v), Shape::from(dims)))
            }
            CpuIntStorage::U32(d) => {
                let (v, dims) = indexing::gather(d, x_l, &ids, idx_l, dim)?;
                Ok((CpuIntStorage::U32(v), Shape::from(dims)))
            }
            CpuIntStorage::U8(d) => {
                let (v, dims) = indexing::gather(d, x_l, &ids, idx_l, dim)?;
                Ok((CpuIntStorage::U8(v), Shape::from(dims)))
            }
        }
    }

    fn i_cat(srcs: &[(&<Cpu as Device>::IntStorage, &Layout)], dim: usize) -> Result<(<Cpu as Device>::IntStorage, Shape)> {
        if srcs.is_empty() {
            return Err(Error::OpRequiresAtLeastOneTensor { op: "cat" });
        }
        let dt = srcs[0].0.dtype();
        for (s, _) in srcs {
            if s.dtype() != dt {
                return Err(Error::DTypeMismatch { lhs: dt, rhs: s.dtype(), op: "cat" });
            }
        }
        macro_rules! cat_variant {
            ($variant:path, $getter:ident) => {{
                let views: Vec<(&[_], &Layout)> = srcs.iter().map(|(s, l)| ($getter(s), *l)).collect();
                let (v, shape) = super::kernels::shape::cat(&views, dim)?;
                Ok(($variant(v), shape))
            }};
        }
        match dt {
            DType::I32 => cat_variant!(CpuIntStorage::I32, as_i32),
            DType::U32 => cat_variant!(CpuIntStorage::U32, as_u32),
            _ => cat_variant!(CpuIntStorage::U8, as_u8),
        }
    }

    fn i_arg_reduce(
        x: &<Cpu as Device>::IntStorage,
        layout: &Layout,
        dim: usize,
        keepdim: bool,
        take_max: bool,
    ) -> Result<(<Cpu as Device>::IntStorage, Shape)> {
        let (indices, shape) = dispatch_int_raw!(x, |d| reduce::arg_reduce(d, layout, dim, keepdim, take_max))?;
        let indices_u32: Vec<u32> = indices.into_iter().map(|i| i as u32).collect();
        Ok((CpuIntStorage::U32(indices_u32), shape))
    }

    fn i_index_add(
        init: &<Cpu as Device>::IntStorage,
        init_l: &Layout,
        idx: &<Cpu as Device>::IntStorage,
        idx_l: &Layout,
        src: &<Cpu as Device>::IntStorage,
        _src_l: &Layout,
        dim: usize,
    ) -> Result<<Cpu as Device>::IntStorage> {
        let idx_u = int_ids_as_usize(idx, idx_l);
        match (init, src) {
            (CpuIntStorage::I32(init_d), CpuIntStorage::I32(src_d)) => {
                let result = indexing::index_add(init_d, init_l, &idx_u, idx_l, src_d, dim)?;
                Ok(CpuIntStorage::I32(result))
            }
            (CpuIntStorage::U32(init_d), CpuIntStorage::U32(src_d)) => {
                let result = indexing::index_add(init_d, init_l, &idx_u, idx_l, src_d, dim)?;
                Ok(CpuIntStorage::U32(result))
            }
            (CpuIntStorage::U8(init_d), CpuIntStorage::U8(src_d)) => {
                let result = indexing::index_add(init_d, init_l, &idx_u, idx_l, src_d, dim)?;
                Ok(CpuIntStorage::U8(result))
            }
            (l, r) => Err(Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: "int index_add" }),
        }
    }

    fn i_scatter_add(
        init: &<Cpu as Device>::IntStorage,
        init_l: &Layout,
        idx: &<Cpu as Device>::IntStorage,
        idx_l: &Layout,
        src: &<Cpu as Device>::IntStorage,
        _src_l: &Layout,
        dim: usize,
    ) -> Result<<Cpu as Device>::IntStorage> {
        let idx_u = int_ids_as_usize(idx, idx_l);
        match (init, src) {
            (CpuIntStorage::I32(init_d), CpuIntStorage::I32(src_d)) => {
                let result = indexing::scatter_add(init_d, init_l, &idx_u, idx_l, src_d, dim)?;
                Ok(CpuIntStorage::I32(result))
            }
            (CpuIntStorage::U32(init_d), CpuIntStorage::U32(src_d)) => {
                let result = indexing::scatter_add(init_d, init_l, &idx_u, idx_l, src_d, dim)?;
                Ok(CpuIntStorage::U32(result))
            }
            (CpuIntStorage::U8(init_d), CpuIntStorage::U8(src_d)) => {
                let result = indexing::scatter_add(init_d, init_l, &idx_u, idx_l, src_d, dim)?;
                Ok(CpuIntStorage::U8(result))
            }
            (l, r) => Err(Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: "int scatter_add" }),
        }
    }

    fn i_pick(
        mask: &<Cpu as Device>::BoolStorage,
        mask_l: &Layout,
        on_true: &<Cpu as Device>::IntStorage,
        true_l: &Layout,
        on_false: &<Cpu as Device>::IntStorage,
        false_l: &Layout,
    ) -> Result<<Cpu as Device>::IntStorage> {
        let m: Vec<bool> = mask_l.storage_indices().map(|i| mask.0[i]).collect();
        match (on_true, on_false) {
            (CpuIntStorage::I32(t), CpuIntStorage::I32(f)) => {
                let tv: Vec<i32> = true_l.storage_indices().map(|i| t[i]).collect();
                let fv: Vec<i32> = false_l.storage_indices().map(|i| f[i]).collect();
                Ok(CpuIntStorage::I32(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { fv[i] }).collect()))
            }
            (CpuIntStorage::U32(t), CpuIntStorage::U32(f)) => {
                let tv: Vec<u32> = true_l.storage_indices().map(|i| t[i]).collect();
                let fv: Vec<u32> = false_l.storage_indices().map(|i| f[i]).collect();
                Ok(CpuIntStorage::U32(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { fv[i] }).collect()))
            }
            (CpuIntStorage::U8(t), CpuIntStorage::U8(f)) => {
                let tv: Vec<u8> = true_l.storage_indices().map(|i| t[i]).collect();
                let fv: Vec<u8> = false_l.storage_indices().map(|i| f[i]).collect();
                Ok(CpuIntStorage::U8(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { fv[i] }).collect()))
            }
            (l, r) => Err(Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: "pick" }),
        }
    }

    fn i_pick_true(
        mask: &<Cpu as Device>::BoolStorage,
        mask_l: &Layout,
        value: i64,
        on_false: &<Cpu as Device>::IntStorage,
        false_l: &Layout,
    ) -> Result<<Cpu as Device>::IntStorage> {
        let m: Vec<bool> = mask_l.storage_indices().map(|i| mask.0[i]).collect();
        match on_false {
            CpuIntStorage::I32(f) => {
                let fv: Vec<i32> = false_l.storage_indices().map(|i| f[i]).collect();
                let val = value as i32;
                Ok(CpuIntStorage::I32(m.iter().enumerate().map(|(i, &c)| if c { val } else { fv[i] }).collect()))
            }
            CpuIntStorage::U32(f) => {
                let fv: Vec<u32> = false_l.storage_indices().map(|i| f[i]).collect();
                let val = value as u32;
                Ok(CpuIntStorage::U32(m.iter().enumerate().map(|(i, &c)| if c { val } else { fv[i] }).collect()))
            }
            CpuIntStorage::U8(f) => {
                let fv: Vec<u8> = false_l.storage_indices().map(|i| f[i]).collect();
                let val = value as u8;
                Ok(CpuIntStorage::U8(m.iter().enumerate().map(|(i, &c)| if c { val } else { fv[i] }).collect()))
            }
        }
    }

    fn i_pick_false(
        mask: &<Cpu as Device>::BoolStorage,
        mask_l: &Layout,
        on_true: &<Cpu as Device>::IntStorage,
        true_l: &Layout,
        value: i64,
    ) -> Result<<Cpu as Device>::IntStorage> {
        let m: Vec<bool> = mask_l.storage_indices().map(|i| mask.0[i]).collect();
        match on_true {
            CpuIntStorage::I32(t) => {
                let tv: Vec<i32> = true_l.storage_indices().map(|i| t[i]).collect();
                let val = value as i32;
                Ok(CpuIntStorage::I32(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { val }).collect()))
            }
            CpuIntStorage::U32(t) => {
                let tv: Vec<u32> = true_l.storage_indices().map(|i| t[i]).collect();
                let val = value as u32;
                Ok(CpuIntStorage::U32(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { val }).collect()))
            }
            CpuIntStorage::U8(t) => {
                let tv: Vec<u8> = true_l.storage_indices().map(|i| t[i]).collect();
                let val = value as u8;
                Ok(CpuIntStorage::U8(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { val }).collect()))
            }
        }
    }

    fn i_allclose(a: &CpuIntStorage, a_l: &Layout, b: &CpuIntStorage, b_l: &Layout) -> Result<bool> {
        match (a, b) {
            (CpuIntStorage::I32(av), CpuIntStorage::I32(bv)) => {
                Ok(a_l.storage_indices().zip(b_l.storage_indices()).all(|(ai, bi)| av[ai] == bv[bi]))
            }
            (CpuIntStorage::U32(av), CpuIntStorage::U32(bv)) => {
                Ok(a_l.storage_indices().zip(b_l.storage_indices()).all(|(ai, bi)| av[ai] == bv[bi]))
            }
            (CpuIntStorage::U8(av), CpuIntStorage::U8(bv)) => {
                Ok(a_l.storage_indices().zip(b_l.storage_indices()).all(|(ai, bi)| av[ai] == bv[bi]))
            }
            _ => Err(crate::Error::DTypeMismatch { lhs: a.dtype(), rhs: b.dtype(), op: "allclose" }),
        }
    }
}

fn as_i32(s: &CpuIntStorage) -> &[i32] {
    match s {
        CpuIntStorage::I32(d) => d,
        _ => unreachable!(),
    }
}
fn as_u32(s: &CpuIntStorage) -> &[u32] {
    match s {
        CpuIntStorage::U32(d) => d,
        _ => unreachable!(),
    }
}
fn as_u8(s: &CpuIntStorage) -> &[u8] {
    match s {
        CpuIntStorage::U8(d) => d,
        _ => unreachable!(),
    }
}
