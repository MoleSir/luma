//! `impl IntOps for Cpu`: dispatch `CpuIntStorage` variants to generic kernels.

use std::borrow::Cow;

use super::kernels::{elementwise as ew, indexing, matmul, reduce};
use super::{Cpu, CpuBoolStorage, CpuFloatStorage, CpuIntStorage, int_ids_as_usize};
use crate::dtype::{BoolDType, FloatDType, IntDType};
use crate::{
    BinaryOp, CmpOp, DType, Device, Error, IntOps, Layout, Result, Shape, Storage, dispatch_int, dispatch_int_raw, dispatch_int2,
    dispatch_int2_raw,
};

fn build_int(n: usize, dtype: IntDType, device: &Cpu, v: i64) -> CpuIntStorage {
    match dtype {
        IntDType::I32 => CpuIntStorage::I32(device.fill_alloc(n, v as i32), device.clone()),
        IntDType::U32 => CpuIntStorage::U32(device.fill_alloc(n, v as u32), device.clone()),
        IntDType::U8 => CpuIntStorage::U8(device.fill_alloc(n, v as u8), device.clone()),
    }
}

impl IntOps<Cpu> for Cpu {
    fn i_zeros(shape: &Shape, device: &Cpu, dtype: IntDType) -> Result<<Cpu as Device>::IntStorage> {
        Ok(build_int(shape.element_count(), dtype, device, 0))
    }

    fn i_ones(shape: &Shape, device: &Cpu, dtype: IntDType) -> Result<<Cpu as Device>::IntStorage> {
        Ok(build_int(shape.element_count(), dtype, device, 1))
    }

    fn i_full(shape: &Shape, value: i64, device: &Cpu, dtype: IntDType) -> Result<<Cpu as Device>::IntStorage> {
        Ok(build_int(shape.element_count(), dtype, device, value))
    }

    fn i_from_i64<'a>(data: impl Into<Cow<'a, [i64]>>, device: &Cpu) -> Result<<Cpu as Device>::IntStorage> {
        let data = data.into();
        Ok(match data {
            Cow::Owned(v) => CpuIntStorage::I32(device.collect_alloc(v.iter().map(|&x| x as i32)), device.clone()),
            Cow::Borrowed(s) => CpuIntStorage::I32(device.collect_alloc(s.iter().map(|&x| x as i32)), device.clone()),
        })
    }

    fn i_from_i32<'a>(data: impl Into<Cow<'a, [i32]>>, device: &Cpu) -> Result<<Cpu as Device>::IntStorage> {
        let data = data.into();
        Ok(match data {
            Cow::Owned(v) => CpuIntStorage::I32(v, device.clone()),
            Cow::Borrowed(s) => CpuIntStorage::I32(device.collect_alloc(s.iter().copied()), device.clone()),
        })
    }

    fn i_from_u32<'a>(data: impl Into<Cow<'a, [u32]>>, device: &Cpu) -> Result<<Cpu as Device>::IntStorage> {
        let data = data.into();
        Ok(match data {
            Cow::Owned(v) => CpuIntStorage::U32(v, device.clone()),
            Cow::Borrowed(s) => CpuIntStorage::U32(device.collect_alloc(s.iter().copied()), device.clone()),
        })
    }

    fn i_from_u8<'a>(data: impl Into<Cow<'a, [u8]>>, device: &Cpu) -> Result<<Cpu as Device>::IntStorage> {
        let data = data.into();
        Ok(match data {
            Cow::Owned(v) => CpuIntStorage::U8(v, device.clone()),
            Cow::Borrowed(s) => CpuIntStorage::U8(device.collect_alloc(s.iter().copied()), device.clone()),
        })
    }

    fn i_from_bytes<'a>(
        bytes: impl Into<Cow<'a, [u8]>>,
        _shape: &Shape,
        device: &Cpu,
        dtype: IntDType,
    ) -> Result<<Cpu as Device>::IntStorage> {
        let bytes = bytes.into();
        Ok(match dtype {
            IntDType::I32 => {
                let v: Vec<i32> = device.collect_alloc(bytes.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())));
                CpuIntStorage::I32(v, device.clone())
            }
            IntDType::U32 => {
                let v: Vec<u32> = device.collect_alloc(bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())));
                CpuIntStorage::U32(v, device.clone())
            }
            IntDType::U8 => match bytes {
                Cow::Owned(b) => CpuIntStorage::U8(b, device.clone()),
                Cow::Borrowed(b) => CpuIntStorage::U8(device.collect_alloc(b.iter().copied()), device.clone()),
            },
        })
    }

    fn i_arange(start: i64, end: i64, step: i64, device: &Cpu, dtype: IntDType) -> Result<(<Cpu as Device>::IntStorage, usize)> {
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
        let storage = match dtype {
            IntDType::I32 => {
                let v: Vec<i32> = device.collect_alloc(data.iter().map(|&x| x as i32));
                Cpu::i_from_i32(v, device)?
            }
            IntDType::U32 => {
                let v: Vec<u32> = device.collect_alloc(data.iter().map(|&x| x as u32));
                Cpu::i_from_u32(v, device)?
            }
            IntDType::U8 => {
                let v: Vec<u8> = device.collect_alloc(data.iter().map(|&x| x as u8));
                Cpu::i_from_u8(v, device)?
            }
        };
        Ok((storage, n))
    }

    fn i_contiguous(x: &<Cpu as Device>::IntStorage, l: &Layout) -> Result<<Cpu as Device>::IntStorage> {
        Ok(dispatch_int!(x, |d| super::kernels::iter::gather(d, l, x.device())))
    }

    fn i_cast_float(x: &CpuIntStorage, layout: &Layout, to: FloatDType) -> Result<CpuFloatStorage> {
        let s = match to {
            FloatDType::F32 => CpuFloatStorage::F32(
                dispatch_int_raw!(x, |d| x.device().collect_alloc(layout.storage_indices().map(|i| d[i] as f32))),
                x.device().clone(),
            ),
            FloatDType::F64 => CpuFloatStorage::F64(
                dispatch_int_raw!(x, |d| x.device().collect_alloc(layout.storage_indices().map(|i| d[i] as f64))),
                x.device().clone(),
            ),
        };
        Ok(s)
    }

    fn i_cast_int(x: &CpuIntStorage, layout: &Layout, to: IntDType) -> Result<CpuIntStorage> {
        let s = match to {
            IntDType::I32 => CpuIntStorage::I32(
                dispatch_int_raw!(x, |d| x.device().collect_alloc(layout.storage_indices().map(|i| d[i] as i32))),
                x.device().clone(),
            ),
            IntDType::U32 => CpuIntStorage::U32(
                dispatch_int_raw!(x, |d| x.device().collect_alloc(layout.storage_indices().map(|i| d[i] as u32))),
                x.device().clone(),
            ),
            IntDType::U8 => CpuIntStorage::U8(
                dispatch_int_raw!(x, |d| x.device().collect_alloc(layout.storage_indices().map(|i| d[i] as u8))),
                x.device().clone(),
            ),
        };
        Ok(s)
    }

    fn i_cast_bool(x: &CpuIntStorage, layout: &Layout, _to: BoolDType) -> Result<CpuBoolStorage> {
        Ok(CpuBoolStorage(
            dispatch_int_raw!(x, |d| x.device().collect_alloc(layout.storage_indices().map(|i| d[i] != 0))),
            x.device().clone(),
        ))
    }

    fn i_to_vec(x: &<Cpu as Device>::IntStorage, layout: &Layout) -> Result<Vec<i64>> {
        Ok(match x {
            CpuIntStorage::I32(d, _) => x.device().collect_alloc(layout.storage_indices().map(|i| d[i] as i64)),
            CpuIntStorage::U32(d, _) => x.device().collect_alloc(layout.storage_indices().map(|i| d[i] as i64)),
            CpuIntStorage::U8(d, _) => x.device().collect_alloc(layout.storage_indices().map(|i| d[i] as i64)),
        })
    }

    fn i_to_bytes<'a>(x: &'a <Cpu as Device>::IntStorage, layout: &Layout) -> Result<Cow<'a, [u8]>> {
        if layout.is_contiguous() {
            Ok(match x {
                CpuIntStorage::I32(d, _) => Cow::Borrowed(bytemuck::cast_slice(d)),
                CpuIntStorage::U32(d, _) => Cow::Borrowed(bytemuck::cast_slice(d)),
                CpuIntStorage::U8(d, _) => Cow::Borrowed(d),
            })
        } else {
            let contig = Self::i_contiguous(x, layout)?;
            Ok(match &contig {
                CpuIntStorage::I32(d, _) => Cow::Owned(x.device().collect_alloc(bytemuck::cast_slice(d).iter().copied())),
                CpuIntStorage::U32(d, _) => Cow::Owned(x.device().collect_alloc(bytemuck::cast_slice(d).iter().copied())),
                CpuIntStorage::U8(d, _) => Cow::Owned(x.device().collect_alloc(d.iter().copied())),
            })
        }
    }

    fn i_binary(
        lhs: &<Cpu as Device>::IntStorage,
        lhs_l: &Layout,
        rhs: &<Cpu as Device>::IntStorage,
        rhs_l: &Layout,
        op: BinaryOp,
    ) -> Result<<Cpu as Device>::IntStorage> {
        dispatch_int2!(lhs, rhs, "int binary", |a, b| ew::num_binary(a, lhs_l, b, rhs_l, op, lhs.device()))
    }

    fn i_binary_scalar(lhs: &<Cpu as Device>::IntStorage, lhs_l: &Layout, rhs: i64, op: BinaryOp) -> Result<<Cpu as Device>::IntStorage> {
        Ok(match lhs {
            CpuIntStorage::I32(d, _) => {
                CpuIntStorage::I32(ew::num_binary_scalar(d, lhs_l, rhs as i32, op, lhs.device()), lhs.device().clone())
            }
            CpuIntStorage::U32(d, _) => {
                CpuIntStorage::U32(ew::num_binary_scalar(d, lhs_l, rhs as u32, op, lhs.device()), lhs.device().clone())
            }
            CpuIntStorage::U8(d, _) => {
                CpuIntStorage::U8(ew::num_binary_scalar(d, lhs_l, rhs as u8, op, lhs.device()), lhs.device().clone())
            }
        })
    }

    fn i_binary_(
        dst: &mut <Cpu as Device>::IntStorage,
        dst_l: &Layout,
        src: &<Cpu as Device>::IntStorage,
        src_l: &Layout,
        op: BinaryOp,
    ) -> Result<()> {
        match (dst, src) {
            (CpuIntStorage::I32(d, _), CpuIntStorage::I32(s, _)) => {
                ew::binary_(d, dst_l, s, src_l, ew::num_binary_fn::<i32>(op));
                Ok(())
            }
            (CpuIntStorage::U32(d, _), CpuIntStorage::U32(s, _)) => {
                ew::binary_(d, dst_l, s, src_l, ew::num_binary_fn::<u32>(op));
                Ok(())
            }
            (CpuIntStorage::U8(d, _), CpuIntStorage::U8(s, _)) => {
                ew::binary_(d, dst_l, s, src_l, ew::num_binary_fn::<u8>(op));
                Ok(())
            }
            (l, r) => Err(Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: "int binary_" }),
        }
    }

    fn i_binary_scalar_(dst: &mut <Cpu as Device>::IntStorage, dst_l: &Layout, rhs: i64, op: BinaryOp) -> Result<()> {
        match dst {
            CpuIntStorage::I32(d, _) => {
                ew::binary_scalar_(d, dst_l, rhs as i32, ew::num_binary_fn::<i32>(op));
                Ok(())
            }
            CpuIntStorage::U32(d, _) => {
                ew::binary_scalar_(d, dst_l, rhs as u32, ew::num_binary_fn::<u32>(op));
                Ok(())
            }
            CpuIntStorage::U8(d, _) => {
                ew::binary_scalar_(d, dst_l, rhs as u8, ew::num_binary_fn::<u8>(op));
                Ok(())
            }
        }
    }

    fn i_binary_scalar_lhs(
        scalar: i64,
        rhs: &<Cpu as Device>::IntStorage,
        rhs_l: &Layout,
        op: BinaryOp,
    ) -> Result<<Cpu as Device>::IntStorage> {
        Ok(match rhs {
            CpuIntStorage::I32(d, _) => {
                CpuIntStorage::I32(ew::num_scalar_binary(scalar as i32, d, rhs_l, op, rhs.device()), rhs.device().clone())
            }
            CpuIntStorage::U32(d, _) => {
                CpuIntStorage::U32(ew::num_scalar_binary(scalar as u32, d, rhs_l, op, rhs.device()), rhs.device().clone())
            }
            CpuIntStorage::U8(d, _) => {
                CpuIntStorage::U8(ew::num_scalar_binary(scalar as u8, d, rhs_l, op, rhs.device()), rhs.device().clone())
            }
        })
    }

    fn i_unary(x: &<Cpu as Device>::IntStorage, l: &Layout, op: crate::UnaryOp<i64>) -> Result<<Cpu as Device>::IntStorage> {
        use super::kernels::element::CpuNum;
        match x {
            CpuIntStorage::I32(d, _) => Ok(CpuIntStorage::I32(
                match op {
                    crate::UnaryOp::Neg => ew::unary(d, l, |v: i32| -v, x.device()),
                    crate::UnaryOp::Abs => ew::unary(d, l, |v: i32| CpuNum::abs(v), x.device()),
                    crate::UnaryOp::Sign => ew::unary(d, l, |v: i32| CpuNum::signum(v), x.device()),
                    crate::UnaryOp::Affine(mul, add) => ew::unary(d, l, |v: i32| (v as i64 * mul + add) as i32, x.device()),
                    crate::UnaryOp::Pow(exp) => ew::unary(d, l, |v: i32| (v as i64).pow(exp as u32) as i32, x.device()),
                    crate::UnaryOp::Clamp(min, max) => {
                        let lo = min.map(|v| v as i32);
                        let hi = max.map(|v| v as i32);
                        ew::unary(
                            d,
                            l,
                            |v: i32| {
                                let mut val = v;
                                if let Some(lo) = lo {
                                    val = val.max(lo);
                                }
                                if let Some(hi) = hi {
                                    val = val.min(hi);
                                }
                                val
                            },
                            x.device(),
                        )
                    }
                },
                x.device().clone(),
            )),
            CpuIntStorage::U32(d, _) => Ok(CpuIntStorage::U32(
                match op {
                    crate::UnaryOp::Neg => ew::unary(d, l, |v: u32| -(v as i32) as u32, x.device()),
                    crate::UnaryOp::Abs => ew::unary(d, l, |v: u32| CpuNum::abs(v), x.device()),
                    crate::UnaryOp::Sign => ew::unary(d, l, |v: u32| CpuNum::signum(v), x.device()),
                    crate::UnaryOp::Affine(mul, add) => ew::unary(d, l, |v: u32| (v as u64 * mul as u64 + add as u64) as u32, x.device()),
                    crate::UnaryOp::Pow(exp) => ew::unary(d, l, |v: u32| (v as u64).pow(exp as u32) as u32, x.device()),
                    crate::UnaryOp::Clamp(min, max) => {
                        let lo = min.map(|v| v.max(0) as u32);
                        let hi = max.map(|v| v as u32);
                        ew::unary(
                            d,
                            l,
                            |v: u32| {
                                let mut val = v;
                                if let Some(lo) = lo {
                                    val = val.max(lo);
                                }
                                if let Some(hi) = hi {
                                    val = val.min(hi);
                                }
                                val
                            },
                            x.device(),
                        )
                    }
                },
                x.device().clone(),
            )),
            CpuIntStorage::U8(d, _) => Ok(CpuIntStorage::U8(
                match op {
                    crate::UnaryOp::Neg => ew::unary(d, l, |v: u8| -(v as i32) as u8, x.device()),
                    crate::UnaryOp::Abs => ew::unary(d, l, |v: u8| CpuNum::abs(v), x.device()),
                    crate::UnaryOp::Sign => ew::unary(d, l, |v: u8| CpuNum::signum(v), x.device()),
                    crate::UnaryOp::Affine(mul, add) => ew::unary(d, l, |v: u8| (v as i64 * mul + add) as u8, x.device()),
                    crate::UnaryOp::Pow(exp) => ew::unary(d, l, |v: u8| (v as u64).pow(exp as u32) as u8, x.device()),
                    crate::UnaryOp::Clamp(min, max) => {
                        let lo = min.map(|v| v.max(0) as u8);
                        let hi = max.map(|v| v as u8);
                        ew::unary(
                            d,
                            l,
                            |v: u8| {
                                let mut val = v;
                                if let Some(lo) = lo {
                                    val = val.max(lo);
                                }
                                if let Some(hi) = hi {
                                    val = val.min(hi);
                                }
                                val
                            },
                            x.device(),
                        )
                    }
                },
                x.device().clone(),
            )),
        }
    }

    fn i_unary_(dst: &mut <Cpu as Device>::IntStorage, dst_l: &Layout, op: crate::UnaryOp<i64>) -> Result<()> {
        use super::kernels::element::CpuNum;
        match dst {
            CpuIntStorage::I32(d, _) => match op {
                crate::UnaryOp::Neg => {
                    ew::unary_(d, dst_l, |v: i32| -v);
                    Ok(())
                }
                crate::UnaryOp::Abs => {
                    ew::unary_(d, dst_l, |v: i32| CpuNum::abs(v));
                    Ok(())
                }
                crate::UnaryOp::Sign => {
                    ew::unary_(d, dst_l, |v: i32| CpuNum::signum(v));
                    Ok(())
                }
                crate::UnaryOp::Affine(mul, add) => {
                    ew::unary_(d, dst_l, |v: i32| (v as i64 * mul + add) as i32);
                    Ok(())
                }
                crate::UnaryOp::Pow(exp) => {
                    ew::unary_(d, dst_l, |v: i32| (v as i64).pow(exp as u32) as i32);
                    Ok(())
                }
                crate::UnaryOp::Clamp(min, max) => {
                    let lo = min.map(|v| v as i32);
                    let hi = max.map(|v| v as i32);
                    ew::unary_(d, dst_l, |v: i32| {
                        let mut val = v;
                        if let Some(lo) = lo {
                            val = val.max(lo);
                        }
                        if let Some(hi) = hi {
                            val = val.min(hi);
                        }
                        val
                    });
                    Ok(())
                }
            },
            CpuIntStorage::U32(d, _) => match op {
                crate::UnaryOp::Neg => {
                    ew::unary_(d, dst_l, |v: u32| -(v as i32) as u32);
                    Ok(())
                }
                crate::UnaryOp::Abs => {
                    ew::unary_(d, dst_l, |v: u32| CpuNum::abs(v));
                    Ok(())
                }
                crate::UnaryOp::Sign => {
                    ew::unary_(d, dst_l, |v: u32| CpuNum::signum(v));
                    Ok(())
                }
                crate::UnaryOp::Affine(mul, add) => {
                    ew::unary_(d, dst_l, |v: u32| (v as u64 * mul as u64 + add as u64) as u32);
                    Ok(())
                }
                crate::UnaryOp::Pow(exp) => {
                    ew::unary_(d, dst_l, |v: u32| (v as u64).pow(exp as u32) as u32);
                    Ok(())
                }
                crate::UnaryOp::Clamp(min, max) => {
                    let lo = min.map(|v| v.max(0) as u32);
                    let hi = max.map(|v| v as u32);
                    ew::unary_(d, dst_l, |v: u32| {
                        let mut val = v;
                        if let Some(lo) = lo {
                            val = val.max(lo);
                        }
                        if let Some(hi) = hi {
                            val = val.min(hi);
                        }
                        val
                    });
                    Ok(())
                }
            },
            CpuIntStorage::U8(d, _) => match op {
                crate::UnaryOp::Neg => {
                    ew::unary_(d, dst_l, |v: u8| -(v as i32) as u8);
                    Ok(())
                }
                crate::UnaryOp::Abs => {
                    ew::unary_(d, dst_l, |v: u8| CpuNum::abs(v));
                    Ok(())
                }
                crate::UnaryOp::Sign => {
                    ew::unary_(d, dst_l, |v: u8| CpuNum::signum(v));
                    Ok(())
                }
                crate::UnaryOp::Affine(mul, add) => {
                    ew::unary_(d, dst_l, |v: u8| (v as i64 * mul + add) as u8);
                    Ok(())
                }
                crate::UnaryOp::Pow(exp) => {
                    ew::unary_(d, dst_l, |v: u8| (v as u64).pow(exp as u32) as u8);
                    Ok(())
                }
                crate::UnaryOp::Clamp(min, max) => {
                    let lo = min.map(|v| v.max(0) as u8);
                    let hi = max.map(|v| v as u8);
                    ew::unary_(d, dst_l, |v: u8| {
                        let mut val = v;
                        if let Some(lo) = lo {
                            val = val.max(lo);
                        }
                        if let Some(hi) = hi {
                            val = val.min(hi);
                        }
                        val
                    });
                    Ok(())
                }
            },
        }
    }

    fn i_matmul(
        lhs: &<Cpu as Device>::IntStorage,
        lhs_l: &Layout,
        rhs: &<Cpu as Device>::IntStorage,
        rhs_l: &Layout,
        out_shape: &Shape,
    ) -> Result<<Cpu as Device>::IntStorage> {
        let result = match (lhs, rhs) {
            (CpuIntStorage::I32(a, _), CpuIntStorage::I32(b, _)) => {
                let (vec, shape) = matmul::matmul(a, lhs_l, b, rhs_l, lhs.device())?;
                debug_assert_eq!(shape.dims(), out_shape.dims(), "cpu i_matmul shape must match the layer");
                CpuIntStorage::I32(vec, lhs.device().clone())
            }
            (CpuIntStorage::U32(a, _), CpuIntStorage::U32(b, _)) => {
                let (vec, shape) = matmul::matmul(a, lhs_l, b, rhs_l, lhs.device())?;
                debug_assert_eq!(shape.dims(), out_shape.dims(), "cpu i_matmul shape must match the layer");
                CpuIntStorage::U32(vec, lhs.device().clone())
            }
            (CpuIntStorage::U8(a, _), CpuIntStorage::U8(b, _)) => {
                let (vec, shape) = matmul::matmul(a, lhs_l, b, rhs_l, lhs.device())?;
                debug_assert_eq!(shape.dims(), out_shape.dims(), "cpu i_matmul shape must match the layer");
                CpuIntStorage::U8(vec, lhs.device().clone())
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
        let v = dispatch_int2_raw!(lhs, rhs, "int cmp", |a, b| ew::num_cmp(a, lhs_l, b, rhs_l, op, lhs.device()))?;
        Ok(CpuBoolStorage(v, lhs.device().clone()))
    }

    fn i_cmp_scalar(lhs: &<Cpu as Device>::IntStorage, lhs_l: &Layout, rhs: i64, op: CmpOp) -> Result<<Cpu as Device>::BoolStorage> {
        Ok(match lhs {
            CpuIntStorage::I32(d, _) => CpuBoolStorage(ew::cmp_scalar(d, lhs_l, rhs as i32, op, lhs.device()), lhs.device().clone()),
            CpuIntStorage::U32(d, _) => CpuBoolStorage(ew::cmp_scalar(d, lhs_l, rhs as u32, op, lhs.device()), lhs.device().clone()),
            CpuIntStorage::U8(d, _) => CpuBoolStorage(ew::cmp_scalar(d, lhs_l, rhs as u8, op, lhs.device()), lhs.device().clone()),
        })
    }

    fn i_reduce(
        x: &<Cpu as Device>::IntStorage,
        l: &Layout,
        dims: &[usize],
        keepdim: bool,
        op: crate::ReduceOp,
        out_shape: &Shape,
    ) -> Result<<Cpu as Device>::IntStorage> {
        let reducer = reduce::Reducer::from(op);
        match x {
            CpuIntStorage::I32(d, _) => {
                let (v, s) = reduce::reduce_dims(d, l, dims, keepdim, reducer, x.device())?;
                debug_assert_eq!(s.dims(), out_shape.dims(), "cpu i_reduce shape must match the layer");
                Ok(CpuIntStorage::I32(v, x.device().clone()))
            }
            CpuIntStorage::U32(d, _) => {
                let (v, s) = reduce::reduce_dims(d, l, dims, keepdim, reducer, x.device())?;
                debug_assert_eq!(s.dims(), out_shape.dims(), "cpu i_reduce shape must match the layer");
                Ok(CpuIntStorage::U32(v, x.device().clone()))
            }
            CpuIntStorage::U8(d, _) => {
                let (v, s) = reduce::reduce_dims(d, l, dims, keepdim, reducer, x.device())?;
                debug_assert_eq!(s.dims(), out_shape.dims(), "cpu i_reduce shape must match the layer");
                Ok(CpuIntStorage::U8(v, x.device().clone()))
            }
        }
    }

    fn i_index_select(
        x: &<Cpu as Device>::IntStorage,
        x_l: &Layout,
        idx: &<Cpu as Device>::IntStorage,
        idx_l: &Layout,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<<Cpu as Device>::IntStorage> {
        let ids = int_ids_as_usize(idx, idx_l);
        match x {
            CpuIntStorage::I32(d, _) => {
                let (v, dims) = indexing::index_select(d, x_l, &ids, idx_l, dim, x.device())?;
                debug_assert_eq!(&dims, out_shape.dims(), "cpu i_index_select shape must match the layer");
                Ok(CpuIntStorage::I32(v, x.device().clone()))
            }
            CpuIntStorage::U32(d, _) => {
                let (v, dims) = indexing::index_select(d, x_l, &ids, idx_l, dim, x.device())?;
                debug_assert_eq!(&dims, out_shape.dims(), "cpu i_index_select shape must match the layer");
                Ok(CpuIntStorage::U32(v, x.device().clone()))
            }
            CpuIntStorage::U8(d, _) => {
                let (v, dims) = indexing::index_select(d, x_l, &ids, idx_l, dim, x.device())?;
                debug_assert_eq!(&dims, out_shape.dims(), "cpu i_index_select shape must match the layer");
                Ok(CpuIntStorage::U8(v, x.device().clone()))
            }
        }
    }

    fn i_gather(
        x: &<Cpu as Device>::IntStorage,
        x_l: &Layout,
        idx: &<Cpu as Device>::IntStorage,
        idx_l: &Layout,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<<Cpu as Device>::IntStorage> {
        let ids = int_ids_as_usize(idx, idx_l);
        match x {
            CpuIntStorage::I32(d, _) => {
                let (v, dims) = indexing::gather(d, x_l, &ids, idx_l, dim, x.device())?;
                debug_assert_eq!(&dims, out_shape.dims(), "cpu i_gather shape must match the layer");
                Ok(CpuIntStorage::I32(v, x.device().clone()))
            }
            CpuIntStorage::U32(d, _) => {
                let (v, dims) = indexing::gather(d, x_l, &ids, idx_l, dim, x.device())?;
                debug_assert_eq!(&dims, out_shape.dims(), "cpu i_gather shape must match the layer");
                Ok(CpuIntStorage::U32(v, x.device().clone()))
            }
            CpuIntStorage::U8(d, _) => {
                let (v, dims) = indexing::gather(d, x_l, &ids, idx_l, dim, x.device())?;
                debug_assert_eq!(&dims, out_shape.dims(), "cpu i_gather shape must match the layer");
                Ok(CpuIntStorage::U8(v, x.device().clone()))
            }
        }
    }

    fn i_cat(srcs: &[(&<Cpu as Device>::IntStorage, &Layout)], dim: usize, out_shape: &Shape) -> Result<<Cpu as Device>::IntStorage> {
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
                let (v, shape) = super::kernels::shape::cat(&views, dim, srcs[0].0.device())?;
                debug_assert_eq!(shape.dims(), out_shape.dims(), "cpu i_cat shape must match the layer");
                Ok($variant(v, srcs[0].0.device().clone()))
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
        out_shape: &Shape,
    ) -> Result<<Cpu as Device>::IntStorage> {
        let (indices, shape) = dispatch_int_raw!(x, |d| reduce::arg_reduce(d, layout, dim, keepdim, take_max, x.device()))?;
        debug_assert_eq!(shape.dims(), out_shape.dims(), "cpu i_arg_reduce shape must match the layer");
        let indices_u32: Vec<u32> = x.device().collect_alloc(indices.into_iter().map(|i| i as u32));
        Ok(CpuIntStorage::U32(indices_u32, x.device().clone()))
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
            (CpuIntStorage::I32(init_d, _), CpuIntStorage::I32(src_d, _)) => {
                let result = indexing::index_add(init_d, init_l, &idx_u, idx_l, src_d, dim, init.device())?;
                Ok(CpuIntStorage::I32(result, init.device().clone()))
            }
            (CpuIntStorage::U32(init_d, _), CpuIntStorage::U32(src_d, _)) => {
                let result = indexing::index_add(init_d, init_l, &idx_u, idx_l, src_d, dim, init.device())?;
                Ok(CpuIntStorage::U32(result, init.device().clone()))
            }
            (CpuIntStorage::U8(init_d, _), CpuIntStorage::U8(src_d, _)) => {
                let result = indexing::index_add(init_d, init_l, &idx_u, idx_l, src_d, dim, init.device())?;
                Ok(CpuIntStorage::U8(result, init.device().clone()))
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
            (CpuIntStorage::I32(init_d, _), CpuIntStorage::I32(src_d, _)) => {
                let result = indexing::scatter_add(init_d, init_l, &idx_u, idx_l, src_d, dim, init.device())?;
                Ok(CpuIntStorage::I32(result, init.device().clone()))
            }
            (CpuIntStorage::U32(init_d, _), CpuIntStorage::U32(src_d, _)) => {
                let result = indexing::scatter_add(init_d, init_l, &idx_u, idx_l, src_d, dim, init.device())?;
                Ok(CpuIntStorage::U32(result, init.device().clone()))
            }
            (CpuIntStorage::U8(init_d, _), CpuIntStorage::U8(src_d, _)) => {
                let result = indexing::scatter_add(init_d, init_l, &idx_u, idx_l, src_d, dim, init.device())?;
                Ok(CpuIntStorage::U8(result, init.device().clone()))
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
        let m: Vec<bool> = mask.device().collect_alloc(mask_l.storage_indices().map(|i| mask.0[i]));
        match (on_true, on_false) {
            (CpuIntStorage::I32(t, _), CpuIntStorage::I32(f, _)) => {
                let tv: Vec<i32> = on_true.device().collect_alloc(true_l.storage_indices().map(|i| t[i]));
                let fv: Vec<i32> = on_false.device().collect_alloc(false_l.storage_indices().map(|i| f[i]));
                Ok(CpuIntStorage::I32(
                    mask.device().collect_alloc(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { fv[i] })),
                    mask.device().clone(),
                ))
            }
            (CpuIntStorage::U32(t, _), CpuIntStorage::U32(f, _)) => {
                let tv: Vec<u32> = on_true.device().collect_alloc(true_l.storage_indices().map(|i| t[i]));
                let fv: Vec<u32> = on_false.device().collect_alloc(false_l.storage_indices().map(|i| f[i]));
                Ok(CpuIntStorage::U32(
                    mask.device().collect_alloc(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { fv[i] })),
                    mask.device().clone(),
                ))
            }
            (CpuIntStorage::U8(t, _), CpuIntStorage::U8(f, _)) => {
                let tv: Vec<u8> = on_true.device().collect_alloc(true_l.storage_indices().map(|i| t[i]));
                let fv: Vec<u8> = on_false.device().collect_alloc(false_l.storage_indices().map(|i| f[i]));
                Ok(CpuIntStorage::U8(
                    mask.device().collect_alloc(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { fv[i] })),
                    mask.device().clone(),
                ))
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
        let m: Vec<bool> = mask.device().collect_alloc(mask_l.storage_indices().map(|i| mask.0[i]));
        match on_false {
            CpuIntStorage::I32(f, _) => {
                let fv: Vec<i32> = on_false.device().collect_alloc(false_l.storage_indices().map(|i| f[i]));
                let val = value as i32;
                Ok(CpuIntStorage::I32(
                    mask.device().collect_alloc(m.iter().enumerate().map(|(i, &c)| if c { val } else { fv[i] })),
                    mask.device().clone(),
                ))
            }
            CpuIntStorage::U32(f, _) => {
                let fv: Vec<u32> = on_false.device().collect_alloc(false_l.storage_indices().map(|i| f[i]));
                let val = value as u32;
                Ok(CpuIntStorage::U32(
                    mask.device().collect_alloc(m.iter().enumerate().map(|(i, &c)| if c { val } else { fv[i] })),
                    mask.device().clone(),
                ))
            }
            CpuIntStorage::U8(f, _) => {
                let fv: Vec<u8> = on_false.device().collect_alloc(false_l.storage_indices().map(|i| f[i]));
                let val = value as u8;
                Ok(CpuIntStorage::U8(
                    mask.device().collect_alloc(m.iter().enumerate().map(|(i, &c)| if c { val } else { fv[i] })),
                    mask.device().clone(),
                ))
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
        let m: Vec<bool> = mask.device().collect_alloc(mask_l.storage_indices().map(|i| mask.0[i]));
        match on_true {
            CpuIntStorage::I32(t, _) => {
                let tv: Vec<i32> = on_true.device().collect_alloc(true_l.storage_indices().map(|i| t[i]));
                let val = value as i32;
                Ok(CpuIntStorage::I32(
                    mask.device().collect_alloc(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { val })),
                    mask.device().clone(),
                ))
            }
            CpuIntStorage::U32(t, _) => {
                let tv: Vec<u32> = on_true.device().collect_alloc(true_l.storage_indices().map(|i| t[i]));
                let val = value as u32;
                Ok(CpuIntStorage::U32(
                    mask.device().collect_alloc(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { val })),
                    mask.device().clone(),
                ))
            }
            CpuIntStorage::U8(t, _) => {
                let tv: Vec<u8> = on_true.device().collect_alloc(true_l.storage_indices().map(|i| t[i]));
                let val = value as u8;
                Ok(CpuIntStorage::U8(
                    mask.device().collect_alloc(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { val })),
                    mask.device().clone(),
                ))
            }
        }
    }

    fn i_allclose(a: &CpuIntStorage, a_l: &Layout, b: &CpuIntStorage, b_l: &Layout) -> Result<bool> {
        match (a, b) {
            (CpuIntStorage::I32(av, _), CpuIntStorage::I32(bv, _)) => {
                Ok(a_l.storage_indices().zip(b_l.storage_indices()).all(|(ai, bi)| av[ai] == bv[bi]))
            }
            (CpuIntStorage::U32(av, _), CpuIntStorage::U32(bv, _)) => {
                Ok(a_l.storage_indices().zip(b_l.storage_indices()).all(|(ai, bi)| av[ai] == bv[bi]))
            }
            (CpuIntStorage::U8(av, _), CpuIntStorage::U8(bv, _)) => {
                Ok(a_l.storage_indices().zip(b_l.storage_indices()).all(|(ai, bi)| av[ai] == bv[bi]))
            }
            _ => Err(crate::Error::DTypeMismatch { lhs: a.dtype(), rhs: b.dtype(), op: "allclose" }),
        }
    }
}

fn as_i32(s: &CpuIntStorage) -> &[i32] {
    match s {
        CpuIntStorage::I32(d, _) => d,
        _ => unreachable!(),
    }
}
fn as_u32(s: &CpuIntStorage) -> &[u32] {
    match s {
        CpuIntStorage::U32(d, _) => d,
        _ => unreachable!(),
    }
}
fn as_u8(s: &CpuIntStorage) -> &[u8] {
    match s {
        CpuIntStorage::U8(d, _) => d,
        _ => unreachable!(),
    }
}
