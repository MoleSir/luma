use std::borrow::Cow;
use rand::rng;
use rand_distr::{Distribution, Normal, Uniform};
use crate::{Cpu, CpuBoolStorage, CpuFloatStorage, CpuIntStorage, dispatch_float, dispatch_float_raw, dispatch_float2, dispatch_float2_raw};
use crate::dispatch::{int_ids_as_usize, usize_to_int_storage};
use crate::kernels::element::CpuNum;
use crate::allocator::AllocVec;
use crate::kernels::{elementwise as ew, indexing, matmul, nn, reduce};
use luma_tensor::{BinaryOp, BoolDType, CmpOp, DType, FloatDType, FloatUnaryOp, IntDType, ReduceOp, UnaryOp};
use luma_tensor::{FloatOps, Device, Error, Layout, Result, Shape, Storage};

/// Build a float storage of `dtype`, filling `n` elements via `f32`/`f64` closures.
fn build(n: usize, dtype: FloatDType, device: &Cpu, f32v: impl Fn() -> f32, f64v: impl Fn() -> f64) -> CpuFloatStorage {
    match dtype {
        FloatDType::F32 => CpuFloatStorage::F32(device.collect_alloc((0..n).map(|_| f32v())), device.clone()),
        FloatDType::F64 => CpuFloatStorage::F64(device.collect_alloc((0..n).map(|_| f64v())), device.clone()),
        FloatDType::F16 => CpuFloatStorage::F16(device.collect_alloc((0..n).map(|_| half::f16::from_f64(f64v()))), device.clone()),
        FloatDType::BF16 => CpuFloatStorage::BF16(device.collect_alloc((0..n).map(|_| half::bf16::from_f64(f64v()))), device.clone()),
    }
}

/// Cast every element of `src` (read in `layout` order) into `Dst` via f64.
fn cast_vec<S: CpuNum, Dst: CpuNum + AllocVec>(src: &[S], layout: &Layout, device: &Cpu) -> Vec<Dst> {
    device.collect_alloc(layout.storage_indices().map(|i| Dst::from_f64(src[i].to_f64())))
}

/// `d[i] != 0` for any element type.
fn nonzero_vec<T: CpuNum>(d: &[T], layout: &Layout, device: &Cpu) -> Vec<bool> {
    device.collect_alloc(layout.storage_indices().map(|i| d[i] != T::ZERO))
}

fn num_binary_scalar_f64<T: CpuNum + AllocVec>(d: &[T], l: &Layout, rhs: f64, op: BinaryOp, device: &Cpu) -> Vec<T> {
    ew::num_binary_scalar(d, l, T::from_f64(rhs), op, device)
}

fn num_scalar_binary_f64<T: CpuNum + AllocVec>(scalar: f64, d: &[T], l: &Layout, op: BinaryOp, device: &Cpu) -> Vec<T> {
    ew::num_scalar_binary(T::from_f64(scalar), d, l, op, device)
}

fn cmp_scalar_f64<T: CpuNum>(d: &[T], l: &Layout, rhs: f64, op: CmpOp, device: &Cpu) -> Vec<bool> {
    ew::cmp_scalar(d, l, T::from_f64(rhs), op, device)
}

fn apply_unary<T: crate::kernels::element::CpuFloat + AllocVec>(d: &[T], l: &Layout, op: UnaryOp<f64>, device: &Cpu) -> Vec<T> {
    match op {
        UnaryOp::Neg => ew::unary(d, l, |v: T| -v, device),
        UnaryOp::Abs => ew::unary(d, l, |v: T| v.abs(), device),
        UnaryOp::Sign => ew::unary(d, l, |v: T| v.signum(), device),
        UnaryOp::Affine(mul, add) => {
            let m = T::from_f64(mul);
            let a = T::from_f64(add);
            ew::unary(d, l, move |v: T| v * m + a, device)
        }
        UnaryOp::Pow(exp) => {
            let e = T::from_f64(exp);
            ew::unary(d, l, move |v: T| v.powf(e), device)
        }
        UnaryOp::Clamp(min, max) => {
            let lo = min.map(T::from_f64);
            let hi = max.map(T::from_f64);
            ew::unary(
                d,
                l,
                move |v: T| {
                    let mut val = v;
                    if let Some(lo) = lo {
                        val = T::maximum(val, lo);
                    }
                    if let Some(hi) = hi {
                        val = T::minimum(val, hi);
                    }
                    val
                },
                device,
            )
        }
    }
}

fn apply_unary_inplace<T: crate::kernels::element::CpuFloat>(d: &mut [T], l: &Layout, op: UnaryOp<f64>) {
    match op {
        UnaryOp::Neg => ew::unary_(d, l, |v: T| -v),
        UnaryOp::Abs => ew::unary_(d, l, |v: T| v.abs()),
        UnaryOp::Sign => ew::unary_(d, l, |v: T| v.signum()),
        UnaryOp::Affine(mul, add) => {
            let m = T::from_f64(mul);
            let a = T::from_f64(add);
            ew::unary_(d, l, move |v: T| v * m + a);
        }
        UnaryOp::Pow(exp) => {
            let e = T::from_f64(exp);
            ew::unary_(d, l, move |v: T| v.powf(e));
        }
        UnaryOp::Clamp(min, max) => {
            let lo = min.map(T::from_f64);
            let hi = max.map(T::from_f64);
            ew::unary_(d, l, move |v: T| {
                let mut val = v;
                if let Some(lo) = lo {
                    val = T::maximum(val, lo);
                }
                if let Some(hi) = hi {
                    val = T::minimum(val, hi);
                }
                val
            });
        }
    }
}

fn apply_float_unary_inplace<T: crate::kernels::element::CpuFloat>(d: &mut [T], l: &Layout, op: FloatUnaryOp) {
    match op {
        FloatUnaryOp::Exp => ew::unary_(d, l, |v: T| v.exp()),
        FloatUnaryOp::Ln => ew::unary_(d, l, |v: T| v.ln()),
        FloatUnaryOp::Sin => ew::unary_(d, l, |v: T| v.sin()),
        FloatUnaryOp::Cos => ew::unary_(d, l, |v: T| v.cos()),
        FloatUnaryOp::Tanh => ew::unary_(d, l, |v: T| v.tanh()),
        FloatUnaryOp::Sqr => ew::unary_(d, l, |v: T| v.sqr()),
        FloatUnaryOp::Sqrt => ew::unary_(d, l, |v: T| v.sqrt()),
        FloatUnaryOp::Recip => ew::unary_(d, l, |v: T| v.recip()),
        FloatUnaryOp::Gelu => ew::unary_(d, l, |v: T| v.gelu()),
        FloatUnaryOp::GeluErf => ew::unary_(d, l, |v: T| v.gelu_erf()),
        FloatUnaryOp::Erf => ew::unary_(d, l, |v: T| v.erf()),
        FloatUnaryOp::Relu => ew::unary_(d, l, |v: T| v.relu()),
        FloatUnaryOp::Silu => ew::unary_(d, l, |v: T| v.silu()),
        FloatUnaryOp::Sigmoid => ew::unary_(d, l, |v: T| v.sigmoid()),
        FloatUnaryOp::Floor => ew::unary_(d, l, |v: T| v.floor()),
        FloatUnaryOp::Ceil => ew::unary_(d, l, |v: T| v.ceil()),
        FloatUnaryOp::Round => ew::unary_(d, l, |v: T| v.round()),
        FloatUnaryOp::LeakyRelu(a) => {
            let a = T::from_f64(a);
            ew::unary_(d, l, move |v: T| v.leaky_relu(a));
        }
    }
}

fn allclose_generic<T: CpuNum>(av: &[T], a_l: &Layout, bv: &[T], b_l: &Layout, rtol: f64, atol: f64) -> bool {
    let rtol = T::from_f64(rtol);
    let atol = T::from_f64(atol);
    a_l.storage_indices().zip(b_l.storage_indices()).all(|(ai, bi)| {
        let diff = (av[ai] - bv[bi]).abs();
        diff <= atol + rtol * bv[bi].abs()
    })
}

impl FloatOps<Cpu> for Cpu {
    fn f_zeros(shape: &Shape, device: &Cpu, dtype: FloatDType) -> Result<<Cpu as Device>::FloatStorage> {
        Ok(build(shape.element_count(), dtype, device, || 0.0, || 0.0))
    }

    fn f_ones(shape: &Shape, device: &Cpu, dtype: FloatDType) -> Result<<Cpu as Device>::FloatStorage> {
        Ok(build(shape.element_count(), dtype, device, || 1.0, || 1.0))
    }

    fn f_full(shape: &Shape, value: f64, device: &Cpu, dtype: FloatDType) -> Result<<Cpu as Device>::FloatStorage> {
        Ok(build(shape.element_count(), dtype, device, || value as f32, || value))
    }

    fn f_from_f64<'a>(data: impl Into<Cow<'a, [f64]>>, device: &Cpu) -> Result<<Cpu as Device>::FloatStorage> {
        let data = data.into();
        Ok(match data {
            Cow::Owned(v) => CpuFloatStorage::F64(v, device.clone()),
            Cow::Borrowed(s) => CpuFloatStorage::F64(device.collect_alloc(s.iter().copied()), device.clone()),
        })
    }

    fn f_from_f32<'a>(data: impl Into<Cow<'a, [f32]>>, device: &Cpu) -> Result<<Cpu as Device>::FloatStorage> {
        let data = data.into();
        Ok(match data {
            Cow::Owned(v) => CpuFloatStorage::F32(v, device.clone()),
            Cow::Borrowed(s) => CpuFloatStorage::F32(device.collect_alloc(s.iter().copied()), device.clone()),
        })
    }

    fn f_from_f16<'a>(data: impl Into<Cow<'a, [half::f16]>>, device: &Cpu) -> Result<<Cpu as Device>::FloatStorage> {
        let data = data.into();
        Ok(match data {
            Cow::Owned(v) => CpuFloatStorage::F16(v, device.clone()),
            Cow::Borrowed(s) => CpuFloatStorage::F16(device.collect_alloc(s.iter().copied()), device.clone()),
        })
    }

    fn f_from_bf16<'a>(data: impl Into<Cow<'a, [half::bf16]>>, device: &Cpu) -> Result<<Cpu as Device>::FloatStorage> {
        let data = data.into();
        Ok(match data {
            Cow::Owned(v) => CpuFloatStorage::BF16(v, device.clone()),
            Cow::Borrowed(s) => CpuFloatStorage::BF16(device.collect_alloc(s.iter().copied()), device.clone()),
        })
    }

    fn f_from_bytes<'a>(
        bytes: impl Into<Cow<'a, [u8]>>,
        _shape: &Shape,
        device: &Cpu,
        dtype: FloatDType,
    ) -> Result<<Cpu as Device>::FloatStorage> {
        let bytes = bytes.into();
        Ok(match dtype {
            FloatDType::F32 => {
                let v: Vec<f32> = device.collect_alloc(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())));
                CpuFloatStorage::F32(v, device.clone())
            }
            FloatDType::F64 => {
                let v: Vec<f64> = device.collect_alloc(bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())));
                CpuFloatStorage::F64(v, device.clone())
            }
            FloatDType::F16 => {
                let v: Vec<half::f16> = device.collect_alloc(bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes(c.try_into().unwrap())));
                CpuFloatStorage::F16(v, device.clone())
            }
            FloatDType::BF16 => {
                let v: Vec<half::bf16> = device.collect_alloc(bytes.chunks_exact(2).map(|c| half::bf16::from_le_bytes(c.try_into().unwrap())));
                CpuFloatStorage::BF16(v, device.clone())
            }
        })
    }

    fn f_rand_uniform(shape: &Shape, lo: f64, hi: f64, device: &Cpu, dtype: FloatDType) -> Result<<Cpu as Device>::FloatStorage> {
        let n = shape.element_count();
        let mut r = rng();
        let s = match dtype {
            FloatDType::F64 => {
                let u = Uniform::new(lo, hi).map_err(|e| Error::Rand(e.to_string()))?;
                CpuFloatStorage::F64(device.collect_alloc((0..n).map(|_| u.sample(&mut r))), device.clone())
            }
            FloatDType::F32 => {
                let u = Uniform::new(lo as f32, hi as f32).map_err(|e| Error::Rand(e.to_string()))?;
                CpuFloatStorage::F32(device.collect_alloc((0..n).map(|_| u.sample(&mut r))), device.clone())
            }
            FloatDType::F16 => {
                let u = Uniform::new(lo as f32, hi as f32).map_err(|e| Error::Rand(e.to_string()))?;
                CpuFloatStorage::F16(device.collect_alloc((0..n).map(|_| half::f16::from_f32(u.sample(&mut r)))), device.clone())
            }
            FloatDType::BF16 => {
                let u = Uniform::new(lo as f32, hi as f32).map_err(|e| Error::Rand(e.to_string()))?;
                CpuFloatStorage::BF16(device.collect_alloc((0..n).map(|_| half::bf16::from_f32(u.sample(&mut r)))), device.clone())
            }
        };
        Ok(s)
    }

    fn f_rand_normal(shape: &Shape, mean: f64, std: f64, device: &Cpu, dtype: FloatDType) -> Result<<Cpu as Device>::FloatStorage> {
        let n = shape.element_count();
        let mut r = rng();
        let s = match dtype {
            FloatDType::F64 => {
                let d = Normal::new(mean, std).map_err(|e| Error::Rand(e.to_string()))?;
                CpuFloatStorage::F64(device.collect_alloc((0..n).map(|_| d.sample(&mut r))), device.clone())
            }
            FloatDType::F32 => {
                let d = Normal::new(mean as f32, std as f32).map_err(|e| Error::Rand(e.to_string()))?;
                CpuFloatStorage::F32(device.collect_alloc((0..n).map(|_| d.sample(&mut r))), device.clone())
            }
            FloatDType::F16 => {
                let d = Normal::new(mean as f32, std as f32).map_err(|e| Error::Rand(e.to_string()))?;
                CpuFloatStorage::F16(device.collect_alloc((0..n).map(|_| half::f16::from_f32(d.sample(&mut r)))), device.clone())
            }
            FloatDType::BF16 => {
                let d = Normal::new(mean as f32, std as f32).map_err(|e| Error::Rand(e.to_string()))?;
                CpuFloatStorage::BF16(device.collect_alloc((0..n).map(|_| half::bf16::from_f32(d.sample(&mut r)))), device.clone())
            }
        };
        Ok(s)
    }

    fn f_contiguous(x: &<Cpu as Device>::FloatStorage, l: &Layout) -> Result<<Cpu as Device>::FloatStorage> {
        Ok(dispatch_float!(x, |d| super::kernels::iter::gather(d, l, x.device())))
    }

    fn f_cast_float(x: &CpuFloatStorage, layout: &Layout, to: FloatDType) -> Result<CpuFloatStorage> {
        let s = match to {
            FloatDType::F32 => CpuFloatStorage::F32(dispatch_float_raw!(x, |d| cast_vec::<_, f32>(d, layout, x.device())), x.device().clone()),
            FloatDType::F64 => CpuFloatStorage::F64(dispatch_float_raw!(x, |d| cast_vec::<_, f64>(d, layout, x.device())), x.device().clone()),
            FloatDType::F16 => CpuFloatStorage::F16(dispatch_float_raw!(x, |d| cast_vec::<_, half::f16>(d, layout, x.device())), x.device().clone()),
            FloatDType::BF16 => CpuFloatStorage::BF16(dispatch_float_raw!(x, |d| cast_vec::<_, half::bf16>(d, layout, x.device())), x.device().clone()),
        };
        Ok(s)
    }

    fn f_cast_int(x: &CpuFloatStorage, layout: &Layout, to: IntDType) -> Result<CpuIntStorage> {
        let s = match to {
            IntDType::I32 => CpuIntStorage::I32(dispatch_float_raw!(x, |d| cast_vec::<_, i32>(d, layout, x.device())), x.device().clone()),
            IntDType::U32 => CpuIntStorage::U32(dispatch_float_raw!(x, |d| cast_vec::<_, u32>(d, layout, x.device())), x.device().clone()),
            IntDType::U8 => CpuIntStorage::U8(dispatch_float_raw!(x, |d| cast_vec::<_, u8>(d, layout, x.device())), x.device().clone()),
        };
        Ok(s)
    }

    fn f_cast_bool(x: &CpuFloatStorage, layout: &Layout, _to: BoolDType) -> Result<CpuBoolStorage> {
        Ok(CpuBoolStorage(dispatch_float_raw!(x, |d| nonzero_vec(d, layout, x.device())), x.device().clone()))
    }

    fn f_to_vec(x: &<Cpu as Device>::FloatStorage, layout: &Layout) -> Result<Vec<f64>> {
        Ok(dispatch_float_raw!(x, |d| x.device().collect_alloc(layout.storage_indices().map(|i| d[i].to_f64()))))
    }

    fn f_to_bytes<'a>(x: &'a <Cpu as Device>::FloatStorage, layout: &Layout) -> Result<Cow<'a, [u8]>> {
        if layout.is_contiguous() {
            Ok(match x {
                CpuFloatStorage::F32(d, _) => Cow::Borrowed(bytemuck::cast_slice(d)),
                CpuFloatStorage::F64(d, _) => Cow::Borrowed(bytemuck::cast_slice(d)),
                CpuFloatStorage::F16(d, _) => Cow::Borrowed(bytemuck::cast_slice(d)),
                CpuFloatStorage::BF16(d, _) => Cow::Borrowed(bytemuck::cast_slice(d)),
            })
        } else {
            let contig = Self::f_contiguous(x, layout)?;
            Ok(match &contig {
                CpuFloatStorage::F32(d, _) => Cow::Owned(bytemuck::cast_slice(d).to_vec()),
                CpuFloatStorage::F64(d, _) => Cow::Owned(bytemuck::cast_slice(d).to_vec()),
                CpuFloatStorage::F16(d, _) => Cow::Owned(bytemuck::cast_slice(d).to_vec()),
                CpuFloatStorage::BF16(d, _) => Cow::Owned(bytemuck::cast_slice(d).to_vec()),
            })
        }
    }

    fn f_binary(
        lhs: &<Cpu as Device>::FloatStorage,
        lhs_l: &Layout,
        rhs: &<Cpu as Device>::FloatStorage,
        rhs_l: &Layout,
        op: BinaryOp,
    ) -> Result<<Cpu as Device>::FloatStorage> {
        dispatch_float2!(lhs, rhs, "binary", |a, b| ew::num_binary(a, lhs_l, b, rhs_l, op, lhs.device()))
    }

    fn f_binary_scalar(
        lhs: &<Cpu as Device>::FloatStorage,
        lhs_l: &Layout,
        rhs: f64,
        op: BinaryOp,
    ) -> Result<<Cpu as Device>::FloatStorage> {
        Ok(dispatch_float!(lhs, |d| num_binary_scalar_f64(d, lhs_l, rhs, op, lhs.device())))
    }

    fn f_binary_scalar_(dst: &mut <Cpu as Device>::FloatStorage, dst_l: &Layout, rhs: f64, op: BinaryOp) -> Result<()> {
        match dst {
            CpuFloatStorage::F32(d, _) => ew::binary_scalar_(d, dst_l, rhs as f32, ew::num_binary_fn::<f32>(op)),
            CpuFloatStorage::F64(d, _) => ew::binary_scalar_(d, dst_l, rhs, ew::num_binary_fn::<f64>(op)),
            CpuFloatStorage::F16(d, _) => ew::binary_scalar_(d, dst_l, half::f16::from_f64(rhs), ew::num_binary_fn::<half::f16>(op)),
            CpuFloatStorage::BF16(d, _) => ew::binary_scalar_(d, dst_l, half::bf16::from_f64(rhs), ew::num_binary_fn::<half::bf16>(op)),
        }
        Ok(())
    }

    fn f_binary_scalar_lhs(scalar: f64, rhs: &CpuFloatStorage, rhs_l: &Layout, op: BinaryOp) -> Result<CpuFloatStorage> {
        Ok(dispatch_float!(rhs, |d| num_scalar_binary_f64(scalar, d, rhs_l, op, rhs.device())))
    }

    fn f_unary(x: &<Cpu as Device>::FloatStorage, l: &Layout, op: UnaryOp<f64>) -> Result<<Cpu as Device>::FloatStorage> {
        Ok(dispatch_float!(x, |d| apply_unary(d, l, op, x.device())))
    }

    fn f_float_unary(x: &<Cpu as Device>::FloatStorage, l: &Layout, op: FloatUnaryOp) -> Result<<Cpu as Device>::FloatStorage> {
        Ok(dispatch_float!(x, |d| ew::float_unary(d, l, op, x.device())))
    }

    fn f_unary_(dst: &mut <Cpu as Device>::FloatStorage, dst_l: &Layout, op: UnaryOp<f64>) -> Result<()> {
        match dst {
            CpuFloatStorage::F32(d, _) => apply_unary_inplace(d, dst_l, op),
            CpuFloatStorage::F64(d, _) => apply_unary_inplace(d, dst_l, op),
            CpuFloatStorage::F16(d, _) => apply_unary_inplace(d, dst_l, op),
            CpuFloatStorage::BF16(d, _) => apply_unary_inplace(d, dst_l, op),
        }
        Ok(())
    }

    fn f_float_unary_(dst: &mut <Cpu as Device>::FloatStorage, dst_l: &Layout, op: FloatUnaryOp) -> Result<()> {
        match dst {
            CpuFloatStorage::F32(d, _) => apply_float_unary_inplace(d, dst_l, op),
            CpuFloatStorage::F64(d, _) => apply_float_unary_inplace(d, dst_l, op),
            CpuFloatStorage::F16(d, _) => apply_float_unary_inplace(d, dst_l, op),
            CpuFloatStorage::BF16(d, _) => apply_float_unary_inplace(d, dst_l, op),
        }
        Ok(())
    }

    fn f_cmp(
        lhs: &<Cpu as Device>::FloatStorage,
        lhs_l: &Layout,
        rhs: &<Cpu as Device>::FloatStorage,
        rhs_l: &Layout,
        op: CmpOp,
    ) -> Result<<Cpu as Device>::BoolStorage> {
        let v = dispatch_float2_raw!(lhs, rhs, "cmp", |a, b| ew::num_cmp(a, lhs_l, b, rhs_l, op, lhs.device()))?;
        Ok(CpuBoolStorage(v, lhs.device().clone()))
    }

    fn f_cmp_scalar(lhs: &<Cpu as Device>::FloatStorage, lhs_l: &Layout, rhs: f64, op: CmpOp) -> Result<<Cpu as Device>::BoolStorage> {
        Ok(CpuBoolStorage(dispatch_float_raw!(lhs, |d| cmp_scalar_f64(d, lhs_l, rhs, op, lhs.device())), lhs.device().clone()))
    }

    fn f_reduce(
        x: &<Cpu as Device>::FloatStorage,
        l: &Layout,
        dims: &[usize],
        keepdim: bool,
        op: ReduceOp,
        out_shape: &Shape,
    ) -> Result<<Cpu as Device>::FloatStorage> {
        let reducer = reduce::Reducer::from(op);
        match x {
            CpuFloatStorage::F32(d, _) => {
                let (v, s) = reduce::reduce_dims(d, l, dims, keepdim, reducer, x.device())?;
                debug_assert_eq!(s.dims(), out_shape.dims(), "cpu f_reduce shape must match the layer");
                Ok(CpuFloatStorage::F32(v, x.device().clone()))
            }
            CpuFloatStorage::F64(d, _) => {
                let (v, s) = reduce::reduce_dims(d, l, dims, keepdim, reducer, x.device())?;
                debug_assert_eq!(s.dims(), out_shape.dims(), "cpu f_reduce shape must match the layer");
                Ok(CpuFloatStorage::F64(v, x.device().clone()))
            }
            CpuFloatStorage::F16(d, _) => {
                let (v, s) = reduce::reduce_dims(d, l, dims, keepdim, reducer, x.device())?;
                debug_assert_eq!(s.dims(), out_shape.dims(), "cpu f_reduce shape must match the layer");
                Ok(CpuFloatStorage::F16(v, x.device().clone()))
            }
            CpuFloatStorage::BF16(d, _) => {
                let (v, s) = reduce::reduce_dims(d, l, dims, keepdim, reducer, x.device())?;
                debug_assert_eq!(s.dims(), out_shape.dims(), "cpu f_reduce shape must match the layer");
                Ok(CpuFloatStorage::BF16(v, x.device().clone()))
            }
        }
    }

    fn f_arg_reduce(
        x: &<Cpu as Device>::FloatStorage,
        l: &Layout,
        dim: usize,
        keepdim: bool,
        take_max: bool,
        out_shape: &Shape,
    ) -> Result<<Cpu as Device>::IntStorage> {
        let (idx, shape) = dispatch_float_raw!(x, |d| reduce::arg_reduce(d, l, dim, keepdim, take_max, x.device()))?;
        debug_assert_eq!(shape.dims(), out_shape.dims(), "cpu f_arg_reduce shape must match the layer");
        Ok(usize_to_int_storage(&idx, DType::U32, x.device()))
    }

    fn f_matmul(
        lhs: &<Cpu as Device>::FloatStorage,
        lhs_l: &Layout,
        rhs: &<Cpu as Device>::FloatStorage,
        rhs_l: &Layout,
        out_shape: &Shape,
    ) -> Result<<Cpu as Device>::FloatStorage> {
        match (lhs, rhs) {
            (CpuFloatStorage::F32(a, _), CpuFloatStorage::F32(b, _)) => {
                let (v, s) = matmul::matmul(a, lhs_l, b, rhs_l, lhs.device())?;
                debug_assert_eq!(s.dims(), out_shape.dims(), "cpu f_matmul shape must match the layer");
                Ok(CpuFloatStorage::F32(v, lhs.device().clone()))
            }
            (CpuFloatStorage::F64(a, _), CpuFloatStorage::F64(b, _)) => {
                let (v, s) = matmul::matmul(a, lhs_l, b, rhs_l, lhs.device())?;
                debug_assert_eq!(s.dims(), out_shape.dims(), "cpu f_matmul shape must match the layer");
                Ok(CpuFloatStorage::F64(v, lhs.device().clone()))
            }
            (CpuFloatStorage::F16(a, _), CpuFloatStorage::F16(b, _)) => {
                let (v, s) = matmul::matmul(a, lhs_l, b, rhs_l, lhs.device())?;
                debug_assert_eq!(s.dims(), out_shape.dims(), "cpu f_matmul shape must match the layer");
                Ok(CpuFloatStorage::F16(v, lhs.device().clone()))
            }
            (CpuFloatStorage::BF16(a, _), CpuFloatStorage::BF16(b, _)) => {
                let (v, s) = matmul::matmul(a, lhs_l, b, rhs_l, lhs.device())?;
                debug_assert_eq!(s.dims(), out_shape.dims(), "cpu f_matmul shape must match the layer");
                Ok(CpuFloatStorage::BF16(v, lhs.device().clone()))
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
            (CpuFloatStorage::F32(d, _), CpuFloatStorage::F32(l, _), CpuFloatStorage::F32(r, _)) => {
                matmul::add_matmul(d, dst_l, l, lhs_l, r, rhs_l)
            }
            (CpuFloatStorage::F64(d, _), CpuFloatStorage::F64(l, _), CpuFloatStorage::F64(r, _)) => {
                matmul::add_matmul(d, dst_l, l, lhs_l, r, rhs_l)
            }
            (CpuFloatStorage::F16(d, _), CpuFloatStorage::F16(l, _), CpuFloatStorage::F16(r, _)) => {
                matmul::add_matmul(d, dst_l, l, lhs_l, r, rhs_l)
            }
            (CpuFloatStorage::BF16(d, _), CpuFloatStorage::BF16(l, _), CpuFloatStorage::BF16(r, _)) => {
                matmul::add_matmul(d, dst_l, l, lhs_l, r, rhs_l)
            }
            (_d, l, r) => Err(Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: "f_add_matmul_" }),
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
            (CpuFloatStorage::F32(d, _), CpuFloatStorage::F32(s, _)) => {
                ew::binary_(d, dst_l, s, src_l, ew::num_binary_fn::<f32>(op));
                Ok(())
            }
            (CpuFloatStorage::F64(d, _), CpuFloatStorage::F64(s, _)) => {
                ew::binary_(d, dst_l, s, src_l, ew::num_binary_fn::<f64>(op));
                Ok(())
            }
            (CpuFloatStorage::F16(d, _), CpuFloatStorage::F16(s, _)) => {
                ew::binary_(d, dst_l, s, src_l, ew::num_binary_fn::<half::f16>(op));
                Ok(())
            }
            (CpuFloatStorage::BF16(d, _), CpuFloatStorage::BF16(s, _)) => {
                ew::binary_(d, dst_l, s, src_l, ew::num_binary_fn::<half::bf16>(op));
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
        out_shape: &Shape,
    ) -> Result<<Cpu as Device>::FloatStorage> {
        let ids = int_ids_as_usize(idx, idx_l);
        match x {
            CpuFloatStorage::F32(d, _) => {
                let (v, dims) = indexing::index_select(d, x_l, &ids, idx_l, dim, x.device())?;
                debug_assert_eq!(&dims, out_shape.dims(), "cpu f_index_select shape must match the layer");
                Ok(CpuFloatStorage::F32(v, x.device().clone()))
            }
            CpuFloatStorage::F64(d, _) => {
                let (v, dims) = indexing::index_select(d, x_l, &ids, idx_l, dim, x.device())?;
                debug_assert_eq!(&dims, out_shape.dims(), "cpu f_index_select shape must match the layer");
                Ok(CpuFloatStorage::F64(v, x.device().clone()))
            }
            CpuFloatStorage::F16(d, _) => {
                let (v, dims) = indexing::index_select(d, x_l, &ids, idx_l, dim, x.device())?;
                debug_assert_eq!(&dims, out_shape.dims(), "cpu f_index_select shape must match the layer");
                Ok(CpuFloatStorage::F16(v, x.device().clone()))
            }
            CpuFloatStorage::BF16(d, _) => {
                let (v, dims) = indexing::index_select(d, x_l, &ids, idx_l, dim, x.device())?;
                debug_assert_eq!(&dims, out_shape.dims(), "cpu f_index_select shape must match the layer");
                Ok(CpuFloatStorage::BF16(v, x.device().clone()))
            }
        }
    }

    fn f_gather(
        x: &<Cpu as Device>::FloatStorage,
        x_l: &Layout,
        idx: &<Cpu as Device>::IntStorage,
        idx_l: &Layout,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<<Cpu as Device>::FloatStorage> {
        let ids = int_ids_as_usize(idx, idx_l);
        match x {
            CpuFloatStorage::F32(d, _) => {
                let (v, dims) = indexing::gather(d, x_l, &ids, idx_l, dim, x.device())?;
                debug_assert_eq!(&dims, out_shape.dims(), "cpu f_gather shape must match the layer");
                Ok(CpuFloatStorage::F32(v, x.device().clone()))
            }
            CpuFloatStorage::F64(d, _) => {
                let (v, dims) = indexing::gather(d, x_l, &ids, idx_l, dim, x.device())?;
                debug_assert_eq!(&dims, out_shape.dims(), "cpu f_gather shape must match the layer");
                Ok(CpuFloatStorage::F64(v, x.device().clone()))
            }
            CpuFloatStorage::F16(d, _) => {
                let (v, dims) = indexing::gather(d, x_l, &ids, idx_l, dim, x.device())?;
                debug_assert_eq!(&dims, out_shape.dims(), "cpu f_gather shape must match the layer");
                Ok(CpuFloatStorage::F16(v, x.device().clone()))
            }
            CpuFloatStorage::BF16(d, _) => {
                let (v, dims) = indexing::gather(d, x_l, &ids, idx_l, dim, x.device())?;
                debug_assert_eq!(&dims, out_shape.dims(), "cpu f_gather shape must match the layer");
                Ok(CpuFloatStorage::BF16(v, x.device().clone()))
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
        dispatch_float2!(init, src, "index-add", |a, b| indexing::index_add(a, init_l, &ids, idx_l, b, dim, init.device())?)
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
        dispatch_float2!(init, src, "scatter-add", |a, b| indexing::scatter_add(a, init_l, &ids, idx_l, b, dim, init.device())?)
    }

    fn f_cat(srcs: &[(&<Cpu as Device>::FloatStorage, &Layout)], dim: usize, out_shape: &Shape) -> Result<<Cpu as Device>::FloatStorage> {
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
                let (v, shape) = super::kernels::shape::cat(&views, dim, srcs[0].0.device())?;
                debug_assert_eq!(shape.dims(), out_shape.dims(), "cpu f_cat shape must match the layer");
                Ok(CpuFloatStorage::F32(v, srcs[0].0.device().clone()))
            }
            DType::F16 => {
                let views: Vec<(&[half::f16], &Layout)> = srcs.iter().map(|(s, l)| (as_f16(s), *l)).collect();
                let (v, shape) = super::kernels::shape::cat(&views, dim, srcs[0].0.device())?;
                debug_assert_eq!(shape.dims(), out_shape.dims(), "cpu f_cat shape must match the layer");
                Ok(CpuFloatStorage::F16(v, srcs[0].0.device().clone()))
            }
            DType::BF16 => {
                let views: Vec<(&[half::bf16], &Layout)> = srcs.iter().map(|(s, l)| (as_bf16(s), *l)).collect();
                let (v, shape) = super::kernels::shape::cat(&views, dim, srcs[0].0.device())?;
                debug_assert_eq!(shape.dims(), out_shape.dims(), "cpu f_cat shape must match the layer");
                Ok(CpuFloatStorage::BF16(v, srcs[0].0.device().clone()))
            }
            _ => {
                let views: Vec<(&[f64], &Layout)> = srcs.iter().map(|(s, l)| (as_f64(s), *l)).collect();
                let (v, shape) = super::kernels::shape::cat(&views, dim, srcs[0].0.device())?;
                debug_assert_eq!(shape.dims(), out_shape.dims(), "cpu f_cat shape must match the layer");
                Ok(CpuFloatStorage::F64(v, srcs[0].0.device().clone()))
            }
        }
    }

    fn f_softmax(x: &<Cpu as Device>::FloatStorage, l: &Layout, dim: usize) -> Result<<Cpu as Device>::FloatStorage> {
        match x {
            CpuFloatStorage::F32(d, _) => Ok(CpuFloatStorage::F32(nn::softmax(d, l, dim, x.device())?, x.device().clone())),
            CpuFloatStorage::F64(d, _) => Ok(CpuFloatStorage::F64(nn::softmax(d, l, dim, x.device())?, x.device().clone())),
            CpuFloatStorage::F16(d, _) => Ok(CpuFloatStorage::F16(nn::softmax(d, l, dim, x.device())?, x.device().clone())),
            CpuFloatStorage::BF16(d, _) => Ok(CpuFloatStorage::BF16(nn::softmax(d, l, dim, x.device())?, x.device().clone())),
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
            (CpuFloatStorage::F32(d, _), CpuFloatStorage::F32(w, _)) => {
                Ok(CpuFloatStorage::F32(nn::rms_norm(d, x_l, w, weight_l, eps as f32, x.device())?, x.device().clone()))
            }
            (CpuFloatStorage::F64(d, _), CpuFloatStorage::F64(w, _)) => {
                Ok(CpuFloatStorage::F64(nn::rms_norm(d, x_l, w, weight_l, eps, x.device())?, x.device().clone()))
            }
            (CpuFloatStorage::F16(d, _), CpuFloatStorage::F16(w, _)) => {
                Ok(CpuFloatStorage::F16(nn::rms_norm(d, x_l, w, weight_l, half::f16::from_f64(eps), x.device())?, x.device().clone()))
            }
            (CpuFloatStorage::BF16(d, _), CpuFloatStorage::BF16(w, _)) => {
                Ok(CpuFloatStorage::BF16(nn::rms_norm(d, x_l, w, weight_l, half::bf16::from_f64(eps), x.device())?, x.device().clone()))
            }
            (l, r) => Err(Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: "rms_norm" }),
        }
    }

    fn f_pick(
        mask: &<Cpu as Device>::BoolStorage,
        mask_l: &Layout,
        on_true: &<Cpu as Device>::FloatStorage,
        true_l: &Layout,
        on_false: &<Cpu as Device>::FloatStorage,
        false_l: &Layout,
    ) -> Result<<Cpu as Device>::FloatStorage> {
        let m: Vec<bool> = mask.device().collect_alloc(mask_l.storage_indices().map(|i| mask.0[i]));
        match (on_true, on_false) {
            (CpuFloatStorage::F32(t, _), CpuFloatStorage::F32(f, _)) => {
                let tv = super::kernels::iter::gather(t, true_l, on_true.device());
                let fv = super::kernels::iter::gather(f, false_l, on_false.device());
                Ok(CpuFloatStorage::F32(
                    mask.device().collect_alloc(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { fv[i] })),
                    mask.device().clone(),
                ))
            }
            (CpuFloatStorage::F64(t, _), CpuFloatStorage::F64(f, _)) => {
                let tv = super::kernels::iter::gather(t, true_l, on_true.device());
                let fv = super::kernels::iter::gather(f, false_l, on_false.device());
                Ok(CpuFloatStorage::F64(
                    mask.device().collect_alloc(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { fv[i] })),
                    mask.device().clone(),
                ))
            }
            (CpuFloatStorage::F16(t, _), CpuFloatStorage::F16(f, _)) => {
                let tv = super::kernels::iter::gather(t, true_l, on_true.device());
                let fv = super::kernels::iter::gather(f, false_l, on_false.device());
                Ok(CpuFloatStorage::F16(
                    mask.device().collect_alloc(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { fv[i] })),
                    mask.device().clone(),
                ))
            }
            (CpuFloatStorage::BF16(t, _), CpuFloatStorage::BF16(f, _)) => {
                let tv = super::kernels::iter::gather(t, true_l, on_true.device());
                let fv = super::kernels::iter::gather(f, false_l, on_false.device());
                Ok(CpuFloatStorage::BF16(
                    mask.device().collect_alloc(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { fv[i] })),
                    mask.device().clone(),
                ))
            }
            (l, r) => Err(Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: "pick" }),
        }
    }

    fn f_pick_true(
        mask: &<Cpu as Device>::BoolStorage,
        mask_l: &Layout,
        value: f64,
        on_false: &<Cpu as Device>::FloatStorage,
        false_l: &Layout,
    ) -> Result<<Cpu as Device>::FloatStorage> {
        let m: Vec<bool> = mask.device().collect_alloc(mask_l.storage_indices().map(|i| mask.0[i]));
        match on_false {
            CpuFloatStorage::F32(f, _) => {
                let fv = super::kernels::iter::gather(f, false_l, on_false.device());
                let val = value as f32;
                Ok(CpuFloatStorage::F32(
                    mask.device().collect_alloc(m.iter().enumerate().map(|(i, &c)| if c { val } else { fv[i] })),
                    mask.device().clone(),
                ))
            }
            CpuFloatStorage::F64(f, _) => {
                let fv = super::kernels::iter::gather(f, false_l, on_false.device());
                Ok(CpuFloatStorage::F64(
                    mask.device().collect_alloc(m.iter().enumerate().map(|(i, &c)| if c { value } else { fv[i] })),
                    mask.device().clone(),
                ))
            }
            CpuFloatStorage::F16(f, _) => {
                let fv = super::kernels::iter::gather(f, false_l, on_false.device());
                let val = half::f16::from_f64(value);
                Ok(CpuFloatStorage::F16(
                    mask.device().collect_alloc(m.iter().enumerate().map(|(i, &c)| if c { val } else { fv[i] })),
                    mask.device().clone(),
                ))
            }
            CpuFloatStorage::BF16(f, _) => {
                let fv = super::kernels::iter::gather(f, false_l, on_false.device());
                let val = half::bf16::from_f64(value);
                Ok(CpuFloatStorage::BF16(
                    mask.device().collect_alloc(m.iter().enumerate().map(|(i, &c)| if c { val } else { fv[i] })),
                    mask.device().clone(),
                ))
            }
        }
    }

    fn f_pick_false(
        mask: &<Cpu as Device>::BoolStorage,
        mask_l: &Layout,
        on_true: &<Cpu as Device>::FloatStorage,
        true_l: &Layout,
        value: f64,
    ) -> Result<<Cpu as Device>::FloatStorage> {
        let m: Vec<bool> = mask.device().collect_alloc(mask_l.storage_indices().map(|i| mask.0[i]));
        match on_true {
            CpuFloatStorage::F32(t, _) => {
                let tv = super::kernels::iter::gather(t, true_l, on_true.device());
                let val = value as f32;
                Ok(CpuFloatStorage::F32(
                    mask.device().collect_alloc(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { val })),
                    mask.device().clone(),
                ))
            }
            CpuFloatStorage::F64(t, _) => {
                let tv = super::kernels::iter::gather(t, true_l, on_true.device());
                Ok(CpuFloatStorage::F64(
                    mask.device().collect_alloc(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { value })),
                    mask.device().clone(),
                ))
            }
            CpuFloatStorage::F16(t, _) => {
                let tv = super::kernels::iter::gather(t, true_l, on_true.device());
                let val = half::f16::from_f64(value);
                Ok(CpuFloatStorage::F16(
                    mask.device().collect_alloc(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { val })),
                    mask.device().clone(),
                ))
            }
            CpuFloatStorage::BF16(t, _) => {
                let tv = super::kernels::iter::gather(t, true_l, on_true.device());
                let val = half::bf16::from_f64(value);
                Ok(CpuFloatStorage::BF16(
                    mask.device().collect_alloc(m.iter().enumerate().map(|(i, &c)| if c { tv[i] } else { val })),
                    mask.device().clone(),
                ))
            }
        }
    }

    fn f_allclose(a: &CpuFloatStorage, a_l: &Layout, b: &CpuFloatStorage, b_l: &Layout, rtol: f64, atol: f64) -> Result<bool> {
        match (a, b) {
            (CpuFloatStorage::F32(av, _), CpuFloatStorage::F32(bv, _)) => Ok(allclose_generic(av, a_l, bv, b_l, rtol, atol)),
            (CpuFloatStorage::F64(av, _), CpuFloatStorage::F64(bv, _)) => Ok(allclose_generic(av, a_l, bv, b_l, rtol, atol)),
            (CpuFloatStorage::F16(av, _), CpuFloatStorage::F16(bv, _)) => Ok(allclose_generic(av, a_l, bv, b_l, rtol, atol)),
            (CpuFloatStorage::BF16(av, _), CpuFloatStorage::BF16(bv, _)) => Ok(allclose_generic(av, a_l, bv, b_l, rtol, atol)),
            _ => return Err(luma_tensor::Error::DTypeMismatch { lhs: a.dtype(), rhs: b.dtype(), op: "allclose" }),
        }
    }
}

fn as_f32(s: &CpuFloatStorage) -> &[f32] {
    match s {
        CpuFloatStorage::F32(d, _) => d,
        _ => unreachable!("dtype checked by caller"),
    }
}

fn as_f64(s: &CpuFloatStorage) -> &[f64] {
    match s {
        CpuFloatStorage::F64(d, _) => d,
        _ => unreachable!("dtype checked by caller"),
    }
}

fn as_f16(s: &CpuFloatStorage) -> &[half::f16] {
    match s {
        CpuFloatStorage::F16(d, _) => d,
        _ => unreachable!("dtype checked by caller"),
    }
}

fn as_bf16(s: &CpuFloatStorage) -> &[half::bf16] {
    match s {
        CpuFloatStorage::BF16(d, _) => d,
        _ => unreachable!("dtype checked by caller"),
    }
}
