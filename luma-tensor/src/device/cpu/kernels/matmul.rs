//! Batched matmul. Ported from luma-core `matmul.rs`. `f32`/`f64` go through
//! the `gemm` crate; other element types use a naive triple loop.

use std::any::TypeId;

use gemm::Parallelism;

use super::element::CpuNum;
use crate::{Error, Layout, Result, Shape};

fn num_threads() -> usize {
    std::env::var("RAYON_NUM_THREADS").ok().and_then(|s| s.parse().ok()).filter(|&n| n > 0).unwrap_or_else(num_cpus::get)
}

/// Offset into a batched tensor's storage for batch index `b`.
fn batch_offset(b: usize, layout: &Layout) -> usize {
    let rank = layout.shape().rank();
    let batch_dims = &layout.dims()[..rank - 2];
    let batch_strides = &layout.stride()[..rank - 2];
    let mut offset = layout.start_offset();
    let mut temp = b;
    for i in (0..batch_dims.len()).rev() {
        let idx = temp % batch_dims[i];
        offset += idx * batch_strides[i];
        temp /= batch_dims[i];
    }
    offset
}

/// Validate matmul shapes and return `(batch, m, n, k, out_shape)`.
fn matmul_dims(lhs_l: &Layout, rhs_l: &Layout) -> Result<(usize, usize, usize, usize, Shape)> {
    let a = lhs_l.dims();
    let b = rhs_l.dims();
    let rank = a.len();
    if rank < 2 || b.len() != rank {
        return Err(Error::ShapeMismatchBinaryOp { lhs: lhs_l.shape().clone(), rhs: rhs_l.shape().clone(), op: "matmul" });
    }
    let m = a[rank - 2];
    let k = a[rank - 1];
    let k2 = b[rank - 2];
    let n = b[rank - 1];
    let batch: usize = a[..rank - 2].iter().product();
    let batch_b: usize = b[..rank - 2].iter().product();
    if k != k2 || batch != batch_b {
        return Err(Error::ShapeMismatchBinaryOp { lhs: lhs_l.shape().clone(), rhs: rhs_l.shape().clone(), op: "matmul" });
    }
    let mut out = a[..rank - 2].to_vec();
    out.push(m);
    out.push(n);
    Ok((batch, m, n, k, Shape::from(out)))
}

fn is_gemm_type<T: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>() || TypeId::of::<T>() == TypeId::of::<f64>()
}

/// `dst = lhs @ rhs` (batched). Returns the output buffer and shape.
pub fn matmul<T: CpuNum>(lhs: &[T], lhs_l: &Layout, rhs: &[T], rhs_l: &Layout) -> Result<(Vec<T>, Shape)> {
    let (batch, m, n, k, out_shape) = matmul_dims(lhs_l, rhs_l)?;
    let mns = m * n;
    let mut dst = vec![T::ZERO; batch * mns];
    if out_shape.element_count() == 0 || k == 0 {
        return Ok((dst, out_shape));
    }

    let rank = lhs_l.shape().rank();
    let l_stride_m = lhs_l.stride()[rank - 2] as isize;
    let l_stride_k = lhs_l.stride()[rank - 1] as isize;
    let r_stride_k = rhs_l.stride()[rank - 2] as isize;
    let r_stride_n = rhs_l.stride()[rank - 1] as isize;

    if is_gemm_type::<T>() {
        let parallelism = Parallelism::Rayon(num_threads());
        for b in 0..batch {
            let l_off = batch_offset(b, lhs_l);
            let r_off = batch_offset(b, rhs_l);
            let dst_slice = &mut dst[b * mns..b * mns + mns];
            unsafe {
                gemm::gemm(
                    m,
                    n,
                    k,
                    dst_slice.as_mut_ptr(),
                    1,
                    n as isize,
                    false,
                    lhs.as_ptr().offset(l_off as isize),
                    l_stride_k,
                    l_stride_m,
                    rhs.as_ptr().offset(r_off as isize),
                    r_stride_n,
                    r_stride_k,
                    T::ZERO,
                    T::ONE,
                    false,
                    false,
                    false,
                    parallelism,
                );
            }
        }
    } else {
        for b in 0..batch {
            let l_off = batch_offset(b, lhs_l);
            let r_off = batch_offset(b, rhs_l);
            let dst_slice = &mut dst[b * mns..b * mns + mns];
            for i in 0..m {
                for p in 0..k {
                    let l_val = lhs[(l_off as isize + i as isize * l_stride_m + p as isize * l_stride_k) as usize];
                    for j in 0..n {
                        let r_val = rhs[(r_off as isize + p as isize * r_stride_k + j as isize * r_stride_n) as usize];
                        dst_slice[i * n + j] = dst_slice[i * n + j] + l_val * r_val;
                    }
                }
            }
        }
    }

    Ok((dst, out_shape))
}

/// Compute output shape for matmul (exposed for reuse).
pub fn out_shape(lhs: &Shape, rhs: &Shape) -> Result<Shape> {
    let (_, _, _, _, shape) = matmul_dims(&Layout::contiguous(lhs.clone()), &Layout::contiguous(rhs.clone()))?;
    Ok(shape)
}

/// `dst += lhs @ rhs` (batched, in-place accumulate). Writes directly into `dst`
/// without allocating a temporary product buffer.
///
/// `dst_l` must be contiguous for best performance (the gemm path requires it);
/// the caller (`f_add_matmul_`) ensures this.
pub fn add_matmul<T: CpuNum>(dst: &mut [T], dst_l: &Layout, lhs: &[T], lhs_l: &Layout, rhs: &[T], rhs_l: &Layout) -> Result<()> {
    let (batch, m, n, k, _out_shape) = matmul_dims(lhs_l, rhs_l)?;
    let mns = m * n;
    let rank = lhs_l.shape().rank();

    let l_stride_m = lhs_l.stride()[rank - 2] as isize;
    let l_stride_k = lhs_l.stride()[rank - 1] as isize;
    let r_stride_k = rhs_l.stride()[rank - 2] as isize;
    let r_stride_n = rhs_l.stride()[rank - 1] as isize;

    if is_gemm_type::<T>() {
        let parallelism = Parallelism::Rayon(num_threads());
        for b in 0..batch {
            let l_off = batch_offset(b, lhs_l);
            let r_off = batch_offset(b, rhs_l);
            let d_off = batch_offset(b, dst_l);
            let dst_slice = &mut dst[d_off..d_off + mns];
            unsafe {
                gemm::gemm(
                    m,
                    n,
                    k,
                    dst_slice.as_mut_ptr(),
                    1,
                    n as isize,
                    false,
                    lhs.as_ptr().offset(l_off as isize),
                    l_stride_k,
                    l_stride_m,
                    rhs.as_ptr().offset(r_off as isize),
                    r_stride_n,
                    r_stride_k,
                    T::ONE, // alpha: accumulate
                    T::ONE, // beta
                    false,
                    false,
                    false,
                    parallelism,
                );
            }
        }
    } else {
        for b in 0..batch {
            let l_off = batch_offset(b, lhs_l);
            let r_off = batch_offset(b, rhs_l);
            let d_off = batch_offset(b, dst_l);
            for i in 0..m {
                for p in 0..k {
                    let l_val = lhs[(l_off as isize + i as isize * l_stride_m + p as isize * l_stride_k) as usize];
                    for j in 0..n {
                        let r_val = rhs[(r_off as isize + p as isize * r_stride_k + j as isize * r_stride_n) as usize];
                        dst[d_off + i * n + j] = dst[d_off + i * n + j] + l_val * r_val;
                    }
                }
            }
        }
    }
    Ok(())
}
