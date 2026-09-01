//! Fused nn kernels: softmax and rms_norm. Ported from luma-core `nn.rs`.

use super::element::CpuFloat;
use super::iter::DimArray;
use crate::device::cpu::allocator::AllocVec;
use crate::{Cpu, Layout, Result};

/// Numerically stable softmax over `dim` using the online max+sum algorithm.
pub fn softmax<T: CpuFloat + AllocVec>(x: &[T], layout: &Layout, dim: usize, device: &Cpu) -> Result<Vec<T>> {
    let reduce_dim_stride = layout.stride()[dim];
    let reduce_dim_size = layout.dims()[dim];
    let mut dst = device.fill_alloc(layout.element_count(), T::ZERO);

    if layout.is_contiguous() && reduce_dim_stride == 1 {
        let x = &x[layout.start_offset()..];
        for (src_chunk, dst_chunk) in
            x.chunks(reduce_dim_size).zip(dst.chunks_mut(reduce_dim_size)).take(layout.element_count() / reduce_dim_size)
        {
            online_softmax(src_chunk, dst_chunk);
        }
    } else {
        let collapsed = layout.narrow(dim, 0, 1)?;
        let mut dst_offset = 0usize;
        for src_index in collapsed.storage_indices() {
            // gather the strided source row
            let row: Vec<T> = device.collect_alloc(DimArray::new(&x[src_index..], reduce_dim_size, reduce_dim_stride).into_iter());
            let mut out_row = device.fill_alloc(reduce_dim_size, T::ZERO);
            online_softmax(&row, &mut out_row);
            for (i, v) in out_row.into_iter().enumerate() {
                dst[dst_offset + i] = v;
            }
            dst_offset += reduce_dim_size;
        }
    }
    Ok(dst)
}

fn online_softmax<T: CpuFloat>(src: &[T], dst: &mut [T]) {
    if src.is_empty() {
        return;
    }
    let mut m = src[0];
    let mut s = T::ONE;
    for &x in src.iter().skip(1) {
        if x <= m {
            s = s + (x - m).exp();
        } else {
            s = s * (m - x).exp() + T::ONE;
            m = x;
        }
    }
    for (i, &x) in src.iter().enumerate() {
        dst[i] = (x - m).exp() / s;
    }
}

/// RMSNorm over the last dim: `out = x / sqrt(mean(x^2) + eps) * weight`.
pub fn rms_norm<T: CpuFloat + AllocVec>(x: &[T], x_l: &Layout, weight: &[T], weight_l: &Layout, eps: T, device: &Cpu) -> Result<Vec<T>> {
    let last_dim = x_l.shape().rank() - 1;
    let last_dim_stride = x_l.stride()[last_dim];
    let last_dim_size = x_l.dims()[last_dim];
    let weight_off = weight_l.start_offset();
    let weight_stride = weight_l.stride()[0];

    let mut out = device.fill_alloc(x_l.element_count(), T::ZERO);

    if x_l.is_contiguous() && last_dim_stride == 1 {
        let x = &x[x_l.start_offset()..];
        let batch = out.len() / last_dim_size;
        for b in 0..batch {
            let src = &x[b * last_dim_size..b * last_dim_size + last_dim_size];
            let dst = &mut out[b * last_dim_size..b * last_dim_size + last_dim_size];
            let variance = src.iter().map(|&v| v.sqr()).sum::<T>() / T::from_usize(last_dim_size);
            let rms = (variance + eps).sqrt();
            for i in 0..last_dim_size {
                dst[i] = src[i] / rms * weight[weight_off + i * weight_stride];
            }
        }
    } else {
        let collapsed = x_l.narrow(last_dim, 0, 1)?;
        let mut dst_offset = 0usize;
        for src_index in collapsed.storage_indices() {
            let row: Vec<T> = device.collect_alloc(DimArray::new(&x[src_index..], last_dim_size, last_dim_stride).into_iter());
            let variance = row.iter().map(|&v| v.sqr()).sum::<T>() / T::from_usize(last_dim_size);
            let rms = (variance + eps).sqrt();
            for i in 0..last_dim_size {
                out[dst_offset + i] = row[i] / rms * weight[weight_off + i * weight_stride];
            }
            dst_offset += last_dim_size;
        }
    }
    Ok(out)
}
