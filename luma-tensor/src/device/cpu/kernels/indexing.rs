//! Indexing kernels: index_select, gather, index_add, scatter_add.
//! Ported from luma-core `storage.rs`. All require contiguous inputs.
//!
//! Indices arrive as `&[usize]` (the caller reads the int storage into usize;
//! `usize::MAX` is the padding sentinel matching luma-core's `I::max_value()`).

use super::element::{CpuDType, CpuNum};
use crate::{Error, Layout, Result};

pub const PAD: usize = usize::MAX;

/// `dst[.., i, ..] = src[.., ids[i], ..]` along `dim`.
pub fn index_select<T: CpuDType>(src: &[T], src_l: &Layout, ids: &[usize], ids_l: &Layout, dim: usize) -> Result<(Vec<T>, Vec<usize>)> {
    if !src_l.is_contiguous() {
        return Err(Error::RequiresContiguous { op: "index-select" });
    }
    let src = &src[src_l.start_offset()..src_l.start_offset() + src_l.shape().element_count()];
    assert_eq!(ids_l.dims().len(), 1);
    let n_ids = ids_l.dims()[0];
    let stride_ids = ids_l.stride()[0];

    let mut dst_dims = src_l.dims().to_vec();
    let src_dim = dst_dims[dim];
    dst_dims[dim] = n_ids;
    let dst_len: usize = dst_dims.iter().product();
    let left_len: usize = dst_dims[..dim].iter().product();
    let right_len: usize = dst_dims[dim + 1..].iter().product();

    let mut dst = vec![T::ZERO; dst_len];
    for left_i in 0..left_len {
        let start_src = left_i * right_len * src_dim;
        let start_dst = left_i * right_len * n_ids;
        for i in 0..n_ids {
            let start_dst = start_dst + i * right_len;
            let index = ids[ids_l.start_offset() + stride_ids * i];
            if index == PAD {
                dst[start_dst..start_dst + right_len].fill(T::ZERO);
            } else {
                if index >= src_dim {
                    return Err(Error::InvalidIndex { index, size: src_dim, op: "index-select" });
                }
                let start_src = start_src + index * right_len;
                dst[start_dst..start_dst + right_len].copy_from_slice(&src[start_src..start_src + right_len]);
            }
        }
    }
    Ok((dst, dst_dims))
}

/// `dst[i,j,k] = src[i, ids[i,j,k], k]` along `dim`.
pub fn gather<T: CpuDType>(src: &[T], src_l: &Layout, ids: &[usize], ids_l: &Layout, dim: usize) -> Result<(Vec<T>, Vec<usize>)> {
    if !src_l.is_contiguous() || !ids_l.is_contiguous() {
        return Err(Error::RequiresContiguous { op: "gather" });
    }
    let src_dims = src_l.dims();
    let ids_dims = ids_l.dims();
    if src_dims.len() != ids_dims.len() {
        return Err(Error::ShapeMismatchBinaryOp { lhs: src_l.shape().clone(), rhs: ids_l.shape().clone(), op: "gather" });
    }
    let dst_len = ids_l.shape().element_count();
    let mut dst = vec![T::ZERO; dst_len];
    let left_len: usize = src_dims[..dim].iter().product();
    let right_len: usize = src_dims[dim + 1..].iter().product();
    let src_dim_size = src_dims[dim];
    let ids_dim_size = ids_dims[dim];

    for i in 0..left_len {
        let src_block = i * src_dim_size * right_len;
        let dst_block = i * ids_dim_size * right_len;
        for j in 0..ids_dim_size {
            for k in 0..right_len {
                let dst_idx = dst_block + j * right_len + k;
                let index = ids[dst_idx];
                if index == PAD {
                    dst[dst_idx] = T::ZERO;
                    continue;
                }
                if index >= src_dim_size {
                    return Err(Error::InvalidIndex { index, size: src_dim_size, op: "gather" });
                }
                dst[dst_idx] = src[src_block + index * right_len + k];
            }
        }
    }
    Ok((dst, ids_dims.to_vec()))
}

/// `out = dst.clone(); out[.., ids[i], ..] += src[.., i, ..]` along `dim`.
pub fn index_add<T: CpuNum>(dst: &[T], dst_l: &Layout, ids: &[usize], ids_l: &Layout, src: &[T], dim: usize) -> Result<Vec<T>> {
    let mut result = dst.to_vec();
    let n_ids = ids_l.dims()[0];
    let stride_ids = ids_l.stride()[0];
    let dst_dims = dst_l.dims();
    let dst_dim_size = dst_dims[dim];
    let left_len: usize = dst_dims[..dim].iter().product();
    let right_len: usize = dst_dims[dim + 1..].iter().product();

    for left_i in 0..left_len {
        let start_src = left_i * n_ids * right_len;
        let start_dst = left_i * dst_dim_size * right_len;
        for i in 0..n_ids {
            let index = ids[ids_l.start_offset() + stride_ids * i];
            if index == PAD {
                continue;
            }
            if index >= dst_dim_size {
                return Err(Error::InvalidIndex { index, size: dst_dim_size, op: "index-add" });
            }
            let src_off = start_src + i * right_len;
            let dst_off = start_dst + index * right_len;
            for k in 0..right_len {
                result[dst_off + k] = result[dst_off + k] + src[src_off + k];
            }
        }
    }
    Ok(result)
}

/// `out = dst.clone(); out[.., ids[i,j,k], k] += src[i,j,k]` along `dim`.
pub fn scatter_add<T: CpuNum>(dst: &[T], dst_l: &Layout, ids: &[usize], ids_l: &Layout, src: &[T], dim: usize) -> Result<Vec<T>> {
    let mut result = dst.to_vec();
    let dst_dims = dst_l.dims();
    let src_dims = ids_l.dims();
    if dst_dims.len() != src_dims.len() {
        return Err(Error::ShapeMismatchBinaryOp { lhs: dst_l.shape().clone(), rhs: ids_l.shape().clone(), op: "scatter-add" });
    }
    let left_len: usize = src_dims[..dim].iter().product();
    let right_len: usize = src_dims[dim + 1..].iter().product();
    let src_dim_size = src_dims[dim];
    let dst_dim_size = dst_dims[dim];

    for i in 0..left_len {
        let src_block = i * src_dim_size * right_len;
        let dst_block = i * dst_dim_size * right_len;
        for j in 0..src_dim_size {
            for k in 0..right_len {
                let linear = src_block + j * right_len + k;
                let index = ids[linear];
                if index == PAD {
                    continue;
                }
                if index >= dst_dim_size {
                    return Err(Error::InvalidIndex { index, size: dst_dim_size, op: "scatter-add" });
                }
                let dst_idx = dst_block + index * right_len + k;
                result[dst_idx] = result[dst_idx] + src[linear];
            }
        }
    }
    Ok(result)
}
