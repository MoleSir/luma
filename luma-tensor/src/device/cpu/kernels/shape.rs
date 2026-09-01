//! Shape-movement kernels that materialize data: concatenation.

use super::iter::gather;
use crate::Cpu;
use crate::device::cpu::allocator::AllocVec;
use crate::{Error, Layout, Result, Shape};

/// Concatenate contiguous logical views along `dim`. Each source is first
/// materialized in logical order, then copied block-by-block into the output.
pub fn cat<T: Copy + Default + AllocVec>(srcs: &[(&[T], &Layout)], dim: usize, device: &Cpu) -> Result<(Vec<T>, Shape)> {
    if srcs.is_empty() {
        return Err(Error::OpRequiresAtLeastOneTensor { op: "cat" });
    }
    let first = srcs[0].1;
    let rank = first.shape().rank();
    // output shape: sum sizes along `dim`, all others must match.
    let mut out_dims = first.dims().to_vec();
    let mut cat_size = 0usize;
    for (n, (_, l)) in srcs.iter().enumerate() {
        let d = l.dims();
        if d.len() != rank {
            return Err(Error::ShapeMismatchCat { dim, first_shape: first.shape().clone(), n, nth_shape: l.shape().clone() });
        }
        for (i, (&a, &b)) in first.dims().iter().zip(d.iter()).enumerate() {
            if i != dim && a != b {
                return Err(Error::ShapeMismatchCat { dim, first_shape: first.shape().clone(), n, nth_shape: l.shape().clone() });
            }
        }
        cat_size += d[dim];
    }
    out_dims[dim] = cat_size;
    let out_shape = Shape::from(out_dims.clone());

    let outer: usize = out_dims[..dim].iter().product();
    let right: usize = out_dims[dim + 1..].iter().product();

    let mut out = device.fill_alloc(out_shape.element_count(), T::default());
    // materialize each source contiguously, then interleave along `dim`.
    let materialized: Vec<Vec<T>> = srcs.iter().map(|(d, l)| gather(d, l, device)).collect();

    for o in 0..outer {
        let mut dst_dim_base = 0usize;
        for (src_idx, (_, l)) in srcs.iter().enumerate() {
            let src_dim = l.dims()[dim];
            let block = src_dim * right;
            let src = &materialized[src_idx][o * block..o * block + block];
            let dst_start = (o * cat_size + dst_dim_base) * right;
            out[dst_start..dst_start + block].copy_from_slice(src);
            dst_dim_base += src_dim;
        }
    }
    Ok((out, out_shape))
}
