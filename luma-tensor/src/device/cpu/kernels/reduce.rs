//! Generic reduction kernels. Ported from luma-core `reduce.rs`.
//!
//! Reduces a single dimension at a time; multi-dim reductions are applied by the
//! caller folding over dims (outermost-first, adjusting indices).

use super::element::CpuNum;
use super::iter::DimArray;
use crate::{Layout, ReduceOp, Result, Shape};

/// Which single-axis reduction to run.
#[derive(Clone, Copy)]
pub enum Reducer {
    Sum,
    Mean,
    Min,
    Max,
    Product,
}

impl From<ReduceOp> for Reducer {
    fn from(op: ReduceOp) -> Self {
        match op {
            ReduceOp::Sum => Reducer::Sum,
            ReduceOp::Mean => Reducer::Mean,
            ReduceOp::Min => Reducer::Min,
            ReduceOp::Max => Reducer::Max,
            ReduceOp::Prod => Reducer::Product,
        }
    }
}

/// Reduce over multiple dims. Reduces from the highest dim down (with keepdim to
/// preserve dim indices), then optionally squeezes the reduced dims out.
pub fn reduce_dims<T: CpuNum>(x: &[T], layout: &Layout, dims: &[usize], keepdim: bool, reducer: Reducer) -> Result<(Vec<T>, Shape)> {
    let mut sorted: Vec<usize> = dims.to_vec();
    sorted.sort_unstable();
    sorted.dedup();

    // First reduction reads from the input storage + layout.
    let mut cur_layout = layout.clone();
    let mut cur_data: Vec<T>;
    let (mut data, mut shape): (Vec<T>, Shape);

    if sorted.is_empty() {
        return Ok((crate::device::cpu::kernels::iter::gather(x, layout), layout.shape().clone()));
    }

    // reduce highest dim first so lower indices stay valid
    let mut iter = sorted.iter().rev();
    let first = *iter.next().unwrap();
    let (d, s) = reduce_dim(x, &cur_layout, first, true, reducer)?;
    data = d;
    shape = s;
    for &dim in iter {
        cur_layout = Layout::contiguous(shape.clone());
        cur_data = data;
        let (d, s) = reduce_dim(&cur_data, &cur_layout, dim, true, reducer)?;
        data = d;
        shape = s;
    }

    if !keepdim {
        let mut out_dims: Vec<usize> = Vec::new();
        for (i, &d) in shape.dims().iter().enumerate() {
            if !sorted.contains(&i) {
                out_dims.push(d);
            }
        }
        shape = Shape::from(out_dims);
    }
    Ok((data, shape))
}

impl Reducer {
    fn apply<T: CpuNum, I: Iterator<Item = T> + ExactSizeIterator>(&self, iter: I) -> T {
        match self {
            Reducer::Sum => iter.sum(),
            Reducer::Mean => {
                let len = iter.len();
                let s: T = iter.sum();
                s / T::from_usize(len.max(1))
            }
            Reducer::Min => iter.reduce(|a, b| T::minimum(a, b)).unwrap_or(T::ZERO),
            Reducer::Max => iter.reduce(|a, b| T::maximum(a, b)).unwrap_or(T::ZERO),
            Reducer::Product => iter.product(),
        }
    }
}

/// Reduce `x`/`layout` over a single `reduce_dim`. Returns the reduced buffer and
/// its shape (with `reduce_dim` kept as size-1 if `keepdim`, else removed).
pub fn reduce_dim<T: CpuNum>(x: &[T], layout: &Layout, reduce_dim: usize, keepdim: bool, reducer: Reducer) -> Result<(Vec<T>, Shape)> {
    let reduce_dim_stride = layout.stride()[reduce_dim];
    let reduce_dim_size = layout.dims()[reduce_dim];

    let dst: Vec<T> = if layout.is_contiguous() && reduce_dim_stride == 1 {
        let x = &x[layout.start_offset()..];
        (0..layout.element_count() / reduce_dim_size)
            .map(|i| {
                let chunk = &x[i * reduce_dim_size..i * reduce_dim_size + reduce_dim_size];
                reducer.apply(chunk.iter().copied())
            })
            .collect()
    } else {
        let dst_len = layout.element_count() / reduce_dim_size;
        let mut dst: Vec<T> = Vec::with_capacity(dst_len);
        let collapsed = layout.narrow(reduce_dim, 0, 1)?;
        if reduce_dim_stride == 1 {
            for src_index in collapsed.storage_indices() {
                let chunk = &x[src_index..src_index + reduce_dim_size];
                dst.push(reducer.apply(chunk.iter().copied()));
            }
        } else {
            for src_index in collapsed.storage_indices() {
                let arr = DimArray::new(&x[src_index..], reduce_dim_size, reduce_dim_stride);
                dst.push(reducer.apply(arr.into_iter()));
            }
        }
        dst
    };

    let mut shape = layout.dims().to_vec();
    if keepdim {
        shape[reduce_dim] = 1;
    } else {
        shape.remove(reduce_dim);
    }
    Ok((dst, Shape::from(shape)))
}

/// argmin/argmax over a single dim. Returns `usize` indices (caller casts to the
/// int storage dtype) and the result shape. Ties keep the first index.
pub fn arg_reduce<T: CpuNum>(x: &[T], layout: &Layout, dim: usize, keepdim: bool, take_max: bool) -> Result<(Vec<usize>, Shape)> {
    let reduce_dim_stride = layout.stride()[dim];
    let reduce_dim_size = layout.dims()[dim];

    let arg = |iter: DimArrayIter<T>| -> usize {
        iter.enumerate()
            .reduce(|(ia, a), (ib, b)| {
                let keep_a = if take_max { a >= b } else { a <= b };
                if keep_a { (ia, a) } else { (ib, b) }
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    };

    let dst_len = layout.element_count() / reduce_dim_size;
    let mut dst: Vec<usize> = Vec::with_capacity(dst_len);
    let collapsed = layout.narrow(dim, 0, 1)?;
    for src_index in collapsed.storage_indices() {
        let arr = DimArray::new(&x[src_index..], reduce_dim_size, reduce_dim_stride);
        dst.push(arg(arr.into_iter()));
    }

    let mut shape = layout.dims().to_vec();
    if keepdim {
        shape[dim] = 1;
    } else {
        shape.remove(dim);
    }
    Ok((dst, Shape::from(shape)))
}

use super::iter::DimArrayIter;
