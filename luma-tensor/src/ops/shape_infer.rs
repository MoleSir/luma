//! Canonical output-shape inference for ops whose result shape is not simply
//! the input shape (reduce / matmul / indexing / cat).
//!
//! These are the *only* shape rules the tensor layer trusts: the layer computes
//! the output shape here, passes it to the device seam, and wraps the result
//! with it. Device kernels still compute shapes internally (they need them for
//! allocation), but the seams `debug_assert` those against the canonical value
//! — a device can never silently disagree with the layer.

use crate::{Error, Result, Shape};

/// Shape after reducing `dims` (duplicates ignored), optionally keeping them.
pub fn reduce_out_shape(input: &Shape, dims: &[usize], keepdim: bool) -> Shape {
    let mut sorted: Vec<usize> = dims.to_vec();
    sorted.sort_unstable();
    sorted.dedup();

    let mut out = input.dims().to_vec();
    if keepdim {
        for &d in &sorted {
            out[d] = 1;
        }
    } else {
        for &d in sorted.iter().rev() {
            out.remove(d);
        }
    }
    Shape::from(out)
}

/// Shape of `lhs @ rhs`: batched matmul with **equal batch products** (the
/// rule both the Cpu and Cuda kernels enforce); batch dims come from `lhs`.
pub fn matmul_out_shape(lhs: &Shape, rhs: &Shape) -> Result<Shape> {
    let a = lhs.dims();
    let b = rhs.dims();
    if a.len() < 2 || b.len() != a.len() {
        return Err(Error::ShapeMismatchBinaryOp { lhs: lhs.clone(), rhs: rhs.clone(), op: "matmul" });
    }
    let (m, k) = (a[a.len() - 2], a[a.len() - 1]);
    let (k2, n) = (b[b.len() - 2], b[b.len() - 1]);
    let batch: usize = a[..a.len() - 2].iter().product();
    let batch_b: usize = b[..b.len() - 2].iter().product();
    if k != k2 || batch != batch_b {
        return Err(Error::ShapeMismatchBinaryOp { lhs: lhs.clone(), rhs: rhs.clone(), op: "matmul" });
    }
    let mut out = a[..a.len() - 2].to_vec();
    out.push(m);
    out.push(n);
    Ok(Shape::from(out))
}

/// Shape after `index_select` along `dim` with `idx_elems` indices.
pub fn index_select_out_shape(x: &Shape, idx_elems: usize, dim: usize) -> Shape {
    let mut dims = x.dims().to_vec();
    dims[dim] = idx_elems;
    Shape::from(dims)
}

/// Shape after `gather`: identical to the index tensor's shape.
pub fn gather_out_shape(idx: &Shape) -> Shape {
    idx.clone()
}

/// Shape after concatenating along `dim` (non-`dim` dims must already match).
pub fn cat_out_shape(srcs: &[&Shape], dim: usize) -> Shape {
    let mut dims = srcs[0].dims().to_vec();
    dims[dim] = srcs.iter().map(|s| s.dims()[dim]).sum();
    Shape::from(dims)
}
