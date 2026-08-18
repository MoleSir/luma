mod bool_ops;
mod float_ops;
mod int_ops;

use crate::{Error, Layout, Result, Shape};

pub(crate) fn cat_compute_shape(layouts: &[&Layout], dim: usize) -> Result<Shape> {
    if layouts.is_empty() {
        return Err(Error::OpRequiresAtLeastOneTensor { op: "cat" });
    }
    let first = layouts[0];
    let rank = first.dims().len();
    let mut out_dims = first.dims().to_vec();
    let mut cat_size = 0usize;
    for (n, l) in layouts.iter().enumerate() {
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
    Ok(Shape::from(out_dims))
}
