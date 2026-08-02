use super::{Dim, DimCoordinates, DimNCoordinates};
use crate::{Error, Result};
use std::vec;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(pub(crate) Vec<usize>);

impl Shape {
    pub fn scalar() -> Self {
        Self(vec![])
    }

    pub fn is_scalar(&self) -> bool {
        self.0.is_empty() || (self.0.len() == 1 && self.0[0] == 1)
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn into_dims(self) -> Vec<usize> {
        self.0
    }

    pub fn dim(&self, dim: impl Dim) -> Result<usize> {
        let index = dim.to_index(self, "get dim")?;
        Ok(self.dims()[index])
    }

    pub fn element_count(&self) -> usize {
        self.dims().iter().product()
    }

    pub fn is_contiguous(&self, stride: &[usize]) -> bool {
        if self.rank() != stride.len() {
            return false;
        }
        // [3, 4, 5] & [20, 5, 1]
        let mut acc = 1;
        for (&stride, &dim) in stride.iter().zip(self.dims().iter()).rev() {
            if dim > 1 && stride != acc {
                return false;
            }
            acc *= dim;
        }
        true
    }

    pub fn extend(mut self, additional_dims: &[usize]) -> Self {
        self.0.extend(additional_dims);
        self
    }

    /// Check whether the two shapes are compatible for broadcast, and if it is the case return the
    /// broadcasted shape. This is to be used for binary pointwise ops.
    /// Copy from https://github.com/huggingface/candle
    pub fn broadcast_shape_binary_op(&self, rhs: &Self, op: &'static str) -> Result<Shape> {
        let lhs = self;
        let lhs_dims = lhs.dims();
        let rhs_dims = rhs.dims();
        let lhs_ndims = lhs_dims.len();
        let rhs_ndims = rhs_dims.len();
        let bcast_ndims = usize::max(lhs_ndims, rhs_ndims);
        let mut bcast_dims = vec![0; bcast_ndims];
        for (idx, bcast_value) in bcast_dims.iter_mut().enumerate() {
            let rev_idx = bcast_ndims - idx;
            let l_value = if lhs_ndims < rev_idx { 1 } else { lhs_dims[lhs_ndims - rev_idx] };
            let r_value = if rhs_ndims < rev_idx { 1 } else { rhs_dims[rhs_ndims - rev_idx] };
            *bcast_value = if l_value == r_value {
                // keep
                l_value
            } else if l_value == 1 {
                // bcast l
                r_value
            } else if r_value == 1 {
                // bcast r
                l_value
            } else {
                Err(Error::ShapeMismatchBinaryOp { lhs: lhs.clone(), rhs: rhs.clone(), op })?
            }
        }
        Ok(Shape::from(bcast_dims))
    }

    /// Returns an iterator over **dimension coordinates**.
    ///
    /// This iterator yields the multi-dimensional coordinates
    /// (e.g., `[i, j, k, ...]`) of each element in the array, independent
    /// of the physical storage layout.
    ///
    /// Example for shape = (2, 2):
    /// yields: `[0, 0], [0, 1], [1, 0], [1, 1]`
    pub fn dim_coordinates(&self) -> DimCoordinates {
        DimCoordinates::from_shape(self)
    }

    pub fn dims_coordinates<const N: usize>(&self) -> Result<DimNCoordinates<N>> {
        DimNCoordinates::<N>::from_shape(self)
    }

    pub fn dim2_coordinates(&self) -> Result<DimNCoordinates<2>> {
        DimNCoordinates::<2>::from_shape(self)
    }

    pub fn dim3_coordinates(&self) -> Result<DimNCoordinates<3>> {
        DimNCoordinates::<3>::from_shape(self)
    }

    pub fn dim4_coordinates(&self) -> Result<DimNCoordinates<4>> {
        DimNCoordinates::<4>::from_shape(self)
    }

    pub fn dim5_coordinates(&self) -> Result<DimNCoordinates<5>> {
        DimNCoordinates::<5>::from_shape(self)
    }

    pub(crate) fn stride_contiguous(&self) -> Vec<usize> {
        let mut stride = self
            .dims()
            .iter()
            .rev()
            .scan(1, |prod, u| {
                let prod_pre_mult = *prod;
                *prod *= u;
                Some(prod_pre_mult)
            })
            .collect::<Vec<_>>();
        stride.reverse();
        stride
    }
}
