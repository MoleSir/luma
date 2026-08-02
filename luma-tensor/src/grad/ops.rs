//! In-place gradient-accumulation helpers used by the backward pass.
//!
//! These bypass autograd (they mutate an accumulator storage directly). Grad
//! accumulators are freshly-allocated, uniquely-owned tensors, so in-place
//! mutation is safe.

use crate::{BinaryOp, Device, Float, Tensor};

impl<D: Device> Tensor<D, Float> {
    /// `self += other` (in place), respecting layouts.
    pub(crate) fn impl_add_(&self, other: &Self) -> crate::Result<()> {
        let other_guard = other.storage_read()?;
        let mut dst = self.storage_write()?;
        D::f_binary_(&mut dst, self.layout(), &other_guard, other.layout(), BinaryOp::Add)
    }

    /// `self -= other` (in place), respecting layouts.
    pub(crate) fn impl_sub_(&self, other: &Self) -> crate::Result<()> {
        let other_guard = other.storage_read()?;
        let mut dst = self.storage_write()?;
        D::f_binary_(&mut dst, self.layout(), &other_guard, other.layout(), BinaryOp::Sub)
    }

    /// `self += lhs @ rhs` (in place). Used by matmul backward.
    pub(crate) fn add_matmul_(&self, lhs: &Self, rhs: &Self) -> crate::Result<()> {
        let lhs_g = lhs.storage_read()?;
        let rhs_g = rhs.storage_read()?;
        let mut dst = self.storage_write()?;
        D::f_add_matmul_(&mut dst, self.layout(), &lhs_g, lhs.layout(), &rhs_g, rhs.layout())
    }
}
