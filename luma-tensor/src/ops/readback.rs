//! Read-back operations: extract tensor data as plain `Vec`s.
//!
//! These are per-kind methods that call directly through the device traits
//! (no Kind dispatch layer) — each kind returns a different element type,
//! so there is no code-sharing benefit to a generic intermediate trait.

use crate::{Bool, Device, Float, Int, Tensor};

impl<D: Device> Tensor<D, Float> {
    /// Read all elements into a `Vec<f64>` in logical (layout) order.
    pub fn to_vec(&self) -> crate::Result<Vec<f64>> {
        D::f_to_vec(&*self.storage_read()?, self.layout())
    }
}

impl<D: Device> Tensor<D, Int> {
    /// Read all elements into a `Vec<i64>` in logical (layout) order.
    pub fn to_vec(&self) -> crate::Result<Vec<i64>> {
        D::i_to_vec(&*self.storage_read()?, self.layout())
    }
}

impl<D: Device> Tensor<D, Bool> {
    /// Read all elements into a `Vec<bool>` in logical (layout) order.
    pub fn to_vec(&self) -> crate::Result<Vec<bool>> {
        D::b_to_vec(&*self.storage_read()?, self.layout())
    }
}
