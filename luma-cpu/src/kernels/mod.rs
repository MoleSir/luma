//! Generic CPU compute kernels, written over the [`element`] traits and plain
//! slices + [`Layout`](crate::Layout). The `impl FloatOps/IntOps/BoolOps for Cpu`
//! blocks dispatch the storage enums to these.

pub mod element;
pub mod elementwise;
pub mod indexing;
pub mod iter;
pub mod matmul;
pub mod nn;
pub mod reduce;
pub mod shape;
