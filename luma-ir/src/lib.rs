//! **luma-ir** — a device/kind-erased graph IR plus a [`Trace`] device.
//!
//! The [`Trace`] device turns any `D: Device`-generic tensor code (including all
//! of `luma-nn`) into a graph recorder: instead of running kernels, it appends
//! nodes to a [`Graph`]. This is the Rust analogue of `torch.jit.trace` and the
//! first step toward TorchScript/ONNX export and graph-level optimization.

pub mod graph;
pub mod trace;

pub use graph::{Graph, Node, NodeOp, Scalar, Value, ValueId};
pub use trace::{Trace, TraceBoolStorage, TraceFloatStorage, TraceIntStorage, Traced};
