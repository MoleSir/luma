//! **luma-jit** — trace, compile, and execute luma modules as portable graphs.
//!
//! The [`Trace`] device turns any `D: Device`-generic tensor code (including all
//! of `luma-nn`) into a graph recorder: instead of running kernels, it appends
//! nodes to a [`Graph`]. A [`GraphExecutor`] then lowers that graph to a concrete
//! device and runs it. This is the Rust analogue of `torch.jit.trace` and the
//! first step toward TorchScript/ONNX export and graph-level optimization.

mod error;
pub mod executor;
pub mod graph;
pub mod opt;
pub mod trace;

pub use error::*;

pub use executor::GraphExecutor;
pub use graph::{ConstData, Graph, Node, NodeOp, Scalar, Value, ValueId};
pub use trace::{Trace, TraceBoolStorage, TraceFloatStorage, TraceInput, TraceIntStorage, Traced, trace};
