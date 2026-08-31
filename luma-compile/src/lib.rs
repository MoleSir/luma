//! **luma-compile** — trace, compile, and execute luma modules as portable graphs.
//!
//! A mini AI compiler stack:
//! - [`frontend`] — device-agnostic IR side: [`trace`] captures a traced module
//!   into a [`Graph`], [`opt`] optimizes it (simplify/fold/cse/dce);
//! - [`backend`] — device-specific side: a [`GraphExecutor`] lowers the graph to
//!   concrete [`Step`](backend::Step)s and runs them.
//!
//! This is the Rust analogue of `torch.jit.trace` plus a graph optimizer and
//! lowering pipeline, a first step toward TorchScript/ONNX export.

pub mod graph;
pub mod trace;
pub mod backend;
pub mod frontend;
pub mod io;

mod error;
pub use error::*;

pub use backend::{GraphExecutor, Step};
pub use trace::{Trace, TraceBoolStorage, TraceFloatStorage, TraceInput, TraceIntStorage, Traced, trace};
pub use graph::{ConstData, Graph, Node, NodeOp, Scalar, Value, ValueId};
