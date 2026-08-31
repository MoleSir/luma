//! Device-agnostic IR side of the compiler: capture and optimize.
//!
//! Everything here works on the [`Graph`](crate::graph::Graph) IR and never
//! touches a concrete device — the counterpart of [`backend`](crate::backend),
//! which lowers the graph to a device and runs it.
//!
//! - [`trace`] — the [`Trace`](crate::trace::Trace) device captures a traced
//!   module as a graph (source → IR);
//! - [`opt`] — front-end optimization passes (IR → IR): simplify/fold/cse/dce,
//!   verified by the pipeline in [`opt::optimize`].

pub mod opt;
pub mod trace;
