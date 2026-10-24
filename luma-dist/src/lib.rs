//! **luma-dist** — collective communication for luma.
//!
//! This crate defines the *device-generic* collective layer. Concrete
//! transports (NCCL under the `nccl` feature; a Gloo-style CPU backend under
//! `gloo`; MPI later) implement the [`DistBackend`] trait, keeping this layer
//! free of `cudarc` or any other native dependency.
//!
//! # Layout
//!
//! - [`base`] — backend-agnostic primitives: the [`DistBackend`] trait,
//!   [`Comm`], [`DistKind`], [`group`], [`DistError`], and [`TensorDistExt`].
//! - [`scheme`] — single-process execution schemes: [`SingleThread`] and
//!   [`MultiThread`].
//! - [`rendezvous`] — how ranks find each other and exchange their bootstrap
//!   payload (e.g. [`Tcp`], via the [`tcp!`](crate::tcp) macro).
//! - [`backends`] — concrete transports (NCCL; a Gloo-style CPU scaffold).
//!
//! # Primitives
//!
//! - [`DistBackend<D>`](DistBackend) — the object-safe trait a transport
//!   implements for a device `D`. It exposes one *typed* method per collective
//!   per kind (`all_reduce_float`/`_int`/`_bool`, …); its signatures only mention
//!   luma types, so native handles stay private.
//! - [`DistKind`] — static-dispatch glue from a tensor kind to the matching
//!   backend method; users never name it.
//! - [`Comm<D>`](Comm) — a cheap-to-clone, user-facing handle wrapping a
//!   `dyn DistBackend<D>`. It exposes a single generic method per collective,
//!   e.g. `all_reduce::<K>`.
//! - [`TensorDistExt`] — optional method syntax on `Tensor<D, K>`.
//!
//! # Execution schemes
//!
//! Compute code is written against a single [`Comm`] (one rank). How you
//! *launch* several ranks is a separate choice, encoded in the entry point:
//!
//! | scheme | entry point | grouping | world |
//! |---|---|---|---|
//! | one thread, many ranks | [`SingleThread`] + [`SingleThread::scope`] | automatic | single process |
//! | one thread per rank | [`MultiThread`] + [`MultiThread::run`] | none needed | single process |
//! | one process per rank | `nccl::init_process_group` | none needed | multi-process |
//!
//! The communicator is always an **explicit owned value**, never global state:
//!
//! ```ignore
//! use luma_dist::{Comm, ReduceOp, TensorDistExt};
//!
//! // backend-specific constructor, from `luma_dist::backends::nccl`:
//! let comms: Vec<Comm<Cuda>> = NcclBackend::from_devices(devices)?;
//! let sum = comms[rank].all_reduce(&x, ReduceOp::Sum)?;
//! // or, with TensorDistExt in scope:
//! let sum = x.all_reduce(&comms[rank], ReduceOp::Sum)?;
//! ```

pub mod backends;
pub mod base;
pub mod rendezvous;
pub mod scheme;
pub mod error;
#[cfg(feature = "nccl")]
pub(crate) mod utils;
pub use error::*;

pub use base::{Comm, DistBackend, DistKind, TensorDistExt, group};
pub use luma_tensor::ReduceOp;
pub use rendezvous::{Rendezvous, Tcp};
pub use scheme::{MultiThread, SingleThread};
