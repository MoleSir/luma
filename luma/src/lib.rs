//! **luma** — the unified facade for the luma ML framework.
//!
//! `luma-tensor` is always available and re-exported flat at the crate root
//! (`luma::Tensor`, `luma::Cpu`, `luma::Device`, …). Higher-level crates are
//! opt-in via features, each re-exported under its own namespace:
//!
//! | feature | enables | module(s) |
//! |---|---|---|
//! | `io` *(default)* | `luma-io` | `luma::io` |
//! | `nn` | `luma-nn`, `luma-optim`, `luma-dataset` | `luma::nn`, `luma::optim`, `luma::dataset` |
//! | `jit` | `luma-jit` (implies `nn`) | `luma::jit` |
//! | `cuda` | CUDA backend for tensor/nn/jit | — |
//! | `full` | `nn` + `jit` | — |
//!
//! ```toml
//! [dependencies]
//! luma = { path = "../luma", features = ["nn"] }
//! ```
//!
//! ```ignore
//! use luma::{Cpu, Device, Tensor};
//! use luma::nn::{Linear, Module};
//! ```

pub use luma_tensor::*;

#[cfg(feature = "nn")]
pub use luma_dataset as dataset;
#[cfg(feature = "io")]
pub use luma_io as io;
#[cfg(feature = "jit")]
pub use luma_jit as jit;
#[cfg(feature = "nn")]
pub use luma_nn as nn;
#[cfg(feature = "nn")]
pub use luma_optim as optim;
