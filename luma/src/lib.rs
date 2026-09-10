//! **luma** — the unified facade for the luma ML framework.
//!
//! `luma-tensor` is always available and re-exported flat at the crate root
//! (`luma::Tensor`, `luma::Device`, …). The backends are opt-in: `Cpu`
//! *(default)* and `Cuda` are re-exported flat as well (`luma::Cpu`,
//! `luma::Cuda`). Higher-level crates are opt-in via features, each
//! re-exported under its own namespace:
//!
//! | feature | enables | module(s) |
//! |---|---|---|
//! | `cpu` *(default)* | `luma-cpu` backend | `luma::Cpu`, … |
//! | `io` *(default)* | `luma-io` | `luma::io` |
//! | `cuda` | `luma-cuda` backend (+ nn/compile cuda) | `luma::Cuda`, … |
//! | `nn` | `luma-nn`, `luma-optim`, `luma-dataset` | `luma::nn`, `luma::optim`, `luma::dataset` |
//! | `compile` | `luma-compile` (implies `nn`) | `luma::compile` |
//! | `full` | `nn` + `compile` | — |
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
#[cfg(feature = "cpu")]
pub use luma_cpu::*;
#[cfg(feature = "cuda")]
pub use luma_cuda::*;

#[cfg(feature = "compile")]
pub use luma_compile as compile;
#[cfg(feature = "nn")]
pub use luma_dataset as dataset;
#[cfg(feature = "io")]
pub use luma_io as io;
#[cfg(feature = "nn")]
pub use luma_nn as nn;
#[cfg(feature = "nn")]
pub use luma_optim as optim;
