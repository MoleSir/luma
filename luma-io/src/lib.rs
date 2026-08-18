//! **luma-io** — model serialization for the luma framework.
//!
//! Currently supports the [safetensors](https://huggingface.co/docs/safetensors/) format.
//!
//! # Example
//!
//! ```no_run
//! use std::collections::HashMap;
//! use luma_io::safetensors;
//! use luma_tensor::{Cpu, DynTensor};
//!
//! let device = Cpu::default();
//! let content = safetensors::load_file("model.safetensors", &device).unwrap();
//! for (name, tensor) in &content.tensors {
//!     println!("{}: dtype={:?}, shape={:?}", name, tensor.dtype(), tensor.dims());
//! }
//! ```

pub mod lpk;
pub mod safetensors;
