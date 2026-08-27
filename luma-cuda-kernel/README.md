# luma-cuda-kernel

Precompiled PTX CUDA kernels that power `luma-tensor`'s GPU backend (binary, unary, reduce, cast, indexing, and nn ops).

## Usage

```rust
use luma_cuda_kernel::{BINARY, Module};

assert_eq!(BINARY.name(), "binary");
println!("{} bytes of PTX", BINARY.ptx().len());
```

Kernels are embedded at compile time via `bindgen_cuda`; enable the `cuda` feature on `luma-tensor` (or the `luma` facade) to use them.

## License

MIT — see [LICENSE](LICENSE).
