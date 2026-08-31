# luma

Unified facade for the luma ML framework: `luma-tensor` is re-exported flat, and higher-level crates are opt-in via features.

## Features

| feature | enables | module(s) |
|---|---|---|
| `io` *(default)* | `luma-io` | `luma::io` |
| `nn` | `luma-nn`, `luma-optim`, `luma-dataset` | `luma::nn`, `luma::optim`, `luma::dataset` |
| `compile` | `luma-compile` (implies `nn`) | `luma::compile` |
| `cuda` | CUDA backend for tensor/nn/compile | — |
| `full` | `nn` + `compile` | — |

## Usage

```toml
[dependencies]
luma = { path = "../luma", features = ["nn"] }
```

```rust
use luma::dtype::FloatDType;
use luma::nn::Linear;
use luma::{Cpu, Tensor};

let linear = Linear::new(3, 4, true, Cpu).unwrap();
let x = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0], (1, 3), FloatDType::F32).unwrap();
let out = linear.forward(&x).unwrap();
println!("{}", out);
```

## Cuda

```toml
luma = { path = "../luma", features = ["nn", "cuda"] }
```

```rust
use luma::{Cuda, Device};

let dev = Cuda::new(0).unwrap();
println!("{}", dev.name());
```

## License

MIT — see [LICENSE](LICENSE).
