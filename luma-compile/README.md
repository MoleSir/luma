# luma-compile

Trace, compile, and execute `luma-nn` modules as portable, device/kind-erased graphs (the Rust analogue of `torch.jit.trace`).

## Usage

```rust
use luma_compile::trace;
use luma_nn::Linear;
use luma_tensor::dtype::FloatDType;
use luma_tensor::{Cpu, Tensor};

let linear = Linear::new(3, 4, true, Cpu).unwrap();
let x = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), FloatDType::F32).unwrap();

// capture the forward pass as a graph, then compile and run it
let graph = trace(&linear, &x).unwrap();
let mut exec = graph.lock().unwrap().compile(&Cpu).unwrap();
let out = exec.run(&[x.clone().into()]).unwrap();
println!("{}", out[0].as_float().unwrap());
```

## Low-level tracing

```rust
use luma_compile::{Trace, Traced};
use luma_tensor::dtype::FloatDType;
use luma_tensor::{Float, Tensor};

let trace = Trace::new();
let a = Tensor::<Trace, Float>::full(&[2, 3], 1.0, (&trace, FloatDType::F32)).unwrap();
let b = a.relu().unwrap();
println!("{}", trace.graph().lock().unwrap());
```

## License

MIT — see [LICENSE](LICENSE).
