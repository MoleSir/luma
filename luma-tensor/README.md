# luma-tensor

A tensor computation library with compile-time kind separation (`Float`/`Int`/`Bool`), runtime precision, a `Cpu`/`Cuda` device abstraction, and tape-based autograd.

## Usage

```rust
use luma_tensor::dtype::FloatDType;
use luma_tensor::{Cpu, Tensor};

// construct a 2×3 tensor and do a matmul
let x = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), FloatDType::F32).unwrap();
let xt = x.transpose(0usize, 1usize).unwrap();
let y = x.matmul(&xt).unwrap();
println!("{}", y);
```

## Autograd

```rust
use luma_tensor::dtype::FloatDType;
use luma_tensor::{Cpu, Tensor};

let x = Tensor::<Cpu>::from_slice(&[2.0, 3.0], (2,), FloatDType::F32).unwrap();
x.set_requires_grad(true);

let y = x.mul(&x).unwrap().sum_all().unwrap();
let grads = y.backward().unwrap();
let gx = grads.get_by_id(x.id()).unwrap();
assert_eq!(gx.to_vec().unwrap(), vec![4.0, 6.0]);
```

## Cuda

```rust
use luma_tensor::dtype::FloatDType;
use luma_tensor::{Cuda, Tensor};

let dev = Cuda::new(0).unwrap();
let x = Tensor::<Cuda>::from_slice(&[1.0, 2.0], (2,), (dev, FloatDType::F32)).unwrap();
```

## License

MIT — see [LICENSE](LICENSE).
