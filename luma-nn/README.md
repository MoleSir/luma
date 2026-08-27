# luma-nn

Neural-network modules (`Linear`, `BatchNorm1d`, `LayerNorm`, `RMSNorm`, `Dropout`, activations) and loss functions built on `luma-tensor`.

## Usage

```rust
use luma_nn::Linear;
use luma_tensor::dtype::FloatDType;
use luma_tensor::{Cpu, Tensor};

let linear = Linear::new(3, 4, true, Cpu).unwrap();
let x = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0], (1, 3), FloatDType::F32).unwrap();
let out = linear.forward(&x).unwrap();
println!("{}", out);
```

## Custom modules with `#[derive(Module)]`

```rust
use luma_macros::Module;
use luma_nn::{Linear, Parameter};
use luma_tensor::{Cpu, Device};

#[derive(Module)]
struct Net<D: Device> {
    fc: Linear<D>,
}
```

## Losses

```rust
use luma_nn::functional::cross_entropy_loss;
use luma_tensor::dtype::{FloatDType, IntDType};
use luma_tensor::{Cpu, Float, Int, Tensor};

let pred = Tensor::<Cpu, Float>::from_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), FloatDType::F32).unwrap();
let target = Tensor::<Cpu, Int>::from_slice(&[0, 1], (2,), IntDType::I32).unwrap();
let loss = cross_entropy_loss(&pred, &target).unwrap();
println!("{}", loss);
```

## License

MIT — see [LICENSE](LICENSE).
