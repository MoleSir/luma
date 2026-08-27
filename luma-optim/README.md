# luma-optim

Optimizers (`SGD`, `SGDM`, `Momentum`, `RMSProp`, `Adam`, `AdamW`) for updating gradients on any `luma-tensor` device.

## Usage

```rust
use luma_nn::{Linear, Module};
use luma_optim::{Optimizer, SGD};
use luma_tensor::Cpu;

let model = Linear::new(3, 4, true, Cpu).unwrap();
let mut optimizer = SGD::new(model.params(), 0.01);

// after a forward pass and `let grads = loss.backward()?`:
// optimizer.step(&grads)?;
```

## Adam

```rust
use luma_optim::{Adam, AdamConfig};

let optimizer = Adam::new(params, AdamConfig::default()).unwrap();
```

## License

MIT — see [LICENSE](LICENSE).
