# luma-macros

Procedural macros for luma, providing the `#[derive(Module)]` macro that walks struct/enum fields to expose parameters, buffers, and sub-modules.

## Usage

```rust
use luma_nn::{Linear, Module, Parameter};
use luma_tensor::Device;

#[derive(Module)]
struct MyNet<D: Device> {
    fc: Linear<D>,
    scale: Parameter<D>,
}
```

## Attributes

- `#[module(skip)]` — skip a field or variant.
- `#[module(display = "fn_name")]` — delegate `extra_display()`.
- `#[module(train = "fn_name")]` — delegate `set_train()`.
- `#[module(reset = "fn_name")]` — delegate `reset_parameters()`.

## License

MIT — see [LICENSE](LICENSE).
