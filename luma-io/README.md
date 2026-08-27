# luma-io

Model serialization for luma, supporting the [safetensors](https://huggingface.co/docs/safetensors/) format and the `lpk` pack format.

## Usage

```rust
use luma_io::safetensors;
use luma_tensor::Cpu;

let device = Cpu::default();
let content = safetensors::load_file("model.safetensors", &device).unwrap();
for (name, tensor) in &content.tensors {
    println!("{name}: {:?} {:?}", tensor.dtype(), tensor.dims());
}
```

## License

MIT — see [LICENSE](LICENSE).
