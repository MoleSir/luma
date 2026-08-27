# luma-dataset

Datasets, batching, and data loaders (MNIST, Iris, …) with optional transforms, all returning `luma-tensor` tensors.

## Usage

```rust
use luma_dataset::vision::{MnistBatcher, MnistDataLoader, MnistDataset};
use luma_tensor::Cpu;

let device = Cpu;
let dataset = MnistDataset::train(Some("cache")).unwrap();
let loader = MnistDataLoader::new(dataset, MnistBatcher::new(device), 64, true);

for batch in loader.iter() {
    // batch.images: (batch, 28, 28), batch.targets: (batch, 1)
    let batch = batch.unwrap();
    println!("{} images", batch.images.dims()[0]);
    break;
}
```

## Custom dataset

```rust
use luma_dataset::{Dataset, NoBatcher, DataLoader};

// implement `Dataset` for your own item type, then:
let loader = DataLoader::from_dataset(my_dataset, 32, true);
```

## License

MIT — see [LICENSE](LICENSE).
