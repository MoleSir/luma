use anyhow::Context;
use luma_dataset::vision::{MnistBatcher, MnistDataLoader, MnistDataset};
use luma_nn::{Linear, Module};
use luma_optim::{Optimizer, SGD};
use luma_tensor::{dtype::FloatDType, no_grad, Cpu, Device, Tensor};
use luma_nn::functional as F;

fn main() {
    let device = Cpu;
    if let Err(e) = result_main(&device) {
        eprintln!("{:?}", e);
    }
}

#[derive(Module)]
pub struct Net<D: Device> {
    pub fc1: Linear<D>,
    pub fc2: Linear<D>,
    pub fc3: Linear<D>,
}

impl<D: Device> Net<D> {
    pub fn new(dtype: FloatDType, device: &D) -> anyhow::Result<Self> {
        let fc1 = Linear::new(784, 512, true, (device, dtype)).context("init fc1")?;
        let fc2 = Linear::new(512, 256, true, (device, dtype)).context("init fc1")?;
        let fc3 = Linear::new(256, 10, true, (device, dtype)).context("init fc1")?;
        Ok(Self { fc1, fc2, fc3 })
    }

    pub fn forward(&self, images: &Tensor<D>) -> anyhow::Result<Tensor<D>> {
        let (batch, height, width) = images.dims3()?;
        let x = images.reshape((batch, height * width)).context("reshape input")?;

        // (batch, 784) => (batch, 512)
        let out = self.fc1.forward(&x).context("fc1 forward")?;
        let out = F::relu(&out).context("relu")?;

        // (batch, 512) => (batch, 256)
        let out = self.fc2.forward(&out).context("fc2 forward")?;
        let out = F::relu(&out).context("relu")?;

        // (batch, 256) => (batch, 10)
        let out = self.fc3.forward(&out).context("fc3 forward")?;

        Ok(out)
    }
}

fn result_main<D: Device>(device: &D) -> anyhow::Result<()> {
    const BATCH_SIZE: usize = 64;
    const LEARNING_RATE: f64 = 0.01;
    const EPOCHS: usize = 1;

    // load dataset 
    let batcher = MnistBatcher::new(device.clone());
    let train_dataset = MnistDataset::train(Some("../cache")).context("download train dataset")?;
    let train_loader = MnistDataLoader::new(train_dataset, batcher, BATCH_SIZE, true);

    let batcher = MnistBatcher::new(device.clone());
    let test_dataset = MnistDataset::test(Some("../cache")).context("download test dataset")?;
    let test_loader = MnistDataLoader::new(test_dataset, batcher, 1000, true);
    
    let model = Net::<D>::new(FloatDType::F32, device).context("create model")?;
    let mut optimizer = SGD::new(model.params(), LEARNING_RATE);

    // train model
    for epoch in 0..EPOCHS {
        train(&model, &train_loader, &mut optimizer, epoch).with_context(|| format!("epoch {epoch} train"))?;
        test(&model, &test_loader).with_context(|| format!("epoch {epoch} test"))?;
    }

    // save model
    model.save_safetensors("../cache/mnist.safetensors").context("save model")?;

    // load from path
    // let model = Net::from_safetensors(&(), "../cache/mnist.safetensors").context("load model")?;
    test(&model, &test_loader).context("test model")?;
    
    Ok(())
}

pub fn train<O, D: Device>(
    model: &Net<D>,
    train_loader: &MnistDataLoader<D>,
    optimizer: &mut O,
    epoch: usize,
) -> anyhow::Result<()> 
where 
    O: Optimizer<Device = D>,
{
    for (batch_idx, batch) in train_loader.iter().enumerate() {
        let batch = batch.with_context(|| format!("parse {batch_idx} batch"))?; 
        let data = batch.images; // (batch, 28, 28)
        let target = batch.targets; // (batch, 1)

        // (batch, 28, 28) => (batch, 10)
        let output = model.forward(&data).context("model forward")?;
        let loss = F::cross_entropy_loss(&output, &target).context("cross entropy loss")?;

        let grads = loss.backward().context("backward")?;
        optimizer.step(&grads)?;

        if batch_idx % 100 == 0 {

            println!(
                "Train Epoch: {} [{}/{} ({:.2}%)]\tLoss: {}",
                epoch, 
                batch_idx * train_loader.batch_size(), 
                train_loader.dataset_len(),
                100.0 * batch_idx as f64 / train_loader.batch_count() as f64, 
                loss.to_scalar()?
            );
        }
    }

    Ok(())
}

pub fn test<D: Device>(model: &Net<D>, test_loader: &MnistDataLoader<D>) -> anyhow::Result<()> {
    let mut test_loss = 0.0;
    let mut correct = 0;
    
    no_grad!();
    for batch in test_loader.iter() {
        let batch = batch.context("parse batch")?;
        let data = batch.images; // (batch, 28, 28)
        let target = batch.targets; // (batch, 1)

        // (batch, 10)
        let output = model.forward(&data).context("model forward")?;
        
        // (batch, 10) and (batch, 1)
        test_loss += F::cross_entropy_loss(&output, &target)
            .context("cal loss")?
            .to_scalar()?;
        
        correct += output
            .argmax_keepdim(1).context("argmax")?
            .eq(&target).context("compare pred and target")?
            .true_count()?;
    }

    let test_loss = test_loss / test_loader.batch_count() as f64;
    let accuracy = 100.0 * correct as f64 / test_loader.dataset_len() as f64;
    
    println!(
        "\n Test set: Average loss: {}, Accuracy: {}/{} ({})", 
        test_loss, correct, test_loader.dataset_len(), accuracy
    );

    Ok(())
}
