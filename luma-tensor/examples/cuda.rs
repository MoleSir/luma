#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("this example requires: cargo run --example cuda --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    use luma_tensor::{Float, Tensor, device::cuda::Cuda};
    let device = Cuda::new(4)?;
    let a = Tensor::<Cuda, Float>::randn(0.0, 1.0, (4, 3), &device)?;
    let b = Tensor::<Cuda, Float>::randn(0.0, 1.0, (4, 3), &device)?;
    a.add(&b)?;
    let c = a.sub(&b)?;
    println!("{}", c);
    Ok(())
}
