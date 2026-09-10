use cudarc::driver::{CudaContext, CudaSlice, DriverError};

fn main() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let _: CudaSlice<f32> = unsafe { stream.alloc::<f32>(10)? };
    let _: CudaSlice<f64> = unsafe { stream.alloc::<f64>(10)? };
    let _: CudaSlice<f64> = stream.alloc_zeros::<f64>(10)?;

    let _: CudaSlice<usize> = stream.clone_htod(&[0; 10])?;
    let _: CudaSlice<u32> = stream.clone_htod(&[1, 2, 3])?;

    Ok(())
}