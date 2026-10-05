use super::super::{Cuda, CudaError, CudaResult, kernel};
use luma_tensor::Layout;
use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};

fn next_pow2(n: u32) -> u32 {
    let mut p: u32 = 1;
    while p < n {
        p <<= 1;
    }
    p
}

pub(crate) fn launch_softmax<T: DeviceRepr>(
    device: &Cuda,
    input: &CudaSlice<T>,
    layout: &Layout,
    dim: usize,
    kernel_name: &str,
) -> CudaResult<CudaSlice<T>> {
    let elem_count = layout.shape().element_count();
    let dims = layout.dims();
    let row_size = dims[dim] as i32;
    let num_rows = (elem_count / row_size as usize) as i32;
    let block_dim = (row_size.min(1024)).max(1) as u32;
    let smem = next_pow2(block_dim) * std::mem::size_of::<T>() as u32;

    let func = device.load_function(kernel_name, &kernel::NN)?;
    let output = device.alloc::<T>(elem_count)?;

    let mut builder = func.builder();
    builder.arg(&num_rows);
    builder.arg(&row_size);
    builder.arg(input);
    builder.arg(&output);

    let config = LaunchConfig { grid_dim: (num_rows as u32, 1, 1), block_dim: (block_dim, 1, 1), shared_mem_bytes: smem };
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(output)
}

pub(crate) fn launch_rms_norm<T: DeviceRepr>(
    device: &Cuda,
    input: &CudaSlice<T>,
    weight: &CudaSlice<T>,
    layout: &Layout,
    _weight_layout: &Layout,
    eps: T,
    kernel_name: &str,
) -> CudaResult<CudaSlice<T>> {
    let elem_count = layout.shape().element_count();
    let dims = layout.dims();
    let last_dim = dims.len() - 1;
    let row_size = dims[last_dim] as i32;
    let num_rows = (elem_count / row_size as usize) as i32;
    let block_dim = (row_size.min(1024)).max(1) as u32;
    let smem = next_pow2(block_dim) * std::mem::size_of::<T>() as u32;

    let func = device.load_function(kernel_name, &kernel::NN)?;
    let output = device.alloc::<T>(elem_count)?;

    let mut builder = func.builder();
    builder.arg(&num_rows);
    builder.arg(&row_size);
    builder.arg(input);
    builder.arg(weight);
    builder.arg(&eps);
    builder.arg(&output);

    let config = LaunchConfig { grid_dim: (num_rows as u32, 1, 1), block_dim: (block_dim, 1, 1), shared_mem_bytes: smem };
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(output)
}
