use super::super::{Cuda, CudaError, CudaResult, kernel};
use crate::Layout;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

fn next_pow2(n: u32) -> u32 {
    let mut p: u32 = 1;
    while p < n {
        p <<= 1;
    }
    p
}

pub(crate) fn launch_softmax_f32(device: &Cuda, input: &CudaSlice<f32>, layout: &Layout, dim: usize) -> CudaResult<CudaSlice<f32>> {
    let elem_count = layout.shape().element_count();
    let dims = layout.dims();
    let row_size = dims[dim] as i32;
    let num_rows = (elem_count / row_size as usize) as i32;
    let block_dim = (row_size.min(1024)).max(1) as u32;
    let smem = next_pow2(block_dim) * 4;

    let func = device.load_function("softmax_f32", &kernel::NN)?;
    let output = device.alloc::<f32>(elem_count)?;

    let mut builder = func.builder();
    builder.arg(&num_rows);
    builder.arg(&row_size);
    builder.arg(input);
    builder.arg(&output);

    let config = LaunchConfig { grid_dim: (num_rows as u32, 1, 1), block_dim: (block_dim, 1, 1), shared_mem_bytes: smem };
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(output)
}

pub(crate) fn launch_softmax_f64(device: &Cuda, input: &CudaSlice<f64>, layout: &Layout, dim: usize) -> CudaResult<CudaSlice<f64>> {
    let elem_count = layout.shape().element_count();
    let dims = layout.dims();
    let row_size = dims[dim] as i32;
    let num_rows = (elem_count / row_size as usize) as i32;
    let block_dim = (row_size.min(1024)).max(1) as u32;
    let smem = next_pow2(block_dim) * 8;

    let func = device.load_function("softmax_f64", &kernel::NN)?;
    let output = device.alloc::<f64>(elem_count)?;

    let mut builder = func.builder();
    builder.arg(&num_rows);
    builder.arg(&row_size);
    builder.arg(input);
    builder.arg(&output);

    let config = LaunchConfig { grid_dim: (num_rows as u32, 1, 1), block_dim: (block_dim, 1, 1), shared_mem_bytes: smem };
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(output)
}

pub(crate) fn launch_rms_norm_f32(
    device: &Cuda,
    input: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    layout: &Layout,
    _weight_layout: &Layout,
    eps: f32,
) -> CudaResult<CudaSlice<f32>> {
    let elem_count = layout.shape().element_count();
    let dims = layout.dims();
    let last_dim = dims.len() - 1;
    let row_size = dims[last_dim] as i32;
    let num_rows = (elem_count / row_size as usize) as i32;
    let block_dim = (row_size.min(1024)).max(1) as u32;
    let smem = next_pow2(block_dim) * 4;

    let func = device.load_function("rms_norm_f32", &kernel::NN)?;
    let output = device.alloc::<f32>(elem_count)?;

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

pub(crate) fn launch_rms_norm_f64(
    device: &Cuda,
    input: &CudaSlice<f64>,
    weight: &CudaSlice<f64>,
    layout: &Layout,
    _weight_layout: &Layout,
    eps: f64,
) -> CudaResult<CudaSlice<f64>> {
    let elem_count = layout.shape().element_count();
    let dims = layout.dims();
    let last_dim = dims.len() - 1;
    let row_size = dims[last_dim] as i32;
    let num_rows = (elem_count / row_size as usize) as i32;
    let block_dim = (row_size.min(1024)).max(1) as u32;
    let smem = next_pow2(block_dim) * 8;

    let func = device.load_function("rms_norm_f64", &kernel::NN)?;
    let output = device.alloc::<f64>(elem_count)?;

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
