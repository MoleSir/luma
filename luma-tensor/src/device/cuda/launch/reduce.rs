use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};
use crate::builder_arg;
use crate::{Layout, ReduceOp, Shape};
use super::super::{Cuda, CudaError, CudaResult, kernel};

pub(crate) fn reduce_kernel_name(op: ReduceOp, suffix: &str) -> String {
    let op_str = match op {
        ReduceOp::Sum => "sum",
        ReduceOp::Mean => "sum",
        ReduceOp::Min => "min",
        ReduceOp::Max => "max",
        ReduceOp::Prod => "prod",
    };
    format!("s{}_{}", op_str, suffix)
}

fn compute_contiguous_strides(dims: &[usize]) -> Vec<usize> {
    let n = dims.len();
    let mut strides = vec![1usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1].max(1);
    }
    strides
}

pub(crate) fn launch_reduce<T: DeviceRepr>(
    device: &Cuda,
    kernel_name: &str,
    module: &kernel::Module,
    src: &CudaSlice<T>,
    dims: &[usize],
    strides: &[usize],
    reduce_dim: usize,
    reduce_size: usize,
    output_block_count: usize,
) -> CudaResult<CudaSlice<T>> {
    let num_dims = dims.len();
    let func = device.load_function(kernel_name, module)?;
    let reduce_stride = strides[reduce_dim];

    let dims = device.memcpy_stod(dims)?;
    let strides = device.memcpy_stod(strides)?;
    let output = device.alloc::<T>(output_block_count)?;

    let mut builder = func.builder();
    builder_arg!(builder, reduce_size);
    builder_arg!(builder, reduce_stride);
    builder_arg!(builder, num_dims);
    builder.arg(&dims);
    builder.arg(&strides);
    builder_arg!(builder, reduce_dim);
    builder.arg(src);
    builder.arg(&output);

    let config = LaunchConfig {
        grid_dim: (output_block_count as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(output)
}

pub(crate) fn launch_multi_reduce_by_kernel_name<T: DeviceRepr>(
    device: &Cuda,
    kernel_name: &str,
    module: &kernel::Module,
    src: &CudaSlice<T>,
    layout: &Layout,
    dims: &[usize],
    keepdim: bool,
) -> CudaResult<(CudaSlice<T>, Shape)> {
    let mut sorted: Vec<usize> = dims.to_vec();
    sorted.sort_unstable();
    sorted.dedup();

    if sorted.is_empty() {
        return Ok((device.memcpy_stod(&device.memcpy_dtov(src)?)?, layout.shape().clone()));
    }

    let mut cur_dims: Vec<usize> = layout.dims().to_vec();
    let mut cur_strides: Vec<usize> = layout.stride().to_vec();
    let mut saved: Vec<CudaSlice<T>> = Vec::new();

    for &dim in sorted.iter().rev() {
        let reduce_size = cur_dims[dim];
        let output_block_count: usize = cur_dims.iter().product::<usize>() / reduce_size;

        let new_out = launch_reduce(
            device, kernel_name, module,
            saved.last().unwrap_or(src),
            &cur_dims, &cur_strides,
            dim, reduce_size, output_block_count,
        )?;
        saved.push(new_out);
        cur_dims.remove(dim);
        cur_strides = compute_contiguous_strides(&cur_dims);
    }

    let result_slice = saved.pop().unwrap();
    let mut shape = cur_dims.clone();

    if keepdim {
        let mut full_shape = layout.dims().to_vec();
        for &d in &sorted {
            full_shape[d] = 1;
        }
        shape = full_shape;
    }

    Ok((result_slice, Shape::from(shape)))
}

pub(crate) fn launch_multi_reduce<T: DeviceRepr>(
    device: &Cuda,
    op: ReduceOp,
    type_name: &str,
    module: &kernel::Module,
    src: &CudaSlice<T>,
    layout: &Layout,
    dims: &[usize],
    keepdim: bool,
) -> CudaResult<(CudaSlice<T>, Shape)> {
    let kernel_name = reduce_kernel_name(op, type_name);
    launch_multi_reduce_by_kernel_name(device, &kernel_name, module, src, layout, dims, keepdim)
}

pub(crate) fn arg_reduce_kernel_name(take_max: bool, suffix: &str) -> String {
    let op = if take_max { "argmax" } else { "argmin" };
    format!("s{}_{}", op, suffix)
}

pub(crate) fn launch_arg_reduce<T: DeviceRepr>(
    device: &Cuda,
    kernel_name: &str,
    module: &kernel::Module,
    src: &CudaSlice<T>,
    dims: &[usize],
    strides: &[usize],
    reduce_dim: usize,
    reduce_size: usize,
    output_block_count: usize,
) -> CudaResult<CudaSlice<i32>> {
    let num_dims = dims.len();
    let func = device.load_function(kernel_name, module)?;
    let reduce_stride = strides[reduce_dim];

    let dims_buf = device.memcpy_stod(dims)?;
    let strides_buf = device.memcpy_stod(strides)?;
    let output = device.alloc::<i32>(output_block_count)?;

    let mut builder = func.builder();
    builder_arg!(builder, reduce_size);
    builder_arg!(builder, reduce_stride);
    builder_arg!(builder, num_dims);
    builder.arg(&dims_buf);
    builder.arg(&strides_buf);
    builder_arg!(builder, reduce_dim);
    builder.arg(src);
    builder.arg(&output);

    let config = LaunchConfig {
        grid_dim: (output_block_count as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(output)
}
