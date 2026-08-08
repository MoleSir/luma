use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};
use crate::builder_arg;
use crate::Layout;
use crate::device::cuda::{Cuda, CudaError, CudaResult};
use crate::device::cuda::kernel;

pub(crate) fn indexing_kernel_name(op: &str, idx_suffix: &str, val_suffix: &str) -> String {
    format!("{}_{}_{}", op, idx_suffix, val_suffix)
}

pub(crate) fn launch_index_select<T: DeviceRepr, I: DeviceRepr>(
    device: &Cuda,
    idx_suffix: &str, val_suffix: &str,
    module: &kernel::Module,
    src: &CudaSlice<T>,
    src_l: &Layout,
    ids: &CudaSlice<I>,
    ids_l: &Layout,
    dim: usize,
) -> CudaResult<CudaSlice<T>> {
    let kernel_name = indexing_kernel_name("is", idx_suffix, val_suffix);
    let src_dims = src_l.dims();
    let num_dims = src_dims.len();

    let left_size: usize = src_dims[..dim].iter().product();
    let right_size: usize = src_dims[dim + 1..].iter().product();
    let src_dim_size = src_dims[dim];
    let ids_dim_size = ids_l.shape().element_count();
    let numel = ids_dim_size * left_size * right_size;

    let func = device.load_function(&kernel_name, module)?;

    let info: Vec<usize> = [src_dims, src_l.stride()].concat();
    let info_dev = device.memcpy_stod(&info)?;
    let out = device.alloc::<T>(numel)?;

    let mut builder = func.builder();
    builder_arg!(builder, numel);
    builder_arg!(builder, num_dims);
    builder.arg(&info_dev);
    builder.arg(ids);
    builder.arg(src);
    builder.arg(&out);
    builder_arg!(builder, left_size, src_dim_size, ids_dim_size, right_size);

    let config = LaunchConfig::for_num_elems(numel as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(out)
}

pub(crate) fn launch_gather<T: DeviceRepr, I: DeviceRepr>(
    device: &Cuda,
    idx_suffix: &str, val_suffix: &str,
    module: &kernel::Module,
    src: &CudaSlice<T>,
    src_l: &Layout,
    ids: &CudaSlice<I>,
    ids_l: &Layout,
    dim: usize,
) -> CudaResult<CudaSlice<T>> {
    let kernel_name = indexing_kernel_name("gather", idx_suffix, val_suffix);
    let src_dims = src_l.dims();

    let left_size: usize = src_dims[..dim].iter().product();
    let right_size: usize = src_dims[dim + 1..].iter().product();
    let src_dim_size = src_dims[dim];
    let ids_dim_size = ids_l.dims()[dim];
    let numel = ids_l.shape().element_count();

    let func = device.load_function(&kernel_name, module)?;
    let out = device.alloc::<T>(numel)?;

    let mut builder = func.builder();
    builder_arg!(builder, numel);
    builder.arg(ids);
    builder.arg(src);
    builder.arg(&out);
    builder_arg!(builder, left_size, src_dim_size, ids_dim_size, right_size);

    let config = LaunchConfig::for_num_elems(numel as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(out)
}

pub(crate) fn launch_index_add<T: DeviceRepr, I: DeviceRepr>(
    device: &Cuda,
    idx_suffix: &str, val_suffix: &str,
    module: &kernel::Module,
    init: &CudaSlice<T>,
    init_l: &Layout,
    ids: &CudaSlice<I>,
    ids_l: &Layout,
    src: &CudaSlice<T>,
    src_l: &Layout,
    dim: usize,
) -> CudaResult<CudaSlice<T>> {
    let kernel_name = indexing_kernel_name("ia", idx_suffix, val_suffix);
    let dst_dims = init_l.dims();
    let src_dims = src_l.dims();

    let left_size: usize = src_dims[..dim].iter().product();
    let right_size: usize = src_dims[dim + 1..].iter().product();
    let src_dim_size = src_dims[dim];
    let dst_dim_size = dst_dims[dim];
    let ids_dim_size = ids_l.dims()[0];
    let numel = left_size * right_size;

    let func = device.load_function(&kernel_name, module)?;

    let init_data = device.memcpy_dtov(init)?;
    let out = device.memcpy_stod(&init_data)?;

    let mut builder = func.builder();
    builder.arg(ids);
    builder_arg!(builder, ids_dim_size);
    builder.arg(src);
    builder.arg(&out);
    builder_arg!(builder, left_size, src_dim_size, dst_dim_size, right_size);

    let config = LaunchConfig::for_num_elems(numel as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(out)
}

pub(crate) fn launch_scatter_add<T: DeviceRepr, I: DeviceRepr>(
    device: &Cuda,
    idx_suffix: &str, val_suffix: &str,
    module: &kernel::Module,
    init: &CudaSlice<T>,
    init_l: &Layout,
    ids: &CudaSlice<I>,
    _ids_l: &Layout,
    src: &CudaSlice<T>,
    src_l: &Layout,
    dim: usize,
) -> CudaResult<CudaSlice<T>> {
    let kernel_name = indexing_kernel_name("sa", idx_suffix, val_suffix);
    let dst_dims = init_l.dims();
    let src_dims = src_l.dims();

    let left_size: usize = src_dims[..dim].iter().product();
    let right_size: usize = src_dims[dim + 1..].iter().product();
    let src_dim_size = src_dims[dim];
    let dst_dim_size = dst_dims[dim];
    let numel = left_size * right_size;

    let func = device.load_function(&kernel_name, module)?;

    let init_data = device.memcpy_dtov(init)?;
    let out = device.memcpy_stod(&init_data)?;

    let mut builder = func.builder();
    builder.arg(ids);
    builder.arg(src);
    builder.arg(&out);
    builder_arg!(builder, left_size, src_dim_size, dst_dim_size, right_size);

    let config = LaunchConfig::for_num_elems(numel as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(out)
}
