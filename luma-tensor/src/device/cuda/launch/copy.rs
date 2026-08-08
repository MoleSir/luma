use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};
use crate::builder_arg;
use crate::Layout;
use crate::device::cuda::{Cuda, CudaError, CudaResult};
use crate::device::cuda::kernel;

pub(crate) fn launch_copy_offset<T: DeviceRepr>(
    device: &Cuda,
    kernel_name: &str,
    module: &kernel::Module,
    src: &CudaSlice<T>,
    layout: &Layout,
    dst: &CudaSlice<T>,
    dst_offset: usize,
) -> CudaResult<()> {
    let dims = layout.dims();
    let elem_count = layout.shape().element_count();
    let num_dims = dims.len();
    let func = device.load_function(kernel_name, module)?;

    let mut builder = func.builder();
    let dims = device.memcpy_stod(dims)?;
    let strides = device.memcpy_stod(layout.stride())?;
    let src_view = src.slice(layout.start_offset()..);
    builder_arg!(builder, elem_count);
    builder_arg!(builder, num_dims);
    builder.arg(&dims);
    builder.arg(&strides);
    builder.arg(&dst_offset);
    builder.arg(&src_view);
    builder.arg(dst);

    let config = LaunchConfig::for_num_elems(elem_count as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(())
}

pub(crate) fn launch_copy2d<T: DeviceRepr>(
    device: &Cuda,
    kernel_name: &str,
    module: &kernel::Module,
    d1: usize,
    d2: usize,
    src_s: usize,
    dst_s: usize,
    src: &CudaSlice<T>,
    src_offset: usize,
    dst: &CudaSlice<T>,
) -> CudaResult<()> {
    let func = device.load_function(kernel_name, module)?;
    let total = d1 * d2;

    let mut builder = func.builder();
    let src_view = src.slice(src_offset..);
    builder_arg!(builder, d1);
    builder_arg!(builder, d2);
    builder_arg!(builder, src_s);
    builder_arg!(builder, dst_s);
    builder.arg(&src_view);
    builder.arg(dst);

    let config = LaunchConfig::for_num_elems(total as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(())
}
