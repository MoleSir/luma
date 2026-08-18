use crate::Layout;
use crate::builder_arg;
use crate::device::cuda::kernel;
use crate::device::cuda::{Cuda, CudaError, CudaResult};
use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};

fn cast_kernel_name(src: &str, dst: &str) -> String {
    format!("ucast_{}_to_{}", src, dst)
}

pub(crate) fn launch_cast<T: DeviceRepr, U: DeviceRepr>(
    device: &Cuda,
    src_type: &str,
    dst_type: &str,
    module: &kernel::Module,
    input: &CudaSlice<T>,
    layout: &Layout,
) -> CudaResult<CudaSlice<U>> {
    let kernel_name = cast_kernel_name(src_type, dst_type);
    let dims = layout.dims();
    let elem_count = layout.shape().element_count();
    let num_dims = dims.len();
    let func = device.load_function(&kernel_name, module)?;

    let mut builder = func.builder();
    let dims_dev = device.memcpy_stod(dims)?;
    let strides_dev = device.memcpy_stod(layout.stride())?;
    let output = device.alloc::<U>(elem_count)?;
    let input_view = input.slice(layout.start_offset()..);
    builder_arg!(builder, elem_count);
    builder_arg!(builder, num_dims);
    builder.arg(&dims_dev);
    builder.arg(&strides_dev);
    builder.arg(&input_view);
    builder.arg(&output);

    let config = LaunchConfig::for_num_elems(elem_count as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(output)
}
