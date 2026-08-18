use crate::Layout;
use crate::device::cuda::kernel;
use crate::device::cuda::{Cuda, CudaError, CudaResult};
use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};

pub(crate) fn launch_pick<T: DeviceRepr>(
    device: &Cuda,
    val_suffix: &str,
    module: &kernel::Module,
    mask: &CudaSlice<u8>,
    mask_l: &Layout,
    t: &CudaSlice<T>,
    true_l: &Layout,
    f: &CudaSlice<T>,
    false_l: &Layout,
) -> CudaResult<CudaSlice<T>> {
    let kernel_name = format!("pick_{}", val_suffix);
    let dims = mask_l.dims();
    let elem_count = mask_l.shape().element_count();
    let num_dims = dims.len();
    let func = device.load_function(&kernel_name, module)?;

    let mut builder = func.builder();

    let dims = device.memcpy_stod(dims)?;
    let mask_strides = device.memcpy_stod(mask_l.stride())?;
    let t_strides = device.memcpy_stod(true_l.stride())?;
    let f_strides = device.memcpy_stod(false_l.stride())?;

    let mask_view = mask.slice(mask_l.start_offset()..);
    let t_view = t.slice(true_l.start_offset()..);
    let f_view = f.slice(false_l.start_offset()..);
    let output = device.alloc::<T>(elem_count)?;

    builder.arg(&elem_count);
    builder.arg(&num_dims);
    builder.arg(&dims);
    builder.arg(&mask_strides);
    builder.arg(&t_strides);
    builder.arg(&f_strides);
    builder.arg(&mask_view);
    builder.arg(&t_view);
    builder.arg(&f_view);
    builder.arg(&output);

    let config = LaunchConfig::for_num_elems(elem_count as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(output)
}

#[allow(unused_variables)]
pub(crate) fn launch_pick_true<T: DeviceRepr, I: DeviceRepr>(
    device: &Cuda,
    val_suffix: &str,
    module: &kernel::Module,
    mask: &CudaSlice<I>,
    mask_l: &Layout,
    val: T,
    f: &CudaSlice<T>,
    false_l: &Layout,
) -> CudaResult<CudaSlice<T>> {
    let kernel_name = format!("pick_true_{}", val_suffix);
    let dims = mask_l.dims();
    let elem_count = mask_l.shape().element_count();
    let num_dims = dims.len();
    let func = device.load_function(&kernel_name, module)?;

    let mut builder = func.builder();

    let dims = device.memcpy_stod(dims)?;
    let mask_strides = device.memcpy_stod(mask_l.stride())?;
    let f_strides = device.memcpy_stod(false_l.stride())?;

    let mask_view = mask.slice(mask_l.start_offset()..);
    let f_view = f.slice(false_l.start_offset()..);
    let output = device.alloc::<T>(elem_count)?;

    builder.arg(&elem_count);
    builder.arg(&num_dims);
    builder.arg(&dims);
    builder.arg(&mask_strides);
    builder.arg(&f_strides);
    builder.arg(&mask_view);
    builder.arg(&val);
    builder.arg(&f_view);
    builder.arg(&output);

    let config = LaunchConfig::for_num_elems(elem_count as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(output)
}

#[allow(unused_variables)]
pub(crate) fn launch_pick_false<T: DeviceRepr, I: DeviceRepr>(
    device: &Cuda,
    val_suffix: &str,
    module: &kernel::Module,
    mask: &CudaSlice<I>,
    mask_l: &Layout,
    t: &CudaSlice<T>,
    true_l: &Layout,
    val: T,
) -> CudaResult<CudaSlice<T>> {
    let kernel_name = format!("pick_false_{}", val_suffix);
    let dims = mask_l.dims();
    let elem_count = mask_l.shape().element_count();
    let num_dims = dims.len();
    let func = device.load_function(&kernel_name, module)?;

    let mut builder = func.builder();

    let dims = device.memcpy_stod(dims)?;
    let mask_strides = device.memcpy_stod(mask_l.stride())?;
    let t_strides = device.memcpy_stod(true_l.stride())?;

    let mask_view = mask.slice(mask_l.start_offset()..);
    let t_view = t.slice(true_l.start_offset()..);
    let output = device.alloc::<T>(elem_count)?;

    builder.arg(&elem_count);
    builder.arg(&num_dims);
    builder.arg(&dims);
    builder.arg(&mask_strides);
    builder.arg(&t_strides);
    builder.arg(&mask_view);
    builder.arg(&t_view);
    builder.arg(&val);
    builder.arg(&output);

    let config = LaunchConfig::for_num_elems(elem_count as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(output)
}
