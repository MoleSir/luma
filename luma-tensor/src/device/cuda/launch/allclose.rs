use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};
use crate::Layout;
use super::super::{Cuda, CudaError, CudaResult, kernel};

pub(crate) fn launch_allclose_float<T: DeviceRepr>(
    device: &Cuda,
    val_suffix: &str,
    a: &CudaSlice<T>, 
    a_l: &Layout,
    b: &CudaSlice<T>, 
    b_l: &Layout,
    rtol: T, 
    atol: T,
) -> CudaResult<bool> {
    let dims = a_l.dims();
    let elem_count = a_l.shape().element_count();
    let num_dims = dims.len();
    let func = device.load_function(&format!("allclose_{}", val_suffix), &kernel::ALLCLOSE)?;

    let mut builder = func.builder();
    let dims_dev = device.memcpy_stod(dims)?;
    let a_strides_dev = device.memcpy_stod(a_l.stride())?;
    let b_strides_dev = device.memcpy_stod(b_l.stride())?;
    let result = device.alloc_zeros::<i32>(1)?;
    let a_view = a.slice(a_l.start_offset()..);
    let b_view = b.slice(b_l.start_offset()..);

    builder.arg(&elem_count);
    builder.arg(&num_dims);
    builder.arg(&dims_dev);
    builder.arg(&a_strides_dev);
    builder.arg(&b_strides_dev);
    builder.arg(&a_view);
    builder.arg(&b_view);
    builder.arg(&rtol);
    builder.arg(&atol);
    builder.arg(&result);

    let config = LaunchConfig::for_num_elems(elem_count as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    let val = device.memcpy_dtov(&result)?;
    Ok(val[0] == 0)
}

pub(crate) fn launch_allclose_int<T: DeviceRepr>(
    device: &Cuda,
    val_suffix: &str,
    a: &CudaSlice<T>, 
    a_l: &Layout,
    b: &CudaSlice<T>, 
    b_l: &Layout,
) -> CudaResult<bool> {
    let dims = a_l.dims();
    let elem_count = a_l.shape().element_count();
    let num_dims = dims.len();
    let func = device.load_function(&format!("allclose_{}", val_suffix), &kernel::ALLCLOSE)?;

    let mut builder = func.builder();
    let dims_dev = device.memcpy_stod(dims)?;
    let a_strides_dev = device.memcpy_stod(a_l.stride())?;
    let b_strides_dev = device.memcpy_stod(b_l.stride())?;
    let result = device.alloc_zeros::<i32>(1)?;
    let a_view = a.slice(a_l.start_offset()..);
    let b_view = b.slice(b_l.start_offset()..);

    builder.arg(&elem_count);
    builder.arg(&num_dims);
    builder.arg(&dims_dev);
    builder.arg(&a_strides_dev);
    builder.arg(&b_strides_dev);
    builder.arg(&a_view);
    builder.arg(&b_view);
    builder.arg(&result);

    let config = LaunchConfig::for_num_elems(elem_count as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    let val = device.memcpy_dtov(&result)?;
    Ok(val[0] == 0)
}
