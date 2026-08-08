use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};
use crate::{BinaryOp, Layout};
use super::super::{Cuda, CudaError, CudaResult, kernel};

pub(crate) fn binary_kernel_name(op: BinaryOp, suffix: &str) -> String {
    let op_str = match op {
        BinaryOp::Add => "add",
        BinaryOp::Sub => "sub",
        BinaryOp::Mul => "mul",
        BinaryOp::Div => "div",
        BinaryOp::Maximum => "max",
        BinaryOp::Minimum => "min",
    };
    format!("b{}_{}", op_str, suffix)
}

pub(crate) fn launch_binary<T: DeviceRepr>(
    device: &Cuda,
    op: BinaryOp,
    type_name: &str,
    module: &kernel::Module,
    lhs: &CudaSlice<T>,
    rhs: &CudaSlice<T>,
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> CudaResult<CudaSlice<T>> {
    let kernel_name = binary_kernel_name(op, type_name);
    launch_binary_by_kernel_name(device, &kernel_name, module, lhs, rhs, lhs_l, rhs_l)
}

pub(crate) fn launch_binary_by_kernel_name<T: DeviceRepr>(
    device: &Cuda,
    kernel_name: &str,
    module: &kernel::Module,
    lhs: &CudaSlice<T>,
    rhs: &CudaSlice<T>,
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> CudaResult<CudaSlice<T>> {
    let dims = lhs_l.dims();
    let elem_count = lhs_l.shape().element_count();
    let num_dims = dims.len();
    let func = device.load_function(&kernel_name, module)?;

    let mut builder = func.builder();

    let dims = device.memcpy_stod(dims)?;
    let lhs_strides = device.memcpy_stod(lhs_l.stride())?;
    let rhs_strides = device.memcpy_stod(rhs_l.stride())?;
    let output = device.alloc::<T>(elem_count)?;
    let lhs_view = lhs.slice(lhs_l.start_offset()..);
    let rhs_view = rhs.slice(rhs_l.start_offset()..);
    
    builder.arg(&elem_count);
    builder.arg(&num_dims);
    builder.arg(&dims);
    builder.arg(&lhs_strides);
    builder.arg(&rhs_strides);
    builder.arg(&lhs_view);
    builder.arg(&rhs_view);
    builder.arg(&output);

    let config = LaunchConfig::for_num_elems(elem_count as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(output)
}

pub(crate) fn launch_binary_inplace<T: DeviceRepr>(
    device: &Cuda,
    op: BinaryOp,
    type_name: &str,
    module: &kernel::Module,
    dst: &CudaSlice<T>,
    src: &CudaSlice<T>,
    dst_l: &Layout,
    src_l: &Layout,
) -> CudaResult<()> {
    let dims = dst_l.dims();
    let elem_count = dst_l.shape().element_count();
    let num_dims = dims.len();
    let kernel_name = binary_kernel_name(op, type_name);
    let func = device.load_function(&kernel_name, module)?;

    let mut builder = func.builder();
    let dims_dev = device.memcpy_stod(dims)?;
    let dst_strides = device.memcpy_stod(dst_l.stride())?;
    let src_strides = device.memcpy_stod(src_l.stride())?;
    let dst_view = dst.slice(dst_l.start_offset()..);
    let src_view = src.slice(src_l.start_offset()..);

    builder.arg(&elem_count);
    builder.arg(&num_dims);
    builder.arg(&dims_dev);
    builder.arg(&dst_strides);
    builder.arg(&src_strides);
    builder.arg(&dst_view);
    builder.arg(&src_view);
    builder.arg(&dst_view);

    let config = LaunchConfig::for_num_elems(elem_count as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(())
}

pub(crate) fn binary_scalar_kernel_name(op: BinaryOp, suffix: &str) -> String {
    let op_str = match op {
        BinaryOp::Add => "add",
        BinaryOp::Sub => "sub",
        BinaryOp::Mul => "mul",
        BinaryOp::Div => "div",
        BinaryOp::Maximum => "max",
        BinaryOp::Minimum => "min",
    };
    format!("bs{}_{}", op_str, suffix)
}

pub(crate) fn launch_binary_scalar<T: DeviceRepr>(
    device: &Cuda,
    op: BinaryOp,
    type_name: &str,
    module: &kernel::Module,
    lhs: &CudaSlice<T>,
    lhs_l: &Layout,
    rhs: T,
) -> CudaResult<CudaSlice<T>> {
    let dims = lhs_l.dims();
    let elem_count = lhs_l.shape().element_count();
    let num_dims = dims.len();
    let kernel_name = binary_scalar_kernel_name(op, type_name);
    let func = device.load_function(&kernel_name, module)?;

    let mut builder = func.builder();
    
    let dims = device.memcpy_stod(dims)?;
    let lhs_strides = device.memcpy_stod(lhs_l.stride())?;
    let output = device.alloc::<T>(elem_count)?;
    let lhs_view = lhs.slice(lhs_l.start_offset()..);
    
    builder.arg(&elem_count);
    builder.arg(&num_dims);
    builder.arg(&dims);
    builder.arg(&lhs_strides);
    builder.arg(&lhs_view);
    builder.arg(&rhs);
    builder.arg(&output);

    let config = LaunchConfig::for_num_elems(elem_count as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(output)
}
