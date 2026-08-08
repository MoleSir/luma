use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};
use crate::builder_arg;
use crate::{CmpOp, Layout};
use super::super::{Cuda, CudaError, CudaResult, kernel};

pub(crate) fn cmp_kernel_name(op: CmpOp, suffix: &str) -> String {
    let op_str = match op {
        CmpOp::Eq => "eq",
        CmpOp::Ne => "ne",
        CmpOp::Lt => "lt",
        CmpOp::Le => "le",
        CmpOp::Gt => "gt",
        CmpOp::Ge => "ge",
    };
    format!("b{}_{}", op_str, suffix)
}

pub(crate) fn launch_cmp<T: DeviceRepr>(
    device: &Cuda,
    op: CmpOp,
    type_name: &str,
    module: &kernel::Module,
    lhs: &CudaSlice<T>,
    rhs: &CudaSlice<T>,
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> CudaResult<CudaSlice<u8>> {
    let dims = lhs_l.dims();
    let elem_count = lhs_l.shape().element_count();
    let num_dims = dims.len();
    let kernel_name = cmp_kernel_name(op, type_name);
    let func = device.load_function(&kernel_name, module)?;

    let mut builder = func.builder();
    let dims_dev = device.memcpy_stod(dims)?;
    let lhs_strides = device.memcpy_stod(lhs_l.stride())?;
    let rhs_strides = device.memcpy_stod(rhs_l.stride())?;
    let output = device.alloc::<u8>(elem_count)?;
    let lhs_view = lhs.slice(lhs_l.start_offset()..);
    let rhs_view = rhs.slice(rhs_l.start_offset()..);
    builder_arg!(builder, elem_count);
    builder_arg!(builder, num_dims);
    builder.arg(&dims_dev);
    builder.arg(&lhs_strides);
    builder.arg(&rhs_strides);
    builder.arg(&lhs_view);
    builder.arg(&rhs_view);
    builder.arg(&output);

    let config = LaunchConfig::for_num_elems(elem_count as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(output)
}
