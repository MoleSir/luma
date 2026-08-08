use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};
use crate::builder_arg;
use crate::{Layout, UnaryOp};
use super::super::{Cuda, CudaError, CudaResult, kernel};

pub(crate) fn unary_kernel_name(op: UnaryOp, suffix: &str) -> String {
    let op_str = match op {
        UnaryOp::Exp => "exp",
        UnaryOp::Ln => "ln",
        UnaryOp::Sin => "sin",
        UnaryOp::Cos => "cos",
        UnaryOp::Tanh => "tanh",
        UnaryOp::Sqr => "sqr",
        UnaryOp::Sqrt => "sqrt",
        UnaryOp::Recip => "recip",
        UnaryOp::Gelu => "gelu",
        UnaryOp::GeluErf => "gelu_erf",
        UnaryOp::Erf => "erf",
        UnaryOp::Relu => "relu",
        UnaryOp::LeakyRelu(_) => "leaky_relu",
        UnaryOp::Silu => "silu",
        UnaryOp::Sigmoid => "sigmoid",
        UnaryOp::Floor => "floor",
        UnaryOp::Ceil => "ceil",
        UnaryOp::Round => "round",
    };
    format!("u{}_{}", op_str, suffix)
}

pub(crate) fn launch_unary_raw_by_kernel_name<T: DeviceRepr>(
    device: &Cuda,
    kernel_name: &str,
    module: &kernel::Module,
    input: &CudaSlice<T>,
    layout: &Layout,
) -> CudaResult<CudaSlice<T>> {
    let dims = layout.dims();
    let elem_count = layout.shape().element_count();
    let num_dims = dims.len();
    let func = device.load_function(kernel_name, module)?;

    let mut builder = func.builder();
    let dims = device.memcpy_stod(dims)?;
    let strides = device.memcpy_stod(layout.stride())?;
    let output = device.alloc::<T>(elem_count)?;
    let input_view = input.slice(layout.start_offset()..);
    builder_arg!(builder, elem_count);
    builder_arg!(builder, num_dims);
    builder.arg(&dims);
    builder.arg(&strides);
    builder.arg(&input_view);
    builder.arg(&output);

    let config = LaunchConfig::for_num_elems(elem_count as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(output)
}

pub(crate) fn launch_unary<T: DeviceRepr>(
    device: &Cuda,
    op: UnaryOp,
    type_name: &str,
    module: &kernel::Module,
    input: &CudaSlice<T>,
    layout: &Layout,
) -> CudaResult<CudaSlice<T>> {
    let kernel_name = unary_kernel_name(op, type_name);
    launch_unary_raw_by_kernel_name(device, &kernel_name, module, input, layout)
}

pub(crate) fn launch_unary_param1<T: DeviceRepr>(
    device: &Cuda,
    op: UnaryOp,
    type_name: &str,
    module: &kernel::Module,
    input: &CudaSlice<T>,
    layout: &Layout,
    param: T,
) -> CudaResult<CudaSlice<T>> {
    let dims = layout.dims();
    let elem_count = layout.shape().element_count();
    let num_dims = dims.len();
    let kernel_name = unary_kernel_name(op, type_name);
    let func = device.load_function(&kernel_name, module)?;

    let mut builder = func.builder();
    let dims = device.memcpy_stod(dims)?;
    let strides = device.memcpy_stod(layout.stride())?;
    let output = device.alloc::<T>(elem_count)?;
    let input_view = input.slice(layout.start_offset()..);
    builder_arg!(builder, elem_count);
    builder_arg!(builder, num_dims);
    builder.arg(&dims);
    builder.arg(&strides);
    builder.arg(&param);
    builder.arg(&input_view);
    builder.arg(&output);

    let config = LaunchConfig::for_num_elems(elem_count as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(output)
}

#[allow(dead_code)]
pub(crate) fn launch_unary_param2<T: DeviceRepr>(
    device: &Cuda,
    op: UnaryOp,
    type_name: &str,
    module: &kernel::Module,
    input: &CudaSlice<T>,
    layout: &Layout,
    param1: T,
    param2: T,
) -> CudaResult<CudaSlice<T>> {
    let dims = layout.dims();
    let elem_count = layout.shape().element_count();
    let num_dims = dims.len();
    let kernel_name = unary_kernel_name(op, type_name);
    let func = device.load_function(&kernel_name, module)?;

    let mut builder = func.builder();
    let dims = device.memcpy_stod(dims)?;
    let strides = device.memcpy_stod(layout.stride())?;
    let output = device.alloc::<T>(elem_count)?;
    let input_view = input.slice(layout.start_offset()..);
    builder_arg!(builder, elem_count);
    builder_arg!(builder, num_dims);
    builder.arg(&dims);
    builder.arg(&strides);
    builder.arg(&param1);
    builder.arg(&param2);
    builder.arg(&input_view);
    builder.arg(&output);

    let config = LaunchConfig::for_num_elems(elem_count as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(output)
}

pub(crate) fn launch_clamp<T: DeviceRepr>(
    device: &Cuda,
    suffix: &str,
    module: &kernel::Module,
    input: &CudaSlice<T>,
    layout: &Layout,
    has_min: bool,
    min_val: T,
    has_max: bool,
    max_val: T,
) -> CudaResult<CudaSlice<T>> {
    let dims = layout.dims();
    let elem_count = layout.shape().element_count();
    let num_dims = dims.len();
    let kernel_name = format!("uclamp_{}", suffix);
    let func = device.load_function(&kernel_name, module)?;

    let mut builder = func.builder();
    let dims_dev = device.memcpy_stod(dims)?;
    let strides = device.memcpy_stod(layout.stride())?;
    let output = device.alloc::<T>(elem_count)?;
    let input_view = input.slice(layout.start_offset()..);
    builder_arg!(builder, elem_count);
    builder_arg!(builder, num_dims);
    builder.arg(&dims_dev);
    builder.arg(&strides);
    builder.arg(&has_min);
    builder.arg(&min_val);
    builder.arg(&has_max);
    builder.arg(&max_val);
    builder.arg(&input_view);
    builder.arg(&output);

    let config = LaunchConfig::for_num_elems(elem_count as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(output)
}

pub(crate) fn launch_affine<T: DeviceRepr>(
    device: &Cuda,
    suffix: &str,
    module: &kernel::Module,
    input: &CudaSlice<T>,
    layout: &Layout,
    mul: T,
    add: T,
) -> CudaResult<CudaSlice<T>> {
    let dims = layout.dims();
    let elem_count = layout.shape().element_count();
    let num_dims = dims.len();
    let kernel_name = format!("uaffine_{}", suffix);
    let func = device.load_function(&kernel_name, module)?;

    let mut builder = func.builder();
    let dims_dev = device.memcpy_stod(dims)?;
    let strides = device.memcpy_stod(layout.stride())?;
    let output = device.alloc::<T>(elem_count)?;
    let input_view = input.slice(layout.start_offset()..);
    builder_arg!(builder, elem_count);
    builder_arg!(builder, num_dims);
    builder.arg(&dims_dev);
    builder.arg(&strides);
    builder.arg(&mul);
    builder.arg(&add);
    builder.arg(&input_view);
    builder.arg(&output);

    let config = LaunchConfig::for_num_elems(elem_count as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(output)
}

pub(crate) fn launch_pow<T: DeviceRepr>(
    device: &Cuda,
    suffix: &str,
    module: &kernel::Module,
    input: &CudaSlice<T>,
    layout: &Layout,
    exp: T,
) -> CudaResult<CudaSlice<T>> {
    let dims = layout.dims();
    let elem_count = layout.shape().element_count();
    let num_dims = dims.len();
    let kernel_name = format!("upow_{}", suffix);
    let func = device.load_function(&kernel_name, module)?;

    let mut builder = func.builder();
    let dims_dev = device.memcpy_stod(dims)?;
    let strides = device.memcpy_stod(layout.stride())?;
    let output = device.alloc::<T>(elem_count)?;
    let input_view = input.slice(layout.start_offset()..);
    builder_arg!(builder, elem_count);
    builder_arg!(builder, num_dims);
    builder.arg(&dims_dev);
    builder.arg(&strides);
    builder.arg(&exp);
    builder.arg(&input_view);
    builder.arg(&output);

    let config = LaunchConfig::for_num_elems(elem_count as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(output)
}
