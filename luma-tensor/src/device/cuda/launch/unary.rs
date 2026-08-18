use super::super::{Cuda, CudaError, CudaResult, kernel};
use crate::builder_arg;
use crate::{FloatUnaryOp, Layout};
use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};

pub(crate) fn float_unary_kernel_name(op: FloatUnaryOp, suffix: &str) -> String {
    let op_str = match op {
        FloatUnaryOp::Exp => "exp",
        FloatUnaryOp::Ln => "ln",
        FloatUnaryOp::Sin => "sin",
        FloatUnaryOp::Cos => "cos",
        FloatUnaryOp::Tanh => "tanh",
        FloatUnaryOp::Sqr => "sqr",
        FloatUnaryOp::Sqrt => "sqrt",
        FloatUnaryOp::Recip => "recip",
        FloatUnaryOp::Gelu => "gelu",
        FloatUnaryOp::GeluErf => "gelu_erf",
        FloatUnaryOp::Erf => "erf",
        FloatUnaryOp::Relu => "relu",
        FloatUnaryOp::LeakyRelu(_) => "leaky_relu",
        FloatUnaryOp::Silu => "silu",
        FloatUnaryOp::Sigmoid => "sigmoid",
        FloatUnaryOp::Floor => "floor",
        FloatUnaryOp::Ceil => "ceil",
        FloatUnaryOp::Round => "round",
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

pub(crate) fn launch_unary_raw_inplace<T: DeviceRepr>(
    device: &Cuda,
    kernel_name: &str,
    module: &kernel::Module,
    dst: &CudaSlice<T>,
    layout: &Layout,
) -> CudaResult<()> {
    let dims = layout.dims();
    let elem_count = layout.shape().element_count();
    let num_dims = dims.len();
    let func = device.load_function(kernel_name, module)?;

    let mut builder = func.builder();
    let dims = device.memcpy_stod(dims)?;
    let strides = device.memcpy_stod(layout.stride())?;
    let dst_view = dst.slice(layout.start_offset()..);
    builder_arg!(builder, elem_count);
    builder_arg!(builder, num_dims);
    builder.arg(&dims);
    builder.arg(&strides);
    builder.arg(&dst_view);
    builder.arg(&dst_view);

    let config = LaunchConfig::for_num_elems(elem_count as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(())
}

pub(crate) fn launch_float_unary_inplace<T: DeviceRepr>(
    device: &Cuda,
    op: FloatUnaryOp,
    type_name: &str,
    module: &kernel::Module,
    dst: &CudaSlice<T>,
    layout: &Layout,
) -> CudaResult<()> {
    let kernel_name = float_unary_kernel_name(op, type_name);
    launch_unary_raw_inplace(device, &kernel_name, module, dst, layout)
}

pub(crate) fn launch_unary_param1_inplace<T: DeviceRepr>(
    device: &Cuda,
    op: FloatUnaryOp,
    type_name: &str,
    module: &kernel::Module,
    dst: &CudaSlice<T>,
    layout: &Layout,
    param: T,
) -> CudaResult<()> {
    let dims = layout.dims();
    let elem_count = layout.shape().element_count();
    let num_dims = dims.len();
    let kernel_name = float_unary_kernel_name(op, type_name);
    let func = device.load_function(&kernel_name, module)?;

    let mut builder = func.builder();
    let dims = device.memcpy_stod(dims)?;
    let strides = device.memcpy_stod(layout.stride())?;
    let dst_view = dst.slice(layout.start_offset()..);
    builder_arg!(builder, elem_count);
    builder_arg!(builder, num_dims);
    builder.arg(&dims);
    builder.arg(&strides);
    builder.arg(&param);
    builder.arg(&dst_view);
    builder.arg(&dst_view);

    let config = LaunchConfig::for_num_elems(elem_count as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(())
}

pub(crate) fn launch_affine_inplace<T: DeviceRepr>(
    device: &Cuda,
    suffix: &str,
    module: &kernel::Module,
    dst: &CudaSlice<T>,
    layout: &Layout,
    mul: T,
    add: T,
) -> CudaResult<()> {
    let dims = layout.dims();
    let elem_count = layout.shape().element_count();
    let num_dims = dims.len();
    let kernel_name = format!("uaffine_{}", suffix);
    let func = device.load_function(&kernel_name, module)?;

    let mut builder = func.builder();
    let dims_dev = device.memcpy_stod(dims)?;
    let strides = device.memcpy_stod(layout.stride())?;
    let dst_view = dst.slice(layout.start_offset()..);
    builder_arg!(builder, elem_count);
    builder_arg!(builder, num_dims);
    builder.arg(&dims_dev);
    builder.arg(&strides);
    builder.arg(&mul);
    builder.arg(&add);
    builder.arg(&dst_view);
    builder.arg(&dst_view);

    let config = LaunchConfig::for_num_elems(elem_count as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(())
}

pub(crate) fn launch_pow_inplace<T: DeviceRepr>(
    device: &Cuda,
    suffix: &str,
    module: &kernel::Module,
    dst: &CudaSlice<T>,
    layout: &Layout,
    exp: T,
) -> CudaResult<()> {
    let dims = layout.dims();
    let elem_count = layout.shape().element_count();
    let num_dims = dims.len();
    let kernel_name = format!("upow_{}", suffix);
    let func = device.load_function(&kernel_name, module)?;

    let mut builder = func.builder();
    let dims_dev = device.memcpy_stod(dims)?;
    let strides = device.memcpy_stod(layout.stride())?;
    let dst_view = dst.slice(layout.start_offset()..);
    builder_arg!(builder, elem_count);
    builder_arg!(builder, num_dims);
    builder.arg(&dims_dev);
    builder.arg(&strides);
    builder.arg(&exp);
    builder.arg(&dst_view);
    builder.arg(&dst_view);

    let config = LaunchConfig::for_num_elems(elem_count as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(())
}

pub(crate) fn launch_clamp_inplace<T: DeviceRepr>(
    device: &Cuda,
    suffix: &str,
    module: &kernel::Module,
    dst: &CudaSlice<T>,
    layout: &Layout,
    has_min: bool,
    min_val: T,
    has_max: bool,
    max_val: T,
) -> CudaResult<()> {
    let dims = layout.dims();
    let elem_count = layout.shape().element_count();
    let num_dims = dims.len();
    let kernel_name = format!("uclamp_{}", suffix);
    let func = device.load_function(&kernel_name, module)?;

    let mut builder = func.builder();
    let dims_dev = device.memcpy_stod(dims)?;
    let strides = device.memcpy_stod(layout.stride())?;
    let dst_view = dst.slice(layout.start_offset()..);
    builder_arg!(builder, elem_count);
    builder_arg!(builder, num_dims);
    builder.arg(&dims_dev);
    builder.arg(&strides);
    builder.arg(&has_min);
    builder.arg(&min_val);
    builder.arg(&has_max);
    builder.arg(&max_val);
    builder.arg(&dst_view);
    builder.arg(&dst_view);

    let config = LaunchConfig::for_num_elems(elem_count as u32);
    unsafe { builder.launch(config) }.map_err(CudaError::CudaDriver)?;
    Ok(())
}

pub(crate) fn launch_float_unary<T: DeviceRepr>(
    device: &Cuda,
    op: FloatUnaryOp,
    type_name: &str,
    module: &kernel::Module,
    input: &CudaSlice<T>,
    layout: &Layout,
) -> CudaResult<CudaSlice<T>> {
    let kernel_name = float_unary_kernel_name(op, type_name);
    launch_unary_raw_by_kernel_name(device, &kernel_name, module, input, layout)
}

pub(crate) fn launch_unary_param1<T: DeviceRepr>(
    device: &Cuda,
    op: FloatUnaryOp,
    type_name: &str,
    module: &kernel::Module,
    input: &CudaSlice<T>,
    layout: &Layout,
    param: T,
) -> CudaResult<CudaSlice<T>> {
    let dims = layout.dims();
    let elem_count = layout.shape().element_count();
    let num_dims = dims.len();
    let kernel_name = float_unary_kernel_name(op, type_name);
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
    op: FloatUnaryOp,
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
    let kernel_name = float_unary_kernel_name(op, type_name);
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
