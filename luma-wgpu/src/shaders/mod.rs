mod cache;
mod shader;
use crate::{Wgpu, WgpuError, WgpuResult};
pub use cache::*;
use luma_tensor::{DType, FloatDType, IntDType};
pub use shader::*;

fn build_shader(device: &Wgpu, kind: ShaderKind) -> WgpuResult<Shader> {
    match kind {
        ShaderKind::Binary(DType::F32) => {
            Shader::new(device, "binary_f32", include_str!("../../shaders/binary/f32.wgsl"), "binary", [256, 1, 1])
        }
        ShaderKind::Binary(DType::F16) => {
            Shader::new(device, "binary_f16", include_str!("../../shaders/binary/f16.wgsl"), "binary", [256, 1, 1])
        }
        ShaderKind::Binary(DType::I32) => {
            Shader::new(device, "binary_i32", include_str!("../../shaders/binary/i32.wgsl"), "binary", [256, 1, 1])
        }
        ShaderKind::Binary(DType::U32) => {
            Shader::new(device, "binary_u32", include_str!("../../shaders/binary/u32.wgsl"), "binary", [256, 1, 1])
        }

        ShaderKind::BinaryScalar(DType::F32) => {
            Shader::new(device, "binary_scalar_f32", include_str!("../../shaders/binary/f32_scalar.wgsl"), "binary_scalar", [256, 1, 1])
        }
        ShaderKind::BinaryScalar(DType::F16) => {
            Shader::new(device, "binary_scalar_f16", include_str!("../../shaders/binary/f16_scalar.wgsl"), "binary_scalar", [256, 1, 1])
        }
        ShaderKind::BinaryScalar(DType::I32) => {
            Shader::new(device, "binary_scalar_i32", include_str!("../../shaders/binary/i32_scalar.wgsl"), "binary_scalar", [256, 1, 1])
        }
        ShaderKind::BinaryScalar(DType::U32) => {
            Shader::new(device, "binary_scalar_u32", include_str!("../../shaders/binary/u32_scalar.wgsl"), "binary_scalar", [256, 1, 1])
        }

        ShaderKind::BinaryInplace(DType::F32) => {
            Shader::new(device, "binary_inplace_f32", include_str!("../../shaders/binary/f32_inplace.wgsl"), "binary_inplace", [256, 1, 1])
        }
        ShaderKind::BinaryInplace(DType::F16) => {
            Shader::new(device, "binary_inplace_f16", include_str!("../../shaders/binary/f16_inplace.wgsl"), "binary_inplace", [256, 1, 1])
        }
        ShaderKind::BinaryInplace(DType::I32) => {
            Shader::new(device, "binary_inplace_i32", include_str!("../../shaders/binary/i32_inplace.wgsl"), "binary_inplace", [256, 1, 1])
        }
        ShaderKind::BinaryInplace(DType::U32) => {
            Shader::new(device, "binary_inplace_u32", include_str!("../../shaders/binary/u32_inplace.wgsl"), "binary_inplace", [256, 1, 1])
        }

        ShaderKind::BinaryScalarInplace(DType::F32) => Shader::new(
            device,
            "binary_scalar_inplace_f32",
            include_str!("../../shaders/binary/f32_scalar_inplace.wgsl"),
            "binary_scalar_inplace",
            [256, 1, 1],
        ),
        ShaderKind::BinaryScalarInplace(DType::F16) => Shader::new(
            device,
            "binary_scalar_inplace_f16",
            include_str!("../../shaders/binary/f16_scalar_inplace.wgsl"),
            "binary_scalar_inplace",
            [256, 1, 1],
        ),
        ShaderKind::BinaryScalarInplace(DType::I32) => Shader::new(
            device,
            "binary_scalar_inplace_i32",
            include_str!("../../shaders/binary/i32_scalar_inplace.wgsl"),
            "binary_scalar_inplace",
            [256, 1, 1],
        ),
        ShaderKind::BinaryScalarInplace(DType::U32) => Shader::new(
            device,
            "binary_scalar_inplace_u32",
            include_str!("../../shaders/binary/u32_scalar_inplace.wgsl"),
            "binary_scalar_inplace",
            [256, 1, 1],
        ),

        ShaderKind::Unary(DType::F32) => {
            Shader::new(device, "unary_f32", include_str!("../../shaders/unary/f32.wgsl"), "unary", [256, 1, 1])
        }
        ShaderKind::Unary(DType::F16) => {
            Shader::new(device, "unary_f16", include_str!("../../shaders/unary/f16.wgsl"), "unary", [256, 1, 1])
        }
        ShaderKind::Unary(DType::I32) => {
            Shader::new(device, "unary_i32", include_str!("../../shaders/unary/i32.wgsl"), "unary", [256, 1, 1])
        }
        ShaderKind::Unary(DType::U32) => {
            Shader::new(device, "unary_u32", include_str!("../../shaders/unary/u32.wgsl"), "unary", [256, 1, 1])
        }

        ShaderKind::UnaryInplace(DType::F32) => {
            Shader::new(device, "unary_inplace_f32", include_str!("../../shaders/unary/f32_inplace.wgsl"), "unary_inplace", [256, 1, 1])
        }
        ShaderKind::UnaryInplace(DType::F16) => {
            Shader::new(device, "unary_inplace_f16", include_str!("../../shaders/unary/f16_inplace.wgsl"), "unary_inplace", [256, 1, 1])
        }
        ShaderKind::UnaryInplace(DType::I32) => {
            Shader::new(device, "unary_inplace_i32", include_str!("../../shaders/unary/i32_inplace.wgsl"), "unary_inplace", [256, 1, 1])
        }
        ShaderKind::UnaryInplace(DType::U32) => {
            Shader::new(device, "unary_inplace_u32", include_str!("../../shaders/unary/u32_inplace.wgsl"), "unary_inplace", [256, 1, 1])
        }

        ShaderKind::FloatUnary(FloatDType::F32) => {
            Shader::new(device, "float_unary_f32", include_str!("../../shaders/float_unary/f32.wgsl"), "float_unary", [256, 1, 1])
        }
        ShaderKind::FloatUnary(FloatDType::F16) => {
            Shader::new(device, "float_unary_f16", include_str!("../../shaders/float_unary/f16.wgsl"), "float_unary", [256, 1, 1])
        }

        ShaderKind::FloatUnaryInplace(FloatDType::F32) => Shader::new(
            device,
            "float_unary_inplace_f32",
            include_str!("../../shaders/float_unary/f32_inplace.wgsl"),
            "float_unary_inplace",
            [256, 1, 1],
        ),
        ShaderKind::FloatUnaryInplace(FloatDType::F16) => Shader::new(
            device,
            "float_unary_inplace_f16",
            include_str!("../../shaders/float_unary/f16_inplace.wgsl"),
            "float_unary_inplace",
            [256, 1, 1],
        ),

        ShaderKind::Cmp(DType::F32) => {
            Shader::new(device, "cmp_f32", include_str!("../../shaders/cmp/f32.wgsl"), "cmp_kernel", [256, 1, 1])
        }
        ShaderKind::Cmp(DType::F16) => {
            Shader::new(device, "cmp_f16", include_str!("../../shaders/cmp/f16.wgsl"), "cmp_kernel", [256, 1, 1])
        }
        ShaderKind::Cmp(DType::I32) => {
            Shader::new(device, "cmp_i32", include_str!("../../shaders/cmp/i32.wgsl"), "cmp_kernel", [256, 1, 1])
        }
        ShaderKind::Cmp(DType::U32) => {
            Shader::new(device, "cmp_u32", include_str!("../../shaders/cmp/u32.wgsl"), "cmp_kernel", [256, 1, 1])
        }

        ShaderKind::CmpScalar(DType::F32) => {
            Shader::new(device, "cmp_scalar_f32", include_str!("../../shaders/cmp/f32_scalar.wgsl"), "cmp_scalar", [256, 1, 1])
        }
        ShaderKind::CmpScalar(DType::F16) => {
            Shader::new(device, "cmp_scalar_f16", include_str!("../../shaders/cmp/f16_scalar.wgsl"), "cmp_scalar", [256, 1, 1])
        }
        ShaderKind::CmpScalar(DType::I32) => {
            Shader::new(device, "cmp_scalar_i32", include_str!("../../shaders/cmp/i32_scalar.wgsl"), "cmp_scalar", [256, 1, 1])
        }
        ShaderKind::CmpScalar(DType::U32) => {
            Shader::new(device, "cmp_scalar_u32", include_str!("../../shaders/cmp/u32_scalar.wgsl"), "cmp_scalar", [256, 1, 1])
        }

        ShaderKind::Matmul(DType::F32) => {
            Shader::new(device, "matmul_f32", include_str!("../../shaders/matmul/f32.wgsl"), "matmul", [16, 16, 1])
        }
        ShaderKind::Matmul(DType::F16) => {
            Shader::new(device, "matmul_f16", include_str!("../../shaders/matmul/f16.wgsl"), "matmul", [16, 16, 1])
        }
        ShaderKind::Matmul(DType::I32) => {
            Shader::new(device, "matmul_i32", include_str!("../../shaders/matmul/i32.wgsl"), "matmul", [16, 16, 1])
        }
        ShaderKind::Matmul(DType::U32) => {
            Shader::new(device, "matmul_u32", include_str!("../../shaders/matmul/u32.wgsl"), "matmul", [16, 16, 1])
        }

        ShaderKind::Reduce(DType::F32) => {
            Shader::new(device, "reduce_f32", include_str!("../../shaders/reduce/f32.wgsl"), "reduce", [256, 1, 1])
        }
        ShaderKind::Reduce(DType::F16) => {
            Shader::new(device, "reduce_f16", include_str!("../../shaders/reduce/f16.wgsl"), "reduce", [256, 1, 1])
        }
        ShaderKind::Reduce(DType::I32) => {
            Shader::new(device, "reduce_i32", include_str!("../../shaders/reduce/i32.wgsl"), "reduce", [256, 1, 1])
        }
        ShaderKind::Reduce(DType::U32) => {
            Shader::new(device, "reduce_u32", include_str!("../../shaders/reduce/u32.wgsl"), "reduce", [256, 1, 1])
        }
        ShaderKind::Reduce(DType::Bool) => {
            Shader::new(device, "reduce_bool", include_str!("../../shaders/reduce/u32.wgsl"), "reduce", [256, 1, 1])
        }

        ShaderKind::ArgReduce(DType::F32) => {
            Shader::new(device, "arg_reduce_f32", include_str!("../../shaders/arg_reduce/f32.wgsl"), "arg_reduce", [256, 1, 1])
        }
        ShaderKind::ArgReduce(DType::F16) => {
            Shader::new(device, "arg_reduce_f16", include_str!("../../shaders/arg_reduce/f16.wgsl"), "arg_reduce", [256, 1, 1])
        }
        ShaderKind::ArgReduce(DType::I32) => {
            Shader::new(device, "arg_reduce_i32", include_str!("../../shaders/arg_reduce/i32.wgsl"), "arg_reduce", [256, 1, 1])
        }
        ShaderKind::ArgReduce(DType::U32) => {
            Shader::new(device, "arg_reduce_u32", include_str!("../../shaders/arg_reduce/u32.wgsl"), "arg_reduce", [256, 1, 1])
        }

        ShaderKind::Random(FloatDType::F32) => {
            Shader::new(device, "random_f32", include_str!("../../shaders/random/f32.wgsl"), "random", [256, 1, 1])
        }
        ShaderKind::Random(FloatDType::F16) => {
            Shader::new(device, "random_f16", include_str!("../../shaders/random/f16.wgsl"), "random", [256, 1, 1])
        }

        ShaderKind::BoolBinary => {
            Shader::new(device, "bool_binary", include_str!("../../shaders/bool/binary.wgsl"), "bool_binary", [256, 1, 1])
        }
        ShaderKind::BoolNot => Shader::new(device, "bool_not", include_str!("../../shaders/bool/not.wgsl"), "bool_not", [256, 1, 1]),

        ShaderKind::CastFloat(DType::F32, FloatDType::F32) => {
            Shader::new(device, "cast_f32_to_f32", include_str!("../../shaders/copy/f32.wgsl"), "copy", [256, 1, 1])
        }
        ShaderKind::CastFloat(DType::F16, FloatDType::F16) => {
            Shader::new(device, "cast_f16_to_f16", include_str!("../../shaders/copy/f16.wgsl"), "copy", [256, 1, 1])
        }
        ShaderKind::CastFloat(DType::F32, FloatDType::F16) => {
            Shader::new(device, "cast_f32_to_f16", include_str!("../../shaders/cast/f32_to_f16.wgsl"), "cast_kernel", [256, 1, 1])
        }
        ShaderKind::CastFloat(DType::F16, FloatDType::F32) => {
            Shader::new(device, "cast_f16_to_f32", include_str!("../../shaders/cast/f16_to_f32.wgsl"), "cast_kernel", [256, 1, 1])
        }
        ShaderKind::CastFloat(DType::I32, FloatDType::F32) => {
            Shader::new(device, "cast_i32_to_f32", include_str!("../../shaders/cast/i32_to_f32.wgsl"), "cast_kernel", [256, 1, 1])
        }
        ShaderKind::CastFloat(DType::I32, FloatDType::F16) => {
            Shader::new(device, "cast_i32_to_f16", include_str!("../../shaders/cast/i32_to_f16.wgsl"), "cast_kernel", [256, 1, 1])
        }
        ShaderKind::CastFloat(DType::U32, FloatDType::F32) => {
            Shader::new(device, "cast_u32_to_f32", include_str!("../../shaders/cast/u32_to_f32.wgsl"), "cast_kernel", [256, 1, 1])
        }
        ShaderKind::CastFloat(DType::U32, FloatDType::F16) => {
            Shader::new(device, "cast_u32_to_f16", include_str!("../../shaders/cast/u32_to_f16.wgsl"), "cast_kernel", [256, 1, 1])
        }
        ShaderKind::CastFloat(DType::Bool, FloatDType::F32) => {
            Shader::new(device, "cast_bool_to_f32", include_str!("../../shaders/cast/bool_to_f32.wgsl"), "cast_kernel", [256, 1, 1])
        }
        ShaderKind::CastFloat(DType::Bool, FloatDType::F16) => {
            Shader::new(device, "cast_bool_to_f16", include_str!("../../shaders/cast/bool_to_f16.wgsl"), "cast_kernel", [256, 1, 1])
        }

        ShaderKind::CastInt(DType::F32, IntDType::I32) => {
            Shader::new(device, "cast_f32_to_i32", include_str!("../../shaders/cast/f32_to_i32.wgsl"), "cast_kernel", [256, 1, 1])
        }
        ShaderKind::CastInt(DType::F32, IntDType::U32) => {
            Shader::new(device, "cast_f32_to_u32", include_str!("../../shaders/cast/f32_to_u32.wgsl"), "cast_kernel", [256, 1, 1])
        }
        ShaderKind::CastInt(DType::F16, IntDType::I32) => {
            Shader::new(device, "cast_f16_to_i32", include_str!("../../shaders/cast/f16_to_i32.wgsl"), "cast_kernel", [256, 1, 1])
        }
        ShaderKind::CastInt(DType::F16, IntDType::U32) => {
            Shader::new(device, "cast_f16_to_u32", include_str!("../../shaders/cast/f16_to_u32.wgsl"), "cast_kernel", [256, 1, 1])
        }
        ShaderKind::CastInt(DType::I32, IntDType::I32) => {
            Shader::new(device, "cast_i32_to_i32", include_str!("../../shaders/copy/i32.wgsl"), "copy", [256, 1, 1])
        }
        ShaderKind::CastInt(DType::I32, IntDType::U32) => {
            Shader::new(device, "cast_i32_to_u32", include_str!("../../shaders/cast/i32_to_u32.wgsl"), "cast_kernel", [256, 1, 1])
        }
        ShaderKind::CastInt(DType::U32, IntDType::I32) => {
            Shader::new(device, "cast_u32_to_i32", include_str!("../../shaders/cast/u32_to_i32.wgsl"), "cast_kernel", [256, 1, 1])
        }
        ShaderKind::CastInt(DType::U32, IntDType::U32) => {
            Shader::new(device, "cast_u32_to_u32", include_str!("../../shaders/copy/u32.wgsl"), "copy", [256, 1, 1])
        }
        ShaderKind::CastInt(DType::Bool, IntDType::I32) => {
            Shader::new(device, "cast_bool_to_i32", include_str!("../../shaders/cast/bool_to_i32.wgsl"), "cast_kernel", [256, 1, 1])
        }
        ShaderKind::CastInt(DType::Bool, IntDType::U32) => {
            Shader::new(device, "cast_bool_to_u32", include_str!("../../shaders/cast/bool_to_u32.wgsl"), "cast_kernel", [256, 1, 1])
        }

        ShaderKind::Copy(DType::F32) => Shader::new(device, "copy_f32", include_str!("../../shaders/copy/f32.wgsl"), "copy", [256, 1, 1]),
        ShaderKind::Copy(DType::F16) => Shader::new(device, "copy_f16", include_str!("../../shaders/copy/f16.wgsl"), "copy", [256, 1, 1]),
        ShaderKind::Copy(DType::I32) => Shader::new(device, "copy_i32", include_str!("../../shaders/copy/i32.wgsl"), "copy", [256, 1, 1]),
        ShaderKind::Copy(DType::U32) => Shader::new(device, "copy_u32", include_str!("../../shaders/copy/u32.wgsl"), "copy", [256, 1, 1]),
        ShaderKind::Copy(DType::Bool) => {
            Shader::new(device, "copy_bool", include_str!("../../shaders/copy/bool.wgsl"), "copy", [256, 1, 1])
        }

        ShaderKind::Pick(DType::F32) => Shader::new(device, "pick_f32", include_str!("../../shaders/pick/f32.wgsl"), "pick", [256, 1, 1]),
        ShaderKind::Pick(DType::F16) => Shader::new(device, "pick_f16", include_str!("../../shaders/pick/f16.wgsl"), "pick", [256, 1, 1]),
        ShaderKind::Pick(DType::I32) => Shader::new(device, "pick_i32", include_str!("../../shaders/pick/i32.wgsl"), "pick", [256, 1, 1]),
        ShaderKind::Pick(DType::U32) => Shader::new(device, "pick_u32", include_str!("../../shaders/pick/u32.wgsl"), "pick", [256, 1, 1]),
        ShaderKind::Pick(DType::Bool) => {
            Shader::new(device, "pick_bool", include_str!("../../shaders/pick/bool.wgsl"), "pick", [256, 1, 1])
        }

        ShaderKind::PickScalar(DType::F32) => {
            Shader::new(device, "pick_scalar_f32", include_str!("../../shaders/pick/f32_scalar.wgsl"), "pick_scalar", [256, 1, 1])
        }
        ShaderKind::PickScalar(DType::F16) => {
            Shader::new(device, "pick_scalar_f16", include_str!("../../shaders/pick/f16_scalar.wgsl"), "pick_scalar", [256, 1, 1])
        }
        ShaderKind::PickScalar(DType::I32) => {
            Shader::new(device, "pick_scalar_i32", include_str!("../../shaders/pick/i32_scalar.wgsl"), "pick_scalar", [256, 1, 1])
        }
        ShaderKind::PickScalar(DType::U32) => {
            Shader::new(device, "pick_scalar_u32", include_str!("../../shaders/pick/u32_scalar.wgsl"), "pick_scalar", [256, 1, 1])
        }
        ShaderKind::PickScalar(DType::Bool) => {
            Shader::new(device, "pick_scalar_bool", include_str!("../../shaders/pick/bool_scalar.wgsl"), "pick_scalar", [256, 1, 1])
        }
        ShaderKind::Cat(DType::F32) => Shader::new(device, "cat_f32", include_str!("../../shaders/cat/f32.wgsl"), "cat", [256, 1, 1]),
        ShaderKind::Cat(DType::F16) => Shader::new(device, "cat_f16", include_str!("../../shaders/cat/f16.wgsl"), "cat", [256, 1, 1]),
        ShaderKind::Cat(DType::I32) => Shader::new(device, "cat_i32", include_str!("../../shaders/cat/i32.wgsl"), "cat", [256, 1, 1]),
        ShaderKind::Cat(DType::U32) => Shader::new(device, "cat_u32", include_str!("../../shaders/cat/u32.wgsl"), "cat", [256, 1, 1]),
        ShaderKind::Cat(DType::Bool) => Shader::new(device, "cat_bool", include_str!("../../shaders/cat/bool.wgsl"), "cat", [256, 1, 1]),
        ShaderKind::IndexSelect(DType::F32) => {
            Shader::new(device, "index_select_f32", include_str!("../../shaders/index_select/f32.wgsl"), "index_select", [256, 1, 1])
        }
        ShaderKind::IndexSelect(DType::F16) => {
            Shader::new(device, "index_select_f16", include_str!("../../shaders/index_select/f16.wgsl"), "index_select", [256, 1, 1])
        }
        ShaderKind::IndexSelect(DType::I32) => {
            Shader::new(device, "index_select_i32", include_str!("../../shaders/index_select/i32.wgsl"), "index_select", [256, 1, 1])
        }
        ShaderKind::IndexSelect(DType::U32) => {
            Shader::new(device, "index_select_u32", include_str!("../../shaders/index_select/u32.wgsl"), "index_select", [256, 1, 1])
        }
        ShaderKind::IndexSelect(DType::Bool) => {
            Shader::new(device, "index_select_bool", include_str!("../../shaders/index_select/bool.wgsl"), "index_select", [256, 1, 1])
        }
        ShaderKind::Gather(DType::F32) => {
            Shader::new(device, "gather_f32", include_str!("../../shaders/gather/f32.wgsl"), "gather", [256, 1, 1])
        }
        ShaderKind::Gather(DType::F16) => {
            Shader::new(device, "gather_f16", include_str!("../../shaders/gather/f16.wgsl"), "gather", [256, 1, 1])
        }
        ShaderKind::Gather(DType::I32) => {
            Shader::new(device, "gather_i32", include_str!("../../shaders/gather/i32.wgsl"), "gather", [256, 1, 1])
        }
        ShaderKind::Gather(DType::U32) => {
            Shader::new(device, "gather_u32", include_str!("../../shaders/gather/u32.wgsl"), "gather", [256, 1, 1])
        }
        ShaderKind::Gather(DType::Bool) => {
            Shader::new(device, "gather_bool", include_str!("../../shaders/gather/bool.wgsl"), "gather", [256, 1, 1])
        }
        ShaderKind::IndexAdd(DType::F32) => {
            Shader::new(device, "index_add_f32", include_str!("../../shaders/index_add/f32.wgsl"), "index_add", [256, 1, 1])
        }
        ShaderKind::IndexAdd(DType::F16) => {
            Shader::new(device, "index_add_f16", include_str!("../../shaders/index_add/f16.wgsl"), "index_add", [256, 1, 1])
        }
        ShaderKind::IndexAdd(DType::I32) => {
            Shader::new(device, "index_add_i32", include_str!("../../shaders/index_add/i32.wgsl"), "index_add", [256, 1, 1])
        }
        ShaderKind::IndexAdd(DType::U32) => {
            Shader::new(device, "index_add_u32", include_str!("../../shaders/index_add/u32.wgsl"), "index_add", [256, 1, 1])
        }
        ShaderKind::ScatterAdd(DType::F32) => {
            Shader::new(device, "scatter_add_f32", include_str!("../../shaders/scatter_add/f32.wgsl"), "scatter_add", [256, 1, 1])
        }
        ShaderKind::ScatterAdd(DType::F16) => {
            Shader::new(device, "scatter_add_f16", include_str!("../../shaders/scatter_add/f16.wgsl"), "scatter_add", [256, 1, 1])
        }
        ShaderKind::ScatterAdd(DType::I32) => {
            Shader::new(device, "scatter_add_i32", include_str!("../../shaders/scatter_add/i32.wgsl"), "scatter_add", [256, 1, 1])
        }
        ShaderKind::ScatterAdd(DType::U32) => {
            Shader::new(device, "scatter_add_u32", include_str!("../../shaders/scatter_add/u32.wgsl"), "scatter_add", [256, 1, 1])
        }

        _ => Err(WgpuError::Msg(format!("unsupported shader kind {kind:?}"))),
    }
}
