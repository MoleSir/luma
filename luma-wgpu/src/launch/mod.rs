//! Host-side launch code for the larger, shape-aware kernels.
//!
//! Unlike the trait plumbing in `ops/`, these kernels need shape/stride
//! metadata, output allocation, and a dedicated dispatch configuration.

mod binary;
mod boolean;
mod cast;
mod cat;
mod cmp;
mod copy;
mod indexing;
mod matmul;
mod pick;
mod random;
mod reduce;
mod unary;

pub(crate) use binary::{launch_binary, launch_binary_inplace, launch_binary_scalar, launch_binary_scalar_inplace};
pub(crate) use boolean::{BoolOp, launch_bool_binary, launch_bool_not};
pub(crate) use cast::{launch_cast_float, launch_cast_int};
pub(crate) use cat::launch_cat;
pub(crate) use cmp::{launch_cmp, launch_cmp_scalar};
pub(crate) use copy::launch_copy;
pub(crate) use indexing::{launch_gather, launch_index_add, launch_index_select, launch_scatter_add};
pub(crate) use matmul::{launch_add_matmul_, launch_matmul};
pub(crate) use pick::{launch_pick, launch_pick_scalar};
pub(crate) use random::{launch_random_normal, launch_random_uniform};
pub(crate) use reduce::{launch_arg_reduce, launch_reduce};
pub(crate) use unary::{launch_float_unary, launch_float_unary_inplace, launch_unary, launch_unary_inplace};

use crate::Wgpu;
use luma_tensor::Layout;

/// `dims | strides` for a single operand.
pub(crate) fn layout_index_meta(layout: &Layout) -> Vec<u32> {
    let mut meta: Vec<u32> = Vec::with_capacity(2 * layout.shape().rank());
    meta.extend(layout.dims().iter().map(|&v| v as u32));
    meta.extend(layout.stride().iter().map(|&v| v as u32));
    meta
}

/// Resolve a layout start offset into `(start_elements_in_shader, buffer_binding_byte_offset)`.
///
/// wgpu storage-buffer binding offsets must satisfy
/// `min_storage_buffer_offset_alignment`. When the layout start is aligned we
/// bind at that offset and the shader can index from zero; otherwise the shader
/// falls back to adding `start_elements`.
pub(crate) fn buffer_start(layout: &Layout, elem_bytes: usize, device: &Wgpu) -> (u32, u64) {
    let byte_offset = (layout.start_offset() * elem_bytes) as u64;
    let align = device.device().limits().min_storage_buffer_offset_alignment as u64;
    if byte_offset.is_multiple_of(align) { (0, byte_offset) } else { (layout.start_offset() as u32, 0) }
}

/// `dims | lhs_strides | rhs_strides` for two same-shaped operands.
pub(crate) fn binary_index_meta(lhs_l: &Layout, rhs_l: &Layout) -> Vec<u32> {
    let rank = lhs_l.shape().rank();
    let mut meta = Vec::with_capacity(3 * rank);
    meta.extend(lhs_l.dims().iter().map(|&v| v as u32));
    meta.extend(lhs_l.stride().iter().map(|&v| v as u32));
    meta.extend(rhs_l.stride().iter().map(|&v| v as u32));
    meta
}
