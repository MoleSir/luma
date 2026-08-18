mod allclose;
mod binary;
mod cast;
mod cmp;
mod copy;
mod indexing;
mod matmul;
mod nn;
mod pick;
mod reduce;
mod unary;

pub(crate) use allclose::{launch_allclose_float, launch_allclose_int};
pub(crate) use binary::{
    launch_binary, launch_binary_by_kernel_name, launch_binary_inplace, launch_binary_scalar, launch_binary_scalar_inplace,
    launch_binary_scalar_lhs,
};
pub(crate) use cast::launch_cast;
pub(crate) use cmp::{launch_cmp, launch_cmp_scalar};
pub(crate) use copy::{launch_copy_offset, launch_copy2d};
pub(crate) use indexing::{launch_gather, launch_index_add, launch_index_select, launch_scatter_add};
pub(crate) use matmul::{launch_add_matmul_, launch_matmul};
pub(crate) use nn::{launch_rms_norm_f32, launch_rms_norm_f64, launch_softmax_f32, launch_softmax_f64};
pub(crate) use pick::{launch_pick, launch_pick_false, launch_pick_true};
pub(crate) use reduce::{arg_reduce_kernel_name, launch_arg_reduce, launch_multi_reduce, launch_multi_reduce_by_kernel_name};
pub(crate) use unary::{
    launch_affine, launch_affine_inplace, launch_clamp, launch_clamp_inplace, launch_float_unary, launch_float_unary_inplace, launch_pow,
    launch_pow_inplace, launch_unary_param1, launch_unary_param1_inplace, launch_unary_raw_by_kernel_name, launch_unary_raw_inplace,
};

#[macro_export]
macro_rules! builder_arg {
    ($b:ident, $($arg:expr),*) => {
        $(
            let __arg = $arg;
            $b.arg(&__arg);
        )*
    };
}
