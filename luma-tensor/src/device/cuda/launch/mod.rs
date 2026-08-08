mod binary;
mod cmp;
mod unary;
mod reduce;
mod cast;
mod matmul;
mod copy;
mod indexing;
mod pick;
mod allclose;
mod nn;

pub(crate) use binary::{launch_binary, launch_binary_inplace, launch_binary_by_kernel_name, launch_binary_scalar};
pub(crate) use cmp::launch_cmp;
pub(crate) use unary::{launch_unary, launch_unary_param1, launch_unary_raw_by_kernel_name, launch_clamp, launch_affine, launch_pow};
pub(crate) use reduce::{launch_multi_reduce, launch_multi_reduce_by_kernel_name, launch_arg_reduce, arg_reduce_kernel_name};
pub(crate) use cast::launch_cast;
pub(crate) use matmul::{launch_matmul, launch_add_matmul_};
pub(crate) use copy::{launch_copy2d, launch_copy_offset};
pub(crate) use indexing::{launch_index_select, launch_gather, launch_index_add, launch_scatter_add};
pub(crate) use pick::{launch_pick, launch_pick_true, launch_pick_false};
pub(crate) use allclose::{launch_allclose_float, launch_allclose_int};
pub(crate) use nn::{launch_softmax_f32, launch_softmax_f64, launch_rms_norm_f32, launch_rms_norm_f64};

#[macro_export]
macro_rules! builder_arg {
    ($b:ident, $($arg:expr),*) => {
        $(
            let __arg = $arg;
            $b.arg(&__arg);
        )*
    };
}
