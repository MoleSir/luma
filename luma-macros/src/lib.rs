use proc_macro::TokenStream;
mod module;
mod utils;

/// Derive macro for the `Module` trait.
///
/// Generates implementations of [`luma_nn::Module`] for structs and enums,
/// automatically traversing all fields/variants to expose parameters, buffers,
/// state, and sub-modules.
///
/// # Examples
///
/// ```ignore
/// use luma_tensor::Device;
/// use luma_nn::{Module, Parameter};
///
/// #[derive(Module)]
/// struct Linear<D: Device> {
///     weight: Parameter<D>,
///     bias: Option<Parameter<D>>,
/// }
/// ```
///
/// # Attributes
///
/// - `#[module(skip)]` — skip a field or variant.
/// - `#[module(display = "fn_name")]` — delegate `extra_display()` to a method.
/// - `#[module(train = "fn_name")]` — delegate `set_train()` to a method.
/// - `#[module(reset = "fn_name")]` — delegate `reset_parameters()` to a method.
#[proc_macro_derive(Module, attributes(module))]
pub fn module_derive(input: TokenStream) -> TokenStream {
    let input = syn::parse(input).unwrap();
    module::derive_impl(&input)
}
