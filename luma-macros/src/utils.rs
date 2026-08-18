use proc_macro_crate::{FoundCrate, crate_name};
use quote::quote;

/// Resolve the path to the `luma-nn` crate from the call site.
///
/// Returns one of `crate` (if the derive is used inside `luma-nn` itself),
/// the renamed identifier (if the user aliased the dependency), or `luma_nn`
/// as the fallback.
pub fn get_luma_nn_path() -> proc_macro2::TokenStream {
    match crate_name("luma-nn") {
        Ok(FoundCrate::Itself) => {
            quote! { crate }
        }
        Ok(FoundCrate::Name(name)) => {
            let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
            quote! { #ident }
        }
        Err(_) => {
            quote! { luma_nn }
        }
    }
}
