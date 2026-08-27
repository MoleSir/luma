use proc_macro_crate::{FoundCrate, crate_name};
use quote::quote;

/// Resolve the path to the `luma-nn` crate from the call site.
///
/// Returns one of `crate` (if the derive is used inside `luma-nn` itself),
/// the renamed identifier (if the user aliased the dependency), `luma::nn`
/// (if the user is using the `luma` facade), or `luma_nn` as the fallback.
pub fn get_luma_nn_path() -> proc_macro2::TokenStream {
    match crate_name("luma-nn") {
        Ok(FoundCrate::Itself) => {
            quote! { crate }
        }
        Ok(FoundCrate::Name(name)) => {
            let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
            quote! { #ident }
        }
        Err(_) => match crate_name("luma-kit") {
            Ok(FoundCrate::Name(_name)) => {
                // let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
                quote! { luma::nn }
            }
            _ => {
                quote! { luma_nn }
            }
        },
    }
}

/// Resolve the path to the `luma-tensor` crate from the call site.
///
/// Returns one of `crate` (if the derive is used inside `luma-tensor` itself),
/// the renamed identifier (if the user aliased the dependency), `luma` (if the
/// user is using the `luma` facade, which re-exports tensor flat), or
/// `luma_tensor` as the fallback.
pub fn get_luma_tensor_path() -> proc_macro2::TokenStream {
    match crate_name("luma-tensor") {
        Ok(FoundCrate::Itself) => {
            quote! { crate }
        }
        Ok(FoundCrate::Name(name)) => {
            let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
            quote! { #ident }
        }
        Err(_) => match crate_name("luma-kit") {
            Ok(FoundCrate::Name(_name)) => {
                // let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
                quote! { luma }
            }
            _ => {
                quote! { luma_tensor }
            }
        },
    }
}
