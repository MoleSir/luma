use crate::utils;
use proc_macro2::TokenStream;
use quote::quote;

use super::common;

pub fn generate_enum(ast: &syn::DeriveInput) -> TokenStream {
    let luma = utils::get_luma_nn_path();
    let name = &ast.ident;

    // ---- validate device generic -------------------------------------------
    let validated_device = match common::validate_and_extract_device_generic(&ast.generics) {
        Ok(res) => res,
        Err(e) => return e.to_compile_error().into(),
    };

    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();
    let (impl_generics_tokens, device_type) = match validated_device.as_ref() {
        Some(user_device_ident) => (quote! { #impl_generics }, quote! { #user_device_ident }),
        None => (quote! { <D: luma_tensor::Device> }, quote! { D }),
    };

    // ---- parse module-level attributes -------------------------------------
    let mut custom_display_fn_name: Option<syn::Ident> = None;
    let mut custom_set_train_name: Option<syn::Ident> = None;
    let mut custom_reset_name: Option<syn::Ident> = None;

    for attr in &ast.attrs {
        if attr.path().is_ident("module") {
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("display") {
                    let value = meta.value()?;
                    let s: syn::LitStr = value.parse()?;
                    custom_display_fn_name = Some(syn::Ident::new(&s.value(), s.span()));
                    return Ok(());
                }
                if meta.path.is_ident("train") {
                    let value = meta.value()?;
                    let s: syn::LitStr = value.parse()?;
                    custom_set_train_name = Some(syn::Ident::new(&s.value(), s.span()));
                    return Ok(());
                }
                if meta.path.is_ident("reset") {
                    let value = meta.value()?;
                    let s: syn::LitStr = value.parse()?;
                    custom_reset_name = Some(syn::Ident::new(&s.value(), s.span()));
                    return Ok(());
                }
                Ok(())
            });
        }
    }

    // ---- extra_display / set_train / reset_parameters ----------------------
    let extra_display_fn = if let Some(fn_name) = custom_display_fn_name {
        quote! { fn extra_display(&self) -> String { self.#fn_name() } }
    } else {
        quote! {}
    };

    let set_train_fn = if let Some(fn_name) = custom_set_train_name {
        quote! { fn set_train(&mut self, mode: bool) { self.#fn_name(mode) } }
    } else {
        quote! {}
    };

    let reset_fn = if let Some(fn_name) = custom_reset_name {
        quote! {
            fn reset_parameters(&mut self) -> std::result::Result<(), #luma::NnError> {
                self.#fn_name()
            }
        }
    } else {
        quote! {}
    };

    // ---- generate match arms for each variant ------------------------------
    let mut param_arms = quote! {};
    let mut param_mut_arms = quote! {};
    let mut state_arms = quote! {};
    let mut state_mut_arms = quote! {};
    let mut buffer_arms = quote! {};
    let mut buffer_mut_arms = quote! {};
    let mut module_arms = quote! {};
    let mut module_mut_arms = quote! {};

    if let syn::Data::Enum(enum_data) = &ast.data {
        for variant in enum_data.variants.iter() {
            let variant_ident = &variant.ident;
            let variant_name_str = variant_ident.to_string();

            // check #[module(skip)]
            let should_skip = variant.attrs.iter().any(|attr| {
                if !attr.path().is_ident("module") {
                    return false;
                }
                let mut found_skip = false;
                let _ = attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("skip") {
                        found_skip = true;
                    }
                    Ok(())
                });
                found_skip
            });

            if should_skip {
                let empty_arm = quote! { Self::#variant_ident(_) => {} };
                param_arms.extend(empty_arm.clone());
                param_mut_arms.extend(empty_arm.clone());
                buffer_arms.extend(empty_arm.clone());
                buffer_mut_arms.extend(empty_arm.clone());
                state_arms.extend(empty_arm.clone());
                state_mut_arms.extend(empty_arm.clone());
                module_arms.extend(empty_arm.clone());
                module_mut_arms.extend(empty_arm.clone());
                continue;
            }

            // each variant must be a single-field tuple variant
            match &variant.fields {
                syn::Fields::Unnamed(fields) if fields.unnamed.len() == 1 => {}
                _ => {
                    return syn::Error::new_spanned(
                        variant,
                        "Enum variants in a Module must have exactly one unnamed field (e.g., `Gelu(Gelu)`).",
                    )
                    .to_compile_error()
                    .into();
                }
            }

            param_arms.extend(quote! {
                Self::#variant_ident(inner) => {
                    visitor.enter_submodule(#variant_name_str, inner);
                    #luma::Module::visit_param(inner, visitor)?;
                    visitor.exit_submodule(#variant_name_str, inner);
                }
            });

            param_mut_arms.extend(quote! {
                Self::#variant_ident(inner) => {
                    visitor.enter_submodule(#variant_name_str, inner);
                    #luma::Module::visit_param_mut(inner, visitor)?;
                    visitor.exit_submodule(#variant_name_str, inner);
                }
            });

            buffer_arms.extend(quote! {
                Self::#variant_ident(inner) => {
                    visitor.enter_submodule(#variant_name_str, inner);
                    #luma::Module::visit_buffer(inner, visitor)?;
                    visitor.exit_submodule(#variant_name_str, inner);
                }
            });

            buffer_mut_arms.extend(quote! {
                Self::#variant_ident(inner) => {
                    visitor.enter_submodule(#variant_name_str, inner);
                    #luma::Module::visit_buffer_mut(inner, visitor)?;
                    visitor.exit_submodule(#variant_name_str, inner);
                }
            });

            state_arms.extend(quote! {
                Self::#variant_ident(inner) => {
                    visitor.enter_submodule(#variant_name_str, inner);
                    #luma::Module::visit_state(inner, visitor)?;
                    visitor.exit_submodule(#variant_name_str, inner);
                }
            });

            state_mut_arms.extend(quote! {
                Self::#variant_ident(inner) => {
                    visitor.enter_submodule(#variant_name_str, inner);
                    #luma::Module::visit_state_mut(inner, visitor)?;
                    visitor.exit_submodule(#variant_name_str, inner);
                }
            });

            module_arms.extend(quote! {
                Self::#variant_ident(inner) => {
                    visitor.enter_submodule(#variant_name_str, inner)?;
                    #luma::Module::visit_module(inner, visitor)?;
                    visitor.exit_submodule(#variant_name_str, inner)?;
                }
            });

            module_mut_arms.extend(quote! {
                Self::#variant_ident(inner) => {
                    visitor.enter_submodule(#variant_name_str, inner)?;
                    #luma::Module::visit_module_mut(inner, visitor)?;
                    visitor.exit_submodule(#variant_name_str, inner)?;
                }
            });
        }
    }

    // ---- codegen -----------------------------------------------------------
    let codegen = quote! {
        impl #impl_generics_tokens #luma::Module<#device_type> for #name #ty_generics #where_clause {

            fn visit_param<Visitor: #luma::TensorVisitor<#device_type>>(
                &self,
                visitor: &mut Visitor,
            ) -> Result<(), Visitor::Error> {
                match self { #param_arms }
                Ok(())
            }

            fn visit_param_mut<Visitor: #luma::TensorVisitorMut<#device_type>>(
                &mut self,
                visitor: &mut Visitor,
            ) -> Result<(), Visitor::Error> {
                match self { #param_mut_arms }
                Ok(())
            }

            fn visit_buffer<Visitor: #luma::TensorVisitor<#device_type>>(
                &self,
                visitor: &mut Visitor,
            ) -> Result<(), Visitor::Error> {
                match self { #buffer_arms }
                Ok(())
            }

            fn visit_buffer_mut<Visitor: #luma::TensorVisitorMut<#device_type>>(
                &mut self,
                visitor: &mut Visitor,
            ) -> Result<(), Visitor::Error> {
                match self { #buffer_mut_arms }
                Ok(())
            }

            fn visit_state<Visitor: #luma::TensorVisitor<#device_type>>(
                &self,
                visitor: &mut Visitor,
            ) -> Result<(), Visitor::Error> {
                match self { #state_arms }
                Ok(())
            }

            fn visit_state_mut<Visitor: #luma::TensorVisitorMut<#device_type>>(
                &mut self,
                visitor: &mut Visitor,
            ) -> Result<(), Visitor::Error> {
                match self { #state_mut_arms }
                Ok(())
            }

            fn visit_module<Visitor: #luma::ModuleVisitor<#device_type>>(
                &self,
                visitor: &mut Visitor,
            ) -> Result<(), Visitor::Error> {
                visitor.visit_module(self)?;
                match self { #module_arms }
                visitor.visit_module_end(self)?;
                Ok(())
            }

            fn visit_module_mut<Visitor: #luma::ModuleVisitorMut<#device_type>>(
                &mut self,
                visitor: &mut Visitor,
            ) -> Result<(), Visitor::Error> {
                visitor.visit_module_mut(self)?;
                match self { #module_mut_arms }
                visitor.visit_module_mut_end(self)?;
                Ok(())
            }

            #extra_display_fn
            #set_train_fn
            #reset_fn
        }

        impl #impl_generics_tokens std::fmt::Display for #name #ty_generics #where_clause {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                use #luma::Module;
                write!(f, "{}", self.display())
            }
        }
    };

    codegen
}
