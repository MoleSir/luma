use crate::utils;
use proc_macro2::TokenStream;
use quote::quote;

use super::common;

pub fn generate_struct(ast: &syn::DeriveInput) -> TokenStream {
    let luma = utils::get_luma_nn_path();
    let tensor = utils::get_luma_tensor_path();
    let name = &ast.ident;

    // ---- validate device generic -------------------------------------------
    let validated_device = match common::validate_and_extract_device_generic(&ast.generics) {
        Ok(res) => res,
        Err(e) => return e.to_compile_error().into(),
    };

    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();
    let (impl_generics_tokens, device_type) = match validated_device.as_ref() {
        Some(user_device_ident) => (quote! { #impl_generics }, quote! { #user_device_ident }),
        None => (quote! { <D: #tensor::Device> }, quote! { D }),
    };

    // ---- generics for the `ToDevice<D2>` impl --------------------------------
    // The struct's own device generic plus the target device `D2`.
    let (to_device_impl_generics, target_type) = match validated_device.as_ref() {
        Some(_) => {
            let mut gens = ast.generics.clone();
            gens.params.push(syn::parse2(quote! { D2: #tensor::Device }).unwrap());
            let (impl_gens, _, _) = gens.split_for_impl();
            let impl_gens = quote::ToTokens::to_token_stream(&impl_gens);
            (impl_gens, quote! { #name::<D2> })
        }
        // No generics on the struct: `D`/`D2` exist only in the impl, and the
        // target type is the non-generic struct itself.
        None => {
            let gens: syn::Generics = syn::parse2(quote! { <D: #tensor::Device, D2: #tensor::Device> }).unwrap();
            let (impl_gens, _, _) = gens.split_for_impl();
            let impl_gens = quote::ToTokens::to_token_stream(&impl_gens);
            (impl_gens, quote! { #name })
        }
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

    // ---- extra_display -----------------------------------------------------
    let extra_display_fn = if let Some(fn_name) = custom_display_fn_name {
        quote! {
            fn extra_display(&self) -> String {
                self.#fn_name()
            }
        }
    } else {
        quote! {}
    };

    // ---- set_train ---------------------------------------------------------
    let set_train_fn = if let Some(fn_name) = custom_set_train_name {
        quote! {
            fn set_train(&mut self, mode: bool) {
                self.#fn_name(mode)
            }
        }
    } else {
        quote! {}
    };

    // ---- reset_parameters --------------------------------------------------
    let reset_fn = if let Some(fn_name) = custom_reset_name {
        quote! {
            fn reset_parameters(&mut self) -> std::result::Result<(), #luma::NnError> {
                self.#fn_name()
            }
        }
    } else {
        quote! {}
    };

    // ---- generate visitor bodies by walking fields --------------------------
    let mut param_body = quote! {};
    let mut param_mut_body = quote! {};
    let mut state_body = quote! {};
    let mut state_mut_body = quote! {};
    let mut buffer_body = quote! {};
    let mut buffer_mut_body = quote! {};
    let mut module_body = quote! {};
    let mut module_mut_body = quote! {};
    // Field initializers for the `ToDevice` impl: sub-modules/params/buffers are
    // recursively transferred, `#[module(skip)]` config fields are cloned
    // (`PhantomData` is reset to `Default` — its type parameter changes with D).
    let mut to_device_fields = quote! {};

    match &ast.data {
        syn::Data::Struct(struct_data) => {
            for field in struct_data.fields.iter() {
                let field_name = field.ident.clone().unwrap();

                // check #[module(skip)]
                let should_skip = field.attrs.iter().any(|attr| {
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
                    if common::is_phantom_data(&field.ty) {
                        to_device_fields.extend(quote! { #field_name: ::core::default::Default::default(), });
                    } else {
                        to_device_fields.extend(quote! { #field_name: ::core::clone::Clone::clone(&self.#field_name), });
                    }
                    continue;
                }

                let name_str = field_name.to_string();
                to_device_fields.extend(quote! { #field_name: #luma::ToDevice::to_device(&self.#field_name, device)?, });

                // -- param --
                let code = quote! {
                    visitor.enter_submodule(#name_str, &self.#field_name);
                    #luma::Module::visit_param(&self.#field_name, visitor)?;
                    visitor.exit_submodule(#name_str, &self.#field_name);
                };
                param_body.extend(code);

                // -- param_mut --
                let code = quote! {
                    visitor.enter_submodule(#name_str, &mut self.#field_name);
                    #luma::Module::visit_param_mut(&mut self.#field_name, visitor)?;
                    visitor.exit_submodule(#name_str, &mut self.#field_name);
                };
                param_mut_body.extend(code);

                // -- buffer --
                let code = quote! {
                    visitor.enter_submodule(#name_str, &self.#field_name);
                    #luma::Module::visit_buffer(&self.#field_name, visitor)?;
                    visitor.exit_submodule(#name_str, &self.#field_name);
                };
                buffer_body.extend(code);

                // -- buffer_mut --
                let code = quote! {
                    visitor.enter_submodule(#name_str, &mut self.#field_name);
                    #luma::Module::visit_buffer_mut(&mut self.#field_name, visitor)?;
                    visitor.exit_submodule(#name_str, &mut self.#field_name);
                };
                buffer_mut_body.extend(code);

                // -- state --
                let code = quote! {
                    visitor.enter_submodule(#name_str, &self.#field_name);
                    #luma::Module::visit_state(&self.#field_name, visitor)?;
                    visitor.exit_submodule(#name_str, &self.#field_name);
                };
                state_body.extend(code);

                // -- state_mut --
                let code = quote! {
                    visitor.enter_submodule(#name_str, &mut self.#field_name);
                    #luma::Module::visit_state_mut(&mut self.#field_name, visitor)?;
                    visitor.exit_submodule(#name_str, &mut self.#field_name);
                };
                state_mut_body.extend(code);

                // -- module --
                let code = quote! {
                    visitor.enter_submodule(#name_str, &self.#field_name)?;
                    #luma::Module::visit_module(&self.#field_name, visitor)?;
                    visitor.exit_submodule(#name_str, &self.#field_name)?;
                };
                module_body.extend(code);

                // -- module_mut --
                let code = quote! {
                    visitor.enter_submodule(#name_str, &mut self.#field_name)?;
                    #luma::Module::visit_module_mut(&mut self.#field_name, visitor)?;
                    visitor.exit_submodule(#name_str, &mut self.#field_name)?;
                };
                module_mut_body.extend(code);
            }
        }
        syn::Data::Enum(_) => panic!("generate_struct called on enum — this is a bug"),
        syn::Data::Union(_) => panic!("Union modules aren't supported"),
    };

    // ---- codegen -----------------------------------------------------------
    let codegen = quote! {
        impl #impl_generics_tokens #luma::Module<#device_type> for #name #ty_generics #where_clause {

            fn visit_param<Visitor: #luma::TensorVisitor<#device_type>>(
                &self,
                visitor: &mut Visitor,
            ) -> Result<(), Visitor::Error> {
                #param_body
                Ok(())
            }

            fn visit_param_mut<Visitor: #luma::TensorVisitorMut<#device_type>>(
                &mut self,
                visitor: &mut Visitor,
            ) -> Result<(), Visitor::Error> {
                #param_mut_body
                Ok(())
            }

            fn visit_buffer<Visitor: #luma::TensorVisitor<#device_type>>(
                &self,
                visitor: &mut Visitor,
            ) -> Result<(), Visitor::Error> {
                #buffer_body
                Ok(())
            }

            fn visit_buffer_mut<Visitor: #luma::TensorVisitorMut<#device_type>>(
                &mut self,
                visitor: &mut Visitor,
            ) -> Result<(), Visitor::Error> {
                #buffer_mut_body
                Ok(())
            }

            fn visit_state<Visitor: #luma::TensorVisitor<#device_type>>(
                &self,
                visitor: &mut Visitor,
            ) -> Result<(), Visitor::Error> {
                #state_body
                Ok(())
            }

            fn visit_state_mut<Visitor: #luma::TensorVisitorMut<#device_type>>(
                &mut self,
                visitor: &mut Visitor,
            ) -> Result<(), Visitor::Error> {
                #state_mut_body
                Ok(())
            }

            fn visit_module<Visitor: #luma::ModuleVisitor<#device_type>>(
                &self,
                visitor: &mut Visitor,
            ) -> Result<(), Visitor::Error> {
                visitor.visit_module(self)?;
                #module_body
                visitor.visit_module_end(self)?;
                Ok(())
            }

            fn visit_module_mut<Visitor: #luma::ModuleVisitorMut<#device_type>>(
                &mut self,
                visitor: &mut Visitor,
            ) -> Result<(), Visitor::Error> {
                visitor.visit_module_mut(self)?;
                #module_mut_body
                visitor.visit_module_mut_end(self)?;
                Ok(())
            }

            #extra_display_fn
            #set_train_fn
            #reset_fn
        }

        impl #to_device_impl_generics #luma::ToDevice<D2> for #name #ty_generics #where_clause {
            type Output = #target_type;

            fn to_device(&self, device: &D2) -> #luma::NnResult<Self::Output> {
                Ok(#target_type {
                    #to_device_fields
                })
            }
        }

        impl #impl_generics_tokens std::fmt::Display for #name #ty_generics #where_clause {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                use #luma::Module;
                write!(f, "{}", self.display())
            }
        }
    };

    // panic!("{}", codegen);
    codegen
}
