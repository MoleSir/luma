/// Validate the generic parameters on a `#[derive(Module)]` item.
///
/// Rules:
/// - At most **one** generic type parameter.
/// - If present, that parameter **must** be bounded by `Device`
///   (e.g. `D: Device` or `D: luma_tensor::Device`).
///
/// Returns `Some(ident)` when the user supplied a device generic, or `None`
/// when there are no generics (the macro will inject its own `D: Device`).
pub fn validate_and_extract_device_generic(generics: &syn::Generics) -> syn::Result<Option<&syn::Ident>> {
    let params: Vec<_> = generics.params.iter().collect();

    // 1. check count
    if params.len() > 1 {
        return Err(syn::Error::new_spanned(generics, "Module allows at most one generic parameter for Device (e.g., <D: Device>)."));
    }

    // no params
    if params.is_empty() {
        return Ok(None);
    }

    // 2. the param must be a type parameter bounded by `Device`
    let param = params[0];
    let type_param = match param {
        syn::GenericParam::Type(tp) => tp,
        _ => return Err(syn::Error::new_spanned(param, "Module generic parameter must be a type parameter, not lifetime or const.")),
    };

    let has_device_bound = type_param.bounds.iter().any(|bound| {
        if let syn::TypeParamBound::Trait(trait_bound) = bound {
            if let Some(segment) = trait_bound.path.segments.last() {
                return segment.ident == "Device";
            }
        }
        false
    });

    if !has_device_bound {
        return Err(syn::Error::new_spanned(
            type_param,
            format!("The generic parameter '{}' must be bound by Device (e.g., {}: Device).", type_param.ident, type_param.ident),
        ));
    }

    Ok(Some(&type_param.ident))
}

/// Whether a field/variant-payload type is `PhantomData` (any path, e.g.
/// `PhantomData<D>`, `std::marker::PhantomData<D>`).
///
/// `PhantomData` is a device-erasable marker: when cloning a module for
/// `to_device` it must be reset to `Default::default()` instead of cloned,
/// since the target module expects `PhantomData<D2>` rather than
/// `PhantomData<D>`.
pub fn is_phantom_data(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Path(type_path) => type_path.path.segments.last().map(|seg| seg.ident == "PhantomData").unwrap_or(false),
        syn::Type::Group(group) => is_phantom_data(&group.elem),
        _ => false,
    }
}
