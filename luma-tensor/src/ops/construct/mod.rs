pub mod bool;
pub mod float;
pub mod from_element;
mod helpers;
pub mod int;
pub mod into_tensor;
pub mod options;

pub use into_tensor::IntoTensor;
pub use options::BytesDTypeKind;
pub use options::ConstructDTypeKind;
pub use options::TensorCreationOptions;

/// Default float precision when unspecified.
pub const DEFAULT_FLOAT: crate::DType = crate::DType::F32;
/// Default int precision.
pub const DEFAULT_INT: crate::DType = crate::DType::I32;
