use std::marker::PhantomData;

use luma_tensor::{Device, TransferDTypeKind};

use super::module::Module;
use crate::{Buffer, NnResult, Parameter, VisitorDispatch};

// ============================================================================================ //
//                        ToDevice
// ============================================================================================ //

/// Move a module to another device, returning the module re-instantiated on
/// the target device type.
///
/// `Module<D>` cannot express this as a trait method (its `Self` carries the
/// device `D`), so this is a separate trait whose [`ToDevice::Output`] names
/// the target instantiation, e.g. `Linear<Cpu>` → `Linear<Cuda>`.
///
/// Implemented by:
/// - the `#[derive(Module)]` macro, which walks every field: sub-modules /
///   `Parameter` / `Buffer` fields are recursively transferred, fields marked
///   `#[module(skip)]` are cloned (they must be `Clone`, or `PhantomData`
///   which is reset to `Default`);
/// - leaf impls here for [`Parameter`] and [`Buffer`];
/// - container impls here for `Option<M>`, `Vec<M>` and `PhantomData<D>`.
pub trait ToDevice<D2: Device>: Sized {
    type Output: Module<D2>;

    fn to_device(&self, device: &D2) -> NnResult<Self::Output>;
}

// ---- leaf: Parameter --------------------------------------------------------

impl<D: Device, D2: Device> ToDevice<D2> for Parameter<D> {
    type Output = Parameter<D2>;

    fn to_device(&self, device: &D2) -> NnResult<Self::Output> {
        Ok(Parameter(self.0.to_device(device)?))
    }
}

// ---- leaf: Buffer -----------------------------------------------------------

impl<D: Device, D2: Device, K: TransferDTypeKind<D, D2> + VisitorDispatch<D2>> ToDevice<D2> for Buffer<D, K> {
    type Output = Buffer<D2, K>;

    fn to_device(&self, device: &D2) -> NnResult<Self::Output> {
        Ok(Buffer(self.0.to_device(device)?))
    }
}

// ---- containers -------------------------------------------------------------

impl<D2: Device, M: ToDevice<D2>> ToDevice<D2> for Option<M> {
    type Output = Option<M::Output>;

    fn to_device(&self, device: &D2) -> NnResult<Self::Output> {
        match self {
            Some(m) => Ok(Some(m.to_device(device)?)),
            None => Ok(None),
        }
    }
}

impl<D2: Device, M: ToDevice<D2>> ToDevice<D2> for Vec<M> {
    type Output = Vec<M::Output>;

    fn to_device(&self, device: &D2) -> NnResult<Self::Output> {
        self.iter().map(|m| m.to_device(device)).collect()
    }
}

impl<D: Device, D2: Device> ToDevice<D2> for PhantomData<D> {
    type Output = PhantomData<D2>;

    fn to_device(&self, _device: &D2) -> NnResult<Self::Output> {
        Ok(PhantomData)
    }
}
