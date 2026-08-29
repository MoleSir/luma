//! Cross-device transfer: move a tensor to another device, e.g. `Cpu` → `Cuda`.

use std::any::TypeId;

#[cfg(feature = "cuda")]
use crate::Cuda;
use crate::ops::construct::BytesDTypeKind;
use crate::{Bool, Cpu, DTypeKind, Device, Float, Int, Tensor};

// ============================================================================
//    TransferDTypeKind: kind dispatch for `to_device`
// ============================================================================

/// Kind-level dispatch for `Tensor::to_device`.
///
/// Implemented for the three closed kinds ([`Float`], [`Int`], [`Bool`]) for
/// every source/target device pair. The transfer is a bytes round-trip in
/// logical order (`to_bytes` → `from_bytes`), so the result is contiguous.
pub trait TransferDTypeKind<D: Device, D2: Device>: DTypeKind<D> + DTypeKind<D2> + BytesDTypeKind<D> + BytesDTypeKind<D2> {
    fn transfer(src: &Tensor<D, Self>, device: &D2) -> crate::Result<Tensor<D2, Self>>;
}

impl<D: Device, D2: Device> TransferDTypeKind<D, D2> for Float {
    fn transfer(src: &Tensor<D, Float>, device: &D2) -> crate::Result<Tensor<D2, Float>> {
        let bytes = src.to_bytes()?;
        let out = Tensor::from_bytes(bytes, src.shape().clone(), (device, src.dtype()))?;
        // The autograd graph is single-device (`Op<D>` holds same-device
        // inputs), so the transfer cannot record a node: the result is a fresh
        // leaf. Preserve the trainability flag so a moved parameter keeps
        // accumulating gradients.
        out.set_requires_grad(src.requires_grad());
        Ok(out)
    }
}

impl<D: Device, D2: Device> TransferDTypeKind<D, D2> for Int {
    fn transfer(src: &Tensor<D, Int>, device: &D2) -> crate::Result<Tensor<D2, Int>> {
        let bytes = src.to_bytes()?;
        Tensor::from_bytes(bytes, src.shape().clone(), (device, src.dtype()))
    }
}

impl<D: Device, D2: Device> TransferDTypeKind<D, D2> for Bool {
    fn transfer(src: &Tensor<D, Bool>, device: &D2) -> crate::Result<Tensor<D2, Bool>> {
        let bytes = src.to_bytes()?;
        Tensor::from_bytes(bytes, src.shape().clone(), (device, src.dtype()))
    }
}

// ============================================================================
//    Public API
// ============================================================================

impl<D: Device, K: DTypeKind<D>> Tensor<D, K> {
    /// Move the tensor to another device.
    ///
    /// - Same device: returns this handle unchanged (shared `Arc`, O(1)) —
    ///   generic code can call this unconditionally.
    /// - Cross device: deep-copies the data (`to_bytes`/`from_bytes`), dtype
    ///   unchanged, result contiguous. For `Float` tensors the autograd graph
    ///   is severed (the result is a leaf) but `requires_grad` is preserved.
    /// - Meta tensors (no storage) error with [`Error::MetaTensor`](crate::Error::MetaTensor).
    pub fn transfer<D2: Device>(&self, device: &D2) -> crate::Result<Tensor<D2, K>>
    where
        K: TransferDTypeKind<D, D2>,
    {
        if TypeId::of::<D>() == TypeId::of::<D2>() {
            // SAFETY: `TypeId` equality means `D` and `D2` are the same concrete
            // type (`Device: 'static`), so these casts are type puns between
            // identical types with identical layout — no aliasing or validity
            // concerns beyond a plain clone.
            let target: &D = unsafe { &*(device as *const D2 as *const D) };
            if self.device().same_device(target) {
                let p = self as *const Tensor<D, K> as *const Tensor<D2, K>;
                return Ok(unsafe { (*p).clone() });
            }
        }
        // Cross-device copy (covers `Cuda` → `Cuda` with different ordinals,
        // which goes through the host — there is no DtoD primitive yet).
        K::transfer(self, device)
    }

    /// Convenience for `to_device(&Cpu)`.
    pub fn cpu(&self) -> crate::Result<Tensor<Cpu, K>>
    where
        K: TransferDTypeKind<D, Cpu>,
    {
        self.transfer(&Cpu)
    }

    /// Convenience for `to_device(&Cuda::new(ordinal))`.
    #[cfg(feature = "cuda")]
    pub fn cuda(&self, ordinal: usize) -> crate::Result<Tensor<Cuda, K>>
    where
        K: TransferDTypeKind<D, Cuda>,
    {
        self.to_device(&Cuda::new(ordinal)?)
    }
}
