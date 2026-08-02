mod dim;
mod layout;
mod shape;
pub use dim::*;
pub use layout::*;
pub use shape::*;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use crate::dtype::Storage;
use crate::{Cpu, DTypeKind, Device, Float};

/// Unique, monotonically increasing tensor identity. Used as a map key during
/// autograd (a tensor's storage/layout may change, but its id never does).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(usize);

impl TensorId {
    pub(crate) fn new() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    pub fn as_usize(&self) -> usize {
        self.0
    }
}

/// The reference-counted tensor handle. Cheap to clone (bumps an `Arc`).
///
/// Two generic parameters:
/// - `D: Device` — where the data lives (e.g. `Cpu`); selects the storage type.
/// - `K: DTypeKind<D>` — the *kind* (`Float`/`Int`/`Bool`), a compile-time marker.
///   The concrete precision within a kind (`f32` vs `f64`, ...) is a runtime
///   [`DType`] carried inside the tensor.
///
/// `Float` is the default kind so `Tensor<Cpu>` means a float tensor.
pub struct Tensor<D: Device = Cpu, K: DTypeKind<D> = Float>(pub(crate) Arc<TensorImpl<D, K>>);

pub struct TensorImpl<D, K>
where
    D: Device,
    K: DTypeKind<D>,
{
    pub(crate) id: TensorId,
    
    pub(crate) storage: Option<Arc<RwLock<K::Storage>>>,
    pub(crate) layout: Layout,
    
    pub(crate) meta: K::Meta,

    pub(crate) dtype: K::DType,
    pub(crate) device: D, 
}

impl<D: Device, K: DTypeKind<D>> Clone for Tensor<D, K> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<D: Device, K: DTypeKind<D>> Tensor<D, K> {
    pub fn id(&self) -> TensorId {
        self.0.id
    }

    pub fn dtype(&self) -> K::DType {
        self.0.dtype
    }

    pub fn device(&self) -> &D {
        &self.0.device
    }

    pub fn layout(&self) -> &Layout {
        &self.0.layout
    }

    pub fn shape(&self) -> &Shape {
        self.0.layout.shape()
    }

    pub fn stride(&self) -> &[usize] {
        self.0.layout.stride()
    }

    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    pub fn dims(&self) -> &[usize] {
        self.shape().dims()
    }

    pub fn dim<Di: Dim>(&self, dim: Di) -> crate::Result<usize> {
        self.shape().dim(dim)
    }

    pub fn element_count(&self) -> usize {
        self.shape().element_count()
    }

    pub fn is_contiguous(&self) -> bool {
        self.0.layout.is_contiguous()
    }

    /// A meta tensor carries shape but no storage.
    pub fn is_meta(&self) -> bool {
        self.0.storage.is_none()
    }

    /// Access the underlying storage, erroring on a meta tensor.
    pub(crate) fn storage(&self) -> crate::Result<&Arc<RwLock<K::Storage>>> {
        self.0.storage.as_ref().ok_or(crate::Error::MetaTensor)
    }

    /// Read-lock the storage, erroring on a meta tensor.
    pub(crate) fn storage_read(&self) -> crate::Result<std::sync::RwLockReadGuard<'_, K::Storage>> {
        Ok(self.storage()?.read().expect("storage read lock"))
    }

    /// Write-lock the storage, erroring on a meta tensor.
    pub(crate) fn storage_write(&self) -> crate::Result<std::sync::RwLockWriteGuard<'_, K::Storage>> {
        Ok(self.storage()?.write().expect("storage write lock"))
    }

    /// Verify `self` and `rhs` have the same shape; return a reference to it.
    pub(crate) fn same_shape(&self, rhs: &Self, op: &'static str) -> crate::Result<&Shape> {
        if self.shape() != rhs.shape() {
            return Err(crate::Error::ShapeMismatchBinaryOp { lhs: self.shape().clone(), rhs: rhs.shape().clone(), op });
        }
        Ok(self.shape())
    }

    // pub(crate) fn same_device(&self, rhs: &Self, op: &'static str) -> crate::Result<&D> {
    //     if self.device() != rhs.device() {
    //         return Err(crate::Error::DeviceMismatchBinaryOp { lhs: self.device().name(), rhs: rhs.device().name(), op });
    //     }
    //     Ok(self.device())
    // }

    /// Build a tensor from a freshly-computed storage + layout + autograd meta.
    pub(crate) fn from_storage<L: Into<Layout>>(storage: K::Storage, layout: L, meta: K::Meta) -> Self {
        let device = storage.device().clone();
        Tensor(Arc::new(TensorImpl {
            id: TensorId::new(),
            dtype: storage.dtype(),
            storage: Some(Arc::new(RwLock::new(storage))),
            layout: layout.into(),
            meta,
            device,
        }))
    }

    /// Build a view that shares `self`'s storage `Arc` but installs a new layout
    /// and meta (used by transpose/slice/broadcast/etc.).
    pub(crate) fn share_storage<L: Into<Layout>>(&self, layout: L, meta: K::Meta) -> Self {
        Tensor(Arc::new(TensorImpl {
            id: TensorId::new(),
            dtype: self.0.dtype,
            storage: self.0.storage.clone(),
            layout: layout.into(),
            meta,
            device: self.device().clone(),
        }))
    }
}

impl<D: Device, K: DTypeKind<D>> std::hash::Hash for Tensor<D, K> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.id.hash(state)
    }
}

impl<D: Device, K: DTypeKind<D>> PartialEq for Tensor<D, K> {
    fn eq(&self, other: &Self) -> bool {
        self.0.id == other.0.id
    }
}

impl<D: Device, K: DTypeKind<D>> Eq for Tensor<D, K> {}

impl<D: Device, K: DTypeKind<D>> AsRef<Tensor<D, K>> for Tensor<D, K> {
    fn as_ref(&self) -> &Tensor<D, K> {
        self
    }
}
