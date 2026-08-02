use crate::{DTypeKind, Device, Float, Int, Layout, Shape, Tensor, TensorMeta, Storage};

pub trait MatmulDTypeKind<D: Device>: DTypeKind<D> + Sized {
    fn matmul_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: &Self::Storage, rhs_l: &Layout) -> crate::Result<(Self::Storage, Shape)>;
}

impl<D: Device> MatmulDTypeKind<D> for Float {
    #[inline]
    fn matmul_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: &Self::Storage, rhs_l: &Layout) -> crate::Result<(Self::Storage, Shape)> {
        D::f_matmul(lhs, lhs_l, rhs, rhs_l)
    }
}

impl<D: Device> MatmulDTypeKind<D> for Int {
    #[inline]
    fn matmul_dispatch(lhs: &Self::Storage, lhs_l: &Layout, rhs: &Self::Storage, rhs_l: &Layout) -> crate::Result<(Self::Storage, Shape)> {
        D::i_matmul(lhs, lhs_l, rhs, rhs_l)
    }
}

impl<D: Device, K: MatmulDTypeKind<D>> Tensor<D, K> {
    pub fn matmul(&self, rhs: &Self) -> crate::Result<Self> {
        let (storage, shape) = K::matmul_dispatch(&*self.storage_read()?, self.layout(), &*rhs.storage_read()?, rhs.layout())?;
        let meta = K::Meta::on_matmul(self, rhs);
        assert_eq!(self.dtype(), storage.dtype());
        Ok(Self::from_storage(storage, shape, meta))
    }
}
