use super::shape_infer::matmul_out_shape;
use crate::{DTypeKind, Device, Float, Int, Layout, Shape, Storage, Tensor, TensorMeta};

pub trait MatmulDTypeKind<D: Device>: DTypeKind<D> + Sized {
    fn matmul_dispatch(
        lhs: &Self::Storage,
        lhs_l: &Layout,
        rhs: &Self::Storage,
        rhs_l: &Layout,
        out_shape: &Shape,
    ) -> crate::Result<Self::Storage>;
}

impl<D: Device> MatmulDTypeKind<D> for Float {
    #[inline]
    fn matmul_dispatch(
        lhs: &Self::Storage,
        lhs_l: &Layout,
        rhs: &Self::Storage,
        rhs_l: &Layout,
        out_shape: &Shape,
    ) -> crate::Result<Self::Storage> {
        D::f_matmul(lhs, lhs_l, rhs, rhs_l, out_shape)
    }
}

impl<D: Device> MatmulDTypeKind<D> for Int {
    #[inline]
    fn matmul_dispatch(
        lhs: &Self::Storage,
        lhs_l: &Layout,
        rhs: &Self::Storage,
        rhs_l: &Layout,
        out_shape: &Shape,
    ) -> crate::Result<Self::Storage> {
        D::i_matmul(lhs, lhs_l, rhs, rhs_l, out_shape)
    }
}

impl<D: Device, K: MatmulDTypeKind<D>> Tensor<D, K> {
    pub fn matmul(&self, rhs: &Self) -> crate::Result<Self> {
        let out_shape = matmul_out_shape(self.shape(), rhs.shape())?;
        let storage = K::matmul_dispatch(&*self.storage_read()?, self.layout(), &*rhs.storage_read()?, rhs.layout(), &out_shape)?;
        let meta = K::Meta::on_matmul(self, rhs);
        assert_eq!(self.dtype(), storage.dtype());
        Ok(Self::from_storage(storage, out_shape, meta))
    }
}
