use crate::{DTypeKind, Device, FloatDType, IntDType, BoolDType, Tensor};
use super::{cast::{Cast, CastDTypeKind}, TransferDTypeKind};

impl<D: Device, K: DTypeKind<D>> Tensor<D, K> {
    #[inline]
    pub fn to_dtype<T: Cast<D, K>>(&self, dtype: T) -> crate::Result<Tensor<D, T::Output>> {
        T::cast(self, dtype)
    }

    #[inline]
    pub fn to_device<D2: Device>(&self, device: &D2) -> crate::Result<Tensor<D2, K>>
    where
        K: TransferDTypeKind<D, D2>,
    {
        self.transfer(device)
    }

    #[inline]
    pub fn to<To>(&self, options: To) -> crate::Result<Tensor<To::OD, To::OK>> 
    where 
        To: TensorToOptions<D, K>
    {
        options.to(self)
    }
}

pub trait TensorToOptions<ID: Device, IK: DTypeKind<ID>> {
    type OD: Device;
    type OK: DTypeKind<Self::OD>;    
    fn to(self, tensor: &Tensor<ID, IK>) -> crate::Result<Tensor<Self::OD, Self::OK>>; 
}

impl<D1: Device, D2: Device, K: TransferDTypeKind<D1, D2>> TensorToOptions<D1, K> for &D2 {
    type OD = D2;
    type OK = K;
    #[inline]
    fn to(self, tensor: &Tensor<D1, K>) -> crate::Result<Tensor<Self::OD, Self::OK>> {
        tensor.to_device(self)
    }
}

macro_rules! impl_dtype {
    ($ty:ty) => {
        impl<D: Device, K: DTypeKind<D>> TensorToOptions<D, K> for $ty 
        where 
            Self: Cast<D, K>
        {
            type OD = D;
            type OK = <Self as Cast<D, K>>::Output;
            #[inline]
            fn to(self, tensor: &Tensor<D, K>) -> crate::Result<Tensor<Self::OD, Self::OK>> {
                tensor.to_dtype(self)
            }
        }
        
        impl<D1: Device, D2: Device, K> TensorToOptions<D1, K> for (&D2, $ty) 
        where 
            Self: Cast<D2, K>,
            K: TransferDTypeKind<D1, D2> + CastDTypeKind<D2>
        {
            type OD = D2;
            type OK = <$ty as Cast<D2, K>>::Output;
        
            fn to(self, tensor: &Tensor<D1, K>) -> crate::Result<Tensor<Self::OD, Self::OK>> {
                tensor.to_device(self.0)?.to_dtype(self.1)
            }
        }
        
        impl<D1: Device, D2: Device, K> TensorToOptions<D1, K> for ($ty, &D2) 
        where 
            Self: Cast<D2, K>,
            K: TransferDTypeKind<D1, D2> + CastDTypeKind<D2>
        {
            type OD = D2;
            type OK = <$ty as Cast<D2, K>>::Output;
        
            fn to(self, tensor: &Tensor<D1, K>) -> crate::Result<Tensor<Self::OD, Self::OK>> {
                tensor.to_device(self.1)?.to_dtype(self.0)
            }
        }
    };
} 

impl_dtype!(FloatDType);
impl_dtype!(IntDType);
impl_dtype!(BoolDType);