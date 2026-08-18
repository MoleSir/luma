use luma_tensor::{Device, Tensor, ops::ShapeDTypeKind};
use std::{convert::Infallible, fmt::Display, marker::PhantomData};

pub trait Batcher {
    type Item;
    type Output;
    type Error: Display;

    fn batch(&self, items: Vec<Self::Item>) -> Result<Self::Output, Self::Error>;
}

#[derive(Default)]
pub struct TensorPairBatcher<D, K1, K2>(PhantomData<D>, PhantomData<K1>, PhantomData<K2>);

impl<D, K1, K2> TensorPairBatcher<D, K1, K2> {
    pub fn new() -> Self {
        Self(Default::default(), Default::default(), Default::default())
    }
}

impl<D: Device, K1: ShapeDTypeKind<D>, K2: ShapeDTypeKind<D>> Batcher for TensorPairBatcher<D, K1, K2> {
    type Item = (Tensor<D, K1>, Tensor<D, K2>);
    type Output = (Tensor<D, K1>, Tensor<D, K2>);
    type Error = luma_tensor::Error;

    fn batch(&self, items: Vec<(Tensor<D, K1>, Tensor<D, K2>)>) -> Result<Self::Output, Self::Error> {
        let (xs, ys): (Vec<_>, Vec<_>) = items.into_iter().unzip();
        let xs = Tensor::stack(&xs, 0)?;
        let ys = Tensor::stack(&ys, 0)?;
        Ok((xs, ys))
    }
}

#[derive(Default)]
pub struct NoBatcher<T>(PhantomData<T>);

impl<T> Batcher for NoBatcher<T> {
    type Error = Infallible;
    type Item = T;
    type Output = Vec<T>;

    fn batch(&self, items: Vec<Self::Item>) -> Result<Self::Output, Self::Error> {
        Ok(items)
    }
}

#[derive(Default)]
pub struct PairBatcher<T1, T2>(PhantomData<T1>, PhantomData<T2>);

impl<T1, T2> Batcher for PairBatcher<T1, T2> {
    type Error = Infallible;
    type Item = (T1, T2);
    type Output = (Vec<T1>, Vec<T2>);

    fn batch(&self, items: Vec<Self::Item>) -> Result<Self::Output, Self::Error> {
        let (xs, ys): (Vec<_>, Vec<_>) = items.into_iter().unzip();
        Ok((xs, ys))
    }
}
