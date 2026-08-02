use super::Shape;
use crate::Result;
use crate::{DTypeKind, Device, Error};
use std::fmt::Display;

pub struct DimCoordinates {
    shape: Vec<usize>,
    current: Vec<usize>,
    done: bool,
}

impl DimCoordinates {
    pub fn from_shape(shape: &Shape) -> Self {
        let rank = shape.rank();
        Self { shape: shape.dims().to_vec(), current: vec![0; rank], done: shape.is_scalar() }
    }
}

impl Iterator for DimCoordinates {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.current.clone();

        for i in (0..self.current.len()).rev() {
            self.current[i] += 1;
            if self.current[i] < self.shape[i] {
                break;
            } else {
                self.current[i] = 0;
                if i == 0 {
                    self.done = true;
                }
            }
        }

        Some(result)
    }
}

pub struct DimNCoordinates<const N: usize> {
    shape: [usize; N],
    current: [usize; N],
    done: bool,
}

impl<const N: usize> DimNCoordinates<N> {
    pub fn from_shape(from_shape: &Shape) -> Result<Self> {
        if from_shape.rank() == N {
            let mut shape = [0usize; N];
            for i in 0..N {
                shape[i] = from_shape.dims()[i];
            }

            let current = [0usize; N];
            Ok(Self { shape, current, done: N == 0 })
        } else {
            Err(Error::UnexpectedNumberOfDims { expected: N, got: from_shape.rank(), shape: Shape::from(from_shape.dims()) })?
        }
    }
}

impl<const N: usize> Iterator for DimNCoordinates<N> {
    type Item = [usize; N];
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.current;

        for i in (0..N).rev() {
            self.current[i] += 1;
            if self.current[i] < self.shape[i] {
                break;
            } else {
                self.current[i] = 0;
                if i == 0 {
                    self.done = true;
                }
            }
        }

        Some(result)
    }
}

impl<const C: usize> From<&[usize; C]> for Shape {
    fn from(dims: &[usize; C]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}

impl From<&Vec<usize>> for Shape {
    fn from(dims: &Vec<usize>) -> Self {
        Self(dims.clone())
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&Shape> for Shape {
    fn from(shape: &Shape) -> Self {
        Self(shape.0.to_vec())
    }
}

impl From<usize> for Shape {
    fn from(d1: usize) -> Self {
        Self([d1].to_vec())
    }
}

impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Self(vec![])
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        for (i, dim) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        if self.0.len() == 1 {
            write!(f, ",")?;
        }
        write!(f, ")")
    }
}

macro_rules! impl_from_tuple {
    ($tuple:ty, $($index:tt),+) => {
        impl From<$tuple> for Shape {
            fn from(d: $tuple) -> Self {
                Self([$(d.$index,)+].to_vec())
            }
        }
    };
}

impl_from_tuple!((usize,), 0);
impl_from_tuple!((usize, usize), 0, 1);
impl_from_tuple!((usize, usize, usize), 0, 1, 2);
impl_from_tuple!((usize, usize, usize, usize), 0, 1, 2, 3);
impl_from_tuple!((usize, usize, usize, usize, usize), 0, 1, 2, 3, 4);
impl_from_tuple!((usize, usize, usize, usize, usize, usize), 0, 1, 2, 3, 4, 5);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum D {
    Minus1,
    Minus2,
    Minus(usize),
    Index(usize),
}

impl D {
    pub fn to_real_index(&self, size: usize, op: &'static str) -> Result<usize> {
        match self {
            Self::Minus1 if size >= 1 => Ok(size - 1),
            Self::Minus2 if size >= 2 => Ok(size - 2),
            Self::Minus(u) if *u > 0 && size >= *u => Ok(size - *u),
            Self::Index(u) if *u < size => Ok(*u),
            _ => Err(crate::Error::DimSizeOutOfRange { size, op })?,
        }
    }
}

impl Display for D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Minus(n) => writeln!(f, "-{}", n),
            Self::Minus1 => writeln!(f, "-1"),
            Self::Minus2 => writeln!(f, "-2"),
            Self::Index(n) => writeln!(f, "{}", n),
        }
    }
}

impl D {
    fn out_of_range(&self, shape: &Shape, op: &'static str) -> Error {
        let dim = match self {
            Self::Minus1 => -1,
            Self::Minus2 => -2,
            Self::Minus(u) => -(*u as i32),
            Self::Index(u) => *u as i32,
        };
        Error::DimOutOfRange { shape: shape.clone(), dim, op }
    }
}

macro_rules! extract_dims {
    ($fn_name:ident, $cnt:tt, $dims:expr, $out_type:ty) => {
        pub fn $fn_name(dims: &[usize]) -> Result<$out_type> {
            if dims.len() != $cnt {
                Err(Error::UnexpectedNumberOfDims { expected: $cnt, got: dims.len(), shape: Shape::from(dims) })?
            } else {
                Ok($dims(dims))
            }
        }

        impl Shape {
            pub fn $fn_name(&self) -> Result<$out_type> {
                $fn_name(self.0.as_slice())
            }
        }

        impl<D: Device, K: DTypeKind<D>> crate::Tensor<D, K> {
            pub fn $fn_name(&self) -> Result<$out_type> {
                self.shape().$fn_name()
            }
        }

        impl std::convert::TryInto<$out_type> for Shape {
            type Error = crate::Error;
            fn try_into(self) -> crate::Result<$out_type> {
                self.$fn_name()
            }
        }
    };
}

extract_dims!(dims0, 0, |_: &[usize]| (), ());
extract_dims!(dims1, 1, |d: &[usize]| d[0], usize);
extract_dims!(dims2, 2, |d: &[usize]| (d[0], d[1]), (usize, usize));
extract_dims!(dims3, 3, |d: &[usize]| (d[0], d[1], d[2]), (usize, usize, usize));
extract_dims!(dims4, 4, |d: &[usize]| (d[0], d[1], d[2], d[3]), (usize, usize, usize, usize));
extract_dims!(dims5, 5, |d: &[usize]| (d[0], d[1], d[2], d[3], d[4]), (usize, usize, usize, usize, usize));

pub trait Dim: Copy {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize>;
    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize>;
}

impl Dim for usize {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let dim = *self;
        if dim >= shape.rank() { Err(Error::DimOutOfRange { shape: shape.clone(), dim: dim as i32, op })? } else { Ok(dim) }
    }

    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let dim = *self;
        if dim > shape.rank() { Err(Error::DimOutOfRange { shape: shape.clone(), dim: dim as i32, op })? } else { Ok(dim) }
    }
}

impl Dim for D {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let rank = shape.rank();
        match self {
            Self::Minus1 if rank >= 1 => Ok(rank - 1),
            Self::Minus2 if rank >= 2 => Ok(rank - 2),
            Self::Minus(u) if *u > 0 && rank >= *u => Ok(rank - *u),
            Self::Index(u) => u.to_index(shape, op),
            _ => Err(self.out_of_range(shape, op))?,
        }
    }

    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let rank = shape.rank();
        match self {
            Self::Minus1 => Ok(rank),
            Self::Minus2 if rank >= 1 => Ok(rank - 1),
            Self::Minus(u) if *u > 0 && rank + 1 >= *u => Ok(rank + 1 - *u),
            Self::Index(u) => u.to_index_plus_one(shape, op),
            _ => Err(self.out_of_range(shape, op))?,
        }
    }
}

pub trait Dims {
    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>>;

    fn check_indexes(dims: &[usize], shape: &Shape, op: &'static str) -> Result<()> {
        for (i, &dim) in dims.iter().enumerate() {
            if dims[..i].contains(&dim) {
                return Err(Error::DuplicateDimIndex { shape: shape.clone(), dims: dims.to_vec(), op })?;
            }
            if dim >= shape.rank() {
                return Err(Error::DimOutOfRange { shape: shape.clone(), dim: dim as i32, op })?;
            }
        }
        Ok(())
    }
}

impl Dims for Vec<usize> {
    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        Self::check_indexes(&self, shape, op)?;
        Ok(self)
    }
}

impl<const N: usize> Dims for [usize; N] {
    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        Self::check_indexes(&self, shape, op)?;
        Ok(self.to_vec())
    }
}

impl Dims for &[usize] {
    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        Self::check_indexes(self, shape, op)?;
        Ok(self.to_vec())
    }
}

impl Dims for () {
    fn to_indexes(self, _: &Shape, _: &'static str) -> Result<Vec<usize>> {
        Ok(vec![])
    }
}

impl<Di: Dim + Sized> Dims for Di {
    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let dim = self.to_index(shape, op)?;
        Ok([dim].to_vec())
    }
}

impl<D1: Dim, D2: Dim> Dims for (D1, D2) {
    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        Ok([d0, d1].to_vec())
    }
}

impl<D1: Dim, D2: Dim, D3: Dim> Dims for (D1, D2, D3) {
    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        let d2 = self.2.to_index(shape, op)?;
        Ok([d0, d1, d2].to_vec())
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim> Dims for (D1, D2, D3, D4) {
    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        let d2 = self.2.to_index(shape, op)?;
        let d3 = self.3.to_index(shape, op)?;
        Ok([d0, d1, d2, d3].to_vec())
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim> Dims for (D1, D2, D3, D4, D5) {
    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        let d2 = self.2.to_index(shape, op)?;
        let d3 = self.3.to_index(shape, op)?;
        let d4 = self.4.to_index(shape, op)?;
        Ok([d0, d1, d2, d3, d4].to_vec())
    }
}
