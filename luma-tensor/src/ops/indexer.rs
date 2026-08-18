use crate::{Bool, DTypeKind, Device, Dim, Float, Int, Layout, Shape, Tensor, TensorMeta, dtype::Storage};

pub trait IndexingDTypeKind<D: Device>: DTypeKind<D> + Sized {
    fn index_select_dispatch(
        x: &Self::Storage,
        x_l: &Layout,
        idx: &D::IntStorage,
        idx_l: &Layout,
        dim: usize,
    ) -> crate::Result<(Self::Storage, Shape)>;

    fn gather_dispatch(
        x: &Self::Storage,
        x_l: &Layout,
        idx: &D::IntStorage,
        idx_l: &Layout,
        dim: usize,
    ) -> crate::Result<(Self::Storage, Shape)>;

    fn index_add_dispatch(
        init: &Self::Storage,
        init_l: &Layout,
        idx: &D::IntStorage,
        idx_l: &Layout,
        src: &Self::Storage,
        src_l: &Layout,
        dim: usize,
    ) -> crate::Result<Self::Storage>;

    fn scatter_add_dispatch(
        init: &Self::Storage,
        init_l: &Layout,
        idx: &D::IntStorage,
        idx_l: &Layout,
        src: &Self::Storage,
        src_l: &Layout,
        dim: usize,
    ) -> crate::Result<Self::Storage>;
}

impl<D: Device> IndexingDTypeKind<D> for Float {
    fn index_select_dispatch(
        x: &Self::Storage,
        x_l: &Layout,
        idx: &D::IntStorage,
        idx_l: &Layout,
        dim: usize,
    ) -> crate::Result<(Self::Storage, Shape)> {
        D::f_index_select(x, x_l, idx, idx_l, dim)
    }

    fn gather_dispatch(
        x: &Self::Storage,
        x_l: &Layout,
        idx: &D::IntStorage,
        idx_l: &Layout,
        dim: usize,
    ) -> crate::Result<(Self::Storage, Shape)> {
        D::f_gather(x, x_l, idx, idx_l, dim)
    }

    fn index_add_dispatch(
        init: &Self::Storage,
        init_l: &Layout,
        idx: &D::IntStorage,
        idx_l: &Layout,
        src: &Self::Storage,
        src_l: &Layout,
        dim: usize,
    ) -> crate::Result<Self::Storage> {
        D::f_index_add(init, init_l, idx, idx_l, src, src_l, dim)
    }

    fn scatter_add_dispatch(
        init: &Self::Storage,
        init_l: &Layout,
        idx: &D::IntStorage,
        idx_l: &Layout,
        src: &Self::Storage,
        src_l: &Layout,
        dim: usize,
    ) -> crate::Result<Self::Storage> {
        D::f_scatter_add(init, init_l, idx, idx_l, src, src_l, dim)
    }
}

impl<D: Device> IndexingDTypeKind<D> for Int {
    fn index_select_dispatch(
        x: &Self::Storage,
        x_l: &Layout,
        idx: &D::IntStorage,
        idx_l: &Layout,
        dim: usize,
    ) -> crate::Result<(Self::Storage, Shape)> {
        D::i_index_select(x, x_l, idx, idx_l, dim)
    }

    fn gather_dispatch(
        x: &Self::Storage,
        x_l: &Layout,
        idx: &D::IntStorage,
        idx_l: &Layout,
        dim: usize,
    ) -> crate::Result<(Self::Storage, Shape)> {
        D::i_gather(x, x_l, idx, idx_l, dim)
    }

    fn index_add_dispatch(
        init: &Self::Storage,
        init_l: &Layout,
        idx: &D::IntStorage,
        idx_l: &Layout,
        src: &Self::Storage,
        src_l: &Layout,
        dim: usize,
    ) -> crate::Result<Self::Storage> {
        D::i_index_add(init, init_l, idx, idx_l, src, src_l, dim)
    }

    fn scatter_add_dispatch(
        init: &Self::Storage,
        init_l: &Layout,
        idx: &D::IntStorage,
        idx_l: &Layout,
        src: &Self::Storage,
        src_l: &Layout,
        dim: usize,
    ) -> crate::Result<Self::Storage> {
        D::i_scatter_add(init, init_l, idx, idx_l, src, src_l, dim)
    }
}

impl<D: Device, K: IndexingDTypeKind<D>> Tensor<D, K> {
    /// Select slices along `dim` at the given 1-D `indices`.
    pub fn index_select<Dm: Dim>(&self, indices: &Tensor<D, Int>, dim: Dm) -> crate::Result<Self> {
        let dim = dim.to_index(self.shape(), "index_select")?;
        let (storage, shape) =
            K::index_select_dispatch(&*self.storage_read()?, self.layout(), &*indices.storage_read()?, indices.layout(), dim)?;
        let meta = K::Meta::on_index_select(self, indices, dim);
        assert_eq!(self.dtype(), storage.dtype());
        Ok(Self::from_storage(storage, shape, meta))
    }

    /// Gather along `dim` using an index tensor of the same rank.
    pub fn gather<Dm: Dim>(&self, indices: &Tensor<D, Int>, dim: Dm) -> crate::Result<Self> {
        let dim = dim.to_index(self.shape(), "gather")?;
        let (storage, shape) = K::gather_dispatch(&*self.storage_read()?, self.layout(), &*indices.storage_read()?, indices.layout(), dim)?;
        let meta = K::Meta::on_gather(self, indices, dim);
        assert_eq!(self.dtype(), storage.dtype());
        Ok(Self::from_storage(storage, shape, meta))
    }

    /// `out = self; out[.., idx[i], ..] += src[.., i, ..]`.
    pub fn index_add<Dm: Dim>(&self, indices: &Tensor<D, Int>, src: &Tensor<D, K>, dim: Dm) -> crate::Result<Self> {
        let dim = dim.to_index(self.shape(), "index_add")?;
        let storage = K::index_add_dispatch(
            &*self.storage_read()?,
            self.layout(),
            &*indices.storage_read()?,
            indices.layout(),
            &*src.storage_read()?,
            src.layout(),
            dim,
        )?;
        let meta = K::Meta::on_index_add(self, indices, src, dim);
        assert_eq!(self.dtype(), storage.dtype());
        Ok(Self::from_storage(storage, self.shape().clone(), meta))
    }

    /// `out = self; out[.., idx[i,j,k], k] += src[i,j,k]`.
    pub fn scatter_add<Dm: Dim>(&self, indices: &Tensor<D, Int>, src: &Tensor<D, K>, dim: Dm) -> crate::Result<Self> {
        let dim = dim.to_index(self.shape(), "scatter_add")?;
        let storage = K::scatter_add_dispatch(
            &*self.storage_read()?,
            self.layout(),
            &*indices.storage_read()?,
            indices.layout(),
            &*src.storage_read()?,
            src.layout(),
            dim,
        )?;
        let meta = K::Meta::on_scatter_add(self, indices, src, dim);
        assert_eq!(self.dtype(), storage.dtype());
        Ok(Self::from_storage(storage, self.shape().clone(), meta))
    }
}

// ---- Slice -------------------------------------------------------------------

/// A slice range with `start`, optional `end` (negative = from end, `None` = to end), and `step`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slice {
    pub start: usize,
    pub end: Option<isize>,
    pub step: usize,
}

impl Slice {
    pub fn new(start: usize, end: Option<isize>, step: usize) -> Self {
        Self { start, end, step }
    }

    /// Resolve `end` against a concrete dimension size.
    pub fn resolve(&self, dim_size: usize) -> (usize, usize, usize) {
        let end_abs = match self.end {
            None => dim_size,
            Some(e) if e < 0 => {
                let abs = (-e) as usize;
                if abs > dim_size { 0 } else { dim_size - abs }
            }
            Some(e) => {
                let e = e as usize;
                if e > dim_size { dim_size } else { e }
            }
        };
        (self.start, end_abs, self.step)
    }

    pub fn len(&self) -> usize {
        self.clone().count()
    }
}

impl Iterator for Slice {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        match self.end {
            Some(end) if end < 0 => {
                let value = self.start;
                self.start += self.step;
                Some(value)
            }
            Some(end) => {
                if self.start < end as usize {
                    let value = self.start;
                    self.start += self.step;
                    Some(value)
                } else {
                    None
                }
            }
            None => {
                let value = self.start;
                self.start += self.step;
                Some(value)
            }
        }
    }
}

impl std::fmt::Display for Slice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let step_part = if self.step == 1 { String::new() } else { format!(":{}", self.step) };
        match self.end {
            Some(end) => write!(f, "{}:{}{}", self.start, end, step_part),
            None => write!(f, "{}:{}", self.start, step_part),
        }
    }
}

// ---- s! macro ----------------------------------------------------------------

/// Create a [`Slice`] with python-like syntax.
///
/// ```ignore
/// s!(1..5)    // Slice { start: 1, end: Some(5), step: 1 }
/// s!(1:)      // Slice { start: 1, end: None, step: 1 }
/// s!(1::2)    // Slice { start: 1, end: None, step: 2 }
/// s!(..5)     // Slice { start: 0, end: Some(5), step: 1 }
/// s!(:)       // Slice { start: 0, end: None, step: 1 }
/// s!(::3)     // Slice { start: 0, end: None, step: 3 }
/// ```
#[macro_export]
macro_rules! s {
    ($start:tt : $end:expr) => {
        $crate::ops::Slice::new($start as usize, Some($end as isize), 1)
    };
    ($start:tt : $end:tt : $step:expr) => {
        $crate::ops::Slice::new($start as usize, Some($end as isize), $step as usize)
    };
    ($start:tt :) => {
        $crate::ops::Slice::new($start as usize, None, 1)
    };
    ($start:tt :: $step:expr) => {
        $crate::ops::Slice::new($start as usize, None, $step as usize)
    };
    (: $end:tt) => {
        $crate::ops::Slice::new(0, Some($end as isize), 1)
    };
    (: $end:tt : $step:expr) => {
        $crate::ops::Slice::new(0, Some($end as isize), $step as usize)
    };
    (:: $step:expr) => {
        $crate::ops::Slice::new(0, None, $step as usize)
    };
    (:) => {
        $crate::ops::Slice::new(0, None, 1)
    };
}

// ---- Indexer -----------------------------------------------------------------

/// One element of a fancy-indexing operation.
#[derive(Clone)]
pub enum Indexer<D: Device> {
    /// Select a single index (removes the dimension).
    Select(usize),
    /// Select via a signed dimension index (removes the dimension).
    SelectD(crate::D),
    /// Slice a range (keeps the dimension).
    Slice(Slice),
    /// Boolean mask filtering (keeps the dimension).
    Boolean(Tensor<D, Bool>),
}

// From impls: single values + ranges → Indexer

impl<D: Device> From<usize> for Indexer<D> {
    fn from(index: usize) -> Self {
        Indexer::Select(index)
    }
}

impl<D: Device> From<crate::D> for Indexer<D> {
    fn from(index: crate::D) -> Self {
        Indexer::SelectD(index)
    }
}

impl<D: Device> From<Slice> for Indexer<D> {
    fn from(value: Slice) -> Self {
        Indexer::Slice(value)
    }
}

impl<D: Device> std::fmt::Debug for Indexer<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Select(n) => f.debug_tuple("Select").field(n).finish(),
            Self::SelectD(d) => f.debug_tuple("SelectD").field(d).finish(),
            Self::Slice(s) => f.debug_tuple("Slice").field(s).finish(),
            Self::Boolean(_) => f.debug_tuple("Boolean").field(&"Tensor<..>").finish(),
        }
    }
}

impl<D: Device> From<Tensor<D, Bool>> for Indexer<D> {
    fn from(value: Tensor<D, Bool>) -> Self {
        Indexer::Boolean(value)
    }
}

impl<D: Device> From<&Tensor<D, Bool>> for Indexer<D> {
    fn from(value: &Tensor<D, Bool>) -> Self {
        Indexer::Boolean(value.clone())
    }
}

impl<D: Device> From<std::ops::Range<usize>> for Indexer<D> {
    fn from(value: std::ops::Range<usize>) -> Self {
        Indexer::Slice(Slice::new(value.start, Some(value.end as isize), 1))
    }
}

impl<D: Device> From<std::ops::RangeFrom<usize>> for Indexer<D> {
    fn from(value: std::ops::RangeFrom<usize>) -> Self {
        Indexer::Slice(Slice::new(value.start, None, 1))
    }
}

impl<D: Device> From<std::ops::RangeTo<usize>> for Indexer<D> {
    fn from(value: std::ops::RangeTo<usize>) -> Self {
        Indexer::Slice(Slice::new(0, Some(value.end as isize), 1))
    }
}

impl<D: Device> From<std::ops::RangeFull> for Indexer<D> {
    fn from(_: std::ops::RangeFull) -> Self {
        Indexer::Slice(Slice::new(0, None, 1))
    }
}

// ---- indexes() ---------------------------------------------------------------

impl<D: Device, K: IndexingDTypeKind<D> + crate::ops::shape::ShapeDTypeKind<D>> Tensor<D, K> {
    /// Apply a sequence of [`Indexer`]s, one per dimension, in order.
    ///
    /// - [`Indexer::Select`] / [`Indexer::SelectD`]: narrow + squeeze (removes dim).
    /// - [`Indexer::Slice`]: narrow (step=1) or slice (step>1) — keeps dim.
    /// - [`Indexer::Boolean`]: boolean mask → `index_select` — keeps dim.
    pub fn indexes(&self, indexers: &[Indexer<D>]) -> crate::Result<Self> {
        let mut x = self.clone();
        let mut current_dim = 0;
        for idx in indexers {
            x = match idx {
                Indexer::Select(n) => x.narrow(current_dim, *n, 1)?.squeeze(current_dim)?,
                Indexer::SelectD(d) => {
                    let dim_size = x.dim(current_dim)?;
                    let n = d.to_real_index(dim_size, "index")?;
                    x.narrow(current_dim, n, 1)?.squeeze(current_dim)?
                }
                Indexer::Slice(s) => {
                    let dim_size = x.dim(current_dim)?;
                    let (start, end, step) = s.resolve(dim_size);
                    let out = if step == 1 { x.narrow(current_dim, start, end - start)? } else { x.slice(current_dim, start, end, step)? };
                    current_dim += 1;
                    out
                }
                Indexer::Boolean(mask) => {
                    let indices: Vec<i64> = mask.to_vec()?.into_iter().enumerate().filter(|(_, v)| *v).map(|(i, _)| i as i64).collect();
                    let idx_tensor = Tensor::<D, Int>::from_slice(&indices, indices.len(), ())?;
                    let out = x.index_select(&idx_tensor, current_dim)?;
                    current_dim += 1;
                    out
                }
            };
        }
        Ok(x)
    }
}

// ---- IndexOp trait (for .i() syntax) -----------------------------------------

/// Trait for fancy indexing via the `.i()` method.
pub trait IndexOp<T, D: Device, K: DTypeKind<D>> {
    fn i(&self, index: T) -> crate::Result<Tensor<D, K>>;
}

// Single indexer → .i(0) or .i(s!(1..3))
impl<I, D, K> IndexOp<I, D, K> for Tensor<D, K>
where
    I: Into<Indexer<D>>,
    D: Device,
    K: IndexingDTypeKind<D> + crate::ops::shape::ShapeDTypeKind<D>,
{
    fn i(&self, index: I) -> crate::Result<Tensor<D, K>> {
        self.indexes(&[index.into()])
    }
}

// Tuple of indexers → .i((0, s!(1..)))
macro_rules! index_op_tuple {
    ($($t:ident),+) => {
        #[allow(non_snake_case)]
        impl<$($t),*, D, K> IndexOp<($($t,)*), D, K> for Tensor<D, K>
        where
            $($t: Into<Indexer<D>>,)*
            D: Device,
            K: IndexingDTypeKind<D> + crate::ops::shape::ShapeDTypeKind<D>,
        {
            fn i(&self, ($($t,)*): ($($t,)*)) -> crate::Result<Tensor<D, K>> {
                self.indexes(&[$($t.into(),)*])
            }
        }
    };
}

index_op_tuple!(I1);
index_op_tuple!(I1, I2);
index_op_tuple!(I1, I2, I3);
index_op_tuple!(I1, I2, I3, I4);
index_op_tuple!(I1, I2, I3, I4, I5);

// Vec<Indexer> → .i(vec![...])
impl<I, D, K> IndexOp<Vec<I>, D, K> for Tensor<D, K>
where
    I: Into<Indexer<D>>,
    D: Device,
    K: IndexingDTypeKind<D> + crate::ops::shape::ShapeDTypeKind<D>,
{
    fn i(&self, index: Vec<I>) -> crate::Result<Tensor<D, K>> {
        let idxs: Vec<Indexer<D>> = index.into_iter().map(|i| i.into()).collect();
        self.indexes(&idxs)
    }
}

// ---- get() convenience -------------------------------------------------------

impl<D: Device, K: crate::ops::shape::ShapeDTypeKind<D>> Tensor<D, K> {
    /// Returns the sub-tensor fixing the index at `i` on the first dimension.
    ///
    /// ```ignore
    /// let t = Tensor::<Cpu>::new(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).unwrap();
    /// let row1 = t.get(1).unwrap(); // [3.0, 4.0]
    /// ```
    pub fn get(&self, i: usize) -> crate::Result<Self> {
        let dims = self.dims();
        if dims.is_empty() { Ok(self.clone()) } else { self.narrow(0, i, 1)?.reshape(&dims[1..]) }
    }
}
