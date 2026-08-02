use crate::{Bool, DTypeKind, Device, Dim, Dims, Error, Float, Int, Layout, Shape, Tensor, TensorMeta, Storage};

pub trait ShapeDTypeKind<D: Device>: DTypeKind<D> {
    fn contiguous_dispatch(s: &Self::Storage, l: &Layout) -> crate::Result<Self::Storage>;
    fn cat_dispatch(srcs: &[(&Self::Storage, &Layout)], dim: usize) -> crate::Result<(Self::Storage, Shape)>;
}

impl<D: Device> ShapeDTypeKind<D> for Float {
    #[inline]
    fn contiguous_dispatch(s: &Self::Storage, l: &Layout) -> crate::Result<Self::Storage> {
        D::f_contiguous(s, l)
    }

    #[inline]
    fn cat_dispatch(srcs: &[(&Self::Storage, &Layout)], dim: usize) -> crate::Result<(Self::Storage, Shape)> {
        D::f_cat(srcs, dim)
    }
}

impl<D: Device> ShapeDTypeKind<D> for Int {
    #[inline]
    fn contiguous_dispatch(s: &Self::Storage, l: &Layout) -> crate::Result<Self::Storage> {
        D::i_contiguous(s, l)
    }

    #[inline]
    fn cat_dispatch(srcs: &[(&Self::Storage, &Layout)], dim: usize) -> crate::Result<(Self::Storage, Shape)> {
        D::i_cat(srcs, dim)
    }
}

impl<D: Device> ShapeDTypeKind<D> for Bool {
    #[inline]
    fn contiguous_dispatch(s: &Self::Storage, l: &Layout) -> crate::Result<Self::Storage> {
        D::b_contiguous(s, l)
    }

    #[inline]
    fn cat_dispatch(srcs: &[(&Self::Storage, &Layout)], dim: usize) -> crate::Result<(Self::Storage, Shape)> {
        D::b_cat(srcs, dim)
    }
}

impl<D: Device, K: ShapeDTypeKind<D>> Tensor<D, K> {
    pub fn reshape<S: Into<Shape>>(&self, shape: S) -> crate::Result<Self> {
        let shape = shape.into();
        if shape.element_count() != self.element_count() {
            return Err(Error::ElementCountMismatchInReshape { origin: self.shape().clone(), target: shape });
        }
        let meta = K::Meta::on_reshape(self);
        if self.is_contiguous() {
            let layout = Layout::contiguous_with_offset(shape, self.layout().start_offset());
            Ok(self.share_storage(layout, meta))
        } else {
            let storage = K::contiguous_dispatch(&*self.storage_read()?, self.layout())?;
            Ok(Self::from_storage(storage, shape, meta))
        }
    }

    pub fn transpose<D1: Dim, D2: Dim>(&self, dim1: D1, dim2: D2) -> crate::Result<Self> {
        let dim1 = dim1.to_index(self.shape(), "transpose")?;
        let dim2 = dim2.to_index(self.shape(), "transpose")?;
        if dim1 == dim2 {
            return Ok(self.clone());
        }
        let layout = self.layout().transpose(dim1, dim2)?;
        Ok(self.share_storage(layout, K::Meta::on_transpose(self, dim1, dim2)))
    }

    pub fn transpose_last(&self) -> crate::Result<Self> {
        self.transpose(crate::D::Minus1, crate::D::Minus2)
    }

    pub fn permute<Ds: Dims>(&self, dims: Ds) -> crate::Result<Self> {
        let dims = dims.to_indexes(self.shape(), "permute")?;
        let layout = self.layout().permute(&dims)?;
        Ok(self.share_storage(layout, K::Meta::on_permute(self, dims)))
    }

    pub fn narrow<Dm: Dim>(&self, dim: Dm, start: usize, len: usize) -> crate::Result<Self> {
        let dim = dim.to_index(self.shape(), "narrow")?;
        let dims = self.dims();
        if start.saturating_add(len) > dims[dim] {
            return Err(Error::NarrowInvalidArgs { shape: self.shape().clone(), dim, start, len, msg: "start + len > dim_len" });
        }
        if start == 0 && dims[dim] == len {
            return Ok(self.clone());
        }
        let layout = self.layout().narrow(dim, start, len)?;
        Ok(self.share_storage(layout, K::Meta::on_narrow(self, dim, start, len)))
    }

    /// Slice along `dim` with `start`, `end`, and `step`.
    ///
    /// Returns a view (no copy) when possible. For `step == 1` this is
    /// equivalent to `narrow`; for `step > 1` the backward pass is not yet
    /// supported on autograd tensors.
    pub fn slice<Dm: Dim>(&self, dim: Dm, start: usize, end: usize, step: usize) -> crate::Result<Self> {
        let dim = dim.to_index(self.shape(), "slice")?;
        let layout = self.layout().slice(dim, start, end, step)?;
        let meta = K::Meta::on_slice(self, dim, start, end, step);
        Ok(self.share_storage(layout, meta))
    }

    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> crate::Result<Self> {
        let layout = self.layout().broadcast_as(shape)?;
        Ok(self.share_storage(layout, K::Meta::on_broadcast(self)))
    }

    pub fn squeeze<Dm: Dim>(&self, dim: Dm) -> crate::Result<Self> {
        let dim = dim.to_index(self.shape(), "squeeze")?;
        let dims = self.dims();
        if dims[dim] != 1 {
            return Err(Error::SqueezeDimNot1 { shape: self.shape().clone(), dim });
        }
        let mut new_dims = dims.to_vec();
        let mut strides = self.stride().to_vec();
        new_dims.remove(dim);
        strides.remove(dim);
        let layout = Layout::new(new_dims, strides, self.layout().start_offset());
        Ok(self.share_storage(layout, K::Meta::on_reshape(self)))
    }

    pub fn unsqueeze<Dm: Dim>(&self, dim: Dm) -> crate::Result<Self> {
        let dim = dim.to_index_plus_one(self.shape(), "unsqueeze")?;
        let mut new_dims = self.dims().to_vec();
        let mut strides = self.stride().to_vec();
        new_dims.insert(dim, 1);
        let stride = if dim < strides.len() { strides[dim] } else { 1 };
        strides.insert(dim, stride);
        let layout = Layout::new(new_dims, strides, self.layout().start_offset());
        Ok(self.share_storage(layout, K::Meta::on_reshape(self)))
    }

    /// Materialize a contiguous copy (records `Op::Copy` for `Float`).
    pub fn contiguous(&self) -> crate::Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        let storage = K::contiguous_dispatch(&*self.storage_read()?, self.layout())?;
        assert_eq!(self.dtype(), storage.dtype());
        Ok(Self::from_storage(storage, self.shape().clone(), K::Meta::on_copy(self)))
    }

    /// Concatenate tensors along `dim` (materializes new storage).
    pub fn cat<A: AsRef<Self>, Dm: Dim>(arrs: &[A], dim: Dm) -> crate::Result<Self> {
        if arrs.is_empty() {
            return Err(Error::OpRequiresAtLeastOneTensor { op: "cat" });
        }
        let first = arrs[0].as_ref();
        let dim = dim.to_index(first.shape(), "cat")?;

        let guards: Vec<_> = arrs.iter().map(|a| a.as_ref().storage_read()).collect::<crate::Result<_>>()?;
        let views: Vec<(&K::Storage, &Layout)> = guards.iter().zip(arrs.iter()).map(|(g, a)| (&**g, a.as_ref().layout())).collect();
        let (storage, shape) = K::cat_dispatch(&views, dim)?;
        drop(guards);

        let meta = K::Meta::on_cat(arrs, dim);
        Ok(Self::from_storage(storage, shape, meta))
    }

    /// Flatten dims `start_dim..=end_dim` into one.
    pub fn flatten<D1: Dim, D2: Dim>(&self, start_dim: D1, end_dim: D2) -> crate::Result<Self> {
        let start = start_dim.to_index(self.shape(), "flatten")?;
        let end = end_dim.to_index(self.shape(), "flatten")?;
        if start > end {
            return Err(Error::NarrowInvalidArgs {
                shape: self.shape().clone(),
                dim: start,
                start,
                len: end,
                msg: "flatten: start_dim > end_dim",
            });
        }
        let merged: usize = self.dims()[start..=end].iter().product();
        let mut new_dims = self.dims().to_vec();
        new_dims.splice(start..=end, [merged]);
        self.reshape(Shape::from(new_dims))
    }

    pub fn flatten_from<Dm: Dim>(&self, start_dim: Dm) -> crate::Result<Self> {
        self.flatten(start_dim, self.rank() - 1)
    }

    pub fn flatten_to<Dm: Dim>(&self, end_dim: Dm) -> crate::Result<Self> {
        self.flatten(0usize, end_dim)
    }

    pub fn flatten_all(&self) -> crate::Result<Self> {
        self.reshape(Shape::from(self.element_count()))
    }

    /// Stack tensors along a new axis at `dim` (unsqueeze each + cat).
    pub fn stack<A: AsRef<Self>, Dm: Dim>(args: &[A], dim: Dm) -> crate::Result<Self> {
        if args.is_empty() {
            return Err(Error::OpRequiresAtLeastOneTensor { op: "stack" });
        }
        let first = args[0].as_ref();
        let dim = dim.to_index_plus_one(first.shape(), "stack")?;
        let unsqueezed: crate::Result<Vec<Self>> = args.iter().map(|a| a.as_ref().unsqueeze(dim)).collect();
        Self::cat(&unsqueezed?, dim)
    }

    /// Split into individual slices along `dim` (one per index).
    pub fn split<Dm: Dim>(&self, dim: Dm) -> crate::Result<Vec<Self>> {
        let dim = dim.to_index(self.shape(), "split")?;
        let n = self.dims()[dim];
        (0..n).map(|i| self.narrow(dim, i, 1)).collect()
    }

    /// Split into `chunks` roughly equal pieces along `dim`.
    pub fn chunk<Dm: Dim>(&self, chunks: usize, dim: Dm) -> crate::Result<Vec<Self>> {
        let dim = dim.to_index(self.shape(), "chunk")?;
        let n = self.dims()[dim];
        if chunks == 0 {
            return Err(Error::Msg("chunk: chunks cannot be 0".into()));
        }
        let size = (n + chunks - 1) / chunks; // ceiling division
        let mut result = Vec::new();
        let mut start = 0;
        while start < n {
            let len = size.min(n - start);
            result.push(self.narrow(dim, start, len)?);
            start += len;
        }
        Ok(result)
    }

    /// Tile `self` by `times` along `dim`.
    pub fn repeat_dim<Dm: Dim>(&self, dim: Dm, times: usize) -> crate::Result<Self> {
        let dim = dim.to_index(self.shape(), "repeat_dim")?;
        let copies: Vec<&Self> = (0..times).map(|_| self).collect();
        Self::cat(&copies, dim)
    }
}
