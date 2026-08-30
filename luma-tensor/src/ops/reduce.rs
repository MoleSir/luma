use super::shape_infer::reduce_out_shape;
use crate::{DTypeKind, Device, Dim, Dims, Float, Int, Layout, ReduceOp, Shape, Tensor, TensorMeta};

pub trait ReduceDTypeKind<D: Device>: DTypeKind<D> {
    fn arg_reduce_dispatch(
        x: &Self::Storage,
        layout: &Layout,
        dim: usize,
        keepdim: bool,
        take_max: bool,
        out_shape: &Shape,
    ) -> crate::Result<D::IntStorage>;
    fn reduce_dispatch(
        x: &Self::Storage,
        l: &Layout,
        dims: &[usize],
        keepdim: bool,
        op: ReduceOp,
        out_shape: &Shape,
    ) -> crate::Result<Self::Storage>;
}

impl<D: Device> ReduceDTypeKind<D> for Float {
    fn arg_reduce_dispatch(
        x: &Self::Storage,
        layout: &crate::Layout,
        dim: usize,
        keepdim: bool,
        take_max: bool,
        out_shape: &Shape,
    ) -> crate::Result<D::IntStorage> {
        D::f_arg_reduce(x, layout, dim, keepdim, take_max, out_shape)
    }

    fn reduce_dispatch(
        x: &Self::Storage,
        l: &Layout,
        dims: &[usize],
        keepdim: bool,
        op: ReduceOp,
        out_shape: &Shape,
    ) -> crate::Result<Self::Storage> {
        D::f_reduce(x, l, dims, keepdim, op, out_shape)
    }
}

impl<D: Device> ReduceDTypeKind<D> for Int {
    fn arg_reduce_dispatch(
        x: &Self::Storage,
        layout: &crate::Layout,
        dim: usize,
        keepdim: bool,
        take_max: bool,
        out_shape: &Shape,
    ) -> crate::Result<D::IntStorage> {
        D::i_arg_reduce(x, layout, dim, keepdim, take_max, out_shape)
    }

    fn reduce_dispatch(
        x: &Self::Storage,
        l: &Layout,
        dims: &[usize],
        keepdim: bool,
        op: ReduceOp,
        out_shape: &Shape,
    ) -> crate::Result<Self::Storage> {
        D::i_reduce(x, l, dims, keepdim, op, out_shape)
    }
}

// ============================================================================
// Generic argmin/argmax for Float and Int
// ============================================================================

impl<D: Device, K: ReduceDTypeKind<D>> Tensor<D, K> {
    pub fn argmin<Dm: Dim>(&self, dim: Dm) -> crate::Result<Tensor<D, Int>> {
        self.arg_reduce_dispatch(dim, false, false)
    }

    pub fn argmin_keepdim<Dm: Dim>(&self, dim: Dm) -> crate::Result<Tensor<D, Int>> {
        self.arg_reduce_dispatch(dim, true, false)
    }

    pub fn argmax<Dm: Dim>(&self, dim: Dm) -> crate::Result<Tensor<D, Int>> {
        self.arg_reduce_dispatch(dim, false, true)
    }

    pub fn argmax_keepdim<Dm: Dim>(&self, dim: Dm) -> crate::Result<Tensor<D, Int>> {
        self.arg_reduce_dispatch(dim, true, true)
    }

    fn arg_reduce_dispatch<Dm: Dim>(&self, dim: Dm, keepdim: bool, take_max: bool) -> crate::Result<Tensor<D, Int>> {
        let d = dim.to_index(self.shape(), "argmin/argmax")?;
        let out_shape = reduce_out_shape(self.shape(), &[d], keepdim);
        let storage = K::arg_reduce_dispatch(&*self.storage_read()?, self.layout(), d, keepdim, take_max, &out_shape)?;
        Ok(Tensor::<D, Int>::from_storage(storage, out_shape, ()))
    }
}

// ============================================================================
// Float-specific: mean and variance
// ============================================================================

impl<D: Device> Tensor<D, Float> {
    pub fn mean<Dm: Dim>(&self, dim: Dm) -> crate::Result<Self> {
        let d = dim.to_index(self.shape(), "mean")?;
        self.f_reduce(&[d], false, ReduceOp::Mean)
    }

    pub fn mean_keepdim<Dm: Dim>(&self, dim: Dm) -> crate::Result<Self> {
        let d = dim.to_index(self.shape(), "mean_keepdim")?;
        self.f_reduce(&[d], true, ReduceOp::Mean)
    }

    pub fn mean_all(&self) -> crate::Result<Self> {
        let dims: Vec<usize> = (0..self.rank()).collect();
        self.f_reduce(&dims, false, ReduceOp::Mean)
    }

    pub fn var<Dm: Dim>(&self, dim: Dm) -> crate::Result<Self> {
        let d = dim.to_index(self.shape(), "var")?;
        self.var_impl(d, false, false)
    }

    pub fn var_keepdim<Dm: Dim>(&self, dim: Dm) -> crate::Result<Self> {
        let d = dim.to_index(self.shape(), "var_keepdim")?;
        self.var_impl(d, true, false)
    }

    pub fn var_all(&self) -> crate::Result<Self> {
        self.flatten_all()?.var(0)
    }

    pub fn var_unbiased<Dm: Dim>(&self, dim: Dm) -> crate::Result<Self> {
        let d = dim.to_index(self.shape(), "var_unbiased")?;
        self.var_impl(d, false, true)
    }

    pub fn var_unbiased_keepdim<Dm: Dim>(&self, dim: Dm) -> crate::Result<Self> {
        let d = dim.to_index(self.shape(), "var_unbiased_keepdim")?;
        self.var_impl(d, true, true)
    }

    pub fn var_unbiased_all(&self) -> crate::Result<Self> {
        self.flatten_all()?.var_unbiased(0)
    }

    /// `var = mean((x - mean(x))^2)`, optionally with Bessel correction.
    fn var_impl(&self, dim: usize, keepdim: bool, unbiased: bool) -> crate::Result<Self> {
        let mean = self.mean_keepdim(dim)?;
        let diff = self.sub(&mean.broadcast_as(self.shape().clone())?)?;
        let sq = diff.sqr()?;
        let v = sq.mean_keepdim(dim)?;
        let result = if keepdim { v.clone() } else { v.squeeze(dim)? };
        if unbiased {
            let n = self.dims()[dim] as f64;
            result.mul_scalar(n / (n - 1.0))
        } else {
            Ok(result)
        }
    }

    fn f_reduce(&self, dims: &[usize], keepdim: bool, op: ReduceOp) -> crate::Result<Self> {
        let out_shape = reduce_out_shape(self.shape(), dims, keepdim);
        let s = D::f_reduce(&*self.storage_read()?, self.layout(), dims, keepdim, op, &out_shape)?;
        let meta = <Float as DTypeKind<D>>::Meta::on_reduce(self, dims, op);
        Ok(Self::from_storage(s, out_shape, meta))
    }
}

impl<D: Device, K: ReduceDTypeKind<D>> Tensor<D, K> {
    fn reduce_impl(&self, dims: &[usize], keepdim: bool, op: ReduceOp) -> crate::Result<Self> {
        let out_shape = reduce_out_shape(self.shape(), dims, keepdim);
        let s = K::reduce_dispatch(&*self.storage_read()?, self.layout(), dims, keepdim, op, &out_shape)?;
        Ok(Self::from_storage(s, out_shape, K::Meta::on_reduce(self, dims, op)))
    }
}

macro_rules! reduce_dispatch {
    ($name:ident, $keep:ident, $all:ident, $variant:ident) => {
        pub fn $name<Dm: Dim>(&self, dim: Dm) -> crate::Result<Self> {
            let d = dim.to_index(self.shape(), stringify!($name))?;
            self.reduce_impl(&[d], false, ReduceOp::$variant)
        }

        pub fn $keep<Dm: Dim>(&self, dim: Dm) -> crate::Result<Self> {
            let d = dim.to_index(self.shape(), stringify!($keep))?;
            self.reduce_impl(&[d], true, ReduceOp::$variant)
        }

        pub fn $all(&self) -> crate::Result<Self> {
            let dims: Vec<usize> = (0..self.rank()).collect();
            self.reduce_impl(&dims, false, ReduceOp::$variant)
        }
    };
}

impl<D: Device, K: ReduceDTypeKind<D>> Tensor<D, K> {
    reduce_dispatch!(sum, sum_keepdim, sum_all, Sum);
    reduce_dispatch!(max, max_keepdim, max_all, Max);
    reduce_dispatch!(min, min_keepdim, min_all, Min);
    reduce_dispatch!(prod, prod_keepdim, prod_all, Prod);

    pub fn sum_dims<Ds: Dims>(&self, dims: Ds, keepdim: bool) -> crate::Result<Self> {
        let dims = dims.to_indexes(self.shape(), "sum_dims")?;
        self.reduce_impl(&dims, keepdim, ReduceOp::Sum)
    }
}

// ---- Float-only: std, logsumexp ----

impl<D: Device> Tensor<D, Float> {
    /// Standard deviation along `dim`.
    pub fn std<Dm: Dim>(&self, dim: Dm) -> crate::Result<Self> {
        self.var(dim)?.sqrt()
    }

    pub fn std_keepdim<Dm: Dim>(&self, dim: Dm) -> crate::Result<Self> {
        self.var_keepdim(dim)?.sqrt()
    }

    pub fn std_all(&self) -> crate::Result<Self> {
        self.var_all()?.sqrt()
    }

    /// Log-sum-exp along `dim`, numerically stable.
    pub fn logsumexp<Dm: Dim>(&self, dim: Dm) -> crate::Result<Self> {
        let m = self.max_keepdim(dim)?;
        let e = self.sub(&m.broadcast_as(self.shape().clone())?)?.exp()?;
        let s = e.sum_keepdim(dim)?;
        s.ln()?.add(&m)?.squeeze(dim)
    }

    pub fn logsumexp_keepdim<Dm: Dim>(&self, dim: Dm) -> crate::Result<Self> {
        let m = self.max_keepdim(dim)?;
        let e = self.sub(&m.broadcast_as(self.shape().clone())?)?.exp()?;
        e.sum_keepdim(dim)?.ln()?.add(&m)
    }

    pub fn logsumexp_all(&self) -> crate::Result<Self> {
        self.flatten_all()?.logsumexp(0)
    }
}
