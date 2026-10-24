use crate::{Comm, DistKind, DistResult};
use luma_tensor::{Device, ReduceOp, Tensor};

pub trait TensorDistExt<D: Device, K: DistKind<D>> {
    /// See [`Comm::all_reduce`].
    fn all_reduce(&self, comm: &Comm<D>, op: ReduceOp) -> DistResult<Tensor<D, K>>;

    /// See [`Comm::all_gather`].
    fn all_gather(&self, comm: &Comm<D>, dim: usize) -> DistResult<Tensor<D, K>>;

    /// See [`Comm::reduce_scatter`].
    fn reduce_scatter(&self, comm: &Comm<D>, op: ReduceOp, dim: usize) -> DistResult<Tensor<D, K>>;

    /// Broadcast this tensor to every rank (this tensor is the source).
    fn broadcast(&self, comm: &Comm<D>) -> DistResult<Tensor<D, K>>;

    /// See [`Comm::send`].
    fn send(&self, comm: &Comm<D>, peer: usize) -> DistResult<()>;
}

impl<D: Device, K: DistKind<D>> TensorDistExt<D, K> for Tensor<D, K> {
    fn all_reduce(&self, comm: &Comm<D>, op: ReduceOp) -> DistResult<Tensor<D, K>> {
        comm.all_reduce(self, op)
    }

    fn all_gather(&self, comm: &Comm<D>, dim: usize) -> DistResult<Tensor<D, K>> {
        comm.all_gather(self, dim)
    }

    fn reduce_scatter(&self, comm: &Comm<D>, op: ReduceOp, dim: usize) -> DistResult<Tensor<D, K>> {
        comm.reduce_scatter(self, op, dim)
    }

    fn broadcast(&self, comm: &Comm<D>) -> DistResult<Tensor<D, K>> {
        comm.broadcast(Some(self))
    }

    fn send(&self, comm: &Comm<D>, peer: usize) -> DistResult<()> {
        comm.send(self, peer)
    }
}
