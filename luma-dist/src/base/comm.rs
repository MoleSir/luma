use crate::{DistError, DistResult};
use luma_tensor::{Bool, DTypeKind, Device, Float, Int, ReduceOp, Tensor};
use std::sync::Arc;

/// A concrete collective-communication transport bound to a device `D`.
pub trait DistBackend<D: Device>: Send + Sync + 'static {
    /// Human-readable backend name (for diagnostics), e.g. `"nccl"`.
    fn name(&self) -> String;

    /// The local device this rank's collective operations run on.
    fn device(&self) -> &D;

    /// This rank's index in `0..world_size`.
    fn rank(&self) -> usize;

    /// Number of ranks in the group.
    fn world_size(&self) -> usize;

    // all_reduce : reduce `x` across all ranks, result on every rank
    fn all_reduce_float(&self, _x: &Tensor<D, Float>, _op: ReduceOp) -> DistResult<Tensor<D, Float>> {
        Err(DistError::Unsupported(format!("all_reduce on {}", self.name())))
    }
    fn all_reduce_int(&self, _x: &Tensor<D, Int>, _op: ReduceOp) -> DistResult<Tensor<D, Int>> {
        Err(DistError::Unsupported(format!("all_reduce on {}", self.name())))
    }
    fn all_reduce_bool(&self, _x: &Tensor<D, Bool>, _op: ReduceOp) -> DistResult<Tensor<D, Bool>> {
        Err(DistError::Unsupported(format!("all_reduce on {}", self.name())))
    }

    // reduce : like all_reduce, but the result is produced only on `root`
    fn reduce_float(&self, _x: &Tensor<D, Float>, _op: ReduceOp, _root: usize) -> DistResult<Option<Tensor<D, Float>>> {
        Err(DistError::Unsupported(format!("reduce on {}", self.name())))
    }
    fn reduce_int(&self, _x: &Tensor<D, Int>, _op: ReduceOp, _root: usize) -> DistResult<Option<Tensor<D, Int>>> {
        Err(DistError::Unsupported(format!("reduce on {}", self.name())))
    }
    fn reduce_bool(&self, _x: &Tensor<D, Bool>, _op: ReduceOp, _root: usize) -> DistResult<Option<Tensor<D, Bool>>> {
        Err(DistError::Unsupported(format!("reduce on {}", self.name())))
    }

    // all_gather : concatenate every rank's `x` along `dim`
    fn all_gather_float(&self, _x: &Tensor<D, Float>, _dim: usize) -> DistResult<Tensor<D, Float>> {
        Err(DistError::Unsupported(format!("all_gather on {}", self.name())))
    }
    fn all_gather_int(&self, _x: &Tensor<D, Int>, _dim: usize) -> DistResult<Tensor<D, Int>> {
        Err(DistError::Unsupported(format!("all_gather on {}", self.name())))
    }
    fn all_gather_bool(&self, _x: &Tensor<D, Bool>, _dim: usize) -> DistResult<Tensor<D, Bool>> {
        Err(DistError::Unsupported(format!("all_gather on {}", self.name())))
    }

    // reduce_scatter : reduce, then keep this rank's chunk along `dim`
    fn reduce_scatter_float(&self, _x: &Tensor<D, Float>, _op: ReduceOp, _dim: usize) -> DistResult<Tensor<D, Float>> {
        Err(DistError::Unsupported(format!("reduce_scatter on {}", self.name())))
    }
    fn reduce_scatter_int(&self, _x: &Tensor<D, Int>, _op: ReduceOp, _dim: usize) -> DistResult<Tensor<D, Int>> {
        Err(DistError::Unsupported(format!("reduce_scatter on {}", self.name())))
    }
    fn reduce_scatter_bool(&self, _x: &Tensor<D, Bool>, _op: ReduceOp, _dim: usize) -> DistResult<Tensor<D, Bool>> {
        Err(DistError::Unsupported(format!("reduce_scatter on {}", self.name())))
    }

    // broadcast : the rank that passes `Some(x)` is the source; others pass `None`
    fn broadcast_float(&self, _x: Option<&Tensor<D, Float>>) -> DistResult<Tensor<D, Float>> {
        Err(DistError::Unsupported(format!("broadcast on {}", self.name())))
    }
    fn broadcast_int(&self, _x: Option<&Tensor<D, Int>>) -> DistResult<Tensor<D, Int>> {
        Err(DistError::Unsupported(format!("broadcast on {}", self.name())))
    }
    fn broadcast_bool(&self, _x: Option<&Tensor<D, Bool>>) -> DistResult<Tensor<D, Bool>> {
        Err(DistError::Unsupported(format!("broadcast on {}", self.name())))
    }

    // point-to-point
    fn send_float(&self, _x: &Tensor<D, Float>, _peer: usize) -> DistResult<()> {
        Err(DistError::Unsupported(format!("send on {}", self.name())))
    }
    fn send_int(&self, _x: &Tensor<D, Int>, _peer: usize) -> DistResult<()> {
        Err(DistError::Unsupported(format!("send on {}", self.name())))
    }
    fn send_bool(&self, _x: &Tensor<D, Bool>, _peer: usize) -> DistResult<()> {
        Err(DistError::Unsupported(format!("send on {}", self.name())))
    }

    fn recv_float(&self, _peer: usize) -> DistResult<Tensor<D, Float>> {
        Err(DistError::Unsupported(format!("recv on {}", self.name())))
    }
    fn recv_int(&self, _peer: usize) -> DistResult<Tensor<D, Int>> {
        Err(DistError::Unsupported(format!("recv on {}", self.name())))
    }
    fn recv_bool(&self, _peer: usize) -> DistResult<Tensor<D, Bool>> {
        Err(DistError::Unsupported(format!("recv on {}", self.name())))
    }

    // grouping : thread-local batching of collectives

    /// Begin a **thread-local** collective group (NCCL: `ncclGroupStart`).
    ///
    /// Collectives issued on this thread before the matching
    /// [`group_end`](Self::group_end) are batched and submitted together. Only
    /// needed when one thread drives more than one [`Comm`], or for matched
    /// point-to-point. With one rank per thread/process it is not required.
    fn group_start(&self) -> DistResult<()> {
        Err(DistError::Unsupported(format!("group_start on {}", self.name())))
    }

    /// End a thread-local collective group (NCCL: `ncclGroupEnd`).
    fn group_end(&self) -> DistResult<()> {
        Err(DistError::Unsupported(format!("group_end on {}", self.name())))
    }

    // barrier : kind-independent, no payload
    fn barrier(&self) -> DistResult<()> {
        Err(DistError::Unsupported(format!("barrier on {}", self.name())))
    }
}

/// Static-dispatch glue from a tensor **kind** to the matching
/// [`DistBackend`] typed method.
///
/// This is what lets [`Comm`] expose one generic `all_reduce<K>` while the
/// backend trait remains object-safe. Implemented for [`Float`], [`Int`],
/// [`Bool`]; users never need to name it.
pub trait DistKind<D: Device>: DTypeKind<D> {
    fn all_reduce(b: &dyn DistBackend<D>, x: &Tensor<D, Self>, op: ReduceOp) -> DistResult<Tensor<D, Self>>;
    fn reduce(b: &dyn DistBackend<D>, x: &Tensor<D, Self>, op: ReduceOp, root: usize) -> DistResult<Option<Tensor<D, Self>>>;
    fn all_gather(b: &dyn DistBackend<D>, x: &Tensor<D, Self>, dim: usize) -> DistResult<Tensor<D, Self>>;
    fn reduce_scatter(b: &dyn DistBackend<D>, x: &Tensor<D, Self>, op: ReduceOp, dim: usize) -> DistResult<Tensor<D, Self>>;
    fn broadcast(b: &dyn DistBackend<D>, x: Option<&Tensor<D, Self>>) -> DistResult<Tensor<D, Self>>;
    fn send(b: &dyn DistBackend<D>, x: &Tensor<D, Self>, peer: usize) -> DistResult<()>;
    fn recv(b: &dyn DistBackend<D>, peer: usize) -> DistResult<Tensor<D, Self>>;
}

macro_rules! impl_dist_kind {
    ($kind:ty, $all_reduce:ident, $reduce:ident, $all_gather:ident, $reduce_scatter:ident, $broadcast:ident, $send:ident, $recv:ident $(,)?) => {
        impl<D: Device> DistKind<D> for $kind {
            fn all_reduce(b: &dyn DistBackend<D>, x: &Tensor<D, Self>, op: ReduceOp) -> DistResult<Tensor<D, Self>> {
                b.$all_reduce(x, op)
            }
            fn reduce(b: &dyn DistBackend<D>, x: &Tensor<D, Self>, op: ReduceOp, root: usize) -> DistResult<Option<Tensor<D, Self>>> {
                b.$reduce(x, op, root)
            }
            fn all_gather(b: &dyn DistBackend<D>, x: &Tensor<D, Self>, dim: usize) -> DistResult<Tensor<D, Self>> {
                b.$all_gather(x, dim)
            }
            fn reduce_scatter(b: &dyn DistBackend<D>, x: &Tensor<D, Self>, op: ReduceOp, dim: usize) -> DistResult<Tensor<D, Self>> {
                b.$reduce_scatter(x, op, dim)
            }
            fn broadcast(b: &dyn DistBackend<D>, x: Option<&Tensor<D, Self>>) -> DistResult<Tensor<D, Self>> {
                b.$broadcast(x)
            }
            fn send(b: &dyn DistBackend<D>, x: &Tensor<D, Self>, peer: usize) -> DistResult<()> {
                b.$send(x, peer)
            }
            fn recv(b: &dyn DistBackend<D>, peer: usize) -> DistResult<Tensor<D, Self>> {
                b.$recv(peer)
            }
        }
    };
}

impl_dist_kind!(Float, all_reduce_float, reduce_float, all_gather_float, reduce_scatter_float, broadcast_float, send_float, recv_float);
impl_dist_kind!(Int, all_reduce_int, reduce_int, all_gather_int, reduce_scatter_int, broadcast_int, send_int, recv_int);
impl_dist_kind!(Bool, all_reduce_bool, reduce_bool, all_gather_bool, reduce_scatter_bool, broadcast_bool, send_bool, recv_bool);

/// A collective-communication handle for device `D`.
///
/// Cheap to clone (bumps an `Arc`). Construct one from a [`DistBackend`] (usually
/// via a backend-specific constructor such as `NcclBackend::from_devices`), then
/// call collectives on it. Every operation is **explicit** about the
/// communication group — there is no global process group.
pub struct Comm<D: Device>(Arc<dyn DistBackend<D>>);

impl<D: Device> Comm<D> {
    /// Wrap a backend in a communicator handle.
    pub fn from_backend(backend: impl DistBackend<D>) -> Self {
        Self(Arc::new(backend))
    }

    /// Backend name, e.g. `"nccl"`.
    pub fn name(&self) -> String {
        self.0.name()
    }

    /// The local device this rank runs on.
    pub fn device(&self) -> &D {
        self.0.device()
    }

    /// This rank's index in `0..world_size`.
    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    /// Number of ranks in the group.
    pub fn world_size(&self) -> usize {
        self.0.world_size()
    }

    /// Reduce `x` across all ranks, returning the result on every rank.
    pub fn all_reduce<K: DistKind<D>>(&self, x: &Tensor<D, K>, op: ReduceOp) -> DistResult<Tensor<D, K>> {
        K::all_reduce(self.0.as_ref(), x, op)
    }

    /// Reduce `x` across all ranks; the result is produced only on `root`
    /// (`Some` there, `None` on every other rank).
    pub fn reduce<K: DistKind<D>>(&self, x: &Tensor<D, K>, op: ReduceOp, root: usize) -> DistResult<Option<Tensor<D, K>>> {
        K::reduce(self.0.as_ref(), x, op, root)
    }

    /// Gather every rank's `x` and concatenate along `dim`.
    pub fn all_gather<K: DistKind<D>>(&self, x: &Tensor<D, K>, dim: usize) -> DistResult<Tensor<D, K>> {
        K::all_gather(self.0.as_ref(), x, dim)
    }

    /// Reduce `x` across all ranks, then keep this rank's chunk along `dim`.
    pub fn reduce_scatter<K: DistKind<D>>(&self, x: &Tensor<D, K>, op: ReduceOp, dim: usize) -> DistResult<Tensor<D, K>> {
        K::reduce_scatter(self.0.as_ref(), x, op, dim)
    }

    /// Broadcast: the rank that passes `Some(x)` is the source; every other rank
    /// passes `None` and receives. Returns the tensor on all ranks.
    pub fn broadcast<K: DistKind<D>>(&self, x: Option<&Tensor<D, K>>) -> DistResult<Tensor<D, K>> {
        K::broadcast(self.0.as_ref(), x)
    }

    /// Send `x` to `peer`.
    pub fn send<K: DistKind<D>>(&self, x: &Tensor<D, K>, peer: usize) -> DistResult<()> {
        K::send(self.0.as_ref(), x, peer)
    }

    /// Receive from `peer`; the shape/dtype are taken from the sender.
    pub fn recv<K: DistKind<D>>(&self, peer: usize) -> DistResult<Tensor<D, K>> {
        K::recv(self.0.as_ref(), peer)
    }

    /// Begin a thread-local collective group (see [`DistBackend::group_start`]).
    pub fn group_start(&self) -> DistResult<()> {
        self.0.group_start()
    }

    /// End a thread-local collective group (see [`DistBackend::group_end`]).
    pub fn group_end(&self) -> DistResult<()> {
        self.0.group_end()
    }

    /// Block until every rank reaches this point.
    pub fn barrier(&self) -> DistResult<()> {
        self.0.barrier()
    }
}

/// Run `f` inside a thread-local collective group on `comm`'s backend.
///
/// Every collective issued inside `f` — on any [`Comm`] sharing this thread —
/// is batched and submitted together when the scope ends. Use this when a
/// single thread drives several comms (e.g. all ranks of a single-process
/// multi-GPU run), or to pair `send`/`recv`.
///
/// With one rank per thread/process, a single collective needs no group at all,
/// so this is usually unnecessary.
///
/// The group is ended even if `f` returns an error.
pub fn group<D: Device, R>(comm: &Comm<D>, f: impl FnOnce() -> DistResult<R>) -> DistResult<R> {
    struct EndGuard<'a, D: Device>(&'a Comm<D>);
    impl<D: Device> Drop for EndGuard<'_, D> {
        fn drop(&mut self) {
            let _ = self.0.group_end();
        }
    }

    comm.group_start()?;
    let _end = EndGuard(comm);
    f()
}

impl<D: Device> Clone for Comm<D> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<D: Device> std::fmt::Debug for Comm<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Comm")
            .field("name", &self.0.name())
            .field("rank", &self.0.rank())
            .field("world_size", &self.0.world_size())
            .finish()
    }
}
