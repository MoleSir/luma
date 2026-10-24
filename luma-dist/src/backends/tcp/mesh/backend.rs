//! `impl DistBackend<Dev> for TcpMeshBackend<Dev>`.
//!
//! Full mesh: every rank has a direct duplex connection to every other rank,
//! so `send`/`recv` work for any peer. The collectives are the simple
//! hub-based ones (rank 0 gathers, reduces, and fans out), which is fine for
//! the small `world_size` this backend targets.

use super::TcpMeshBackend;
use super::super::utils;
use crate::{DistBackend, DistError, DistResult, backends::tcp::TcpError};
use luma_tensor::{Bool, Device, Float, Int, ReduceOp, Tensor, ops::{BaseOpsDTypeKind, NumOpsDTypeKind, construct::BytesDTypeKind}};

impl<Dev: Device> DistBackend<Dev> for TcpMeshBackend<Dev> {
    fn name(&self) -> String {
        format!("tcp-mesh-{}", self.device.name())
    }

    fn device(&self) -> &Dev {
        &self.device
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn all_reduce_float(&self, x: &Tensor<Dev, Float>, op: ReduceOp) -> DistResult<Tensor<Dev, Float>> {
        self.all_reduce_impl(x, op)
    }

    fn all_reduce_int(&self, x: &Tensor<Dev, Int>, op: ReduceOp) -> DistResult<Tensor<Dev, Int>> {
        self.all_reduce_impl(x, op)
    }

    fn all_reduce_bool(&self, _x: &Tensor<Dev, Bool>, _op: ReduceOp) -> DistResult<Tensor<Dev, Bool>> {
        Err(DistError::Unsupported("tcp mesh all_reduce_bool".into()))
    }

    fn reduce_float(&self, x: &Tensor<Dev, Float>, op: ReduceOp, root: usize) -> DistResult<Option<Tensor<Dev, Float>>> {
        self.reduce_impl(x, op, root)
    }

    fn reduce_int(&self, x: &Tensor<Dev, Int>, op: ReduceOp, root: usize) -> DistResult<Option<Tensor<Dev, Int>>> {
        self.reduce_impl(x, op, root)
    }

    fn reduce_bool(&self, _x: &Tensor<Dev, Bool>, _op: ReduceOp, _root: usize) -> DistResult<Option<Tensor<Dev, Bool>>> {
        Err(DistError::Unsupported("tcp mesh reduce_bool".into()))
    }

    fn all_gather_float(&self, x: &Tensor<Dev, Float>, dim: usize) -> DistResult<Tensor<Dev, Float>> {
        self.all_gather_impl(x, dim)
    }

    fn all_gather_int(&self, x: &Tensor<Dev, Int>, dim: usize) -> DistResult<Tensor<Dev, Int>> {
        self.all_gather_impl(x, dim)
    }

    fn all_gather_bool(&self, x: &Tensor<Dev, Bool>, dim: usize) -> DistResult<Tensor<Dev, Bool>> {
        self.all_gather_impl(x, dim)
    }

    fn reduce_scatter_float(&self, x: &Tensor<Dev, Float>, op: ReduceOp, dim: usize) -> DistResult<Tensor<Dev, Float>> {
        self.reduce_scatter_impl(x, op, dim)
    }

    fn reduce_scatter_int(&self, x: &Tensor<Dev, Int>, op: ReduceOp, dim: usize) -> DistResult<Tensor<Dev, Int>> {
        self.reduce_scatter_impl(x, op, dim)
    }

    fn reduce_scatter_bool(&self, _x: &Tensor<Dev, Bool>, _op: ReduceOp, _dim: usize) -> DistResult<Tensor<Dev, Bool>> {
        Err(DistError::Unsupported("tcp mesh reduce_scatter_bool".into()))
    }

    fn broadcast_float(&self, x: Option<&Tensor<Dev, Float>>) -> DistResult<Tensor<Dev, Float>> {
        self.broadcast_impl(x)
    }

    fn broadcast_int(&self, x: Option<&Tensor<Dev, Int>>) -> DistResult<Tensor<Dev, Int>> {
        self.broadcast_impl(x)
    }

    fn broadcast_bool(&self, x: Option<&Tensor<Dev, Bool>>) -> DistResult<Tensor<Dev, Bool>> {
        self.broadcast_impl(x)
    }

    fn send_float(&self, x: &Tensor<Dev, Float>, peer: usize) -> DistResult<()> {
        self.send_impl(x, peer)
    }

    fn send_int(&self, x: &Tensor<Dev, Int>, peer: usize) -> DistResult<()> {
        self.send_impl(x, peer)
    }

    fn send_bool(&self, x: &Tensor<Dev, Bool>, peer: usize) -> DistResult<()> {
        self.send_impl(x, peer)
    }

    fn recv_float(&self, peer: usize) -> DistResult<Tensor<Dev, Float>> {
        self.recv_impl(peer)
    }

    fn recv_int(&self, peer: usize) -> DistResult<Tensor<Dev, Int>> {
        self.recv_impl(peer)
    }

    fn recv_bool(&self, peer: usize) -> DistResult<Tensor<Dev, Bool>> {
        self.recv_impl(peer)
    }

    fn group_start(&self) -> DistResult<()> {
        // One rank per thread/process blocks per call, so no grouping is needed.
        Ok(())
    }

    fn group_end(&self) -> DistResult<()> {
        Ok(())
    }

    fn barrier(&self) -> DistResult<()> {
        // A cheap all-reduce doubles as a rendezvous.
        let token = Tensor::<Dev, Int>::from_vec_i64(vec![0i64], (1,), &self.device)?;
        self.all_reduce_impl(&token, ReduceOp::Sum)?;
        Ok(())
    }
}

impl<Dev: Device> TcpMeshBackend<Dev> {
    /// Point-to-point send to any other rank (direct link).
    fn send_impl<K: BytesDTypeKind<Dev>>(&self, x: &Tensor<Dev, K>, peer: usize) -> DistResult<()> {
        let stream = self.peers.get(&peer).ok_or(DistError::Tcp(TcpError::NoPeer(peer)))?;
        utils::send_tensor(&mut stream.lock().expect("lock stream"), x)
    }

    /// Point-to-point receive from any other rank (direct link).
    fn recv_impl<K: BytesDTypeKind<Dev>>(&self, peer: usize) -> DistResult<Tensor<Dev, K>> {
        let stream = self.peers.get(&peer).ok_or(DistError::Tcp(TcpError::NoPeer(peer)))?;
        utils::recv_tensor(&mut stream.lock().expect("lock stream"), &self.device)
    }

    /// Every rank sends to rank 0; rank 0 reduces and fans the result back out.
    fn all_reduce_impl<K: NumOpsDTypeKind<Dev>>(&self, x: &Tensor<Dev, K>, op: ReduceOp) -> DistResult<Tensor<Dev, K>> {
        let world = self.world_size;
        let x = x.contiguous()?;
        if self.rank == 0 {
            let acc = x.copy()?;
            for j in 1..world {
                let got: Tensor<Dev, K> = self.recv_impl(j)?;
                reduce_op(&acc, &got, op)?;
            }
            if let ReduceOp::Mean = op {
                acc.div_scalar_(K::from_i64(world as i64))?;
            }
            for j in 1..world {
                self.send_impl(&acc, j)?;
            }
            Ok(acc)
        } else {
            self.send_impl(&x, 0)?;
            self.recv_impl(0)
        }
    }

    /// Every rank sends to `root`; `root` reduces and returns `Some`.
    fn reduce_impl<K: NumOpsDTypeKind<Dev>>(&self, x: &Tensor<Dev, K>, op: ReduceOp, root: usize) -> DistResult<Option<Tensor<Dev, K>>> {
        let world = self.world_size;
        let x = x.contiguous()?;
        if self.rank == root {
            let acc = x.copy()?;
            for j in 0..world {
                if j == root {
                    continue;
                }
                let got: Tensor<Dev, K> = self.recv_impl(j)?;
                reduce_op(&acc, &got, op)?;
            }
            if let ReduceOp::Mean = op {
                acc.div_scalar_(K::from_i64(world as i64))?;
            }
            Ok(Some(acc))
        } else {
            self.send_impl(&x, root)?;
            Ok(None)
        }
    }

    /// Every rank sends to rank 0; rank 0 concatenates and fans the result out.
    fn all_gather_impl<K: BaseOpsDTypeKind<Dev>>(&self, x: &Tensor<Dev, K>, dim: usize) -> DistResult<Tensor<Dev, K>> {
        let world = self.world_size;
        let x = x.contiguous()?;
        if self.rank == 0 {
            let mut parts: Vec<Tensor<Dev, K>> = Vec::with_capacity(world);
            parts.push(x);
            for j in 1..world {
                parts.push(self.recv_impl(j)?);
            }
            let res = Tensor::cat(&parts, dim)?;
            for j in 1..world {
                self.send_impl(&res, j)?;
            }
            Ok(res)
        } else {
            self.send_impl(&x, 0)?;
            self.recv_impl(0)
        }
    }

    /// Rank 0 reduces everything, then hands rank `j` its own chunk.
    fn reduce_scatter_impl<K: NumOpsDTypeKind<Dev>>(&self, x: &Tensor<Dev, K>, op: ReduceOp, dim: usize) -> DistResult<Tensor<Dev, K>> {
        let world = self.world_size;
        let x = x.contiguous()?;
        let size = x.dim(dim)?;
        if size % world != 0 {
            return Err(DistError::Shape(format!(
                "tcp mesh reduce_scatter: dim {dim} size {size} not divisible by world_size {world}"
            )));
        }
        let part = size / world;
        if self.rank == 0 {
            let acc = x.copy()?;
            for j in 1..world {
                let got: Tensor<Dev, K> = self.recv_impl(j)?;
                reduce_op(&acc, &got, op)?;
            }
            if let ReduceOp::Mean = op {
                acc.div_scalar_(K::from_i64(world as i64))?;
            }
            for j in 1..world {
                let chunk = acc.narrow(dim, j * part, part)?;
                self.send_impl(&chunk, j)?;
            }
            Ok(acc.narrow(dim, 0, part)?)
        } else {
            self.send_impl(&x, 0)?;
            self.recv_impl(0)
        }
    }

    /// The rank passing `Some` is the source; detect it via an all-gather and
    /// then have it send the tensor directly to everyone else.
    fn broadcast_impl<K: BaseOpsDTypeKind<Dev>>(&self, x: Option<&Tensor<Dev, K>>) -> DistResult<Tensor<Dev, K>> {
        let world = self.world_size;
        let me = if x.is_some() { self.rank as i64 } else { -1 };
        let indicator = Tensor::<Dev, Int>::from_vec_i64(vec![me], (1,), &self.device)?;
        let vals = self.all_gather_impl(&indicator, 0)?.to_vec()?;
        let roots: Vec<usize> = vals.iter().filter(|v| **v != -1).map(|v| *v as usize).collect();
        if roots.is_empty() {
            return Err(DistError::NoBroadcastRoot);
        }
        if roots.len() > 1 {
            return Err(DistError::MultiBroadcastRoot(roots));
        }

        let source = roots[0];
        if self.rank == source {
            let x = x.expect("broadcast source must pass Some").contiguous()?;
            for j in 0..world {
                if j != self.rank {
                    self.send_impl(&x, j)?;
                }
            }
            Ok(x)
        } else {
            self.recv_impl(source)
        }
    }
}

/// Combine two equally-shaped tensors elementwise for a reduce op.
///
/// `Mean` accumulates as `Sum`; the division by `world_size` happens once, at
/// the hub, before the result is handed back out.
fn reduce_op<Dev: Device, K: NumOpsDTypeKind<Dev>>(a: &Tensor<Dev, K>, b: &Tensor<Dev, K>, op: ReduceOp) -> DistResult<()> {
    match op {
        ReduceOp::Sum | ReduceOp::Mean => { a.add_(b)?; }
        ReduceOp::Prod => { a.mul_(b)?; }
        ReduceOp::Max => { a.maximum_(b)?; }
        ReduceOp::Min => { a.minimum_(b)?; }
    }
    Ok(())
}
