use super::TcpRingBackend;
use crate::{DistBackend, DistError, DistResult};
use luma_tensor::{Bool, Device, Float, IndexOp, Int, ReduceOp, Tensor, ops::{BaseOpsDTypeKind, NumOpsDTypeKind, construct::BytesDTypeKind}};
use super::super::utils;

impl<Dev: Device> DistBackend<Dev> for TcpRingBackend<Dev> {
    fn name(&self) -> String {
        format!("tcp-ring-{}", self.device.name())
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
        Err(DistError::Unsupported("tcp ring all_reduce_bool".into()))
    }

    fn reduce_float(&self, x: &Tensor<Dev, Float>, op: ReduceOp, root: usize) -> DistResult<Option<Tensor<Dev, Float>>> {
        self.reduce_impl(x, op, root)
    }

    fn reduce_int(&self, x: &Tensor<Dev, Int>, op: ReduceOp, root: usize) -> DistResult<Option<Tensor<Dev, Int>>> {
        self.reduce_impl(x, op, root)
    }

    fn reduce_bool(&self, _x: &Tensor<Dev, Bool>, _op: ReduceOp, _root: usize) -> DistResult<Option<Tensor<Dev, Bool>>> {
        Err(DistError::Unsupported("tcp ring reduce_bool".into()))
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
        Err(DistError::Unsupported("tcp ring reduce_scatter_bool".into()))
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
        Ok(())
    }

    fn group_end(&self) -> DistResult<()> {
        Ok(())
    }

    fn barrier(&self) -> DistResult<()> {
        let token = Tensor::<Dev, Int>::from_vec_i64(vec![0i64; self.world_size], (self.world_size,), &self.device)?;
        self.all_reduce_impl(&token, ReduceOp::Sum)?;
        Ok(())
    }
}

impl<Dev: Device> TcpRingBackend<Dev> {
    fn send_next<K: BytesDTypeKind<Dev>>(&self, tensor: &Tensor<Dev, K>) -> DistResult<()> {
        utils::send_tensor(&mut self.next.lock().expect("lock stream"), tensor)
    }

    fn recv_prev<K: BytesDTypeKind<Dev>>(&self) -> DistResult<Tensor<Dev, K>> {
        utils::recv_tensor(&mut self.prev.lock().expect("lock stream"), &self.device)
    }

    /// Point-to-point send to a neighbour.
    ///
    /// The ring only has direct links to `next`/`prev`; any other peer errors.
    /// `next` is tested first so that, when `world_size == 2` (`next == prev`),
    /// the send goes over the `next` socket — the one `recv_impl` reads there.
    fn send_impl<K: BytesDTypeKind<Dev>>(&self, x: &Tensor<Dev, K>, peer: usize) -> DistResult<()> {
        let next = (self.rank + 1) % self.world_size;
        let prev = (self.rank + self.world_size - 1) % self.world_size;
        if peer == next {
            utils::send_tensor(&mut self.next.lock().expect("lock stream"), x)
        } else if peer == prev {
            utils::send_tensor(&mut self.prev.lock().expect("lock stream"), x)
        } else {
            Err(DistError::Unsupported(format!("tcp ring: no direct link to peer {peer}")))
        }
    }

    /// Point-to-point receive from a neighbour.
    ///
    /// `prev` is tested first so that, when `world_size == 2`, the receive pairs
    /// with `send_impl`'s `next` socket.
    fn recv_impl<K: BytesDTypeKind<Dev>>(&self, peer: usize) -> DistResult<Tensor<Dev, K>> {
        let next = (self.rank + 1) % self.world_size;
        let prev = (self.rank + self.world_size - 1) % self.world_size;
        if peer == prev {
            utils::recv_tensor(&mut self.prev.lock().expect("lock stream"), &self.device)
        } else if peer == next {
            utils::recv_tensor(&mut self.next.lock().expect("lock stream"), &self.device)
        } else {
            Err(DistError::Unsupported(format!("tcp ring: no direct link to peer {peer}")))
        }
    }

    /// Send `send` to `next` and receive from `prev` at the same time.
    ///
    /// The send runs on a scoped thread so neither direction can block the
    /// other; done sequentially, a large message can wedge the whole ring.
    fn exchange<K: BytesDTypeKind<Dev>>(&self, send: &Tensor<Dev, K>) -> DistResult<Tensor<Dev, K>> {
        std::thread::scope(|scope| {
            let handle = scope.spawn(|| self.send_next(send));
            let received = self.recv_prev::<K>();
            handle.join().expect("ring send thread panicked")?;
            received
        })
    }

    fn all_reduce_impl<K: NumOpsDTypeKind<Dev>>(&self, x: &Tensor<Dev, K>, op: ReduceOp) -> DistResult<Tensor<Dev, K>> {
        let world = self.world_size;
        let x = x.contiguous()?;

        // Split into `world` equal chunks along dim 0.
        let dim0 = x.dim(0)?;
        if dim0 % world != 0 {
            return Err(DistError::Shape(format!(
                "ring all_reduce: dim 0 size {dim0} not divisible by world_size {world}"
            )));
        }
        let block = dim0 / world;
        let mut blocks: Vec<Tensor<Dev, K>> = (0..world)
            .map(|i| Ok(x.i(i * block..(i + 1) * block)?.copy()?))
            .collect::<DistResult<_>>()?;

        // Phase 1 — reduce-scatter: N-1 steps, accumulate into the received chunk.
        for i in 0..world - 1 {
            let send_id = (self.rank + world - i) % world;
            let recv_id = (self.rank + world - i - 1) % world;
            let got = self.exchange(&blocks[send_id])?;
            reduce_op(&blocks[recv_id], &got, op)?;
        }

        // Phase 2 — all-gather: N-1 steps, overwrite the received chunk.
        for i in 0..world - 1 {
            let send_id = (self.rank + world - i + 1) % world;
            let recv_id = (self.rank + world - i) % world;
            let got = self.exchange(&blocks[send_id])?;
            blocks[recv_id] = got;
        }

        let out = Tensor::cat(&blocks, 0)?;
        if let ReduceOp::Mean = op {
            out.div_scalar_(K::from_i64(world as i64))?;
        }
        Ok(out)
    }

    /// Ring all-gather: `x` is this rank's slice of the result along `dim`.
    ///
    /// Note the index origin differs from `all_reduce`'s second phase: here the
    /// chunk a rank starts with is its own (`chunk[rank] = x`), so the chunk
    /// travels with `send = (rank - i)`, `recv = (rank - i - 1)`.
    fn all_gather_impl<K: BaseOpsDTypeKind<Dev>>(&self, x: &Tensor<Dev, K>, dim: usize) -> DistResult<Tensor<Dev, K>> {
        let world = self.world_size;
        let x = x.contiguous()?;

        let mut chunks: Vec<Option<Tensor<Dev, K>>> = (0..world).map(|_| None).collect();
        chunks[self.rank] = Some(x);

        // N-1 steps: every chunk hops one rank per step until all ranks have all.
        for i in 0..world - 1 {
            let send_id = (self.rank + world - i) % world;
            let recv_id = (self.rank + world - i - 1) % world;
            let send = chunks[send_id].as_ref().expect("ring all_gather: send chunk missing");
            let got = self.exchange(send)?;
            chunks[recv_id] = Some(got);
        }

        let chunks: Vec<Tensor<Dev, K>> = chunks
            .into_iter()
            .map(|c| c.expect("ring all_gather: chunk missing"))
            .collect();
        Ok(Tensor::cat(&chunks, dim)?)
    }

    /// Ring reduce-scatter: reduce `x` and keep this rank's slice along `dim`.
    ///
    /// The chunks are indexed so rank `r` ends up holding the fully reduced
    /// chunk `r` (one slot off from `all_reduce`'s first phase, which lands on
    /// `r + 1`).
    fn reduce_scatter_impl<K: NumOpsDTypeKind<Dev>>(&self, x: &Tensor<Dev, K>, op: ReduceOp, dim: usize) -> DistResult<Tensor<Dev, K>> {
        let world = self.world_size;
        let x = x.contiguous()?;

        let size = x.dim(dim)?;
        if size % world != 0 {
            return Err(DistError::Shape(format!(
                "ring reduce_scatter: dim {dim} size {size} not divisible by world_size {world}"
            )));
        }
        let part = size / world;
        let chunks: Vec<Tensor<Dev, K>> = (0..world)
            .map(|i| Ok(x.narrow(dim, i * part, part)?.copy()?))
            .collect::<DistResult<_>>()?;

        // N-1 steps, accumulating as each chunk hops; rank r lands on chunk r.
        for i in 0..world - 1 {
            let send_id = (self.rank + world - i - 1) % world;
            let recv_id = (self.rank + world - i - 2) % world;
            let got = self.exchange(&chunks[send_id])?;
            reduce_op(&chunks[recv_id], &got, op)?;
        }

        let out = chunks
            .into_iter()
            .nth(self.rank)
            .expect("ring reduce_scatter: chunk missing");
        if let ReduceOp::Mean = op {
            out.div_scalar_(K::from_i64(world as i64))?;
        }
        Ok(out)
    }

    /// Ring reduce: accumulate every rank's `x` along a chain that ends at
    /// `root`; the result is `Some` on `root` and `None` elsewhere.
    ///
    /// Chain: `root+1 -> root+2 -> ... -> root-1 -> root`. Each rank adds its
    /// own tensor to the partial it receives and forwards it on, so every rank
    /// contributes exactly once and the total lands on `root`. Uses only
    /// `send_next`/`recv_prev`, and is a linear chain, so it cannot deadlock.
    fn reduce_impl<K: NumOpsDTypeKind<Dev>>(&self, x: &Tensor<Dev, K>, op: ReduceOp, root: usize) -> DistResult<Option<Tensor<Dev, K>>> {
        let world = self.world_size;
        let x = x.contiguous()?;
        let start = (root + 1) % world;

        if self.rank == root {
            let partial = self.recv_prev::<K>()?;
            let res = x.copy()?;
            reduce_op(&res, &partial, op)?;
            if let ReduceOp::Mean = op {
                res.div_scalar_(K::from_i64(world as i64))?;
            }
            Ok(Some(res))
        } else if self.rank == start {
            // Head of the chain: contributes its own tensor and forwards it.
            self.send_next(&x)?;
            Ok(None)
        } else {
            let partial = self.recv_prev::<K>()?;
            let acc = x.copy()?;
            reduce_op(&acc, &partial, op)?;
            self.send_next(&acc)?;
            Ok(None)
        }
    }

    fn broadcast_impl<K: BaseOpsDTypeKind<Dev>>(&self, x: Option<&Tensor<Dev, K>>) -> DistResult<Tensor<Dev, K>> {
        // 收集所有调用信息，选择发送的 send rank
        let source_rank = Tensor::<Dev, Int>::new(if x.is_some() { self.rank as i64 } else { -1i64 }, &self.device)?;
        let result = self.all_gather_impl(&source_rank, 0)?;
        let result = result.to_vec()?;
        let source_ranks: Vec<usize> = result.into_iter().filter(|sr| *sr != -1).map(|sr| sr as usize).collect();
        if source_ranks.is_empty() {
            Err(DistError::NoBroadcastRoot)?;
        }
        if source_ranks.len() > 1 {
            Err(DistError::MultiBroadcastRoot(source_ranks.clone()))?;
        }

        let source_rank = source_ranks[0];
        let source_last = (source_rank + self.world_size - 1) % self.world_size;
        match x {
            Some(x) => {
                let x = x.contiguous()?;
                assert_eq!(source_rank, self.rank);
                // source 负责发送到下一个
                self.send_next(&x)?;
                Ok(x)
            }
            None => {
                // s -> s + 1 -> s + 2 ...
                // 需要等待上一个发送
                let res = self.recv_prev()?;
                // 如果 rank 不不是 source 的上一个，需要负责把 tensor 继续转发给自己的下一个，如果是就不需要，因为 source 本来就有
                if self.rank != source_last {
                    self.send_next(&res)?;
                }
                Ok(res)
            }

        }
    }
}

/// Combine two equally-shaped tensors elementwise for a reduce op.
///
/// `Mean` accumulates as `Sum`; the division by `world_size` happens once,
/// after the all-gather.
fn reduce_op<Dev: Device, K: NumOpsDTypeKind<Dev>>(a: &Tensor<Dev, K>, b: &Tensor<Dev, K>, op: ReduceOp) -> DistResult<()> {
    match op {
        ReduceOp::Sum | ReduceOp::Mean => { a.add_(b)?; }
        ReduceOp::Prod => { a.mul_(b)?; }
        ReduceOp::Max => { a.maximum_(b)?; }
        ReduceOp::Min => { a.minimum_(b)?; }
    }
    Ok(())
}