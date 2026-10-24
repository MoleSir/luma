use std::io::{Read, Write};
use super::{TcpStarBackend, StarContext};
use crate::{DistBackend, DistError, DistResult, backends::tcp::TcpError};
use luma_tensor::{Bool, Device, Float, Int, ReduceOp, Tensor, ops::{BaseOpsDTypeKind, NumOpsDTypeKind, construct::BytesDTypeKind}};
use super::super::utils;

impl<Dev: Device> DistBackend<Dev> for TcpStarBackend<Dev> {
    fn name(&self) -> String {
        format!("tcp-{}", self.device.name())
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
        Err(DistError::Tcp(TcpError::BoolTensorNotSupportReduce))
    }

    fn reduce_float(&self, x: &Tensor<Dev, Float>, op: ReduceOp, dst: usize) -> DistResult<Option<Tensor<Dev, Float>>> {
        self.reduce_impl(x, op, dst)
    }

    fn reduce_int(&self, x: &Tensor<Dev, Int>, op: ReduceOp, dst: usize) -> DistResult<Option<Tensor<Dev, Int>>> {
        self.reduce_impl(x, op, dst)
    }

    fn reduce_bool(&self, _x: &Tensor<Dev, Bool>, _op: ReduceOp, _dst: usize) -> DistResult<Option<Tensor<Dev, Bool>>> {
        Err(DistError::Tcp(TcpError::BoolTensorNotSupportReduce))
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
        Err(DistError::Tcp(TcpError::BoolTensorNotSupportReduce))
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
        // Rank 0 waits for every client's byte before releasing — a rendezvous.
        self.sync_bytes([0u8; 1], "barrier")
    }
}

impl<Dev: Device> TcpStarBackend<Dev> {
    fn all_reduce_impl<K: NumOpsDTypeKind<Dev>>(&self, x: &Tensor<Dev, K>, op: ReduceOp) -> DistResult<Tensor<Dev, K>> {
        let x = x.contiguous()?;
        let res = match &self.ctx {
            StarContext::Server(clients) => {
                let mut tensors = vec![];
                tensors.push(x.clone());

                // 对 client 等待 tensor 计算
                for (_, stream) in clients.iter() {
                    let mut stream = stream.lock().expect("lock stream");
                    let tensor = utils::recv_tensor(&mut stream, &self.device)?;
                    tensors.push(tensor);
                }

                // 计算
                let res = calculate_reduce(&tensors, op)?;

                // 发送回其他 client
                for (_, stream) in clients.iter() {
                    let mut stream = stream.lock().expect("lock stream");
                    utils::send_tensor(&mut stream, &res)?;
                }

                res
            },
            StarContext::Client(stream) => {
                let mut stream = stream.lock().expect("lock stream");
                // 发送给 rank 0 计算
                utils::send_tensor(&mut stream, &x)?;
                // 等待 rank 0 的计算结果
                utils::recv_tensor(&mut stream, &self.device)?
            }
        };

        Ok(res)
    }
    
    fn reduce_impl<K: NumOpsDTypeKind<Dev>>(&self, x: &Tensor<Dev, K>, op: ReduceOp, dst: usize) -> DistResult<Option<Tensor<Dev, K>>> {
        self.sync_bytes((dst as u64).to_be_bytes(), "dst")?;
        
        let x = x.contiguous()?;
        let res = match &self.ctx {
            StarContext::Server(clients) => {
                assert_eq!(0, self.rank);
                let mut tensors = vec![];
                tensors.push(x.clone());

                // 对 client 等待 tensor 计算
                for (_, stream) in clients.iter() {
                    let mut stream = stream.lock().expect("lock stream");
                    let tensor = utils::recv_tensor(&mut stream, &self.device)?;
                    tensors.push(tensor);
                }

                // 计算
                let res = calculate_reduce(&tensors, op)?;

                // 自己返回
                if dst == 0 {
                    Some(res)
                } else {
                    // 发给 dst
                    let stream = clients.get(&dst).expect("get dst rank");
                    let mut stream = stream.lock().expect("lock stream");
                    utils::send_tensor(&mut stream, &res)?;
                    None
                }
            },
            StarContext::Client(stream) => {
                let mut stream = stream.lock().expect("lock stream");
                // 确认 dst 一致 
                // 发送给 rank 0 计算
                utils::send_tensor(&mut stream, &x)?;
                if dst == self.rank {
                    // 等待 rank 0 的计算结果
                    Some(utils::recv_tensor(&mut stream, &self.device)?)
                } else {
                    None
                }
            }
        };

        Ok(res)
    }

    fn reduce_scatter_impl<K: NumOpsDTypeKind<Dev>>(&self, x: &Tensor<Dev, K>, op: ReduceOp, dim: usize) -> DistResult<Tensor<Dev, K>> {
        self.sync_bytes((dim as u64).to_be_bytes(), "dim")?;
        
        let x = x.contiguous()?;
        let size = *x.dims().get(dim).ok_or_else(|| DistError::Shape(format!("reduce_scatter: dim {dim} out of range")))?;
        let world = self.world_size;
        if size % world != 0 {
            return Err(DistError::Shape(format!("reduce_scatter: dim {dim} size {size} not divisible by world_size {world}")));
        }
        let part_size = size / world;

        let res = match &self.ctx {
            StarContext::Server(clients) => {
                let mut tensors = vec![];
                tensors.push(x.clone());

                // 对 client 等待 tensor 计算
                for (_, stream) in clients.iter() {
                    let mut stream = stream.lock().expect("lock stream");
                    let tensor = utils::recv_tensor(&mut stream, &self.device)?;
                    tensors.push(tensor);
                }

                // 计算
                let res = calculate_reduce(&tensors, op)?;
                // 切分：第 rank 个块给 rank
                let rank0_tensor = res.narrow(dim, 0, part_size)?;

                // 发送回其他 client
                for (&client_rank, stream) in clients.iter() {
                    let mut stream = stream.lock().expect("lock stream");
                    let rank_tensor = res.narrow(dim, client_rank*part_size, part_size)?;
                    utils::send_tensor(&mut stream, &rank_tensor)?;
                }

                rank0_tensor
            },
            StarContext::Client(stream) => {
                let mut stream = stream.lock().expect("lock stream");
                // 发送给 rank 0 计算
                utils::send_tensor(&mut stream, &x)?;
                // 等待 rank 0 的计算结果
                utils::recv_tensor(&mut stream, &self.device)?
            }
        };

        Ok(res)
    }

    fn send_impl<K: BytesDTypeKind<Dev>>(&self, x: &Tensor<Dev, K>, peer: usize) -> DistResult<()> {
        match &self.ctx {
            StarContext::Server(clients) => {
                let stream = clients.get(&peer).ok_or(DistError::Tcp(TcpError::NoPeer(peer)))?;
                let mut stream = stream.lock().expect("lock stream");
                utils::send_tensor(&mut stream, x)
            }
            StarContext::Client(stream) => {
                if peer != 0 {
                    return Err(DistError::Tcp(TcpError::UnsupportedPeer(peer)));
                }
                let mut stream = stream.lock().expect("lock stream");
                utils::send_tensor(&mut stream, x)
            }
        }
    }

    fn recv_impl<K: BytesDTypeKind<Dev>>(&self, peer: usize) -> DistResult<Tensor<Dev, K>> {
        match &self.ctx {
            StarContext::Server(clients) => {
                let stream = clients.get(&peer).ok_or(DistError::Tcp(TcpError::NoPeer(peer)))?;
                let mut stream = stream.lock().expect("lock stream");
                utils::recv_tensor(&mut stream, &self.device)
            }
            StarContext::Client(stream) => {
                if peer != 0 {
                    return Err(DistError::Tcp(TcpError::UnsupportedPeer(peer)));
                }
                let mut stream = stream.lock().expect("lock stream");
                utils::recv_tensor(&mut stream, &self.device)
            }
        }
    }

    fn all_gather_impl<K: BaseOpsDTypeKind<Dev>>(&self, x: &Tensor<Dev, K>, dim: usize) -> DistResult<Tensor<Dev, K>> {
        let x = x.contiguous()?;
        let res = match &self.ctx {
            StarContext::Server(clients) => {
                let mut tensors = vec![None; self.world_size];
                tensors[0] = Some(x.clone());

                // 对 client 等待 tensor 计算
                for (&client_rank, stream) in clients.iter() {
                    let mut stream = stream.lock().expect("lock stream");
                    let tensor = utils::recv_tensor(&mut stream, &self.device)?;
                    assert!(client_rank != 0);
                    assert!(tensors[client_rank].is_none());
                    tensors[client_rank] = Some(tensor);
                }

                // 计算
                let tensors: Vec<_> = tensors.into_iter().map(|t| t.unwrap()).collect();
                let res = Tensor::cat(&tensors, dim)?;

                // 发送回其他 client
                for (_, stream) in clients.iter() {
                    let mut stream = stream.lock().expect("lock stream");
                    utils::send_tensor(&mut stream, &res)?;
                }

                res
            },
            StarContext::Client(stream) => {
                let mut stream = stream.lock().expect("lock stream");
                // 发送给 rank 0 计算
                utils::send_tensor(&mut stream, &x)?;
                // 等待 rank 0 的计算结果
                utils::recv_tensor(&mut stream, &self.device)?
            }
        };

        Ok(res)
    }

    fn broadcast_impl<K: BaseOpsDTypeKind<Dev>>(&self, x: Option<&Tensor<Dev, K>>) -> DistResult<Tensor<Dev, K>> {
        const SUCCESS: u8 = 0;
        const MULTI_ROOT: u8 = 1;
        const NO_ROOT: u8 = 2;

        match &self.ctx {
            StarContext::Server(clients) => {
                // 主进程收集下是否发送
                let mut send_ranks = if x.is_some() { vec![0] } else { vec![] };
                for (&client_rank, stream) in clients.iter() {
                    let mut stream = stream.lock().expect("lock stream");
                    let mut bytes = [0u8; 1];
                    stream.read_exact(&mut bytes)?;
                    if bytes[0] != 0 { 
                        send_ranks.push(client_rank);
                    }
                }

                // 处理错误
                match send_ranks.len() {
                    0 => {
                        for (_, stream) in clients.iter() {
                            let mut stream = stream.lock().expect("lock stream");
                            stream.write_all(&[NO_ROOT])?;
                        }
                        Err(DistError::NoBroadcastRoot)?;
                    },
                    1 => {
                        for (_, stream) in clients.iter() {
                            let mut stream = stream.lock().expect("lock stream");
                            stream.write_all(&[SUCCESS])?;
                        }
                    },
                    len => {
                        for (_, stream) in clients.iter() {
                            let mut stream = stream.lock().expect("lock stream");
                            stream.write_all(&[MULTI_ROOT])?;
                            stream.write_all(&(len as u64).to_be_bytes())?;
                            for &r in send_ranks.iter() {
                                let bytes = (r as u64).to_be_bytes();
                                stream.write_all(&bytes)?;
                            }
                        }
                        Err(DistError::MultiBroadcastRoot(send_ranks.clone()))?;
                    }
                }             

                assert!(send_ranks.len() == 1);
                let send_rank = send_ranks[0];
                
                let tensor = match x {
                    // 1. 自己发给所有其他 rank
                    Some(x) => {
                        assert!(send_rank == 0);
                        let x = x.contiguous()?;
                        for (_, stream) in clients.iter() {
                            let mut stream = stream.lock().expect("lock stream");
                            utils::send_tensor(&mut stream, &x)?;
                        }
                        x
                    }
                    // 2. 先等待 rank x，然后发送给其他 rank
                    None => {
                        assert!(send_rank != 0);
                        assert!(send_rank < self.world_size);

                        // 等待 root stream 发送
                        let root_stream = clients.get(&send_rank).expect("get send rank");
                        let mut root_stream = root_stream.lock().expect("lock stream");
                        let tensor = utils::recv_tensor(&mut root_stream, &self.device)?;
                        for (&client_rank, stream) in clients.iter() {
                            if client_rank == send_rank {
                                continue;
                            }
                            let mut stream = stream.lock().expect("lock stream");
                            utils::send_tensor(&mut stream, &tensor)?;
                        }

                        tensor
                    }
                };

                Ok(tensor)
            }
            StarContext::Client(stream) => {
                let mut stream = stream.lock().expect("lock stream");
                let byte = if x.is_some() { 1u8 } else { 0u8 };
                stream.write_all(&[byte])?;

                let mut status = [0u8; 1];
                stream.read_exact(&mut status)?;
                let status = status[0];
                if status == NO_ROOT {
                    Err(DistError::NoBroadcastRoot)?;
                } 
                if status == MULTI_ROOT {
                    let mut bytes = [0u8; 8];
                    stream.read_exact(&mut bytes)?;
                    let len = u64::from_be_bytes(bytes) as usize;

                    let mut send_ranks = vec![];
                    for _ in 0..len {
                        let mut bytes = [0u8; 8];
                        stream.read_exact(&mut bytes)?;
                        send_ranks.push(u64::from_be_bytes(bytes) as usize);
                    }
                    Err(DistError::MultiBroadcastRoot(send_ranks))?;
                }

                // 状态正常，等待 tensor（如果是自己发的，先发）
                let tensor = match x {
                    Some(x) => {
                        let x = x.contiguous()?;
                        utils::send_tensor(&mut stream, &x)?;
                        x
                    }
                    None => {
                        utils::recv_tensor(&mut stream, &self.device)?
                    }
                };

                Ok(tensor)
            }
        }
    }
}

impl<Dev: Device> TcpStarBackend<Dev> {
    fn sync_bytes<const N: usize>(&self, bytes: [u8; N], msg: &'static str) -> DistResult<()> {
        let equals = match &self.ctx {
            StarContext::Server(clients) => {
                // 等待所有 client，比较
                let mut equals = true;
                for (_, stream) in clients.iter() {
                    let mut stream = stream.lock().expect("lock stream");
                    let mut buff = [0u8; N];
                    stream.read_exact(&mut buff)?;
                    if buff != bytes {
                        equals = false;
                    }
                }

                let resp = if equals { 0u8 } else { 1u8 };
                for (_, stream) in clients.iter() {
                    let mut stream = stream.lock().expect("lock stream");
                    stream.write_all(&[resp])?;
                }

                equals
            }
            StarContext::Client(stream) => {
                // 全部发给 server 进行同步
                let mut stream = stream.lock().expect("lock stream");
                stream.write_all(&bytes)?;
                // 等待确认消息
                let mut byte = [0u8; 1];
                stream.read_exact(&mut byte)?;
                
                byte[0] == 0
            }
        };

        if equals {
            Ok(())
        } else {
            Err(DistError::Tcp(TcpError::SyncBytes(msg)))?
        }
    }
}

fn calculate_reduce<Dev: Device, K: NumOpsDTypeKind<Dev>>(tensors: &[Tensor<Dev, K>], op: ReduceOp) -> DistResult<Tensor<Dev, K>> {
    let res: Tensor<Dev, K> = tensors[0].copy()?;
    match op {
        ReduceOp::Max => {
            for tensor in tensors.iter().skip(1) {
                res.maximum_(tensor)?;
            }
        },
        ReduceOp::Min => {
            for tensor in tensors.iter().skip(1) {
                res.minimum_(tensor)?;
            }
        },
        ReduceOp::Sum => {
            for tensor in tensors.iter().skip(1) {
                res.add_(tensor)?;
            }
        },
        ReduceOp::Mean => {
            for tensor in tensors.iter().skip(1) {
                res.add_(tensor)?;
            }
            let len = K::from_i64(tensors.len() as i64);
            res.div_scalar_(len)?;
        },
        ReduceOp::Prod => {
            for tensor in tensors.iter().skip(1) {
                res.mul_(tensor)?;
            }
        },
    }
    Ok(res)
}
