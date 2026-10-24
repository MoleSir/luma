use std::{collections::{HashMap, hash_map::Entry}, io::{Read, Write}, net::{TcpListener, TcpStream}, sync::Mutex};
use luma_tensor::Device;
use crate::{Comm, DistError, DistResult};
use super::{TcpStarBackend, super::TcpError, StarContext};

pub fn init_process_group<Dev: Device>(addr: &str, world_size: usize, rank: usize, device: Dev) -> DistResult<Comm<Dev>> {
    let ctx = if rank == 0 {
        let mut clients = HashMap::new();
        if world_size > 1 {
            let listen = TcpListener::bind(addr)?;
            for stream in listen.incoming() {
                let mut stream = stream?;
                // 等待接受 rank/world
                let mut buf = [0u8; 8];
                stream.read_exact(&mut buf)?;
                let client_rank = u64::from_be_bytes(buf) as usize;
                if client_rank == 0 || client_rank >= world_size {
                    Err(DistError::InvalidRank { rank: client_rank, world_size })?;
                }

                let mut buf = [0u8; 8];
                stream.read_exact(&mut buf)?;
                let client_world_size = u64::from_be_bytes(buf) as usize;
                if client_world_size != world_size {
                    Err(DistError::UnmatchWorldSize(world_size, client_world_size))?;
                }

                // 收集 client
                let entry = clients.entry(client_rank);
                match entry {
                    Entry::Occupied(_) => Err(DistError::DuplicateRank(client_rank))?,
                    Entry::Vacant(v) => v.insert(Mutex::new(stream)),
                };
                if clients.len() == world_size - 1 {
                    break;
                }
            }
            // 通知 client 完成，返回 rank 表示确认
            for (client_rank, stream) in clients.iter_mut() {
                let buf = client_rank.to_be_bytes();
                stream.get_mut().expect("lock stream").write_all(&buf)?;
            }
        }

        StarContext::Server(clients)
    } else {
        let mut stream = connect_retry(addr)?;
        // 发送 rank/world
        let buf = rank.to_be_bytes();
        stream.write_all(&buf)?;
        let buf = world_size.to_be_bytes();
        stream.write_all(&buf)?;

        // 等待直到接收到 ack
        let mut buf = [0u8; 8];
        stream.read_exact(&mut buf)?;
        let ack_rank = u64::from_be_bytes(buf) as usize;
        if ack_rank != rank {
            Err(DistError::Tcp(TcpError::RankAck(rank, ack_rank)))?;
        }

        StarContext::Client(Mutex::new(stream))
    };

    let backend = TcpStarBackend { ctx, world_size, rank, device, };
    Ok(Comm::from_backend(backend))
}

/// Connect to rank 0, retrying briefly: peers may start before rank 0 has bound.
fn connect_retry(addr: &str) -> DistResult<TcpStream> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match TcpStream::connect(addr) {
            Ok(stream) => return Ok(stream),
            Err(e) => {
                if std::time::Instant::now() >= deadline {
                    return Err(e.into());
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
    }
}