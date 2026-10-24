use std::{collections::{HashMap, HashSet}, io::{Read, Write}, net::{SocketAddr, TcpListener, TcpStream}, sync::Mutex};
use luma_tensor::Device;
use crate::{Comm, DistError, DistResult, backends::tcp::ring::TcpRingBackend};

pub fn init_process_group<Dev: Device>(addr: &str, world_size: usize, rank: usize, device: Dev) -> DistResult<Comm<Dev>> {
    // 所有线程得到了：一个 listener + 下一个结点的地址
    let (listener, next_addr) = bootstrap_ring(addr, world_size, rank)?;

    // 连接 下一个 rank，接受上一个 rank 
    let next_stream = TcpStream::connect(next_addr)?;
    let (prev_stream, _peer) = listener.accept()?;

    let backend = TcpRingBackend {
        next: Mutex::new(next_stream),
        prev: Mutex::new(prev_stream),
        device, rank, world_size,
    };
    Ok(Comm::from_backend(backend))
}

// ================================================================================================== //
//                    bootstrap ring
// ================================================================================================== //

fn bootstrap_ring(addr: &str, world_size: usize, rank: usize) -> DistResult<(TcpListener, String)> {
    // A ring needs at least two ranks; a single rank has no neighbours.
    if world_size < 2 {
        return Err(DistError::Unsupported(format!("tcp ring requires world_size >= 2, got {world_size}")));
    }

    // const DUPLICAT_RANK
    enum State {
        Success,
        UnmatchWorldSize(usize, usize),
        DuplicateRank(usize),
        InvalidRank { rank: usize, world_size: usize },
    }
    const SUCCESS: u8 = 0;
    const UNMATCH_WORLDSIZE: u8 = 1;
    const DUPLICAT_RANK: u8 = 2;
    const INVALID_RANK: u8 = 3;

    let listener = TcpListener::bind("127.0.0.1:0")?;
    let listen_addr = listener.local_addr()?;
    
    let next_addr = if rank == 0 {
        let accept_listener = TcpListener::bind(addr)?;

        // 等待所有 client 收集 rank 
        let mut clients = vec![];
        for stream in accept_listener.incoming().take(world_size - 1) {
            let mut stream = stream?;
            // 等待 stream 发送 rank
            let client_rank = recv_u64(&mut stream)? as usize;
            // 等待 stream 发送 world size
            let client_world_size = recv_u64(&mut stream)? as usize;
            // 等待 stream 发送 addr
            let client_addr = recv_addr(&mut stream)?;
            clients.push((client_rank, client_world_size, client_addr, stream));
        }

        // 检查信息
        let mut used_ranks = HashSet::new();
        used_ranks.insert(0);
        let mut state = State::Success;
        for (client_rank, client_world_size, _, _) in clients.iter() {
            if world_size != *client_world_size {
                state = State::UnmatchWorldSize(world_size, *client_world_size);
                break;
            }
            if *client_rank >= *client_world_size {
                state = State::InvalidRank { rank: *client_rank, world_size };
                break;
            }
            if used_ranks.contains(client_rank) {
                state = State::DuplicateRank(*client_rank);
                break;
            }
            used_ranks.insert(*client_rank);
        };
        
        // 如果有问题。统一通知 client 报错
        match state {
            State::Success => {
                for (_, _, _, stream) in clients.iter_mut() {
                    stream.write_all(&[SUCCESS])?;
                }
            }
            State::UnmatchWorldSize(world_size1, world_size2) => {
                for (_, _, _, stream) in clients.iter_mut() {
                    stream.write_all(&[UNMATCH_WORLDSIZE])?;
                    stream.write_all(&(world_size1 as u64).to_be_bytes())?;
                    stream.write_all(&(world_size2 as u64).to_be_bytes())?;
                }
                Err(DistError::UnmatchWorldSize(world_size1, world_size2))?;
            }
            State::DuplicateRank(drank) => {
                for (_, _, _, stream) in clients.iter_mut() {
                    stream.write_all(&[DUPLICAT_RANK])?;
                    stream.write_all(&(drank as u64).to_be_bytes())?;
                }
                Err(DistError::DuplicateRank(drank))?;
            }
            State::InvalidRank { rank, world_size } => {
                for (_, _, _, stream) in clients.iter_mut() {
                    stream.write_all(&[INVALID_RANK])?;
                    stream.write_all(&(rank as u64).to_be_bytes())?;
                    stream.write_all(&(world_size as u64).to_be_bytes())?;
                }
                Err(DistError::InvalidRank{rank, world_size})?;
            }
        }

        // 收集 stream/addr
        let mut streams = HashMap::new();
        let mut addrs = HashMap::new();
        addrs.insert(0, listen_addr.to_string());
        for (client_rank, _, client_addr, client_stream) in clients.into_iter() {
            streams.insert(client_rank, client_stream);
            addrs.insert(client_rank, client_addr);
        }

        // 给其他 client 发送他们 prev/next 的地址信息
        for (&rank, stream) in streams.iter_mut() {
            // 每个 rank 作为 server 连接 rank next
            let next_rank = (rank + 1) % world_size;
            send_str(stream, &addrs[&next_rank])?;
            // 通知所有 client 同步完成
            stream.write_all(&(rank as u64).to_be_bytes())?;
        }

        addrs[&1].to_string()
    } else {
        let mut rank0_stream = connect_retry(addr)?;
        // 给 rank 0 发送自己的 rank
        rank0_stream.write_all(&(rank as u64).to_be_bytes())?;
        // 给 rank 0 发送自己的 world size
        rank0_stream.write_all(&(world_size as u64).to_be_bytes())?;
        // 给 rank 0 发送自己的地址
        send_addr(&mut rank0_stream, &listen_addr)?;

        // 等待检查结果
        let code = recv_u8(&mut rank0_stream)?;
        match code {
            SUCCESS => {}
            UNMATCH_WORLDSIZE => {
                let world_size1 = recv_u64(&mut rank0_stream)? as usize;
                let world_size2 = recv_u64(&mut rank0_stream)? as usize;
                Err(DistError::UnmatchWorldSize(world_size1, world_size2))?;
            }
            DUPLICAT_RANK => {
                let drank = recv_u64(&mut rank0_stream)? as usize;
                Err(DistError::DuplicateRank(drank))?;
            }
            INVALID_RANK => {
                let rank = recv_u64(&mut rank0_stream)? as usize;
                let world_size = recv_u64(&mut rank0_stream)? as usize;
                Err(DistError::InvalidRank{rank, world_size})?;
            }
            _ => unreachable!(),
        }

        // 等待主线程发送 next rank 的地址
        let next_addr = recv_addr(&mut rank0_stream)?;
        // 等待 ack
        let self_rank = recv_u64(&mut rank0_stream)? as usize;
        assert_eq!(self_rank, rank);

        next_addr
    };

    Ok((listener, next_addr))
}

// ================================================================================================== //
//                    Utils
// ================================================================================================== //

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

fn send_addr(stream: &mut TcpStream, addr: &SocketAddr) -> DistResult<()> {
    send_str(stream, &addr.to_string())
}

fn send_str(stream: &mut TcpStream, s: &str) -> DistResult<()> {
    stream.write_all(&(s.len() as u64).to_be_bytes())?;
    stream.write_all(s.as_bytes())?;
    Ok(())
}

fn recv_addr(stream: &mut TcpStream) -> DistResult<String> {
    let mut buf = [0u8; 8];
    stream.read_exact(&mut buf)?;
    let addr_len = u64::from_be_bytes(buf) as usize;
    let mut addr_buf = vec![0u8; addr_len];
    stream.read_exact(addr_buf.as_mut())?;
    let addr = String::from_utf8(addr_buf).expect("invalid string");
    Ok(addr)
}

fn recv_u64(stream: &mut TcpStream) -> DistResult<u64> {
    let mut buf = [0u8; 8];
    stream.read_exact(&mut buf)?;
    let val = u64::from_be_bytes(buf);
    Ok(val)
}

fn recv_u8(stream: &mut TcpStream) -> DistResult<u8> {
    let mut buf = [0u8; 1];
    stream.read_exact(&mut buf)?;
    Ok(buf[0])
}