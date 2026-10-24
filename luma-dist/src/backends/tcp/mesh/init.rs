use std::{collections::{HashMap, HashSet}, io::{Read, Write}, net::{SocketAddr, TcpListener, TcpStream}, sync::Mutex};
use luma_tensor::Device;
use super::TcpMeshBackend;
use crate::{Comm, DistError, DistResult};

pub fn init_process_group<Dev: Device>(addr: &str, world_size: usize, rank: usize, device: Dev) -> DistResult<Comm<Dev>> {
    // A full mesh needs at least two ranks.
    if world_size < 2 {
        return Err(DistError::Unsupported(format!("tcp mesh requires world_size >= 2, got {world_size}")));
    }

    // Everyone learns everyone's listen address; the node then dials the ranks
    // above it and accepts from the ranks below it.
    let (listener, addrs) = bootstrap_mesh(addr, world_size, rank)?;

    let mut peers = HashMap::new();
    for (peer, addr) in addrs.iter().enumerate().skip(rank + 1) {
        let mut stream = connect_retry(addr)?;
        // Hand the acceptor our rank so it can key this stream.
        stream.write_all(&(rank as u64).to_be_bytes())?;
        peers.insert(peer, Mutex::new(stream));
    }
    for _ in 0..rank {
        let (mut stream, _) = listener.accept()?;
        let peer = recv_u64(&mut stream)? as usize;
        assert!(peer < rank, "tcp mesh: unexpected inbound peer {peer} at rank {rank}");
        peers.insert(peer, Mutex::new(stream));
    }

    let backend = TcpMeshBackend { peers, device, rank, world_size };
    Ok(Comm::from_backend(backend))
}

/// Bind a loopback listener and exchange every rank's address through rank 0.
///
/// Returns the data listener plus the full address table indexed by rank.
fn bootstrap_mesh(addr: &str, world_size: usize, rank: usize) -> DistResult<(TcpListener, Vec<String>)> {
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

    // Same-host only for now: advertise a loopback address. Multi-host support
    // would take the host from `addr` and only let the OS pick the port.
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let listen_addr = listener.local_addr()?;

    let addrs = if rank == 0 {
        let accept_listener = TcpListener::bind(addr)?;

        // Wait for every client to report rank / world size / address.
        let mut clients = vec![];
        for stream in accept_listener.incoming().take(world_size - 1) {
            let mut stream = stream?;
            let client_rank = recv_u64(&mut stream)? as usize;
            let client_world_size = recv_u64(&mut stream)? as usize;
            let client_addr = recv_addr(&mut stream)?;
            clients.push((client_rank, client_world_size, client_addr, stream));
        }

        // Validate the collected info.
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
        }

        // On failure, notify every client uniformly.
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
                Err(DistError::InvalidRank { rank, world_size })?;
            }
        }

        // Build the address table and send it to every client.
        let mut table = vec![String::new(); world_size];
        table[0] = listen_addr.to_string();
        for (client_rank, _, client_addr, _) in clients.iter() {
            table[*client_rank] = client_addr.clone();
        }
        for (client_rank, _, _, stream) in clients.iter_mut() {
            stream.write_all(&(world_size as u64).to_be_bytes())?;
            for a in &table {
                send_str(stream, a)?;
            }
            // Ack with the client's own rank.
            stream.write_all(&(*client_rank as u64).to_be_bytes())?;
        }

        table
    } else {
        let mut rank0_stream = connect_retry(addr)?;
        rank0_stream.write_all(&(rank as u64).to_be_bytes())?;
        rank0_stream.write_all(&(world_size as u64).to_be_bytes())?;
        send_addr(&mut rank0_stream, &listen_addr)?;

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
                Err(DistError::InvalidRank { rank, world_size })?;
            }
            _ => unreachable!(),
        }

        let n = recv_u64(&mut rank0_stream)? as usize;
        let mut table = Vec::with_capacity(n);
        for _ in 0..n {
            table.push(recv_addr(&mut rank0_stream)?);
        }
        let self_rank = recv_u64(&mut rank0_stream)? as usize;
        assert_eq!(self_rank, rank);

        table
    };

    Ok((listener, addrs))
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
    Ok(u64::from_be_bytes(buf))
}

fn recv_u8(stream: &mut TcpStream) -> DistResult<u8> {
    let mut buf = [0u8; 1];
    stream.read_exact(&mut buf)?;
    Ok(buf[0])
}
