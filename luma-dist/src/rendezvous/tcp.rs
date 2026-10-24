use super::Rendezvous;
use crate::{DistError, DistResult};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

/// A rendezvous over a TCP socket: rank 0 listens on `addr`, every other rank
/// connects to it and reads rank 0's payload.
///
/// The payload is framed with an 8-byte little-endian length prefix, so the
/// receiver does not need to know the size up front.
pub struct Tcp {
    addr: String,
}

impl Tcp {
    /// Create a rendezvous bound to `addr` (`"host:port"`).
    pub fn new(addr: impl Into<String>) -> Self {
        Self { addr: addr.into() }
    }

    /// The `"host:port"` this rendezvous talks to.
    pub fn addr(&self) -> &str {
        &self.addr
    }
}

impl Rendezvous for Tcp {
    fn exchange(&self, rank: usize, world_size: usize, payload: &[u8]) -> DistResult<Vec<u8>> {
        if rank == 0 {
            let listener = TcpListener::bind(&self.addr).map_err(map_io)?;
            // Serve the payload to every other rank, then return our own copy.
            for _ in 0..world_size.saturating_sub(1) {
                let (mut stream, _) = listener.accept().map_err(map_io)?;
                write_frame(&mut stream, payload)?;
            }
            Ok(payload.to_vec())
        } else {
            let mut stream = TcpStream::connect(&self.addr).map_err(map_io)?;
            read_frame(&mut stream)
        }
    }

    fn root_addr(&self) -> Option<&str> {
        Some(&self.addr)
    }
}

fn write_frame(stream: &mut TcpStream, data: &[u8]) -> DistResult<()> {
    stream.write_all(&(data.len() as u64).to_le_bytes()).map_err(map_io)?;
    stream.write_all(data).map_err(map_io)?;
    stream.flush().map_err(map_io)?;
    Ok(())
}

fn read_frame(stream: &mut TcpStream) -> DistResult<Vec<u8>> {
    let mut len = [0u8; 8];
    stream.read_exact(&mut len).map_err(map_io)?;
    let n = u64::from_le_bytes(len) as usize;
    let mut buf = vec![0u8; n];
    stream.read_exact(&mut buf).map_err(map_io)?;
    Ok(buf)
}

fn map_io(e: std::io::Error) -> DistError {
    DistError::Rendezvous(e.to_string())
}
