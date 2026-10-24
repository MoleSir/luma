mod star;
mod ring;
mod mesh;

mod error;
mod utils;

pub use error::*;
use luma_tensor::Device;
use star::TcpStarBackend;
use ring::TcpRingBackend;
use mesh::TcpMeshBackend;
use crate::{Comm, DistResult};

#[cfg(test)]
mod tests;

pub enum TcpBackend<Dev: Device> {
    Star(TcpStarBackend<Dev>),
    Ring(TcpRingBackend<Dev>),
    Mesh(TcpMeshBackend<Dev>),
}

#[derive(Copy, Clone)]
pub enum TcpBackendImpl {
    Star,
    Ring,
    Mesh,
}

pub fn init_process_group<Dev: Device>(impll: TcpBackendImpl, addr: &str, world_size: usize, rank: usize, device: Dev) -> DistResult<Comm<Dev>> {
    match impll {
        TcpBackendImpl::Star => star::init_process_group(addr, world_size, rank, device),
        TcpBackendImpl::Ring => ring::init_process_group(addr, world_size, rank, device),
        TcpBackendImpl::Mesh => mesh::init_process_group(addr, world_size, rank, device),
    }
}