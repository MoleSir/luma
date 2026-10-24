mod backend;
mod init;
use std::{collections::HashMap, net::TcpStream, sync::Mutex};
use luma_tensor::Device;
pub use init::init_process_group;

pub struct TcpStarBackend<Dev: Device> {
    ctx: StarContext,
    /// The local CPU device (carries the allocator used for result tensors).
    device: Dev,
    /// This rank's index in `0..world_size`.
    rank: usize,
    /// Number of ranks in the group.
    world_size: usize,
}

pub enum StarContext {
    Server(HashMap<usize, Mutex<TcpStream>>),
    Client(Mutex<TcpStream>),
}