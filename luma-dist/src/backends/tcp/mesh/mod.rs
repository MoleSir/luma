mod backend;
mod init;
use std::{collections::HashMap, net::TcpStream, sync::Mutex};
use luma_tensor::Device;
pub use init::init_process_group;

pub struct TcpMeshBackend<Dev: Device> {
    /// One duplex connection to every other rank (keyed by that rank).
    peers: HashMap<usize, Mutex<TcpStream>>,
    /// The local CPU device (carries the allocator used for result tensors).
    device: Dev,
    /// This rank's index in `0..world_size`.
    rank: usize,
    /// Number of ranks in the group.
    world_size: usize,
}
