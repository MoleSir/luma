use luma_cuda::Cuda;
use luma_dist::backends::nccl::{self, NcclError};
use luma_dist::{ReduceOp, tcp};
use luma_tensor::Tensor;

/// One process per rank. Launch `WORLD_SIZE` of these, each with its own `RANK`
/// and the shared `MASTER_ADDR`/`MASTER_PORT` (see `run_multiprocess.sh`).
fn main() -> luma_dist::DistResult<()> {
    let world_size = env_usize("WORLD_SIZE", 2);
    let rank = env_usize("RANK", 0);
    let addr = std::env::var("MASTER_ADDR").unwrap_or_else(|_| "127.0.0.1".into());
    let port = std::env::var("MASTER_PORT").unwrap_or_else(|_| "29500".into());

    // Build this rank's device, then rendezvous + join the NCCL group in one call.
    let device = Cuda::new(rank).map_err(NcclError::Cuda)?;
    let comm = nccl::init_process_group(tcp!(format!("{addr}:{port}")), world_size, rank, device)?;

    // Every rank fills with (rank + 1); the sum is 1 + 2 + ... + world_size.
    let x = Tensor::<Cuda>::full((4,), (rank + 1) as f64, comm.device())?;
    let y = comm.all_reduce(&x, ReduceOp::Sum)?;
    println!("{:?}", y.to_vec()?);

    Ok(())
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
