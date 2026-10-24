//! Execution schemes: how several ranks are launched and driven *within one
//! process*.
//!
//! Compute code is written against a single [`Comm`] (one rank). These schemes
//! drive several ranks locally:
//!
//! | scheme | type / entry | grouping |
//! |---|---|---|
//! | one thread, many ranks | [`SingleThread`] | automatic |
//! | one thread per rank | [`MultiThread`] | none needed |
//!
//! The *one process per rank* case is not a scheme — it is a backend concern:
//! each process builds its own [`Comm`] via a backend constructor such as
//! `backends::nccl::init_process_group`.
//!
//! The same single-rank compute code runs unchanged under [`MultiThread`] and
//! the multi-process case; only [`SingleThread`] adds the `scope` wrapper.
use crate::{Comm, DistError, DistResult};
use luma_tensor::Device;
mod multi_thread;
mod single_thread;

pub use multi_thread::MultiThread;
pub use single_thread::SingleThread;

/// Validate that `comms` form a complete local group: non-empty, every comm
/// knows the same `world_size == comms.len()`, and ranks are exactly `0..len`.
fn validate<D: Device>(comms: &[Comm<D>]) -> DistResult<()> {
    if comms.is_empty() {
        return Err(DistError::InvalidGroup("empty local group".into()));
    }
    let n = comms.len();
    let mut seen = vec![false; n];
    for comm in comms {
        if comm.world_size() != n {
            return Err(DistError::InvalidGroup(format!("world_size {} != local rank count {}", comm.world_size(), n)));
        }
        let rank = comm.rank();
        if rank >= n || seen[rank] {
            return Err(DistError::InvalidGroup(format!("rank {rank} out of 0..{n} (or duplicated)")));
        }
        seen[rank] = true;
    }
    Ok(())
}
