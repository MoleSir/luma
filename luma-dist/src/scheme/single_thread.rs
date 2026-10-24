use super::validate;
use crate::{Comm, DistResult};
use luma_tensor::Device;

/// One thread driving several local ranks, batched under a single NCCL group.
pub struct SingleThread<D: Device> {
    comms: Vec<Comm<D>>,
}

impl<D: Device> SingleThread<D> {
    pub fn new(comms: Vec<Comm<D>>) -> DistResult<Self> {
        validate(&comms)?;
        Ok(Self { comms })
    }

    pub fn world_size(&self) -> usize {
        self.comms.len()
    }

    pub fn device(&self, rank: usize) -> &D {
        self.comms[rank].device()
    }

    /// Run `f` inside one thread-local group, handing it every local rank.
    pub fn scope<R>(&self, f: impl FnOnce(&[Comm<D>]) -> DistResult<R>) -> DistResult<R> {
        crate::group(&self.comms[0], || f(&self.comms))
    }

    /// Block until every rank reaches this point (auto-grouped).
    pub fn barrier(&self) -> DistResult<()> {
        self.scope(|comms| {
            for comm in comms {
                comm.barrier()?;
            }
            Ok(())
        })
    }
}
