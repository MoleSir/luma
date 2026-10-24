use super::validate;
use crate::{Comm, DistError, DistResult};
use luma_tensor::Device;

/// One thread per local rank; each thread drives exactly one [`Comm`].
pub struct MultiThread<D: Device> {
    comms: Vec<Comm<D>>,
}

impl<D: Device> MultiThread<D> {
    pub fn new(comms: Vec<Comm<D>>) -> DistResult<Self> {
        validate(&comms)?;
        Ok(Self { comms })
    }

    pub fn world_size(&self) -> usize {
        self.comms.len()
    }

    pub fn comm(&self, rank: usize) -> &Comm<D> {
        &self.comms[rank]
    }

    /// Spawn one thread per rank, run `f(rank, comm)` in each, and collect the
    pub fn run<R: Send>(&self, f: impl Fn(usize, Comm<D>) -> DistResult<R> + Send + Sync) -> DistResult<Vec<R>> {
        let f = &f;
        std::thread::scope(|s| {
            let handles: Vec<_> = self
                .comms
                .iter()
                .enumerate()
                .map(|(rank, comm)| {
                    let comm = comm.clone();
                    (rank, s.spawn(move || f(rank, comm)))
                })
                .collect();

            let mut out: Vec<(usize, R)> = Vec::with_capacity(handles.len());
            for (rank, handle) in handles {
                match handle.join() {
                    Ok(result) => out.push((rank, result?)),
                    Err(_) => return Err(DistError::Backend("rank thread panicked".into())),
                }
            }
            out.sort_by_key(|(rank, _)| *rank);
            Ok(out.into_iter().map(|(_, value)| value).collect())
        })
    }
}
