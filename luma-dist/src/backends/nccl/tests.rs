use super::{Id, NcclBackend};
use crate::{MultiThread, SingleThread};
use luma_cuda::Cuda;
use luma_tensor::{ReduceOp, Tensor};

/// Two devices (0, 1), or `None` when the machine has fewer than two GPUs.
fn two_devices() -> Option<(Cuda, Cuda)> {
    if cudarc::driver::CudaContext::device_count().ok()? < 2 {
        return None;
    }
    Some((Cuda::new(0).ok()?, Cuda::new(1).ok()?))
}

/// One thread driving both ranks via an automatic group.
#[test]
fn test_single_thread() -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let Some((cuda0, cuda1)) = two_devices() else {
        eprintln!("skip: need >= 2 GPUs");
        return Ok(());
    };
    let group = SingleThread::new(NcclBackend::from_devices(vec![cuda0, cuda1])?)?;

    let xs: Vec<Tensor<Cuda>> = (0..group.world_size())
        .map(|rank| Tensor::<Cuda>::full((4,), (rank + 1) as f64, group.device(rank)))
        .collect::<std::result::Result<_, _>>()?;

    let ys: Vec<Tensor<Cuda>> = group.scope(|comms| comms.iter().zip(&xs).map(|(comm, x)| comm.all_reduce(x, ReduceOp::Sum)).collect())?;

    for y in &ys {
        assert_eq!(y.to_vec()?, vec![3.0; 4]);
    }
    Ok(())
}

/// One thread per rank, no grouping.
#[test]
fn test_thread_per_rank() -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let Some((cuda0, cuda1)) = two_devices() else {
        eprintln!("skip: need >= 2 GPUs");
        return Ok(());
    };
    let group = MultiThread::new(NcclBackend::from_devices(vec![cuda0, cuda1])?)?;

    let outs: Vec<Vec<f64>> = group.run(|rank, comm| {
        let x = Tensor::<Cuda>::full((4,), (rank + 1) as f64, comm.device())?;
        Ok(comm.all_reduce(&x, ReduceOp::Sum)?.to_vec()?)
    })?;

    for out in &outs {
        assert_eq!(out, &vec![3.0; 4]);
    }
    Ok(())
}

/// `all_gather` concatenates each rank's tensor along the requested dim.
#[test]
fn test_all_gather() -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let Some((cuda0, cuda1)) = two_devices() else {
        eprintln!("skip: need >= 2 GPUs");
        return Ok(());
    };
    let group = MultiThread::new(NcclBackend::from_devices(vec![cuda0, cuda1])?)?;

    // rank r fills with (r+1); dim 0  -> [rank0 rows | rank1 rows]
    let outs = group.run(|rank, comm| {
        let x = Tensor::<Cuda>::full((2, 2), (rank + 1) as f64, comm.device())?;
        Ok(comm.all_gather(&x, 0)?.to_vec()?)
    })?;
    for out in &outs {
        assert_eq!(out, &vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]);
    }

    // dim 1 -> interleave columns
    let outs = group.run(|rank, comm| {
        let x = Tensor::<Cuda>::full((2, 2), (rank + 1) as f64, comm.device())?;
        Ok(comm.all_gather(&x, 1)?.to_vec()?)
    })?;
    for out in &outs {
        assert_eq!(out, &vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]);
    }
    Ok(())
}

/// `reduce` produces the result only on the root.
#[test]
fn test_reduce() -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let Some((cuda0, cuda1)) = two_devices() else {
        eprintln!("skip: need >= 2 GPUs");
        return Ok(());
    };
    let group = MultiThread::new(NcclBackend::from_devices(vec![cuda0, cuda1])?)?;

    let outs = group.run(|rank, comm| {
        let x = Tensor::<Cuda>::full((4,), (rank + 1) as f64, comm.device())?;
        let reduced = comm.reduce(&x, ReduceOp::Sum, 0)?;
        Ok(reduced.map(|t| t.to_vec()).transpose()?)
    })?;

    assert_eq!(outs[0].as_ref().unwrap(), &vec![3.0; 4]);
    assert!(outs[1].is_none());
    Ok(())
}

/// `reduce_scatter` sums across ranks then hands each rank its chunk.
#[test]
fn test_reduce_scatter() -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let Some((cuda0, cuda1)) = two_devices() else {
        eprintln!("skip: need >= 2 GPUs");
        return Ok(());
    };
    let group = MultiThread::new(NcclBackend::from_devices(vec![cuda0, cuda1])?)?;

    // every rank sends [1,2,3,4]; sum = [2,4,6,8]; rank0 gets [2,4], rank1 [6,8]
    let outs = group.run(|_rank, comm| {
        let x = Tensor::<Cuda>::from_vec_f64(&[1.0f64, 2.0, 3.0, 4.0], (4,), comm.device())?;
        Ok(comm.reduce_scatter(&x, ReduceOp::Sum, 0)?.to_vec()?)
    })?;
    assert_eq!(outs[0], vec![2.0, 4.0]);
    assert_eq!(outs[1], vec![6.0, 8.0]);
    Ok(())
}

/// `broadcast` copies root's tensor to every rank.
#[test]
fn test_broadcast() -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let Some((cuda0, cuda1)) = two_devices() else {
        eprintln!("skip: need >= 2 GPUs");
        return Ok(());
    };
    let group = MultiThread::new(NcclBackend::from_devices(vec![cuda0, cuda1])?)?;

    let outs = group.run(|rank, comm| {
        let x = if rank == 0 {
            Some(Tensor::<Cuda>::from_vec_f32(&[1.0f32, 2.0, 3.0], (3,), comm.device())?)
        } else {
            None
        };
        Ok(comm.broadcast(x.as_ref())?.to_vec()?)
    })?;
    for out in &outs {
        assert_eq!(out, &vec![1.0, 2.0, 3.0]);
    }
    Ok(())
}

/// Point-to-point `send` / `recv`.
#[test]
fn test_send_recv() -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let Some((cuda0, cuda1)) = two_devices() else {
        eprintln!("skip: need >= 2 GPUs");
        return Ok(());
    };
    let group = MultiThread::new(NcclBackend::from_devices(vec![cuda0, cuda1])?)?;

    let outs = group.run(|rank, comm| {
        if rank == 0 {
            let x = Tensor::<Cuda>::from_vec_f32(&[10.0f32, 20.0, 30.0, 40.0], (4,), comm.device())?;
            comm.send(&x, 1)?;
            Ok(Vec::new())
        } else {
            let got: Tensor<Cuda> = comm.recv(0)?;
            Ok(got.to_vec()?)
        }
    })?;
    assert_eq!(outs[1], vec![10.0, 20.0, 30.0, 40.0]);
    Ok(())
}

/// `barrier` does not deadlock.
#[test]
fn test_barrier() -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let Some((cuda0, cuda1)) = two_devices() else {
        eprintln!("skip: need >= 2 GPUs");
        return Ok(());
    };
    let group = MultiThread::new(NcclBackend::from_devices(vec![cuda0, cuda1])?)?;
    group.run(|_rank, comm| comm.barrier())?;
    Ok(())
}

/// Multi-process building block: each rank bootstraps via `from_rank`.
#[test]
fn test_from_rank() -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if two_devices().is_none() {
        eprintln!("skip: need >= 2 GPUs");
        return Ok(());
    }
    let world_size = 2usize;
    let id = Id::new().expect("nccl unique id");

    std::thread::scope(|scope| {
        for rank in 0..world_size {
            scope.spawn(move || {
                let device = Cuda::new(rank).unwrap();
                let comm = NcclBackend::from_rank(device, rank, world_size, id).unwrap();
                let x = Tensor::<Cuda>::full((4,), (rank + 1) as f64, comm.device()).unwrap();
                let y = comm.all_reduce(&x, ReduceOp::Sum).unwrap();
                assert_eq!(y.to_vec().unwrap(), vec![3.0; 4]);
            });
        }
    });
    Ok(())
}
