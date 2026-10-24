//! TCP backend tests.
//!
//! Every case is generic over the device `Dev`; the `#[test]` wrappers
//! instantiate them with a concrete device (`Cpu` here). To exercise another
//! device, add a wrapper that passes a matching `make_dev`.

use std::net::TcpListener;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

use luma_cpu::Cpu;
use luma_tensor::{Device, Int, ReduceOp, Tensor};

use super::init_process_group;
use crate::backends::tcp::TcpBackendImpl;
use crate::{Comm, MultiThread};

/// A distinct loopback `host:port` per call (base from a free port, then
/// incremented), so tests running in parallel never collide.
fn unique_addr() -> String {
    static BASE: OnceLock<u16> = OnceLock::new();
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let base = *BASE.get_or_init(|| {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind :0");
        listener.local_addr().expect("local_addr").port()
    });
    let idx = NEXT.fetch_add(1, Ordering::Relaxed);
    // Stay below the OS ephemeral range (32768..60999 here): a bootstrap bind
    // must not collide with an outgoing connection's local port, which caused
    // flaky `ConnectionRefused` under parallel test execution.
    let port = 10000 + ((base as u32 + idx) % 10000);
    format!("127.0.0.1:{port}")
}

/// Build a `world`-rank group over localhost; `make_dev(rank)` supplies each
/// rank's device. Works for any `Dev: Device` (all ranks run as threads here).
fn group<D, F>(impll: TcpBackendImpl, world: usize, make_dev: F) -> MultiThread<D>
where
    D: Device,
    F: Fn(usize) -> D + Send + Sync,
{
    let addr = unique_addr();
    let comms: Vec<Comm<D>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..world)
            .map(|rank| {
                let addr = addr.clone();
                let dev = make_dev(rank);
                scope.spawn(move || init_process_group(impll, &addr, world, rank, dev).expect("init_process_group"))
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("join")).collect()
    });
    MultiThread::new(comms).expect("MultiThread::new")
}

fn all_reduce_case<D: Device>(impll: TcpBackendImpl, make_dev: impl Fn(usize) -> D + Send + Sync) {
    let group = group(impll, 2, make_dev);
    let outs = group
        .run(|rank, comm| {
            let x = Tensor::<D>::full((4,), (rank + 1) as f64, comm.device())?;
            Ok(comm.all_reduce(&x, ReduceOp::Sum)?.to_vec()?)
        })
        .unwrap();
    for out in &outs {
        assert_eq!(out, &vec![3.0; 4]);
    }
}

fn all_reduce_int_case<D: Device>(impll: TcpBackendImpl, make_dev: impl Fn(usize) -> D + Send + Sync) {
    let group = group(impll, 2, make_dev);
    let outs = group
        .run(|rank, comm| {
            let x = Tensor::<D, Int>::from_vec_i32(&[(rank + 1) as i32; 4], (4,), comm.device())?;
            Ok(comm.all_reduce(&x, ReduceOp::Sum)?.to_vec()?)
        })
        .unwrap();
    for out in &outs {
        assert_eq!(out, &vec![3i64; 4]);
    }
}

fn reduce_case<D: Device>(impll: TcpBackendImpl, make_dev: impl Fn(usize) -> D + Send + Sync) {
    let group = group(impll, 2, make_dev);
    let outs = group
        .run(|rank, comm| {
            let x = Tensor::<D>::full((4,), (rank + 1) as f64, comm.device())?;
            Ok(comm.reduce(&x, ReduceOp::Sum, 0)?.map(|t| t.to_vec()).transpose()?)
        })
        .unwrap();
    assert_eq!(outs[0].as_ref().unwrap(), &vec![3.0; 4]);
    assert!(outs[1].is_none());
}

fn all_gather_case<D: Device>(impll: TcpBackendImpl, make_dev: impl Fn(usize) -> D + Send + Sync) {
    let group = group(impll, 2, make_dev);
    let outs = group
        .run(|rank, comm| {
            let x = Tensor::<D>::full((2, 2), (rank + 1) as f64, comm.device())?;
            Ok(comm.all_gather(&x, 0)?.to_vec()?)
        })
        .unwrap();
    for out in &outs {
        assert_eq!(out, &vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]);
    }
}

fn reduce_scatter_case<D: Device>(impll: TcpBackendImpl, make_dev: impl Fn(usize) -> D + Send + Sync) {
    let group = group(impll, 2, make_dev);
    let outs = group
        .run(|_rank, comm| {
            let x = Tensor::<D>::from_vec_f64(&[1.0f64, 2.0, 3.0, 4.0], (4,), comm.device())?;
            Ok(comm.reduce_scatter(&x, ReduceOp::Sum, 0)?.to_vec()?)
        })
        .unwrap();
    assert_eq!(outs[0], vec![2.0, 4.0]);
    assert_eq!(outs[1], vec![6.0, 8.0]);
}

fn broadcast_case<D: Device>(impll: TcpBackendImpl, make_dev: impl Fn(usize) -> D + Send + Sync) {
    let group = group(impll, 2, make_dev);
    let outs = group
        .run(|rank, comm| {
            let x = if rank == 0 {
                Some(Tensor::<D>::from_vec_f64(&[1.0f64, 2.0, 3.0], (3,), comm.device())?)
            } else {
                None
            };
            Ok(comm.broadcast(x.as_ref())?.to_vec()?)
        })
        .unwrap();
    for out in &outs {
        assert_eq!(out, &vec![1.0, 2.0, 3.0]);
    }
}

fn send_recv_case<D: Device>(impll: TcpBackendImpl, make_dev: impl Fn(usize) -> D + Send + Sync) {
    let group = group(impll, 2, make_dev);
    let outs = group
        .run(|rank, comm| {
            if rank == 0 {
                let x = Tensor::<D>::from_vec_f64(&[10.0f64, 20.0, 30.0, 40.0], (4,), comm.device())?;
                comm.send(&x, 1)?;
                Ok(Vec::new())
            } else {
                let got: Tensor<D> = comm.recv(0)?;
                Ok(got.to_vec()?)
            }
        })
        .unwrap();
    assert_eq!(outs[1], vec![10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_all_reduce() {
    all_reduce_case(TcpBackendImpl::Star, |_| Cpu::default());
}

#[test]
fn test_ring_all_reduce() {
    all_reduce_case(TcpBackendImpl::Ring, |_| Cpu::default());
}

#[test]
fn test_ring_all_reduce_int() {
    all_reduce_int_case(TcpBackendImpl::Ring, |_| Cpu::default());
}

#[test]
fn test_ring_all_gather() {
    all_gather_case(TcpBackendImpl::Ring, |_| Cpu::default());
}

#[test]
fn test_ring_reduce_scatter() {
    reduce_scatter_case(TcpBackendImpl::Ring, |_| Cpu::default());
}

#[test]
fn test_ring_world4_collectives() {
    let group = group(TcpBackendImpl::Ring, 4, |_| Cpu::default());
    let outs = group
        .run(|rank, comm| {
            let base = (rank * 10) as f64;
            let x = Tensor::<Cpu>::from_vec_f64(
                &[base, base + 1.0, base + 2.0, base + 3.0],
                (4,),
                comm.device(),
            )?;
            let sum = comm.all_reduce(&x, ReduceOp::Sum)?.to_vec()?;
            let gathered = comm.all_gather(&x, 0)?.to_vec()?;
            let scattered = comm.reduce_scatter(&x, ReduceOp::Sum, 0)?.to_vec()?;
            Ok((sum, gathered, scattered))
        })
        .unwrap();

    for (sum, _, _) in &outs {
        assert_eq!(sum, &vec![60.0, 64.0, 68.0, 72.0]);
    }
    for (_, gathered, _) in &outs {
        assert_eq!(
            gathered,
            &vec![
                0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0, 30.0, 31.0,
                32.0, 33.0
            ]
        );
    }
    assert_eq!(outs[0].2, vec![60.0]);
    assert_eq!(outs[1].2, vec![64.0]);
    assert_eq!(outs[2].2, vec![68.0]);
    assert_eq!(outs[3].2, vec![72.0]);
}

#[test]
fn test_ring_reduce() {
    reduce_case(TcpBackendImpl::Ring, |_| Cpu::default());
}

#[test]
fn test_ring_world4_reduce() {
    let group = group(TcpBackendImpl::Ring, 4, |_| Cpu::default());
    let outs = group
        .run(|rank, comm| {
            let x = Tensor::<Cpu>::from_vec_f64(&[(rank + 1) as f64; 4], (4,), comm.device())?;
            Ok(comm.reduce(&x, ReduceOp::Sum, 2)?.map(|t| t.to_vec()).transpose()?)
        })
        .unwrap();
    // 1 + 2 + 3 + 4 = 10, only on root = 2.
    assert_eq!(outs[2].as_ref().unwrap(), &vec![10.0; 4]);
    for r in [0, 1, 3] {
        assert!(outs[r].is_none());
    }
}

#[test]
fn test_barrier() {
    let group = group(TcpBackendImpl::Star, 4, |_| Cpu::default());
    let outs = group
        .run(|rank, comm| {
            comm.barrier()?;
            // A collective after a barrier must still line up.
            let x = Tensor::<Cpu>::from_vec_f64(&[(rank + 1) as f64], (1,), comm.device())?;
            Ok(comm.all_reduce(&x, ReduceOp::Sum)?.to_vec()?)
        })
        .unwrap();
    for out in &outs {
        assert_eq!(out, &vec![10.0]);
    }
}

#[test]
fn test_ring_barrier() {
    let group = group(TcpBackendImpl::Ring, 4, |_| Cpu::default());
    let outs = group
        .run(|rank, comm| {
            comm.barrier()?;
            // A collective after a barrier must still line up.
            let x = Tensor::<Cpu>::from_vec_f64(&[(rank + 1) as f64; 4], (4,), comm.device())?;
            Ok(comm.all_reduce(&x, ReduceOp::Sum)?.to_vec()?)
        })
        .unwrap();
    for out in &outs {
        assert_eq!(out, &vec![10.0; 4]);
    }
}

#[test]
fn test_ring_send_recv() {
    send_recv_case(TcpBackendImpl::Ring, |_| Cpu::default());
}

#[test]
fn test_ring_world4_send_recv() {
    let group = group(TcpBackendImpl::Ring, 4, |_| Cpu::default());
    let outs = group
        .run(|rank, comm| {
            let world = 4;
            let next = (rank + 1) % world;
            let prev = (rank + world - 1) % world;

            // Forward edge: send my rank to `next`, receive the rank before me.
            let fwd = Tensor::<Cpu>::from_vec_f64(&[rank as f64], (1,), comm.device())?;
            comm.send(&fwd, next)?;
            let got_fwd: Tensor<Cpu> = comm.recv(prev)?;

            // Backward edge: send rank+100 to `prev`, receive from `next`.
            let bwd = Tensor::<Cpu>::from_vec_f64(&[rank as f64 + 100.0], (1,), comm.device())?;
            comm.send(&bwd, prev)?;
            let got_bwd: Tensor<Cpu> = comm.recv(next)?;

            Ok((got_fwd.to_vec()?, got_bwd.to_vec()?))
        })
        .unwrap();

    for rank in 0..4 {
        let prev = (rank + 3) % 4;
        let next = (rank + 1) % 4;
        assert_eq!(outs[rank].0, vec![prev as f64]);
        assert_eq!(outs[rank].1, vec![next as f64 + 100.0]);
    }
}

#[test]
fn test_all_reduce_int() {
    all_reduce_int_case(TcpBackendImpl::Star, |_| Cpu::default());
}

#[test]
fn test_reduce() {
    reduce_case(TcpBackendImpl::Star, |_| Cpu::default());
}

#[test]
fn test_all_gather() {
    all_gather_case(TcpBackendImpl::Star, |_| Cpu::default());
}

#[test]
fn test_reduce_scatter() {
    reduce_scatter_case(TcpBackendImpl::Star, |_| Cpu::default());
}

#[test]
fn test_broadcast() {
    broadcast_case(TcpBackendImpl::Star, |_| Cpu::default());
}

#[test]
fn test_send_recv() {
    send_recv_case(TcpBackendImpl::Star, |_| Cpu::default());
}

// ---- full mesh ----

#[test]
fn test_mesh_all_reduce() {
    all_reduce_case(TcpBackendImpl::Mesh, |_| Cpu::default());
}

#[test]
fn test_mesh_all_reduce_int() {
    all_reduce_int_case(TcpBackendImpl::Mesh, |_| Cpu::default());
}

#[test]
fn test_mesh_reduce() {
    reduce_case(TcpBackendImpl::Mesh, |_| Cpu::default());
}

#[test]
fn test_mesh_all_gather() {
    all_gather_case(TcpBackendImpl::Mesh, |_| Cpu::default());
}

#[test]
fn test_mesh_reduce_scatter() {
    reduce_scatter_case(TcpBackendImpl::Mesh, |_| Cpu::default());
}

#[test]
fn test_mesh_broadcast() {
    broadcast_case(TcpBackendImpl::Mesh, |_| Cpu::default());
}

#[test]
fn test_mesh_send_recv() {
    send_recv_case(TcpBackendImpl::Mesh, |_| Cpu::default());
}

#[test]
fn test_mesh_all_to_all() {
    let group = group(TcpBackendImpl::Mesh, 4, |_| Cpu::default());
    let outs = group
        .run(|rank, comm| {
            let x = Tensor::<Cpu>::from_vec_f64(&[rank as f64], (1,), comm.device())?;
            for peer in 0..4 {
                if peer != rank {
                    comm.send(&x, peer)?;
                }
            }
            let mut got = vec![];
            for peer in 0..4 {
                if peer != rank {
                    let t: Tensor<Cpu> = comm.recv(peer)?;
                    got.push(t.to_vec()?);
                }
            }
            Ok(got)
        })
        .unwrap();
    for rank in 0..4usize {
        let expected: Vec<Vec<f64>> = (0..4).filter(|p| *p != rank).map(|p| vec![p as f64]).collect();
        assert_eq!(outs[rank], expected);
    }
}

#[test]
fn test_mesh_world4_collectives() {
    let group = group(TcpBackendImpl::Mesh, 4, |_| Cpu::default());
    let outs = group
        .run(|rank, comm| {
            let base = (rank * 10) as f64;
            let x = Tensor::<Cpu>::from_vec_f64(&[base, base + 1.0, base + 2.0, base + 3.0], (4,), comm.device())?;
            let sum = comm.all_reduce(&x, ReduceOp::Sum)?.to_vec()?;
            let gathered = comm.all_gather(&x, 0)?.to_vec()?;
            let scattered = comm.reduce_scatter(&x, ReduceOp::Sum, 0)?.to_vec()?;
            let reduced = comm.reduce(&x, ReduceOp::Sum, 2)?.map(|t| t.to_vec()).transpose()?;
            let bc = if rank == 2 {
                Some(Tensor::<Cpu>::from_vec_f64(&[7.0, 8.0], (2,), comm.device())?)
            } else {
                None
            };
            let bcast = comm.broadcast(bc.as_ref())?.to_vec()?;
            Ok((sum, gathered, scattered, reduced, bcast))
        })
        .unwrap();

    for (sum, gathered, _, _, bcast) in &outs {
        assert_eq!(sum, &vec![60.0, 64.0, 68.0, 72.0]);
        assert_eq!(
            gathered,
            &vec![
                0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0, 30.0, 31.0,
                32.0, 33.0
            ]
        );
        assert_eq!(bcast, &vec![7.0, 8.0]);
    }
    assert_eq!(outs[0].2, vec![60.0]);
    assert_eq!(outs[1].2, vec![64.0]);
    assert_eq!(outs[2].2, vec![68.0]);
    assert_eq!(outs[3].2, vec![72.0]);
    assert_eq!(outs[2].3.as_ref().unwrap(), &vec![60.0, 64.0, 68.0, 72.0]);
    for r in [0, 1, 3] {
        assert!(outs[r].3.is_none());
    }
}
