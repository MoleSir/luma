use crate::rendezvous::Rendezvous;
use crate::{Comm, DistBackend, DistError, DistResult};
use cudarc::driver::{CudaSlice, DevicePtr, DeviceRepr, ValidAsZeroBits};
use cudarc::nccl;
pub use cudarc::nccl::Id;
use luma_cuda::Cuda;
use luma_tensor::{DType, DTypeKind, Shape, Tensor};

mod backend;
mod error;
#[cfg(test)]
mod tests;

pub use error::NcclError;
use error::map_nccl_err;

/// A one-rank-per-device NCCL communicator.
pub struct NcclBackend {
    comm: nccl::Comm,
    device: Cuda,
}

impl NcclBackend {
    /// Single-process, multi-GPU construction: one [`Comm`] per device, with
    /// `rank == index` in `cudas`.
    pub fn from_devices(cudas: Vec<Cuda>) -> DistResult<Vec<Comm<Cuda>>> {
        let streams = cudas.iter().map(|cuda| cuda.stream()).collect();
        let nccl_comms = nccl::Comm::from_devices(streams).map_err(map_nccl_err)?;
        Ok(nccl_comms.into_iter().zip(cudas).map(|(comm, device)| Comm::from_backend(NcclBackend { comm, device })).collect())
    }

    /// One rank of a group, bound to `device`.
    ///
    /// `id` must be obtained from [`init_process_group`]; use that instead unless
    /// you already manage the unique-id exchange yourself.
    pub fn from_rank(device: Cuda, rank: usize, world_size: usize, id: Id) -> DistResult<Comm<Cuda>> {
        if rank >= world_size {
            return Err(DistError::InvalidRank { rank, world_size });
        }
        let comm = nccl::Comm::from_rank(device.stream(), rank, world_size, id).map_err(map_nccl_err)?;
        Ok(Comm::from_backend(NcclBackend { comm, device }))
    }

    // --- collective primitives, on plain slices -----------------------------

    fn all_reduce_into<T, S>(&self, src: &S, count: usize, op: &nccl::ReduceOp) -> DistResult<CudaSlice<T>>
    where
        T: nccl::NcclType + DeviceRepr + ValidAsZeroBits,
        S: DevicePtr<T>,
    {
        let mut dst = self.device.alloc_zeros::<T>(count).map_err(NcclError::Cuda)?;
        self.comm.all_reduce(src, &mut dst, op).map_err(map_nccl_err)?;
        Ok(dst)
    }

    fn all_gather_into<T, S>(&self, src: &S, count: usize) -> DistResult<CudaSlice<T>>
    where
        T: nccl::NcclType + DeviceRepr + ValidAsZeroBits,
        S: DevicePtr<T>,
    {
        let mut dst = self.device.alloc_zeros::<T>(count * self.comm.world_size()).map_err(NcclError::Cuda)?;
        self.comm.all_gather(src, &mut dst).map_err(map_nccl_err)?;
        Ok(dst)
    }

    fn reduce_scatter_into<T, S>(&self, src: &S, count: usize, op: &nccl::ReduceOp) -> DistResult<CudaSlice<T>>
    where
        T: nccl::NcclType + DeviceRepr + ValidAsZeroBits,
        S: DevicePtr<T>,
    {
        let mut dst = self.device.alloc_zeros::<T>(count / self.comm.world_size()).map_err(NcclError::Cuda)?;
        self.comm.reduce_scatter(src, &mut dst, op).map_err(map_nccl_err)?;
        Ok(dst)
    }

    fn broadcast_into<T, S>(&self, src: &S, count: usize, root: usize) -> DistResult<CudaSlice<T>>
    where
        T: nccl::NcclType + DeviceRepr + ValidAsZeroBits,
        S: DevicePtr<T>,
    {
        let mut dst = self.device.alloc_zeros::<T>(count).map_err(NcclError::Cuda)?;
        self.comm.broadcast(Some(src), &mut dst, root as i32).map_err(map_nccl_err)?;
        Ok(dst)
    }

    fn reduce_into<T, S>(&self, src: &S, count: usize, op: &nccl::ReduceOp, root: usize) -> DistResult<Option<CudaSlice<T>>>
    where
        T: nccl::NcclType + DeviceRepr + ValidAsZeroBits,
        S: DevicePtr<T>,
    {
        if self.comm.rank() == root {
            let mut dst = self.device.alloc_zeros::<T>(count).map_err(NcclError::Cuda)?;
            self.comm.reduce(src, Some(&mut dst), op, root as i32).map_err(map_nccl_err)?;
            Ok(Some(dst))
        } else {
            self.comm.reduce(src, None::<&mut CudaSlice<T>>, op, root as i32).map_err(map_nccl_err)?;
            Ok(None)
        }
    }

    fn send_slice<T, S>(&self, src: &S, peer: usize) -> DistResult<()>
    where
        T: nccl::NcclType,
        S: DevicePtr<T>,
    {
        self.comm.send(src, peer as i32).map_err(map_nccl_err)?;
        Ok(())
    }

    fn recv_slice<T>(&self, count: usize, peer: usize) -> DistResult<CudaSlice<T>>
    where
        T: nccl::NcclType + DeviceRepr + ValidAsZeroBits,
    {
        let mut dst = self.device.alloc_zeros::<T>(count).map_err(NcclError::Cuda)?;
        self.comm.recv(&mut dst, peer as i32).map_err(map_nccl_err)?;
        Ok(dst)
    }

    // --- broadcast support ------------------------------------------------
    //
    // `broadcast` takes no explicit root: exactly one rank passes `Some(x)` and
    // the others pass `None`. NCCL needs the root index up front and every rank
    // needs a same-sized buffer, so we (1) find the source via an all-reduce of
    // an indicator, then (2) broadcast a small shape/dtype header so receivers
    // can allocate, then (3) broadcast the data.

    /// The rank that passed `Some` (via an all-reduce MAX of an indicator).
    fn find_source(&self, has_data: bool) -> DistResult<usize> {
        let v = if has_data { self.rank() as i32 } else { -1 };
        let out = self.all_reduce_i32(&[v], &nccl::ReduceOp::Max)?;
        if out[0] < 0 {
            Err(DistError::Backend("broadcast: exactly one rank must pass Some(tensor)".into()))
        } else {
            Ok(out[0] as usize)
        }
    }

    /// Broadcast `(shape, dtype)` from the source so every rank can allocate.
    fn broadcast_header<K: DTypeKind<Cuda>>(&self, x: Option<&Tensor<Cuda, K>>, root: usize) -> DistResult<(Shape, K::DType)> {
        let hdr = match x {
            Some(t) if self.rank() == root => encode_header(t),
            _ => vec![0i32; HEADER_I32],
        };
        let hdr = self.all_reduce_i32(&hdr, &nccl::ReduceOp::Sum)?;
        decode_header::<K>(&hdr)
    }

    /// Point-to-point variants of the header: `send` prepends it, `recv` reads it.
    fn send_header<K: DTypeKind<Cuda>>(&self, x: &Tensor<Cuda, K>, peer: usize) -> DistResult<()> {
        self.send_i32(&encode_header(x), peer)
    }

    fn recv_header<K: DTypeKind<Cuda>>(&self, peer: usize) -> DistResult<(Shape, K::DType)> {
        let hdr = self.recv_i32(HEADER_I32, peer)?;
        decode_header::<K>(&hdr)
    }

    fn all_reduce_i32(&self, vals: &[i32], op: &nccl::ReduceOp) -> DistResult<Vec<i32>> {
        let src = self.device.memcpy_stod(vals).map_err(NcclError::Cuda)?;
        let mut dst = self.device.alloc_zeros::<i32>(vals.len()).map_err(NcclError::Cuda)?;
        self.comm.all_reduce(&src, &mut dst, op).map_err(map_nccl_err)?;
        Ok(self.device.memcpy_dtov(&dst).map_err(NcclError::Cuda)?)
    }

    fn send_i32(&self, vals: &[i32], peer: usize) -> DistResult<()> {
        let src = self.device.memcpy_stod(vals).map_err(NcclError::Cuda)?;
        self.send_slice(&src, peer)
    }

    fn recv_i32(&self, count: usize, peer: usize) -> DistResult<Vec<i32>> {
        let dst = self.recv_slice::<i32>(count, peer)?;
        Ok(self.device.memcpy_dtov(&dst).map_err(NcclError::Cuda)?)
    }
}

const MAX_DIMS: usize = 8;
const HEADER_I32: usize = 2 + MAX_DIMS;

fn encode_header<K: DTypeKind<Cuda>>(x: &Tensor<Cuda, K>) -> Vec<i32> {
    let shape = x.shape();
    let mut hdr = vec![0i32; HEADER_I32];
    hdr[0] = shape.rank() as i32;
    hdr[1] = dtype_to_i32(x.dtype().into());
    for (i, &d) in shape.dims().iter().enumerate() {
        hdr[2 + i] = d as i32;
    }
    hdr
}

fn decode_header<K: DTypeKind<Cuda>>(hdr: &[i32]) -> DistResult<(Shape, K::DType)> {
    let nd = hdr[0] as usize;
    let dtype = K::DType::try_from(i32_to_dtype(hdr[1])?)?;
    let dims: Vec<usize> = hdr[2..2 + nd].iter().map(|&d| d as usize).collect();
    Ok((Shape::from(dims), dtype))
}

fn dtype_to_i32(dtype: DType) -> i32 {
    match dtype {
        DType::F32 => 0,
        DType::F64 => 1,
        DType::F16 => 2,
        DType::BF16 => 3,
        DType::I32 => 4,
        DType::U32 => 5,
        DType::U8 => 6,
        DType::Bool => 7,
    }
}

fn i32_to_dtype(v: i32) -> DistResult<DType> {
    Ok(match v {
        0 => DType::F32,
        1 => DType::F64,
        2 => DType::F16,
        3 => DType::BF16,
        4 => DType::I32,
        5 => DType::U32,
        6 => DType::U8,
        7 => DType::Bool,
        _ => return Err(DistError::Backend(format!("broadcast: bad dtype code {v}"))),
    })
}

/// One-call, one-rank bootstrap for the process-per-rank scheme.
///
/// Every process calls this with its own `rank`, the shared `world_size` and a
/// `device` (which may be the same on several ranks — e.g. several ranks
/// multiplexed onto one GPU). Rank 0 generates the NCCL unique id and publishes
/// it through `rndv`; every other rank retrieves it; then all ranks join the
/// group on their device.
///
/// ```ignore
/// use luma_dist::{tcp, backends::nccl};
/// use luma_cuda::Cuda;
///
/// let comm = nccl::init_process_group(tcp!("localhost:2333"), world_size, rank, Cuda::new(rank)?)?;
/// ```
pub fn init_process_group(rndv: impl Rendezvous, world_size: usize, rank: usize, device: Cuda) -> DistResult<Comm<Cuda>> {
    if rank >= world_size {
        return Err(DistError::InvalidRank { rank, world_size });
    }

    let id = if rank == 0 {
        let id = Id::new().map_err(map_nccl_err)?;
        rndv.exchange(rank, world_size, &id_to_bytes(&id))?;
        id
    } else {
        bytes_to_id(&rndv.exchange(rank, world_size, &[])?)?
    };

    NcclBackend::from_rank(device, rank, world_size, id)
}

pub fn init_group(cudas: Vec<Cuda>) -> DistResult<Vec<Comm<Cuda>>> {
    NcclBackend::from_devices(cudas)
}

/// The NCCL unique id is exactly 128 bytes; copy it out for transport.
fn id_to_bytes(id: &Id) -> [u8; 128] {
    let mut bytes = [0u8; 128];
    for (dst, src) in bytes.iter_mut().zip(id.internal()) {
        *dst = *src as u8;
    }
    bytes
}

/// Rebuild an `Id` from a transported payload.
fn bytes_to_id(bytes: &[u8]) -> DistResult<Id> {
    if bytes.len() != 128 {
        return Err(DistError::Rendezvous(format!("nccl unique id must be 128 bytes, got {}", bytes.len())));
    }
    let mut id = [0 as std::ffi::c_char; 128];
    for (dst, src) in id.iter_mut().zip(bytes) {
        *dst = *src as std::ffi::c_char;
    }
    Ok(Id::uninit(id))
}

unsafe impl Send for NcclBackend {}
unsafe impl Sync for NcclBackend {}
