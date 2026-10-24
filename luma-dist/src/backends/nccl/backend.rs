use super::NcclBackend;
use super::error::{map_nccl_err, NcclError};
use crate::utils::{front_layout, restore_layout};
use crate::{DistBackend, DistError, DistResult};
use cudarc::nccl;
use luma_cuda::{Cuda, CudaBoolStorage, CudaFloatSlice, CudaFloatStorage, CudaIntSlice, CudaIntStorage};
use luma_tensor::{Bool, Device, Float, FloatDType, FloatMeta, Int, IntDType, ReduceOp, Shape, Tensor};

macro_rules! float_map {
    ($storage:expr, $offset:expr, $body:expr) => {{
        match &$storage.slice {
            CudaFloatSlice::F32(s) => CudaFloatSlice::F32($body(s.slice($offset..))?),
            CudaFloatSlice::F64(s) => CudaFloatSlice::F64($body(s.slice($offset..))?),
            CudaFloatSlice::F16(s) => CudaFloatSlice::F16($body(s.slice($offset..))?),
            CudaFloatSlice::BF16(s) => CudaFloatSlice::BF16($body(s.slice($offset..))?),
        }
    }};
}

macro_rules! int_map {
    ($storage:expr, $offset:expr, $body:expr) => {{
        match &$storage.slice {
            CudaIntSlice::I32(s) => CudaIntSlice::I32($body(s.slice($offset..))?),
            CudaIntSlice::U32(s) => CudaIntSlice::U32($body(s.slice($offset..))?),
            CudaIntSlice::U8(s) => CudaIntSlice::U8($body(s.slice($offset..))?),
        }
    }};
}

macro_rules! float_each {
    ($storage:expr, $offset:expr, $body:expr) => {{
        match &$storage.slice {
            CudaFloatSlice::F32(s) => $body(&s.slice($offset..))?,
            CudaFloatSlice::F64(s) => $body(&s.slice($offset..))?,
            CudaFloatSlice::F16(s) => $body(&s.slice($offset..))?,
            CudaFloatSlice::BF16(s) => $body(&s.slice($offset..))?,
        }
    }};
}

macro_rules! int_each {
    ($storage:expr, $offset:expr, $body:expr) => {{
        match &$storage.slice {
            CudaIntSlice::I32(s) => $body(&s.slice($offset..))?,
            CudaIntSlice::U32(s) => $body(&s.slice($offset..))?,
            CudaIntSlice::U8(s) => $body(&s.slice($offset..))?,
        }
    }};
}

macro_rules! float_map_opt {
    ($storage:expr, $offset:expr, $body:expr) => {{
        match &$storage.slice {
            CudaFloatSlice::F32(s) => $body(s.slice($offset..))?.map(CudaFloatSlice::F32),
            CudaFloatSlice::F64(s) => $body(s.slice($offset..))?.map(CudaFloatSlice::F64),
            CudaFloatSlice::F16(s) => $body(s.slice($offset..))?.map(CudaFloatSlice::F16),
            CudaFloatSlice::BF16(s) => $body(s.slice($offset..))?.map(CudaFloatSlice::BF16),
        }
    }};
}

macro_rules! int_map_opt {
    ($storage:expr, $offset:expr, $body:expr) => {{
        match &$storage.slice {
            CudaIntSlice::I32(s) => $body(s.slice($offset..))?.map(CudaIntSlice::I32),
            CudaIntSlice::U32(s) => $body(s.slice($offset..))?.map(CudaIntSlice::U32),
            CudaIntSlice::U8(s) => $body(s.slice($offset..))?.map(CudaIntSlice::U8),
        }
    }};
}

impl DistBackend<Cuda> for NcclBackend {
    fn name(&self) -> String {
        format!("nccl-{}", self.device.name())
    }

    fn device(&self) -> &Cuda {
        &self.device
    }

    fn rank(&self) -> usize {
        self.comm.rank()
    }

    fn world_size(&self) -> usize {
        self.comm.world_size()
    }

    fn all_reduce_float(&self, x: &Tensor<Cuda, Float>, op: ReduceOp) -> DistResult<Tensor<Cuda, Float>> {
        let x = x.contiguous()?;
        let count = x.element_count();
        let offset = x.layout().start_offset();
        let op = to_nccl_reduce_op(op);

        let guard = x.storage_read()?;
        let slice = float_map!(&*guard, offset, |src| self.all_reduce_into(&src, count, &op));
        drop(guard);

        Ok(Tensor::from_storage(CudaFloatStorage { slice, device: self.device.clone() }, x.shape(), FloatMeta::val()))
    }

    fn all_reduce_int(&self, x: &Tensor<Cuda, Int>, op: ReduceOp) -> DistResult<Tensor<Cuda, Int>> {
        let x = x.contiguous()?;
        let count = x.element_count();
        let offset = x.layout().start_offset();
        let op = to_nccl_reduce_op(op);

        let guard = x.storage_read()?;
        let slice = int_map!(&*guard, offset, |src| self.all_reduce_into(&src, count, &op));
        drop(guard);

        Ok(Tensor::from_storage(CudaIntStorage { slice, device: self.device.clone() }, x.shape(), ()))
    }

    fn all_reduce_bool(&self, x: &Tensor<Cuda, Bool>, op: ReduceOp) -> DistResult<Tensor<Cuda, Bool>> {
        let x = x.contiguous()?;
        let count = x.element_count();
        let offset = x.layout().start_offset();
        let op = to_nccl_reduce_op(op);

        let guard = x.storage_read()?;
        let slice = self.all_reduce_into(&guard.slice.slice(offset..), count, &op)?;
        drop(guard);

        Ok(Tensor::from_storage(CudaBoolStorage { slice, device: self.device.clone() }, x.shape(), ()))
    }

    fn reduce_float(&self, x: &Tensor<Cuda, Float>, op: ReduceOp, root: usize) -> DistResult<Option<Tensor<Cuda, Float>>> {
        let x = x.contiguous()?;
        let count = x.element_count();
        let offset = x.layout().start_offset();
        let op = to_nccl_reduce_op(op);

        let guard = x.storage_read()?;
        let slice = float_map_opt!(&*guard, offset, |src| self.reduce_into(&src, count, &op, root));
        drop(guard);

        Ok(slice.map(|slice| Tensor::from_storage(CudaFloatStorage { slice, device: self.device.clone() }, x.shape(), FloatMeta::val())))
    }

    fn reduce_int(&self, x: &Tensor<Cuda, Int>, op: ReduceOp, root: usize) -> DistResult<Option<Tensor<Cuda, Int>>> {
        let x = x.contiguous()?;
        let count = x.element_count();
        let offset = x.layout().start_offset();
        let op = to_nccl_reduce_op(op);

        let guard = x.storage_read()?;
        let slice = int_map_opt!(&*guard, offset, |src| self.reduce_into(&src, count, &op, root));
        drop(guard);

        Ok(slice.map(|slice| Tensor::from_storage(CudaIntStorage { slice, device: self.device.clone() }, x.shape(), ())))
    }

    fn reduce_bool(&self, x: &Tensor<Cuda, Bool>, op: ReduceOp, root: usize) -> DistResult<Option<Tensor<Cuda, Bool>>> {
        let x = x.contiguous()?;
        let count = x.element_count();
        let offset = x.layout().start_offset();
        let op = to_nccl_reduce_op(op);

        let guard = x.storage_read()?;
        let slice = self.reduce_into(&guard.slice.slice(offset..), count, &op, root)?;
        drop(guard);

        Ok(slice.map(|slice| Tensor::from_storage(CudaBoolStorage { slice, device: self.device.clone() }, x.shape(), ())))
    }

    fn all_gather_float(&self, x: &Tensor<Cuda, Float>, dim: usize) -> DistResult<Tensor<Cuda, Float>> {
        let (xp, inv) = front_layout(x, dim)?;
        let count = xp.element_count();
        let offset = xp.layout().start_offset();

        let guard = xp.storage_read()?;
        let slice = float_map!(&*guard, offset, |src| self.all_gather_into(&src, count));
        drop(guard);

        let mut dims = xp.dims().to_vec();
        dims[0] *= self.comm.world_size();
        let gathered = Tensor::from_storage(CudaFloatStorage { slice, device: self.device.clone() }, Shape::from(dims), FloatMeta::val());
        restore_layout(gathered, inv)
    }

    fn all_gather_int(&self, x: &Tensor<Cuda, Int>, dim: usize) -> DistResult<Tensor<Cuda, Int>> {
        let (xp, inv) = front_layout(x, dim)?;
        let count = xp.element_count();
        let offset = xp.layout().start_offset();

        let guard = xp.storage_read()?;
        let slice = int_map!(&*guard, offset, |src| self.all_gather_into(&src, count));
        drop(guard);

        let mut dims = xp.dims().to_vec();
        dims[0] *= self.comm.world_size();
        let gathered = Tensor::from_storage(CudaIntStorage { slice, device: self.device.clone() }, Shape::from(dims), ());
        restore_layout(gathered, inv)
    }

    fn all_gather_bool(&self, x: &Tensor<Cuda, Bool>, dim: usize) -> DistResult<Tensor<Cuda, Bool>> {
        let (xp, inv) = front_layout(x, dim)?;
        let count = xp.element_count();
        let offset = xp.layout().start_offset();

        let guard = xp.storage_read()?;
        let slice = self.all_gather_into(&guard.slice.slice(offset..), count)?;
        drop(guard);

        let mut dims = xp.dims().to_vec();
        dims[0] *= self.comm.world_size();
        let gathered = Tensor::from_storage(CudaBoolStorage { slice, device: self.device.clone() }, Shape::from(dims), ());
        restore_layout(gathered, inv)
    }

    fn reduce_scatter_float(&self, x: &Tensor<Cuda, Float>, op: ReduceOp, dim: usize) -> DistResult<Tensor<Cuda, Float>> {
        let world = self.comm.world_size();
        let (xp, inv) = front_layout(x, dim)?;
        check_divisible(xp.dims()[0], world, dim)?;
        let count = xp.element_count();
        let offset = xp.layout().start_offset();
        let op = to_nccl_reduce_op(op);

        let guard = xp.storage_read()?;
        let slice = float_map!(&*guard, offset, |src| self.reduce_scatter_into(&src, count, &op));
        drop(guard);

        let mut dims = xp.dims().to_vec();
        dims[0] /= world;
        let reduced = Tensor::from_storage(CudaFloatStorage { slice, device: self.device.clone() }, Shape::from(dims), FloatMeta::val());
        restore_layout(reduced, inv)
    }

    fn reduce_scatter_int(&self, x: &Tensor<Cuda, Int>, op: ReduceOp, dim: usize) -> DistResult<Tensor<Cuda, Int>> {
        let world = self.comm.world_size();
        let (xp, inv) = front_layout(x, dim)?;
        check_divisible(xp.dims()[0], world, dim)?;
        let count = xp.element_count();
        let offset = xp.layout().start_offset();
        let op = to_nccl_reduce_op(op);

        let guard = xp.storage_read()?;
        let slice = int_map!(&*guard, offset, |src| self.reduce_scatter_into(&src, count, &op));
        drop(guard);

        let mut dims = xp.dims().to_vec();
        dims[0] /= world;
        let reduced = Tensor::from_storage(CudaIntStorage { slice, device: self.device.clone() }, Shape::from(dims), ());
        restore_layout(reduced, inv)
    }

    fn reduce_scatter_bool(&self, x: &Tensor<Cuda, Bool>, op: ReduceOp, dim: usize) -> DistResult<Tensor<Cuda, Bool>> {
        let world = self.comm.world_size();
        let (xp, inv) = front_layout(x, dim)?;
        check_divisible(xp.dims()[0], world, dim)?;
        let count = xp.element_count();
        let offset = xp.layout().start_offset();
        let op = to_nccl_reduce_op(op);

        let guard = xp.storage_read()?;
        let slice = self.reduce_scatter_into(&guard.slice.slice(offset..), count, &op)?;
        drop(guard);

        let mut dims = xp.dims().to_vec();
        dims[0] /= world;
        let reduced = Tensor::from_storage(CudaBoolStorage { slice, device: self.device.clone() }, Shape::from(dims), ());
        restore_layout(reduced, inv)
    }

    fn broadcast_float(&self, x: Option<&Tensor<Cuda, Float>>) -> DistResult<Tensor<Cuda, Float>> {
        let root = self.find_source(x.is_some())?;
        let (shape, dtype) = self.broadcast_header(x, root)?;
        let src = match x {
            Some(x) => x.contiguous()?,
            None => Tensor::<Cuda, Float>::zeros(shape.clone(), (&self.device, dtype))?,
        };
        let count = shape.element_count();
        let offset = src.layout().start_offset();

        let guard = src.storage_read()?;
        let slice = float_map!(&*guard, offset, |s| self.broadcast_into(&s, count, root));
        drop(guard);

        Ok(Tensor::from_storage(CudaFloatStorage { slice, device: self.device.clone() }, shape, FloatMeta::val()))
    }

    fn broadcast_int(&self, x: Option<&Tensor<Cuda, Int>>) -> DistResult<Tensor<Cuda, Int>> {
        let root = self.find_source(x.is_some())?;
        let (shape, dtype) = self.broadcast_header(x, root)?;
        let src = match x {
            Some(x) => x.contiguous()?,
            None => Tensor::<Cuda, Int>::zeros(shape.clone(), (&self.device, dtype))?,
        };
        let count = shape.element_count();
        let offset = src.layout().start_offset();

        let guard = src.storage_read()?;
        let slice = int_map!(&*guard, offset, |s| self.broadcast_into(&s, count, root));
        drop(guard);

        Ok(Tensor::from_storage(CudaIntStorage { slice, device: self.device.clone() }, shape, ()))
    }

    fn broadcast_bool(&self, x: Option<&Tensor<Cuda, Bool>>) -> DistResult<Tensor<Cuda, Bool>> {
        let root = self.find_source(x.is_some())?;
        let (shape, dtype) = self.broadcast_header(x, root)?;
        let src = match x {
            Some(x) => x.contiguous()?,
            None => Tensor::<Cuda, Bool>::zeros(shape.clone(), (&self.device, dtype))?,
        };
        let count = shape.element_count();
        let offset = src.layout().start_offset();

        let guard = src.storage_read()?;
        let slice = self.broadcast_into(&guard.slice.slice(offset..), count, root)?;
        drop(guard);

        Ok(Tensor::from_storage(CudaBoolStorage { slice, device: self.device.clone() }, shape, ()))
    }

    fn send_float(&self, x: &Tensor<Cuda, Float>, peer: usize) -> DistResult<()> {
        self.send_header(x, peer)?;
        let x = x.contiguous()?;
        let offset = x.layout().start_offset();
        let guard = x.storage_read()?;
        float_each!(&*guard, offset, |src| self.send_slice(src, peer));
        Ok(())
    }

    fn send_int(&self, x: &Tensor<Cuda, Int>, peer: usize) -> DistResult<()> {
        self.send_header(x, peer)?;
        let x = x.contiguous()?;
        let offset = x.layout().start_offset();
        let guard = x.storage_read()?;
        int_each!(&*guard, offset, |src| self.send_slice(src, peer));
        Ok(())
    }

    fn send_bool(&self, x: &Tensor<Cuda, Bool>, peer: usize) -> DistResult<()> {
        self.send_header(x, peer)?;
        let x = x.contiguous()?;
        let offset = x.layout().start_offset();
        let guard = x.storage_read()?;
        self.send_slice(&guard.slice.slice(offset..), peer)?;
        Ok(())
    }

    fn recv_float(&self, peer: usize) -> DistResult<Tensor<Cuda, Float>> {
        let (shape, dtype) = self.recv_header::<Float>(peer)?;
        let count = shape.element_count();
        let slice = match dtype {
            FloatDType::F32 => CudaFloatSlice::F32(self.recv_slice::<f32>(count, peer)?),
            FloatDType::F64 => CudaFloatSlice::F64(self.recv_slice::<f64>(count, peer)?),
            FloatDType::F16 => CudaFloatSlice::F16(self.recv_slice::<luma_tensor::f16>(count, peer)?),
            FloatDType::BF16 => CudaFloatSlice::BF16(self.recv_slice::<luma_tensor::bf16>(count, peer)?),
        };
        Ok(Tensor::from_storage(CudaFloatStorage { slice, device: self.device.clone() }, shape, FloatMeta::val()))
    }

    fn recv_int(&self, peer: usize) -> DistResult<Tensor<Cuda, Int>> {
        let (shape, dtype) = self.recv_header::<Int>(peer)?;
        let count = shape.element_count();
        let slice = match dtype {
            IntDType::I32 => CudaIntSlice::I32(self.recv_slice::<i32>(count, peer)?),
            IntDType::U32 => CudaIntSlice::U32(self.recv_slice::<u32>(count, peer)?),
            IntDType::U8 => CudaIntSlice::U8(self.recv_slice::<u8>(count, peer)?),
        };
        Ok(Tensor::from_storage(CudaIntStorage { slice, device: self.device.clone() }, shape, ()))
    }

    fn recv_bool(&self, peer: usize) -> DistResult<Tensor<Cuda, Bool>> {
        let (shape, _dtype) = self.recv_header::<Bool>(peer)?;
        let slice = self.recv_slice::<u8>(shape.element_count(), peer)?;
        Ok(Tensor::from_storage(CudaBoolStorage { slice, device: self.device.clone() }, shape, ()))
    }

    fn group_start(&self) -> DistResult<()> {
        nccl::result::group_start().map_err(map_nccl_err)?;
        Ok(())
    }

    fn group_end(&self) -> DistResult<()> {
        nccl::result::group_end().map_err(map_nccl_err)?;
        Ok(())
    }

    fn barrier(&self) -> DistResult<()> {
        // NCCL has no barrier primitive; a 1-element all-reduce is the idiom.
        let src = self.device.alloc_zeros::<f32>(1).map_err(NcclError::Cuda)?;
        let mut dst = self.device.alloc_zeros::<f32>(1).map_err(NcclError::Cuda)?;
        self.comm.all_reduce(&src, &mut dst, &nccl::ReduceOp::Sum).map_err(map_nccl_err)?;
        Ok(())
    }
}

fn check_divisible(size: usize, world: usize, dim: usize) -> DistResult<()> {
    if !size.is_multiple_of(world) {
        return Err(DistError::Shape(format!("reduce_scatter: dim {dim} size {size} not divisible by world_size {world}")));
    }
    Ok(())
}

fn to_nccl_reduce_op(op: ReduceOp) -> nccl::ReduceOp {
    match op {
        ReduceOp::Sum => nccl::ReduceOp::Sum,
        ReduceOp::Min => nccl::ReduceOp::Min,
        ReduceOp::Max => nccl::ReduceOp::Max,
        ReduceOp::Prod => nccl::ReduceOp::Prod,
        ReduceOp::Mean => nccl::ReduceOp::Avg,
    }
}
