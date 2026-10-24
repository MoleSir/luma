use crate::Wgpu;
use crate::launch::buffer_start;
use crate::ops::BindGroupBuilder;
use crate::shaders::ShaderKind;
use crate::storage::WgpuStorage;
use luma_tensor::{DType, DTypeKindT, Error, Layout, Result, Shape};

// Must match BM/BN in the matmul shaders.
const TILE: u32 = 64;

#[derive(Clone, Copy)]
struct MatmulDims {
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
}

impl MatmulDims {
    fn from_layouts(lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let a = lhs_l.dims();
        let b = rhs_l.dims();
        let rank = a.len();
        if rank < 2 || b.len() != rank {
            return Err(Error::ShapeMismatchBinaryOp { lhs: lhs_l.shape().clone(), rhs: rhs_l.shape().clone(), op: "matmul" });
        }

        let (m, k) = (a[rank - 2], a[rank - 1]);
        let (k2, n) = (b[rank - 2], b[rank - 1]);
        let batch: usize = a[..rank - 2].iter().product();
        let batch_b: usize = b[..rank - 2].iter().product();
        if k != k2 || batch != batch_b {
            return Err(Error::ShapeMismatchBinaryOp { lhs: lhs_l.shape().clone(), rhs: rhs_l.shape().clone(), op: "matmul" });
        }

        Ok(Self { batch, m, n, k })
    }
}

/// Uniform payload for tiled matmul.
///
/// Batch dims/strides live in a read-only `u32` storage buffer; the offsets
/// `lhs_meta`/`rhs_meta` point at the reversed batch dims followed by the
/// reversed batch strides for each operand.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulParams {
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
    lhs_start: u32,
    rhs_start: u32,
    dst_start: u32,
    lhs_meta: u32,
    rhs_meta: u32,
    lhs_batch_ndim: u32,
    rhs_batch_ndim: u32,
    lhs_sm: u32,
    lhs_sk: u32,
    rhs_sm: u32,
    rhs_sk: u32,
    accumulate: u32,
}

/// Batch dims/strides in reverse order, matching the shader's `batch_offset`.
fn batch_meta(layout: &Layout) -> Vec<u32> {
    let rank = layout.shape().rank();
    let dims = &layout.dims()[..rank - 2];
    let strides = &layout.stride()[..rank - 2];
    dims.iter().rev().chain(strides.iter().rev()).map(|&v| v as u32).collect()
}

/// Everything needed to dispatch one matmul: resolved binding offsets, uniform
/// params, and the batch metadata buffer.
struct MatmulPlan {
    dtype: DType,
    dims: MatmulDims,
    lhs_offset: u64,
    rhs_offset: u64,
    out_offset: u64,
    params: MatmulParams,
    meta: Vec<u32>,
}

struct MatmulPlanArgs<'a> {
    dtype: DType,
    elem_bytes: usize,
    dims: MatmulDims,
    lhs_l: &'a Layout,
    rhs_l: &'a Layout,
    dst_l: Option<&'a Layout>,
    accumulate: bool,
}

impl MatmulPlan {
    fn new(device: &Wgpu, args: MatmulPlanArgs<'_>) -> Result<Self> {
        let MatmulPlanArgs { dtype, elem_bytes, dims, lhs_l, rhs_l, dst_l, accumulate } = args;
        let (lhs_start, lhs_offset) = buffer_start(lhs_l, elem_bytes, device);
        let (rhs_start, rhs_offset) = buffer_start(rhs_l, elem_bytes, device);
        let (dst_start, out_offset) = match dst_l {
            Some(layout) => buffer_start(layout, elem_bytes, device),
            None => (0, 0),
        };

        let rank = lhs_l.shape().rank();
        let mut meta = batch_meta(lhs_l);
        let lhs_meta = 0u32;
        let rhs_meta = meta.len() as u32;
        meta.extend(batch_meta(rhs_l));

        let params = MatmulParams {
            batch: dims.batch as u32,
            m: dims.m as u32,
            n: dims.n as u32,
            k: dims.k as u32,
            lhs_start,
            rhs_start,
            dst_start,
            lhs_meta,
            rhs_meta,
            lhs_batch_ndim: (rank - 2) as u32,
            rhs_batch_ndim: (rank - 2) as u32,
            lhs_sm: lhs_l.stride()[rank - 2] as u32,
            lhs_sk: lhs_l.stride()[rank - 1] as u32,
            rhs_sm: rhs_l.stride()[rank - 2] as u32,
            rhs_sk: rhs_l.stride()[rank - 1] as u32,
            accumulate: u32::from(accumulate),
        };

        Ok(Self { dtype, dims, lhs_offset, rhs_offset, out_offset, params, meta })
    }

    fn dispatch(&self, device: &Wgpu, lhs: &wgpu::Buffer, rhs: &wgpu::Buffer, out: &wgpu::Buffer) -> Result<()> {
        let key = ShaderKind::Matmul(self.dtype);
        let params = device.upload_uniform(&self.params)?;
        let meta = device.upload_storage_u32(&self.meta)?;

        let shader = device.shader(key)?;
        let bind_group = BindGroupBuilder::new()
            .storage_read_at(0, lhs, self.lhs_offset)
            .storage_read_at(1, rhs, self.rhs_offset)
            .storage_rw_at(2, out, self.out_offset)
            .uniform(3, &params)
            .storage_read(4, &meta)
            .build(device, shader.layout());

        let dims = self.dims;
        let workgroups = [(dims.n as u32).div_ceil(TILE), (dims.m as u32).div_ceil(TILE), dims.batch as u32];
        shader.run(device, &bind_group, workgroups);
        Ok(())
    }
}

pub(crate) fn launch_matmul<S: WgpuStorage>(lhs: &S, lhs_l: &Layout, rhs: &S, rhs_l: &Layout, out_shape: &Shape) -> Result<S> {
    let dtype = lhs.wgpu_dtype();
    if dtype != rhs.wgpu_dtype() {
        return Err(Error::DTypeMismatch { lhs: dtype.into(), rhs: rhs.wgpu_dtype().into(), op: "matmul" });
    }
    let elem_bytes = dtype.size_of();

    let dims = MatmulDims::from_layouts(lhs_l, rhs_l)?;
    let total = dims.batch * dims.m * dims.n;
    debug_assert_eq!(out_shape.element_count(), total, "wgpu f_matmul shape must match the layer");

    let device = lhs.wgpu_device();
    let out = device.alloc_storage_output(total * elem_bytes)?;
    if total == 0 || dims.k == 0 {
        // A fresh wgpu buffer is zero-initialized, which is the k == 0 result.
        return Ok(S::wrap_wgpu(out, dtype, device));
    }

    let plan =
        MatmulPlan::new(device, MatmulPlanArgs { dtype: dtype.into(), elem_bytes, dims, lhs_l, rhs_l, dst_l: None, accumulate: false })?;
    plan.dispatch(device, lhs.wgpu_buffer(), rhs.wgpu_buffer(), &out)?;
    Ok(S::wrap_wgpu(out, dtype, device))
}

pub(crate) fn launch_add_matmul_<S: WgpuStorage>(dst: &S, dst_l: &Layout, lhs: &S, lhs_l: &Layout, rhs: &S, rhs_l: &Layout) -> Result<()> {
    if !dst_l.is_contiguous() {
        return Err(Error::RequiresContiguous { op: "wgpu f_add_matmul_" });
    }

    let dtype = lhs.wgpu_dtype();
    if dtype != rhs.wgpu_dtype() || dtype != dst.wgpu_dtype() {
        return Err(Error::DTypeMismatch { lhs: dtype.into(), rhs: rhs.wgpu_dtype().into(), op: "add_matmul" });
    }
    let elem_bytes = dtype.size_of();

    let dims = MatmulDims::from_layouts(lhs_l, rhs_l)?;
    let total = dims.batch * dims.m * dims.n;
    debug_assert_eq!(dst_l.shape().element_count(), total, "wgpu f_add_matmul_ shape must match the layer");
    if total == 0 || dims.k == 0 {
        return Ok(());
    }

    let device = dst.wgpu_device();
    let plan = MatmulPlan::new(
        device,
        MatmulPlanArgs { dtype: dtype.into(), elem_bytes, dims, lhs_l, rhs_l, dst_l: Some(dst_l), accumulate: true },
    )?;
    plan.dispatch(device, lhs.wgpu_buffer(), rhs.wgpu_buffer(), dst.wgpu_buffer())
}
