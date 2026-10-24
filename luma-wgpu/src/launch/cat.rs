use crate::launch::{buffer_start, layout_index_meta};
use crate::ops::BindGroupBuilder;
use crate::shaders::ShaderKind;
use crate::storage::WgpuStorage;
use luma_tensor::{Error, Layout, Result, Shape};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CatParams {
    n: u32,
    rank: u32,
    dim: u32,
    start: u32,
    dims_base: u32,
    strides_base: u32,
    contiguous: u32,
    dst_dim_base: u32,
    cat_size: u32,
    right: u32,
    _pad0: u32,
    _pad1: u32,
}

fn cat_compute_shape(layouts: &[&Layout], dim: usize) -> Result<Shape> {
    if layouts.is_empty() {
        return Err(Error::OpRequiresAtLeastOneTensor { op: "cat" });
    }

    let first = layouts[0];
    let rank = first.shape().rank();
    if dim >= rank {
        return Err(Error::DimOutOfRange { shape: first.shape().clone(), dim: dim as i32, op: "cat" });
    }

    let mut out_dims = first.dims().to_vec();
    let mut cat_size = 0usize;
    for (n, layout) in layouts.iter().enumerate() {
        let dims = layout.dims();
        if dims.len() != rank {
            return Err(Error::ShapeMismatchCat { dim, first_shape: first.shape().clone(), n, nth_shape: layout.shape().clone() });
        }
        for (i, (&a, &b)) in first.dims().iter().zip(dims.iter()).enumerate() {
            if i != dim && a != b {
                return Err(Error::ShapeMismatchCat { dim, first_shape: first.shape().clone(), n, nth_shape: layout.shape().clone() });
            }
        }
        cat_size += dims[dim];
    }

    out_dims[dim] = cat_size;
    Ok(Shape::from(out_dims))
}

pub(crate) fn launch_cat<S: WgpuStorage>(srcs: &[(&S, &Layout)], dim: usize, out_shape: &Shape) -> Result<S> {
    if srcs.is_empty() {
        return Err(Error::OpRequiresAtLeastOneTensor { op: "cat" });
    }

    let dtype = srcs[0].0.wgpu_dtype();
    for (storage, _) in srcs {
        if storage.wgpu_dtype() != dtype {
            return Err(Error::DTypeMismatch { lhs: dtype.into(), rhs: storage.wgpu_dtype().into(), op: "cat" });
        }
    }

    let layouts: Vec<&Layout> = srcs.iter().map(|(_, layout)| *layout).collect();
    let internal_shape = cat_compute_shape(&layouts, dim)?;
    debug_assert_eq!(internal_shape.dims(), out_shape.dims(), "wgpu cat shape must match the layer");

    let device = srcs[0].0.wgpu_device();
    let elem_bytes = srcs[0].0.wgpu_elem_bytes();
    let out_count = out_shape.element_count();
    let out = device.alloc_storage_output(out_count * elem_bytes)?;
    let wrapped = S::wrap_wgpu(out, dtype, device);
    if out_count == 0 {
        return Ok(wrapped);
    }

    let key = ShaderKind::Cat(dtype.into());
    let shader = device.shader(key)?;
    let cat_size = out_shape.dims()[dim] as u32;
    let right: usize = out_shape.dims()[dim + 1..].iter().product();
    let mut dst_dim_base = 0u32;

    for (storage, layout) in srcs {
        let n = layout.shape().element_count();
        if n > 0 {
            let rank = layout.shape().rank();
            let (start, offset) = buffer_start(layout, elem_bytes, device);
            let params = device.upload_uniform(&CatParams {
                n: n as u32,
                rank: rank as u32,
                dim: dim as u32,
                start,
                dims_base: 0,
                strides_base: rank as u32,
                contiguous: u32::from(layout.is_contiguous()),
                dst_dim_base,
                cat_size,
                right: right as u32,
                _pad0: 0,
                _pad1: 0,
            })?;
            let meta = device.upload_storage_u32(&layout_index_meta(layout))?;

            let bind_group = BindGroupBuilder::new()
                .storage_read_at(0, storage.wgpu_buffer(), offset)
                .storage_rw(1, wrapped.wgpu_buffer())
                .uniform(2, &params)
                .storage_read(3, &meta)
                .build(device, shader.layout());
            shader.run(device, &bind_group, shader.workgroups_1d(n));
        }
        dst_dim_base += layout.dims()[dim] as u32;
    }

    Ok(wrapped)
}
