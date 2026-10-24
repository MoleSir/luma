use crate::DistError;
use luma_cuda::Cuda;
use luma_tensor::Tensor;
use luma_tensor::ops::BaseOpsDTypeKind;


/// A contiguous tensor with `dim` moved to the front, plus the inverse
/// permutation to restore the original axis order (or `None` if no reorder).
pub(crate) type FrontLayout<K> = (Tensor<Cuda, K>, Option<Vec<usize>>);

/// Move `dim` to the front and materialize the tensor contiguously.
pub(crate) fn front_layout<K: BaseOpsDTypeKind<Cuda>>(x: &Tensor<Cuda, K>, dim: usize) -> crate::DistResult<FrontLayout<K>> {
    let rank = x.rank();
    if dim >= rank {
        return Err(DistError::Shape(format!("dim {dim} out of range for rank {rank}")));
    }
    if dim == 0 {
        return Ok((x.contiguous()?, None));
    }
    let mut perm: Vec<usize> = Vec::with_capacity(rank);
    perm.push(dim);
    perm.extend((0..rank).filter(|&i| i != dim));

    let mut inv = vec![0usize; rank];
    for (k, &p) in perm.iter().enumerate() {
        inv[p] = k;
    }
    Ok((x.permute(perm)?.contiguous()?, Some(inv)))
}

/// Undo [`front_layout`]'s reorder on the gathered/scattered result.
pub(crate) fn restore_layout<K: BaseOpsDTypeKind<Cuda>>(g: Tensor<Cuda, K>, inv: Option<Vec<usize>>) -> crate::DistResult<Tensor<Cuda, K>> {
    match inv {
        None => Ok(g),
        Some(inv) => Ok(g.permute(inv)?.contiguous()?),
    }
}
