//! Shared strided-iteration helpers used by the generic CPU kernels, plus the
//! `DimArray` view for iterating a single (possibly strided) dimension.

use crate::Cpu;
use crate::Layout;
use crate::device::cpu::allocator::AllocVec;

/// A strided view of one dimension: `get(i) = src[i * stride]`, for `size` steps.
/// Used by reductions / nn kernels to walk the reduced axis.
pub struct DimArray<'a, T> {
    pub src: &'a [T],
    pub size: usize,
    pub stride: usize,
}

impl<'a, T: Copy> DimArray<'a, T> {
    pub fn new(src: &'a [T], size: usize, stride: usize) -> Self {
        Self { src, size, stride }
    }
}

impl<'a, T: Copy> IntoIterator for DimArray<'a, T> {
    type Item = T;
    type IntoIter = DimArrayIter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        DimArrayIter { arr: self, index: 0 }
    }
}

pub struct DimArrayIter<'a, T> {
    arr: DimArray<'a, T>,
    index: usize,
}

impl<'a, T: Copy> Iterator for DimArrayIter<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.index >= self.arr.size {
            None
        } else {
            let v = self.arr.src[self.index * self.arr.stride];
            self.index += 1;
            Some(v)
        }
    }
}

impl<'a, T: Copy> ExactSizeIterator for DimArrayIter<'a, T> {
    fn len(&self) -> usize {
        self.arr.size - self.index
    }
}

/// Materialize the logical elements of `data` under `layout` (in row-major
/// logical order) into a fresh `Vec`, applying `f` to each.
pub fn gather_map<T: Copy, U: AllocVec, F: Fn(T) -> U>(data: &[T], layout: &Layout, f: F, device: &Cpu) -> Vec<U> {
    device.collect_alloc(layout.storage_indices().map(|i| f(data[i])))
}

/// Materialize the logical elements of `data` under `layout` into a fresh `Vec`.
pub fn gather<T: Copy + AllocVec>(data: &[T], layout: &Layout, device: &Cpu) -> Vec<T> {
    device.collect_alloc(layout.storage_indices().map(|i| data[i]))
}
