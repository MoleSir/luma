mod bool_ops;
mod float_ops;
mod int_ops;
pub mod kernels;
mod storage;

pub use storage::*;

use crate::{DType, Device};

/// The CPU device: a zero-sized type tag. All ops are associated functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Cpu;

impl Device for Cpu {
    type FloatStorage = CpuFloatStorage;
    type IntStorage = CpuIntStorage;
    type BoolStorage = CpuBoolStorage;

    fn name(&self) -> String {
        "cpu".into()
    }
}

// ============================================================================
// Dispatch macros: map a storage enum to its concrete element type, run the
// generic kernel, and re-wrap the result.
// ============================================================================

/// Run `$body` with `$data` bound to the inner `&Vec<t>` of a `CpuFloatStorage`,
/// then wrap the returned `Vec<t>` back into a `CpuFloatStorage`.
#[macro_export]
macro_rules! dispatch_float {
    ($storage:expr, |$data:ident| $body:expr) => {
        match $storage {
            $crate::CpuFloatStorage::F32($data) => $crate::CpuFloatStorage::F32($body),
            $crate::CpuFloatStorage::F64($data) => $crate::CpuFloatStorage::F64($body),
        }
    };
}

/// Like [`dispatch_float`] but `$body` yields a non-storage value (e.g. Vec<bool>).
#[macro_export]
macro_rules! dispatch_float_raw {
    ($storage:expr, |$data:ident| $body:expr) => {
        match $storage {
            $crate::CpuFloatStorage::F32($data) => $body,
            $crate::CpuFloatStorage::F64($data) => $body,
        }
    };
}

/// Dispatch two float storages of the SAME variant; errors on mismatch.
#[macro_export]
macro_rules! dispatch_float2 {
    ($lhs:expr, $rhs:expr, $op:literal, |$a:ident, $b:ident| $body:expr) => {
        match ($lhs, $rhs) {
            ($crate::CpuFloatStorage::F32($a), $crate::CpuFloatStorage::F32($b)) => Ok($crate::CpuFloatStorage::F32($body)),
            ($crate::CpuFloatStorage::F64($a), $crate::CpuFloatStorage::F64($b)) => Ok($crate::CpuFloatStorage::F64($body)),
            (l, r) => Err($crate::Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: $op }),
        }
    };
}

/// Dispatch two float storages of the SAME variant, `$body` yields a raw value.
#[macro_export]
macro_rules! dispatch_float2_raw {
    ($lhs:expr, $rhs:expr, $op:literal, |$a:ident, $b:ident| $body:expr) => {
        match ($lhs, $rhs) {
            ($crate::CpuFloatStorage::F32($a), $crate::CpuFloatStorage::F32($b)) => Ok($body),
            ($crate::CpuFloatStorage::F64($a), $crate::CpuFloatStorage::F64($b)) => Ok($body),
            (l, r) => Err($crate::Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: $op }),
        }
    };
}

#[macro_export]
macro_rules! dispatch_int {
    ($storage:expr, |$data:ident| $body:expr) => {
        match $storage {
            $crate::CpuIntStorage::I32($data) => $crate::CpuIntStorage::I32($body),
            $crate::CpuIntStorage::U32($data) => $crate::CpuIntStorage::U32($body),
            $crate::CpuIntStorage::U8($data) => $crate::CpuIntStorage::U8($body),
        }
    };
}

#[macro_export]
macro_rules! dispatch_int_raw {
    ($storage:expr, |$data:ident| $body:expr) => {
        match $storage {
            $crate::CpuIntStorage::I32($data) => $body,
            $crate::CpuIntStorage::U32($data) => $body,
            $crate::CpuIntStorage::U8($data) => $body,
        }
    };
}

#[macro_export]
macro_rules! dispatch_int2 {
    ($lhs:expr, $rhs:expr, $op:literal, |$a:ident, $b:ident| $body:expr) => {
        match ($lhs, $rhs) {
            ($crate::CpuIntStorage::I32($a), $crate::CpuIntStorage::I32($b)) => Ok($crate::CpuIntStorage::I32($body)),
            ($crate::CpuIntStorage::U32($a), $crate::CpuIntStorage::U32($b)) => Ok($crate::CpuIntStorage::U32($body)),
            ($crate::CpuIntStorage::U8($a), $crate::CpuIntStorage::U8($b)) => Ok($crate::CpuIntStorage::U8($body)),
            (l, r) => Err($crate::Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: $op }),
        }
    };
}

#[macro_export]
macro_rules! dispatch_int2_raw {
    ($lhs:expr, $rhs:expr, $op:literal, |$a:ident, $b:ident| $body:expr) => {
        match ($lhs, $rhs) {
            ($crate::CpuIntStorage::I32($a), $crate::CpuIntStorage::I32($b)) => Ok($body),
            ($crate::CpuIntStorage::U32($a), $crate::CpuIntStorage::U32($b)) => Ok($body),
            ($crate::CpuIntStorage::U8($a), $crate::CpuIntStorage::U8($b)) => Ok($body),
            (l, r) => Err($crate::Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: $op }),
        }
    };
}

/// Read an int storage's elements (in `layout` order) as `usize`, mapping the
/// dtype's MAX sentinel to `kernels::indexing::PAD`. Used by indexing kernels.
pub(crate) fn int_ids_as_usize(storage: &CpuIntStorage, layout: &crate::Layout) -> Vec<usize> {
    use kernels::element::{CpuInt, CpuNum};
    use kernels::indexing::PAD;
    macro_rules! collect {
        ($data:expr) => {
            layout
                .storage_indices()
                .map(|i| {
                    let v = $data[i];
                    if v == CpuInt::MAX { PAD } else { v.to_usize() }
                })
                .collect()
        };
    }
    match storage {
        CpuIntStorage::I32(d) => collect!(d),
        CpuIntStorage::U32(d) => collect!(d),
        CpuIntStorage::U8(d) => collect!(d),
    }
}

/// Build an int storage of the given dtype from `usize` indices.
pub(crate) fn usize_to_int_storage(data: &[usize], dtype: DType) -> CpuIntStorage {
    match dtype {
        DType::I32 => CpuIntStorage::I32(data.iter().map(|&v| v as i32).collect()),
        DType::U32 => CpuIntStorage::U32(data.iter().map(|&v| v as u32).collect()),
        DType::U8 => CpuIntStorage::U8(data.iter().map(|&v| v as u8).collect()),
        _ => CpuIntStorage::U32(data.iter().map(|&v| v as u32).collect()),
    }
}
