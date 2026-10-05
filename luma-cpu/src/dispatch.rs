// ============================================================================
// Dispatch macros: map a storage enum to its concrete element type, run the
// generic kernel, and re-wrap the result.
// ============================================================================

use luma_tensor::{DType, Layout};

use crate::{Cpu, CpuIntStorage};

/// Run `$body` with `$data` bound to the inner `&Vec<t>` of a `CpuFloatStorage`,
/// then wrap the returned `Vec<t>` back into a `CpuFloatStorage`.
///
/// The rebuild preserves the storage's device instance (result tensors inherit
/// it via `from_storage`).
#[macro_export]
macro_rules! dispatch_float {
    ($storage:expr, |$data:ident| $body:expr) => {
        match $storage {
            $crate::CpuFloatStorage::F32($data, _) => $crate::CpuFloatStorage::F32($body, $storage.device().clone()),
            $crate::CpuFloatStorage::F64($data, _) => $crate::CpuFloatStorage::F64($body, $storage.device().clone()),
            $crate::CpuFloatStorage::F16($data, _) => $crate::CpuFloatStorage::F16($body, $storage.device().clone()),
            $crate::CpuFloatStorage::BF16($data, _) => $crate::CpuFloatStorage::BF16($body, $storage.device().clone()),
        }
    };
}

/// Like [`dispatch_float`] but `$body` yields a non-storage value (e.g. Vec<bool>).
#[macro_export]
macro_rules! dispatch_float_raw {
    ($storage:expr, |$data:ident| $body:expr) => {
        match $storage {
            $crate::CpuFloatStorage::F32($data, _) => $body,
            $crate::CpuFloatStorage::F64($data, _) => $body,
            $crate::CpuFloatStorage::F16($data, _) => $body,
            $crate::CpuFloatStorage::BF16($data, _) => $body,
        }
    };
}

/// Dispatch two float storages of the SAME variant; errors on mismatch.
///
/// The result inherits the LHS's device (the `&self` storage in op code).
#[macro_export]
macro_rules! dispatch_float2 {
    ($lhs:expr, $rhs:expr, $op:literal, |$a:ident, $b:ident| $body:expr) => {
        match ($lhs, $rhs) {
            ($crate::CpuFloatStorage::F32($a, _), $crate::CpuFloatStorage::F32($b, _)) => {
                Ok($crate::CpuFloatStorage::F32($body, $lhs.device().clone()))
            }
            ($crate::CpuFloatStorage::F64($a, _), $crate::CpuFloatStorage::F64($b, _)) => {
                Ok($crate::CpuFloatStorage::F64($body, $lhs.device().clone()))
            }
            ($crate::CpuFloatStorage::F16($a, _), $crate::CpuFloatStorage::F16($b, _)) => {
                Ok($crate::CpuFloatStorage::F16($body, $lhs.device().clone()))
            }
            ($crate::CpuFloatStorage::BF16($a, _), $crate::CpuFloatStorage::BF16($b, _)) => {
                Ok($crate::CpuFloatStorage::BF16($body, $lhs.device().clone()))
            }
            (l, r) => Err(luma_tensor::Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: $op }),
        }
    };
}

/// Dispatch two float storages of the SAME variant, `$body` yields a raw value.
#[macro_export]
macro_rules! dispatch_float2_raw {
    ($lhs:expr, $rhs:expr, $op:literal, |$a:ident, $b:ident| $body:expr) => {
        match ($lhs, $rhs) {
            ($crate::CpuFloatStorage::F32($a, _), $crate::CpuFloatStorage::F32($b, _)) => Ok($body),
            ($crate::CpuFloatStorage::F64($a, _), $crate::CpuFloatStorage::F64($b, _)) => Ok($body),
            ($crate::CpuFloatStorage::F16($a, _), $crate::CpuFloatStorage::F16($b, _)) => Ok($body),
            ($crate::CpuFloatStorage::BF16($a, _), $crate::CpuFloatStorage::BF16($b, _)) => Ok($body),
            (l, r) => Err(luma_tensor::Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: $op }),
        }
    };
}

#[macro_export]
macro_rules! dispatch_int {
    ($storage:expr, |$data:ident| $body:expr) => {
        match $storage {
            $crate::CpuIntStorage::I32($data, _) => $crate::CpuIntStorage::I32($body, $storage.device().clone()),
            $crate::CpuIntStorage::U32($data, _) => $crate::CpuIntStorage::U32($body, $storage.device().clone()),
            $crate::CpuIntStorage::U8($data, _) => $crate::CpuIntStorage::U8($body, $storage.device().clone()),
        }
    };
}

#[macro_export]
macro_rules! dispatch_int_raw {
    ($storage:expr, |$data:ident| $body:expr) => {
        match $storage {
            $crate::CpuIntStorage::I32($data, _) => $body,
            $crate::CpuIntStorage::U32($data, _) => $body,
            $crate::CpuIntStorage::U8($data, _) => $body,
        }
    };
}

#[macro_export]
macro_rules! dispatch_int2 {
    ($lhs:expr, $rhs:expr, $op:literal, |$a:ident, $b:ident| $body:expr) => {
        match ($lhs, $rhs) {
            ($crate::CpuIntStorage::I32($a, _), $crate::CpuIntStorage::I32($b, _)) => {
                Ok($crate::CpuIntStorage::I32($body, $lhs.device().clone()))
            }
            ($crate::CpuIntStorage::U32($a, _), $crate::CpuIntStorage::U32($b, _)) => {
                Ok($crate::CpuIntStorage::U32($body, $lhs.device().clone()))
            }
            ($crate::CpuIntStorage::U8($a, _), $crate::CpuIntStorage::U8($b, _)) => {
                Ok($crate::CpuIntStorage::U8($body, $lhs.device().clone()))
            }
            (l, r) => Err(luma_tensor::Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: $op }),
        }
    };
}

#[macro_export]
macro_rules! dispatch_int2_raw {
    ($lhs:expr, $rhs:expr, $op:literal, |$a:ident, $b:ident| $body:expr) => {
        match ($lhs, $rhs) {
            ($crate::CpuIntStorage::I32($a, _), $crate::CpuIntStorage::I32($b, _)) => Ok($body),
            ($crate::CpuIntStorage::U32($a, _), $crate::CpuIntStorage::U32($b, _)) => Ok($body),
            ($crate::CpuIntStorage::U8($a, _), $crate::CpuIntStorage::U8($b, _)) => Ok($body),
            (l, r) => Err(luma_tensor::Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: $op }),
        }
    };
}

/// Read an int storage's elements (in `layout` order) as `usize`, mapping the
/// dtype's MAX sentinel to `kernels::indexing::PAD`. Used by indexing kernels.
pub(crate) fn int_ids_as_usize(storage: &CpuIntStorage, layout: &Layout) -> Vec<usize> {
    use crate::kernels::element::{CpuInt, CpuNum};
    use crate::kernels::indexing::PAD;
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
        CpuIntStorage::I32(d, _) => collect!(d),
        CpuIntStorage::U32(d, _) => collect!(d),
        CpuIntStorage::U8(d, _) => collect!(d),
    }
}

/// Build an int storage of the given dtype from `usize` indices.
pub(crate) fn usize_to_int_storage(data: &[usize], dtype: DType, device: &Cpu) -> CpuIntStorage {
    match dtype {
        DType::I32 => CpuIntStorage::I32(data.iter().map(|&v| v as i32).collect(), device.clone()),
        DType::U32 => CpuIntStorage::U32(data.iter().map(|&v| v as u32).collect(), device.clone()),
        DType::U8 => CpuIntStorage::U8(data.iter().map(|&v| v as u8).collect(), device.clone()),
        _ => CpuIntStorage::U32(data.iter().map(|&v| v as u32).collect(), device.clone()),
    }
}
