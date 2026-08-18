//! Safetensors format I/O.
//!
//! Format layout:
//! ```text
//! +-----------------------+------------------------------+---------------------+
//! |  N (8 bytes, u64 LE)  |  Header (N bytes, JSON)     |  Data (rest)        |
//! +-----------------------+------------------------------+---------------------+
//! ```
//!
//! The header is a JSON object mapping tensor names to `{dtype, shape, data_offsets}`.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use luma_tensor::{DType, Device, DynTensor};
use memmap2::MmapOptions;
use serde::{Deserialize, Serialize};

// ============================================================================
//    DType mapping
// ============================================================================

/// Safetensors element type string as it appears in the JSON header.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SafeTensorsDType {
    #[serde(rename = "BOOL")]
    Bool,
    #[serde(rename = "U8")]
    U8,
    #[serde(rename = "I32")]
    I32,
    #[serde(rename = "U32")]
    U32,
    #[serde(rename = "F32")]
    F32,
    #[serde(rename = "F64")]
    F64,
}

impl TryFrom<SafeTensorsDType> for DType {
    type Error = SafeTensorsError;

    fn try_from(value: SafeTensorsDType) -> Result<Self, Self::Error> {
        match value {
            SafeTensorsDType::Bool => Ok(DType::Bool),
            SafeTensorsDType::U8 => Ok(DType::U8),
            SafeTensorsDType::I32 => Ok(DType::I32),
            SafeTensorsDType::U32 => Ok(DType::U32),
            SafeTensorsDType::F32 => Ok(DType::F32),
            SafeTensorsDType::F64 => Ok(DType::F64),
        }
    }
}

impl From<DType> for SafeTensorsDType {
    fn from(value: DType) -> Self {
        match value {
            DType::Bool => Self::Bool,
            DType::U8 => Self::U8,
            DType::I32 => Self::I32,
            DType::U32 => Self::U32,
            DType::F32 => Self::F32,
            DType::F64 => Self::F64,
        }
    }
}

// ============================================================================
//    Header types
// ============================================================================

#[derive(Debug, Deserialize, Serialize)]
struct SafeTensorsInfo {
    dtype: SafeTensorsDType,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

// ============================================================================
//    Public API
// ============================================================================

/// Result of loading a safetensors file.
pub struct SafeTensorsContent<D: Device> {
    /// Optional `__metadata__` map from the header.
    pub metadata: Option<HashMap<String, String>>,
    /// Loaded tensors, keyed by name.
    pub tensors: HashMap<String, DynTensor<D>>,
}

impl<D: Device> SafeTensorsContent<D> {
    /// Save all tensors (and optional metadata) to a safetensors file.
    pub fn save_file<P: AsRef<Path>>(&self, path: P) -> Result<(), SafeTensorsError> {
        save_file(&self.tensors, self.metadata.as_ref(), path)
    }
}

/// Load tensors from a memory-mapped safetensors file.
///
/// Tensors are constructed directly on `device` from the raw file bytes
/// (zero-copy where possible).
pub fn load_file<D: Device, P: AsRef<Path>>(path: P, device: &D) -> Result<SafeTensorsContent<D>, SafeTensorsError> {
    let file = File::open(path).map_err(SafeTensorsError::Io)?;
    let mmap = unsafe { MmapOptions::new().map(&file).map_err(SafeTensorsError::Io)? };

    // header size (8 bytes, little-endian u64)
    let header_size = u64::from_le_bytes(mmap[0..8].try_into()?) as usize;

    // parse JSON header
    let header_slice = &mmap[8..8 + header_size];
    let header: HashMap<String, serde_json::Value> = serde_json::from_slice(header_slice)?;

    let data_start = 8 + header_size;
    let mut metadata = None;
    let mut tensors = HashMap::new();

    for (name, value) in header {
        if name == "__metadata__" {
            metadata = Some(serde_json::from_value(value)?);
            continue;
        }

        let info: SafeTensorsInfo = serde_json::from_value(value)?;
        let (start_offset, end_offset) = info.data_offsets;
        let raw_bytes = &mmap[data_start + start_offset..data_start + end_offset];

        let dtype: DType = info.dtype.try_into()?;
        let tensor = DynTensor::from_bytes(raw_bytes, dtype, info.shape, device)?;
        tensors.insert(name, tensor);
    }

    Ok(SafeTensorsContent { metadata, tensors })
}

/// Load tensors from a reader (non-mmap path).
pub fn load<D: Device, R: Read>(reader: &mut R, device: &D) -> Result<SafeTensorsContent<D>, SafeTensorsError> {
    // header size
    let mut header_size_bytes = [0u8; 8];
    reader.read_exact(&mut header_size_bytes).map_err(SafeTensorsError::Io)?;
    let header_size = usize::from_le_bytes(header_size_bytes);

    // JSON header
    let mut json_bytes = vec![0u8; header_size];
    reader.read_exact(&mut json_bytes).map_err(SafeTensorsError::Io)?;
    let header: HashMap<String, serde_json::Value> = serde_json::from_slice(&json_bytes)?;

    // data
    let mut data_buffer = Vec::new();
    reader.read_to_end(&mut data_buffer).map_err(SafeTensorsError::Io)?;

    let mut metadata = None;
    let mut tensors = HashMap::new();

    for (name, value) in header {
        if name == "__metadata__" {
            metadata = Some(serde_json::from_value(value)?);
            continue;
        }

        let info: SafeTensorsInfo = serde_json::from_value(value)?;
        let (start, end) = info.data_offsets;
        if end > data_buffer.len() {
            return Err(SafeTensorsError::DataOffsetOutOfRange(data_buffer.len(), end));
        }

        let dtype: DType = info.dtype.try_into()?;
        let tensor = DynTensor::from_bytes(&data_buffer[start..end], dtype, info.shape, device)?;
        tensors.insert(name, tensor);
    }

    Ok(SafeTensorsContent { metadata, tensors })
}

/// Save tensors to a file.
pub fn save_file<D: Device, P: AsRef<Path>>(
    tensors: &HashMap<String, DynTensor<D>>,
    metadata: Option<&HashMap<String, String>>,
    path: P,
) -> Result<(), SafeTensorsError> {
    let file = File::create(path).map_err(SafeTensorsError::Io)?;
    let mut writer = BufWriter::new(file);
    save(tensors, metadata, &mut writer)
}

/// Save tensors to a writer.
pub fn save<D: Device, W: Write>(
    tensors: &HashMap<String, DynTensor<D>>,
    metadata: Option<&HashMap<String, String>>,
    writer: &mut W,
) -> Result<(), SafeTensorsError> {
    let mut header_map = BTreeMap::new();
    let mut current_offset = 0;
    let tensors_ordered: BTreeMap<&String, &DynTensor<D>> = tensors.iter().collect();

    // build header info
    for (name, tensor) in tensors_ordered.iter() {
        let n_bytes = tensor.shape().element_count() * tensor.dtype().size_in_bytes();

        let info = SafeTensorsInfo {
            dtype: tensor.dtype().into(),
            shape: tensor.dims().to_vec(),
            data_offsets: (current_offset, current_offset + n_bytes),
        };

        current_offset += n_bytes;
        header_map.insert(name.as_str(), info);
    }

    // insert metadata
    let mut header_value = serde_json::to_value(&header_map)?;
    if let Some(metadata) = metadata {
        let meta_value = serde_json::to_value(metadata)?;
        if let Some(obj) = header_value.as_object_mut() {
            obj.insert("__metadata__".to_string(), meta_value);
        }
    }

    // write header
    let header_bytes = serde_json::to_vec(&header_value)?;
    let header_size_u64 = header_bytes.len() as u64;
    writer.write_all(&header_size_u64.to_le_bytes()).map_err(SafeTensorsError::Io)?;
    writer.write_all(&header_bytes).map_err(SafeTensorsError::Io)?;

    // write tensor data
    for (_, tensor) in tensors_ordered {
        writer.write_all(&tensor.to_bytes()?).map_err(SafeTensorsError::Io)?;
    }

    writer.flush()?;
    Ok(())
}

// ============================================================================
//    Error
// ============================================================================

#[derive(Debug, thiserror::Error)]
pub enum SafeTensorsError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("tensor error: {0}")]
    Tensor(#[from] luma_tensor::Error),

    #[error("byte slice error: {0}")]
    Slice(#[from] std::array::TryFromSliceError),

    #[error("data offset out of range: buffer len {0}, tried to access {1}")]
    DataOffsetOutOfRange(usize, usize),
}

// ============================================================================
//    Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use luma_tensor::dtype::FloatDType;
    use luma_tensor::{Cpu, Int, Tensor};

    #[test]
    fn test_safetensors_roundtrip_memory() {
        let device = Cpu::default();
        let w = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), FloatDType::F32).unwrap();
        let b = Tensor::<Cpu>::from_slice(&[0.1, 0.2], (2,), FloatDType::F32).unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("weight".to_string(), DynTensor::Float(w));
        tensors.insert("bias".to_string(), DynTensor::Float(b));

        let mut buf = Vec::new();
        save(&tensors, None, &mut buf).unwrap();

        let mut cursor = std::io::Cursor::new(buf);
        let content = load(&mut cursor, &device).unwrap();
        assert_eq!(content.tensors.len(), 2);

        let w2 = content.tensors["weight"].as_float().unwrap();
        assert_eq!(w2.dims(), &[2, 3]);
        let expected_w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        for (a, b) in w2.to_vec().unwrap().iter().zip(expected_w.iter()) {
            assert!((a - b).abs() < 1e-5);
        }

        let b2 = content.tensors["bias"].as_float().unwrap();
        assert_eq!(b2.dims(), &[2]);
        assert!((b2.to_vec().unwrap()[0] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_safetensors_roundtrip_mixed_kinds() {
        let device = Cpu::default();
        let f = Tensor::<Cpu>::from_slice(&[1.0, 2.0], (2,), FloatDType::F32).unwrap();
        let i = Tensor::<Cpu, Int>::from_slice(&[10i64, 20, 30], (3,), luma_tensor::dtype::IntDType::I32).unwrap();
        let b = Tensor::<Cpu, luma_tensor::Bool>::from_slice(&[true, false], (2,), luma_tensor::dtype::BoolDType::Bool).unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("f".to_string(), DynTensor::Float(f));
        tensors.insert("i".to_string(), DynTensor::Int(i));
        tensors.insert("b".to_string(), DynTensor::Bool(b));

        let mut buf = Vec::new();
        save(&tensors, None, &mut buf).unwrap();

        let mut cursor = std::io::Cursor::new(buf);
        let content = load(&mut cursor, &device).unwrap();
        assert_eq!(content.tensors.len(), 3);
        assert_eq!(content.tensors["f"].dtype(), luma_tensor::DType::F32);
        assert_eq!(content.tensors["i"].dtype(), luma_tensor::DType::I32);
        assert_eq!(content.tensors["b"].dtype(), luma_tensor::DType::Bool);
        assert_eq!(content.tensors["i"].as_int().unwrap().to_vec().unwrap(), vec![10i64, 20, 30]);
        assert_eq!(content.tensors["b"].as_bool().unwrap().to_vec().unwrap(), vec![true, false]);
    }

    #[test]
    fn test_safetensors_file_roundtrip() {
        let device = Cpu::default();
        let t = Tensor::<Cpu>::from_slice(&[42.0, 99.0], (2,), FloatDType::F32).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert("t".to_string(), DynTensor::Float(t));

        let dir = std::env::temp_dir();
        let path = dir.join("luma_test.safetensors");
        save_file(&tensors, None, &path).unwrap();

        let content = load_file(&path, &device).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(content.tensors.len(), 1);
        let v = content.tensors["t"].as_float().unwrap().to_vec().unwrap();
        assert!((v[0] - 42.0).abs() < 1e-5);
        assert!((v[1] - 99.0).abs() < 1e-5);
    }
}
