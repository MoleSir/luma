//! LumaPack — a key-value container for tensors, scalars, and metadata.
//!
//! # File format (`.lpk`)
//!
//! ```text
//! +------------------------+-----------------------------+---------------------+
//! | N (8 bytes, u64 LE)    | Header (N bytes, JSON)      | Tensor data         |
//! +------------------------+-----------------------------+---------------------+
//! ```
//!
//! JSON header:
//! ```json
//! {
//!   "tensors": {
//!     "weight": {"dtype": "F32", "shape": [64, 128], "data_offsets": [0, 32768]}
//!   },
//!   "scalars": {
//!     "lr":     {"dtype": "F64", "value": 0.001},
//!     "step_t": {"dtype": "I64", "value": 100}
//!   },
//!   "metadata": {"epoch": "10"}
//! }
//! ```
//!

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use luma_tensor::{Device, DynTensor, Scalar};
use memmap2::MmapOptions;
use serde::{Deserialize, Serialize};
use serde_json;

// ============================================================================
//   LumaPack
// ============================================================================

/// A heterogeneous key-value container for model checkpoints.
///
/// Holds three collections under one roof:
/// - **tensors**  — named multi-dimensional arrays (model weights, optimizer state).
/// - **scalars**  — typed scalar values (learning rate, step counter, …).
/// - **metadata** — free-form string key/value pairs.
pub struct LumaPack<D: Device> {
    pub tensors: HashMap<String, DynTensor<D>>,
    pub scalars: HashMap<String, Scalar>,
    pub metadata: HashMap<String, String>,
}

impl<D: Device> LumaPack<D> {
    pub fn new() -> Self {
        Self { tensors: HashMap::new(), scalars: HashMap::new(), metadata: HashMap::new() }
    }

    /// Convenience: save this pack to `path` via [`save_file`].
    pub fn save_file(&self, path: impl AsRef<Path>) -> Result<(), LumaPackError> {
        save_file(self, path)
    }

    /// Serialize this pack into the returned bytes (header + tensor data).
    pub fn to_bytes(&self) -> Result<Vec<u8>, LumaPackError> {
        let mut buf = Vec::new();
        save(self, &mut buf)?;
        Ok(buf)
    }
}

// ============================================================================
//   Error
// ============================================================================

/// Errors that can occur during LumaPack I/O.
#[derive(Debug, thiserror::Error)]
pub enum LumaPackError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("tensor error: {0}")]
    Tensor(#[from] luma_tensor::Error),

    #[error("byte slice error: {0}")]
    Slice(#[from] std::array::TryFromSliceError),

    #[error("unknown scalar dtype: {0}")]
    UnknownScalarDType(String),

    #[error("scalar value missing for key: {0}")]
    MissingScalarValue(String),
}

// ============================================================================
//   Free functions — load
// ============================================================================

/// Load a `.lpk` file from `path`, placing tensors on `device`.
pub fn load_file<D: Device, P: AsRef<Path>>(path: P, device: &D) -> Result<LumaPack<D>, LumaPackError> {
    let file = File::open(path).map_err(LumaPackError::Io)?;
    let mmap = unsafe { MmapOptions::new().map(&file).map_err(LumaPackError::Io)? };

    let header_size = u64::from_le_bytes(mmap[0..8].try_into()?) as usize;
    let header_slice = &mmap[8..8 + header_size];
    let header: serde_json::Value = serde_json::from_slice(header_slice)?;
    let data_start = 8 + header_size;

    let obj = header.as_object().ok_or_else(|| LumaPackError::Json(json_error("header must be a JSON object")))?;
    load_lpk(&mmap, data_start, obj, device)
}

/// Load a `.lpk` pack from a reader.
pub fn load<D: Device, R: Read>(reader: &mut R, device: &D) -> Result<LumaPack<D>, LumaPackError> {
    let mut header_size_bytes = [0u8; 8];
    reader.read_exact(&mut header_size_bytes).map_err(LumaPackError::Io)?;
    let header_size = u64::from_le_bytes(header_size_bytes) as usize;

    let mut json_bytes = vec![0u8; header_size];
    reader.read_exact(&mut json_bytes).map_err(LumaPackError::Io)?;
    let header: serde_json::Value = serde_json::from_slice(&json_bytes)?;

    let mut data_buffer = Vec::new();
    reader.read_to_end(&mut data_buffer).map_err(LumaPackError::Io)?;

    let obj = header.as_object().ok_or_else(|| LumaPackError::Json(json_error("header must be a JSON object")))?;
    load_lpk(&data_buffer, 0, obj, device)
}

fn load_lpk<D: Device>(
    data: &[u8],
    data_start: usize,
    header_obj: &serde_json::Map<String, serde_json::Value>,
    device: &D,
) -> Result<LumaPack<D>, LumaPackError> {
    let mut pack = LumaPack::new();

    // tensors
    if let Some(tensors_val) = header_obj.get("tensors") {
        let tensors_map = tensors_val.as_object().ok_or_else(|| LumaPackError::Json(json_error("'tensors' must be an object")))?;
        for (name, info_val) in tensors_map {
            let info: TensorEntry = serde_json::from_value(info_val.clone())?;
            let (start, end) = info.data_offsets;
            let raw = &data[data_start + start..data_start + end];
            let dtype = dtype_from_str(&info.dtype)?;
            let tensor = DynTensor::from_bytes(raw, dtype, info.shape, device)?;
            pack.tensors.insert(name.clone(), tensor);
        }
    }

    // scalars
    if let Some(scalars_val) = header_obj.get("scalars") {
        let scalars_map = scalars_val.as_object().ok_or_else(|| LumaPackError::Json(json_error("'scalars' must be an object")))?;
        for (name, val) in scalars_map {
            pack.scalars.insert(name.clone(), scalar_from_json(val)?);
        }
    }

    // metadata
    if let Some(meta_val) = header_obj.get("metadata") {
        if let Some(meta_obj) = meta_val.as_object() {
            for (k, v) in meta_obj {
                if let Some(s) = v.as_str() {
                    pack.metadata.insert(k.clone(), s.to_string());
                }
            }
        }
    }

    Ok(pack)
}

// ============================================================================
//   Free functions — save
// ============================================================================

/// Save a [`LumaPack`] to a `.lpk` file.
pub fn save_file<D: Device, P: AsRef<Path>>(pack: &LumaPack<D>, path: P) -> Result<(), LumaPackError> {
    let file = File::create(path).map_err(LumaPackError::Io)?;
    let mut writer = BufWriter::new(file);
    save(pack, &mut writer)
}

/// Save a [`LumaPack`] to a writer.
pub fn save<D: Device, W: Write>(pack: &LumaPack<D>, writer: &mut W) -> Result<(), LumaPackError> {
    // --- build tensor entries & accumulate binary data ---
    let tensor_entries: BTreeMap<&String, &DynTensor<D>> = pack.tensors.iter().collect();
    let mut header_tensors = BTreeMap::new();
    let mut current_offset = 0usize;

    for (name, tensor) in tensor_entries.iter() {
        let n_bytes = tensor.shape().element_count() * tensor.dtype().size_in_bytes();
        header_tensors.insert(
            name.as_str(),
            TensorEntry {
                dtype: dtype_to_str(tensor.dtype()).to_string(),
                shape: tensor.dims().to_vec(),
                data_offsets: (current_offset, current_offset + n_bytes),
            },
        );
        current_offset += n_bytes;
    }

    // --- build scalars section ---
    let mut header_scalars = BTreeMap::new();
    for (name, scalar) in &pack.scalars {
        header_scalars.insert(name.as_str(), scalar_to_json(scalar));
    }

    // --- build metadata section ---
    let mut header_metadata = BTreeMap::new();
    for (k, v) in &pack.metadata {
        header_metadata.insert(k.as_str(), serde_json::Value::String(v.clone()));
    }

    // --- assemble JSON header ---
    let mut header_root = serde_json::Map::new();
    header_root.insert("tensors".to_string(), serde_json::to_value(&header_tensors)?);
    if !header_scalars.is_empty() {
        header_root.insert(
            "scalars".to_string(),
            serde_json::Value::Object(header_scalars.into_iter().map(|(k, v)| (k.to_string(), v)).collect()),
        );
    }
    if !header_metadata.is_empty() {
        header_root.insert(
            "metadata".to_string(),
            serde_json::Value::Object(header_metadata.into_iter().map(|(k, v)| (k.to_string(), v)).collect()),
        );
    }

    let header_bytes = serde_json::to_vec(&header_root)?;
    let header_size_u64 = header_bytes.len() as u64;

    writer.write_all(&header_size_u64.to_le_bytes())?;
    writer.write_all(&header_bytes)?;

    for (_, tensor) in tensor_entries {
        writer.write_all(&tensor.to_bytes()?)?;
    }

    writer.flush()?;
    Ok(())
}

// ============================================================================
//   Internal types & helpers
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
struct TensorEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

fn json_error(msg: impl Into<String>) -> serde_json::Error {
    serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::InvalidData, msg.into()))
}

fn dtype_to_str(dt: luma_tensor::DType) -> &'static str {
    use luma_tensor::DType;
    match dt {
        DType::F32 => "F32",
        DType::F64 => "F64",
        DType::I32 => "I32",
        DType::U32 => "U32",
        DType::U8 => "U8",
        DType::Bool => "Bool",
    }
}

fn dtype_from_str(s: &str) -> Result<luma_tensor::DType, LumaPackError> {
    use luma_tensor::DType;
    match s {
        "F32" => Ok(DType::F32),
        "F64" => Ok(DType::F64),
        "I32" => Ok(DType::I32),
        "U32" => Ok(DType::U32),
        "U8" => Ok(DType::U8),
        "BOOL" | "Bool" => Ok(DType::Bool),
        other => Err(LumaPackError::UnknownScalarDType(other.into())),
    }
}

fn scalar_to_json(s: &Scalar) -> serde_json::Value {
    use serde_json::json;
    match s {
        Scalar::F32(v) => json!({"dtype": "F32", "value": v}),
        Scalar::F64(v) => json!({"dtype": "F64", "value": v}),
        Scalar::I32(v) => json!({"dtype": "I32", "value": v}),
        Scalar::U32(v) => json!({"dtype": "U32", "value": v}),
        Scalar::U8(v) => json!({"dtype": "U8",  "value": v}),
        Scalar::Bool(v) => json!({"dtype": "Bool", "value": v}),
    }
}

fn scalar_from_json(v: &serde_json::Value) -> Result<Scalar, LumaPackError> {
    let dtype = v["dtype"].as_str().ok_or_else(|| LumaPackError::UnknownScalarDType("missing dtype field".into()))?;

    let val = &v["value"];
    match dtype {
        "F32" => Ok(Scalar::F32(val.as_f64().ok_or_else(|| LumaPackError::MissingScalarValue("F32".into()))? as f32)),
        "F64" => Ok(Scalar::F64(val.as_f64().ok_or_else(|| LumaPackError::MissingScalarValue("F64".into()))?)),
        "I32" => Ok(Scalar::I32(val.as_i64().ok_or_else(|| LumaPackError::MissingScalarValue("I32".into()))? as i32)),
        "U32" => Ok(Scalar::U32(val.as_i64().ok_or_else(|| LumaPackError::MissingScalarValue("U32".into()))? as u32)),
        "U8" => Ok(Scalar::U8(val.as_i64().ok_or_else(|| LumaPackError::MissingScalarValue("U8".into()))? as u8)),
        "Bool" => Ok(Scalar::Bool(val.as_bool().ok_or_else(|| LumaPackError::MissingScalarValue("Bool".into()))?)),
        other => Err(LumaPackError::UnknownScalarDType(other.into())),
    }
}

// ============================================================================
//   Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use luma_tensor::DType;
    use luma_tensor::dtype::{BoolDType, FloatDType, IntDType};
    use luma_tensor::{Bool, Cpu, Int, Tensor};

    fn device() -> Cpu {
        Cpu::default()
    }

    #[test]
    fn test_lpk_roundtrip_tensors_only() {
        let dev = device();
        let a = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), FloatDType::F32).unwrap();
        let b = Tensor::<Cpu>::from_slice(&[0.1, 0.2], (2,), FloatDType::F32).unwrap();

        let mut pack = LumaPack::new();
        pack.tensors.insert("weight".into(), DynTensor::Float(a));
        pack.tensors.insert("bias".into(), DynTensor::Float(b));

        let bytes = pack.to_bytes().unwrap();
        assert!(!bytes.is_empty());

        // file round-trip via free functions
        let tmp = std::env::temp_dir().join("luma_test_roundtrip.lpk");
        save_file(&pack, &tmp).unwrap();
        let loaded = load_file(&tmp, &dev).unwrap();
        let _ = std::fs::remove_file(&tmp);

        assert_eq!(loaded.tensors.len(), 2);
        let w = loaded.tensors["weight"].as_float().unwrap();
        assert_eq!(w.dims(), &[2, 2]);
        assert!((w.to_vec().unwrap()[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_lpk_roundtrip_via_reader() {
        let dev = device();
        let t = Tensor::<Cpu>::from_slice(&[3.0, 4.0], (2,), FloatDType::F32).unwrap();
        let mut pack = LumaPack::new();
        pack.tensors.insert("x".into(), DynTensor::Float(t));
        pack.scalars.insert("lr".into(), Scalar::F64(0.01));

        let bytes = pack.to_bytes().unwrap();
        let mut cursor = std::io::Cursor::new(bytes);
        let loaded: LumaPack<Cpu> = load(&mut cursor, &dev).unwrap();
        assert_eq!(loaded.tensors.len(), 1);
        assert!((loaded.scalars["lr"].to_f64().unwrap() - 0.01).abs() < 1e-8);
    }

    #[test]
    fn test_lpk_roundtrip_with_scalars_and_metadata() {
        let dev = device();
        let t = Tensor::<Cpu>::from_slice(&[42.0], (1,), FloatDType::F32).unwrap();

        let mut pack = LumaPack::new();
        pack.tensors.insert("t".into(), DynTensor::Float(t));
        pack.scalars.insert("lr".into(), Scalar::F64(0.001));
        pack.scalars.insert("step_t".into(), Scalar::I32(100));
        pack.scalars.insert("flag".into(), Scalar::Bool(true));
        pack.metadata.insert("epoch".into(), "5".into());

        let tmp = std::env::temp_dir().join("luma_test_scalars.lpk");
        save_file(&pack, &tmp).unwrap();
        let loaded = load_file(&tmp, &dev).unwrap();
        let _ = std::fs::remove_file(&tmp);

        assert_eq!(loaded.tensors.len(), 1);
        assert_eq!(loaded.scalars.len(), 3);
        assert!((loaded.scalars["lr"].to_f64().unwrap() - 0.001).abs() < 1e-8);
        assert_eq!(loaded.scalars["step_t"].to_i64().unwrap(), 100);
        assert_eq!(loaded.scalars["flag"].to_bool().unwrap(), true);
        assert_eq!(loaded.metadata["epoch"], "5");
    }

    #[test]
    fn test_lpk_roundtrip_mixed_kinds() {
        let dev = device();
        let f = Tensor::<Cpu>::from_slice(&[1.0], (1,), FloatDType::F32).unwrap();
        let i = Tensor::<Cpu, Int>::from_slice(&[10i64], (1,), IntDType::I32).unwrap();
        let b = Tensor::<Cpu, Bool>::from_slice(&[true], (1,), BoolDType::Bool).unwrap();

        let mut pack = LumaPack::new();
        pack.tensors.insert("f".into(), DynTensor::Float(f));
        pack.tensors.insert("i".into(), DynTensor::Int(i));
        pack.tensors.insert("b".into(), DynTensor::Bool(b));

        let tmp = std::env::temp_dir().join("luma_test_mixed.lpk");
        save_file(&pack, &tmp).unwrap();
        let loaded = load_file(&tmp, &dev).unwrap();
        let _ = std::fs::remove_file(&tmp);

        assert_eq!(loaded.tensors.len(), 3);
        assert_eq!(loaded.tensors["f"].dtype(), DType::F32);
        assert_eq!(loaded.tensors["i"].dtype(), DType::I32);
        assert_eq!(loaded.tensors["b"].dtype(), DType::Bool);
        assert_eq!(loaded.tensors["i"].as_int().unwrap().to_vec().unwrap(), vec![10i64]);
        assert_eq!(loaded.tensors["b"].as_bool().unwrap().to_vec().unwrap(), vec![true]);
    }

    #[test]
    fn test_lpk_empty_pack() {
        let dev = device();
        let pack = LumaPack::<Cpu>::new();
        let tmp = std::env::temp_dir().join("luma_test_empty.lpk");
        save_file(&pack, &tmp).unwrap();
        let loaded = load_file(&tmp, &dev).unwrap();
        let _ = std::fs::remove_file(&tmp);
        assert!(loaded.tensors.is_empty());
        assert!(loaded.scalars.is_empty());
        assert!(loaded.metadata.is_empty());
    }

    #[test]
    fn test_lpk_save_file_method() {
        let dev = device();
        let mut pack = LumaPack::<Cpu>::new();
        pack.scalars.insert("k".into(), Scalar::I32(77));

        let tmp = std::env::temp_dir().join("luma_test_method.lpk");
        pack.save_file(&tmp).unwrap();
        let loaded = load_file(&tmp, &dev).unwrap();
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(loaded.scalars["k"].to_i64().unwrap(), 77);
    }
}
