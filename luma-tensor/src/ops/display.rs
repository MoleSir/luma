//! Display, to_scalar, allclose, false_count, and other utility ops.

use crate::dtype::{FloatDType, IntDType};
use crate::{Bool, Device, Float, Int, Tensor};

// ---- to_scalar ----

impl<D: Device> Tensor<D, Float> {
    /// Read the single element of a scalar (0-d or 1-element) tensor.
    pub fn to_scalar(&self) -> crate::Result<f64> {
        if self.element_count() != 1 {
            return Err(crate::Error::NotScalar);
        }
        Ok(self.to_vec()?[0])
    }
}

impl<D: Device> Tensor<D, Int> {
    pub fn to_scalar(&self) -> crate::Result<i64> {
        if self.element_count() != 1 {
            return Err(crate::Error::NotScalar);
        }
        Ok(self.to_vec()?[0])
    }
}

// ---- allclose ----

impl<D: Device> Tensor<D, Float> {
    pub fn allclose(&self, other: &Self, rtol: f64, atol: f64) -> crate::Result<bool> {
        if self.element_count() != other.element_count() {
            return Ok(false);
        }
        D::f_allclose(&*self.storage_read()?, self.layout(), &*other.storage_read()?, other.layout(), rtol, atol)
    }
}

impl<D: Device> Tensor<D, Int> {
    pub fn allclose(&self, other: &Self) -> crate::Result<bool> {
        if self.element_count() != other.element_count() {
            return Ok(false);
        }
        D::i_allclose(&*self.storage_read()?, self.layout(), &*other.storage_read()?, other.layout())
    }
}

impl<D: Device> Tensor<D, Bool> {
    pub fn allclose(&self, other: &Self) -> crate::Result<bool> {
        if self.element_count() != other.element_count() {
            return Ok(false);
        }
        D::b_allclose(&*self.storage_read()?, self.layout(), &*other.storage_read()?, other.layout())
    }
}

// ---- false_count (Bool) ----

impl<D: Device> Tensor<D, Bool> {
    pub fn false_count(&self) -> crate::Result<usize> {
        Ok(self.element_count() - self.true_count()?)
    }
}

// ---- Display ----
//
// Prints in NumPy/PyTorch style:
//   scalar:   tensor(3.14)
//   1-D:      tensor([1., 2., 3.])
//   2-D:      tensor([[1., 2.],
//                     [3., 4.]])
//   Higher:   nested [ ... ]
//
// Precision: 4 significant digits for float, exact for int/bool.

fn fmt_f64(v: f64, precision: usize) -> String {
    let abs = v.abs();
    if abs == 0.0 || (abs >= 0.001 && abs < 1e5) {
        format!("{:.prec$}", v, prec = precision)
    } else {
        format!("{:.prec$e}", v, prec = precision)
    }
}

fn write_nd(buf: &mut String, data: &[String], dims: &[usize], depth: usize, indent: usize) {
    if dims.is_empty() {
        // scalar
        buf.push_str(&data[0]);
        return;
    }
    if dims.len() == 1 {
        buf.push('[');
        for (i, v) in data.iter().enumerate() {
            if i > 0 {
                buf.push_str(", ");
            }
            buf.push_str(v);
        }
        buf.push(']');
        return;
    }
    // multi-dim: recurse
    let stride: usize = dims[1..].iter().product();
    buf.push('[');
    for i in 0..dims[0] {
        if i > 0 {
            buf.push(',');
            buf.push('\n');
            for _ in 0..indent + depth + 1 {
                buf.push(' ');
            }
        }
        let slice = &data[i * stride..(i + 1) * stride];
        write_nd(buf, slice, &dims[1..], depth + 1, indent);
    }
    buf.push(']');
}

impl<D: Device> std::fmt::Display for Tensor<D, Float> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const PREC: usize = 4;
        let vals = match self.to_vec() {
            Ok(v) => v,
            Err(e) => return write!(f, "<Tensor error: {}>", e),
        };
        let strs: Vec<String> = vals.iter().map(|&v| fmt_f64(v, PREC)).collect();
        let dtype_str = match self.dtype() {
            FloatDType::F32 => "f32",
            FloatDType::F64 => "f64",
        };
        let mut buf = format!("Tensor<{}>(", dtype_str);
        let indent = buf.len();
        write_nd(&mut buf, &strs, self.shape().dims(), 0, indent);
        buf.push(')');
        write!(f, "{}", buf)
    }
}

impl<D: Device> std::fmt::Display for Tensor<D, Int> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let vals = match self.to_vec() {
            Ok(v) => v,
            Err(e) => return write!(f, "<Tensor error: {}>", e),
        };
        let strs: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
        let dtype_str = match self.dtype() {
            IntDType::I32 => "i32",
            IntDType::U32 => "u32",
            IntDType::U8 => "u8",
        };
        let mut buf = format!("Tensor<{}>(", dtype_str);
        let indent = buf.len();
        write_nd(&mut buf, &strs, self.shape().dims(), 0, indent);
        buf.push(')');
        write!(f, "{}", buf)
    }
}

impl<D: Device> std::fmt::Display for Tensor<D, Bool> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let vals = match self.to_vec() {
            Ok(v) => v,
            Err(e) => return write!(f, "<Tensor error: {}>", e),
        };
        let strs: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
        let mut buf = "Tensor<bool>(".to_string();
        let indent = buf.len();
        write_nd(&mut buf, &strs, self.shape().dims(), 0, indent);
        buf.push(')');
        write!(f, "{}", buf)
    }
}
