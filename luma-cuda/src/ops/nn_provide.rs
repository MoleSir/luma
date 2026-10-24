//! Fused CUDA implementations of the high-level NN operators.

use crate::launch;
use crate::{Cuda, CudaFloatSlice, CudaFloatStorage};
use luma_tensor::{
    CustomOp1, CustomOp2, CustomOpError, Float, FloatDType, FloatOps, Layout, NnOpDefault, NnProvide, Result, Shape,
    Tensor,
};

fn softmax_fwd(x: &CudaFloatStorage, layout: &Layout, dim: usize) -> Result<CudaFloatStorage> {
    if !layout.is_contiguous() {
        return Err(luma_tensor::Error::RequiresContiguous { op: "softmax" });
    }
    let last_dim = layout.dims().len() - 1;
    if dim != last_dim {
        let data = Cuda::f_to_vec(x, layout)?;
        let dims = layout.dims();
        let reduce_size = dims[dim];
        let outer = dims[..dim].iter().product::<usize>();
        let inner = dims[dim + 1..].iter().product::<usize>();
        let mut out = vec![0f64; data.len()];
        for o in 0..outer {
            for k in 0..inner {
                let mut row = vec![0f64; reduce_size];
                for r in 0..reduce_size {
                    row[r] = data[o * reduce_size * inner + r * inner + k];
                }
                let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_sum: f64 = row.iter().map(|&v| (v - max_val).exp()).sum();
                for r in 0..reduce_size {
                    out[o * reduce_size * inner + r * inner + k] = ((row[r] - max_val).exp()) / exp_sum;
                }
            }
        }
        let dtype = x.dtype().as_float();
        let storage = match dtype {
            FloatDType::F64 => Cuda::f_from_f64(&out, &x.device)?,
            FloatDType::F32 => {
                let v: Vec<f32> = out.iter().map(|&x| x as f32).collect();
                Cuda::f_from_f32(&v, &x.device)?
            }
            FloatDType::F16 => {
                let v: Vec<half::f16> = out.iter().map(|&x| half::f16::from_f64(x)).collect();
                Cuda::f_from_f16(&v, &x.device)?
            }
            FloatDType::BF16 => {
                let v: Vec<half::bf16> = out.iter().map(|&x| half::bf16::from_f64(x)).collect();
                Cuda::f_from_bf16(&v, &x.device)?
            }
        };
        return Ok(storage);
    }
    let device = &x.device;
    match &x.slice {
        CudaFloatSlice::F32(data) => {
            let slice = launch::launch_softmax(device, data, layout, dim, "softmax_f32")?;
            Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(slice), device: device.clone() })
        }
        CudaFloatSlice::F64(data) => {
            let slice = launch::launch_softmax(device, data, layout, dim, "softmax_f64")?;
            Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(slice), device: device.clone() })
        }
        CudaFloatSlice::F16(data) => {
            let slice = launch::launch_softmax(device, data, layout, dim, "softmax_f16")?;
            Ok(CudaFloatStorage { slice: CudaFloatSlice::F16(slice), device: device.clone() })
        }
        CudaFloatSlice::BF16(data) => {
            let slice = launch::launch_softmax(device, data, layout, dim, "softmax_bf16")?;
            Ok(CudaFloatStorage { slice: CudaFloatSlice::BF16(slice), device: device.clone() })
        }
    }
}

fn rms_norm_fwd(x: &CudaFloatStorage, x_l: &Layout, weight: &CudaFloatStorage, weight_l: &Layout, eps: f64) -> Result<CudaFloatStorage> {
    if !x_l.is_contiguous() || !weight_l.is_contiguous() {
        return Err(luma_tensor::Error::RequiresContiguous { op: "rms_norm" });
    }
    x.device.same_ordinal(&weight.device, "rms_norm")?;
    let device = &x.device;
    match (&x.slice, &weight.slice) {
        (CudaFloatSlice::F32(xs), CudaFloatSlice::F32(ws)) => {
            let slice = launch::launch_rms_norm(device, xs, ws, x_l, weight_l, eps as f32, "rms_norm_f32")?;
            Ok(CudaFloatStorage { slice: CudaFloatSlice::F32(slice), device: device.clone() })
        }
        (CudaFloatSlice::F64(xs), CudaFloatSlice::F64(ws)) => {
            let slice = launch::launch_rms_norm(device, xs, ws, x_l, weight_l, eps, "rms_norm_f64")?;
            Ok(CudaFloatStorage { slice: CudaFloatSlice::F64(slice), device: device.clone() })
        }
        (CudaFloatSlice::F16(xs), CudaFloatSlice::F16(ws)) => {
            let slice = launch::launch_rms_norm(device, xs, ws, x_l, weight_l, half::f16::from_f64(eps), "rms_norm_f16")?;
            Ok(CudaFloatStorage { slice: CudaFloatSlice::F16(slice), device: device.clone() })
        }
        (CudaFloatSlice::BF16(xs), CudaFloatSlice::BF16(ws)) => {
            let slice = launch::launch_rms_norm(device, xs, ws, x_l, weight_l, half::bf16::from_f64(eps), "rms_norm_bf16")?;
            Ok(CudaFloatStorage { slice: CudaFloatSlice::BF16(slice), device: device.clone() })
        }
        _ => Err(luma_tensor::Error::DTypeMismatch { lhs: x.dtype(), rhs: weight.dtype(), op: "rms_norm" }),
    }
}

struct CudaSoftmaxOp {
    dim: usize,
}

impl CustomOp1<Cuda> for CudaSoftmaxOp {
    fn name(&self) -> String {
        "cuda.softmax".into()
    }

    fn forward(&self, x: &Tensor<Cuda, Float>) -> std::result::Result<(CudaFloatStorage, Shape), CustomOpError> {
        let guard = x.storage_read()?;
        Ok((softmax_fwd(&guard, x.layout(), self.dim)?, x.shape().clone()))
    }

    fn backward(&self, _x: &Tensor<Cuda, Float>, y: &Tensor<Cuda, Float>, grad: &Tensor<Cuda, Float>) -> std::result::Result<Tensor<Cuda, Float>, CustomOpError> {
        Ok(NnOpDefault::softmax_backward(y, grad, self.dim, true)?)
    }
}

struct CudaRmsNormOp {
    eps: f64,
}

impl CustomOp2<Cuda> for CudaRmsNormOp {
    fn name(&self) -> String {
        "cuda.rms_norm".into()
    }

    fn forward(&self, x: &Tensor<Cuda, Float>, weight: &Tensor<Cuda, Float>) -> std::result::Result<(CudaFloatStorage, Shape), CustomOpError> {
        let x_guard = x.storage_read()?;
        let w_guard = weight.storage_read()?;
        Ok((rms_norm_fwd(&x_guard, x.layout(), &w_guard, weight.layout(), self.eps)?, x.shape().clone()))
    }

    fn backward(
        &self,
        x: &Tensor<Cuda, Float>,
        weight: &Tensor<Cuda, Float>,
        y: &Tensor<Cuda, Float>,
        grad: &Tensor<Cuda, Float>,
    ) -> std::result::Result<(Tensor<Cuda, Float>, Tensor<Cuda, Float>), CustomOpError> {
        let (dx, dweight, _) = NnOpDefault::rms_norm_backward(x, Some(weight), None, y, grad, self.eps)?;
        Ok((dx, dweight.expect("rms_norm weight is present")))
    }
}

impl NnProvide<Cuda> for Cuda {
    fn softmax(x: &Tensor<Cuda, Float>, dim: usize, keepdim: bool) -> luma_tensor::Result<Tensor<Cuda, Float>> {
        if !x.is_contiguous() {
            return NnOpDefault::softmax_forward(x, dim, keepdim);
        }
        let y = x.custom_op1(Box::new(CudaSoftmaxOp { dim }))?;
        if keepdim { Ok(y) } else { y.squeeze(dim) }
    }

    fn rms_norm(
        x: &Tensor<Cuda, Float>,
        weight: Option<&Tensor<Cuda, Float>>,
        bias: Option<&Tensor<Cuda, Float>>,
        eps: f64,
    ) -> luma_tensor::Result<Tensor<Cuda, Float>> {
        match (weight, bias) {
            // The fused kernel covers the common weight-only case; bias or a
            // layout it cannot handle falls back to the composed default.
            (Some(weight), None) => {
                if !x.is_contiguous() || !weight.is_contiguous() {
                    return NnOpDefault::rms_norm_forward(x, Some(weight), None, eps);
                }
                x.custom_op2(weight, Box::new(CudaRmsNormOp { eps }))
            }
            (weight, bias) => NnOpDefault::rms_norm_forward(x, weight, bias, eps),
        }
    }
}
