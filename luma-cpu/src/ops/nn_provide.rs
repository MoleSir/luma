//! Fused CPU implementations of the high-level NN operators.
//!
//! The default [`NnProvide`] implementations compose primitive ops; these
//! overrides call the hand-written kernels and provide matching backward
//! through [`CustomOp1`] / [`CustomOp2`].

use crate::kernels::nn;
use crate::{Cpu, CpuFloatStorage};
use luma_tensor::{CustomOp1, CustomOp2, CustomOpError, Float, NnOpDefault, NnProvide, Shape, Tensor};

struct CpuSoftmaxOp {
    dim: usize,
}

impl CustomOp1<Cpu> for CpuSoftmaxOp {
    fn name(&self) -> String {
        "cpu.softmax".into()
    }

    fn forward(&self, x: &Tensor<Cpu, Float>) -> Result<(CpuFloatStorage, Shape), CustomOpError> {
        let guard = x.storage_read()?;
        let storage = match &*guard {
            CpuFloatStorage::F32(d, device) => {
                CpuFloatStorage::F32(nn::softmax(d, x.layout(), self.dim, device)?, device.clone())
            }
            CpuFloatStorage::F64(d, device) => {
                CpuFloatStorage::F64(nn::softmax(d, x.layout(), self.dim, device)?, device.clone())
            }
            CpuFloatStorage::F16(d, device) => {
                CpuFloatStorage::F16(nn::softmax(d, x.layout(), self.dim, device)?, device.clone())
            }
            CpuFloatStorage::BF16(d, device) => {
                CpuFloatStorage::BF16(nn::softmax(d, x.layout(), self.dim, device)?, device.clone())
            }
        };
        Ok((storage, x.shape().clone()))
    }

    fn backward(&self, _x: &Tensor<Cpu, Float>, y: &Tensor<Cpu, Float>, grad: &Tensor<Cpu, Float>) -> Result<Tensor<Cpu, Float>, CustomOpError> {
        Ok(NnOpDefault::softmax_backward(y, grad, self.dim, true)?)
    }
}

struct CpuRmsNormOp {
    eps: f64,
}

impl CustomOp2<Cpu> for CpuRmsNormOp {
    fn name(&self) -> String {
        "cpu.rms_norm".into()
    }

    fn forward(&self, x: &Tensor<Cpu, Float>, weight: &Tensor<Cpu, Float>) -> Result<(CpuFloatStorage, Shape), CustomOpError> {
        let x_guard = x.storage_read()?;
        let w_guard = weight.storage_read()?;
        let storage = match (&*x_guard, &*w_guard) {
            (CpuFloatStorage::F32(d, device), CpuFloatStorage::F32(w, _)) => CpuFloatStorage::F32(
                nn::rms_norm(d, x.layout(), w, weight.layout(), self.eps as f32, device)?,
                device.clone(),
            ),
            (CpuFloatStorage::F64(d, device), CpuFloatStorage::F64(w, _)) => {
                CpuFloatStorage::F64(nn::rms_norm(d, x.layout(), w, weight.layout(), self.eps, device)?, device.clone())
            }
            (CpuFloatStorage::F16(d, device), CpuFloatStorage::F16(w, _)) => CpuFloatStorage::F16(
                nn::rms_norm(d, x.layout(), w, weight.layout(), half::f16::from_f64(self.eps), device)?,
                device.clone(),
            ),
            (CpuFloatStorage::BF16(d, device), CpuFloatStorage::BF16(w, _)) => CpuFloatStorage::BF16(
                nn::rms_norm(d, x.layout(), w, weight.layout(), half::bf16::from_f64(self.eps), device)?,
                device.clone(),
            ),
            (l, r) => {
                return Err(luma_tensor::Error::DTypeMismatch { lhs: l.dtype(), rhs: r.dtype(), op: "rms_norm" }.into());
            }
        };
        Ok((storage, x.shape().clone()))
    }

    fn backward(
        &self,
        x: &Tensor<Cpu, Float>,
        weight: &Tensor<Cpu, Float>,
        y: &Tensor<Cpu, Float>,
        grad: &Tensor<Cpu, Float>,
    ) -> Result<(Tensor<Cpu, Float>, Tensor<Cpu, Float>), CustomOpError> {
        let (dx, dweight, _) = NnOpDefault::rms_norm_backward(x, Some(weight), None, y, grad, self.eps)?;
        Ok((dx, dweight.expect("rms_norm weight is present")))
    }
}

impl NnProvide<Cpu> for Cpu {
    fn softmax(x: &Tensor<Cpu, Float>, dim: usize, keepdim: bool) -> luma_tensor::Result<Tensor<Cpu, Float>> {
        let y = x.custom_op1(Box::new(CpuSoftmaxOp { dim }))?;
        if keepdim { Ok(y) } else { y.squeeze(dim) }
    }

    fn rms_norm(
        x: &Tensor<Cpu, Float>,
        weight: Option<&Tensor<Cpu, Float>>,
        bias: Option<&Tensor<Cpu, Float>>,
        eps: f64,
    ) -> luma_tensor::Result<Tensor<Cpu, Float>> {
        match (weight, bias) {
            // The fused kernel covers the common weight-only case; bias falls
            // back to the composed default.
            (Some(weight), None) => x.custom_op2(weight, Box::new(CpuRmsNormOp { eps })),
            (weight, bias) => NnOpDefault::rms_norm_forward(x, weight, bias, eps),
        }
    }
}
