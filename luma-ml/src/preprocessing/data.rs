use std::f64;

use luma_tensor::{no_grad, Device, Tensor, D};

use crate::{TransformFit, TransformModel};


pub struct StandardScaler {
    pub eps: f64,
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self { eps: f64::EPSILON }
    }
}

pub struct StandardScalerModel<Dev: Device> {
    pub mean: Tensor<Dev>,
    pub std: Tensor<Dev>,
}

impl<Dev: Device> TransformFit<Tensor<Dev>> for StandardScaler {
    type Output = Tensor<Dev>;
    type Model = StandardScalerModel<Dev>;

    fn fit(&self, x: &Tensor<Dev>) -> crate::MlResult<Self::Model> {
        no_grad!();
        let mean = x.mean_keepdim(D::Minus1)?;
        let var = x.var_keepdim(D::Minus1)?;
        let std = (var + self.eps).sqrt()?;
        Ok(StandardScalerModel { mean, std })
    }
}

impl<Dev: Device> TransformModel for StandardScalerModel<Dev> {
    type Input = Tensor<Dev>;
    type Output = Tensor<Dev>;

    fn transform(&self, x: &Self::Input) -> crate::MlResult<Self::Output> {
        let y = x
            .broadcast_sub(&self.mean)?
            .broadcast_div(&self.std)?;
        Ok(y)
    }
}

// ====================================================================== //
//                               Min Max
// ====================================================================== //

pub struct MinMaxScaler {
    pub eps: f64,
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self { eps: f64::EPSILON }
    }
}

pub struct MinMaxScalerModel<Dev: Device> {
    pub min: Tensor<Dev>,
    pub delta: Tensor<Dev>,
}

impl<Dev: Device> TransformFit<Tensor<Dev>> for MinMaxScaler {
    type Output = Tensor<Dev>;
    type Model = MinMaxScalerModel<Dev>;

    fn fit(&self, x: &Tensor<Dev>) -> crate::MlResult<Self::Model> {
        no_grad!();
        let min = x.min_keepdim(D::Minus1)?;
        let max = x.max_keepdim(D::Minus1)?;
        let delta = (&max - &min) + self.eps;
        Ok(MinMaxScalerModel { min, delta })
    }
}

impl<Dev: Device> TransformModel for MinMaxScalerModel<Dev> {
    type Input = Tensor<Dev>;
    type Output = Tensor<Dev>;

    fn transform(&self, x: &Self::Input) -> crate::MlResult<Self::Output> {
        let y = x
            .broadcast_sub(&self.min)?
            .broadcast_div(&self.delta)?;
        Ok(y)
    }
}