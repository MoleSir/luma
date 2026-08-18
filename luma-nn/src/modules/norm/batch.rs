use luma_macros::Module;
use luma_tensor::ops::construct::TensorCreationOptions;
use luma_tensor::{Device, Float, Tensor};

use crate::init::Init;
use crate::{Buffer, NnError, NnResult, Parameter};

// ============================================================================
//   BatchNorm1dConfig
// ============================================================================

#[derive(Debug, Clone)]
pub struct BatchNorm1dConfig {
    pub num_features: usize,
    pub eps: f64,
    pub momentum: f64,
    pub weight_init: Init,
    pub bias_init: Init,
}

impl BatchNorm1dConfig {
    pub fn new(num_features: usize) -> Self {
        Self { num_features, eps: 1e-5, momentum: 0.1, weight_init: Init::Ones, bias_init: Init::Zeros }
    }
}

// ============================================================================
//   BatchNorm1d
// ============================================================================

/// Applies Batch Normalization over a 2D or 3D input.
#[derive(Module)]
#[module(display = "display")]
#[module(train = "train")]
pub struct BatchNorm1d<Dev: Device> {
    pub gamma: Parameter<Dev>,
    pub beta: Parameter<Dev>,

    pub running_mean: Buffer<Dev>,
    pub running_var: Buffer<Dev>,

    #[module(skip)]
    pub num_features: usize,
    #[module(skip)]
    pub eps: f64,
    #[module(skip)]
    pub momentum: f64,
    #[module(skip)]
    pub training: bool,
}

impl<Dev: Device> BatchNorm1d<Dev> {
    pub fn new(num_features: usize, options: impl Into<TensorCreationOptions<Dev, Float>>) -> NnResult<Self> {
        let config = BatchNorm1dConfig::new(num_features);
        Self::from_config(&config, options)
    }

    pub fn from_config(config: &BatchNorm1dConfig, options: impl Into<TensorCreationOptions<Dev, Float>>) -> NnResult<Self> {
        let options: TensorCreationOptions<Dev, Float> = options.into();
        let opts = (&options.device, options.dtype);

        let gamma = config.weight_init.init_param((config.num_features,), opts)?;
        let beta = config.bias_init.init_param((config.num_features,), opts)?;
        let running_mean = Init::Zeros.init_buffer((config.num_features,), opts)?;
        let running_var = Init::Ones.init_buffer((config.num_features,), opts)?;

        Ok(Self {
            gamma,
            beta,
            running_mean,
            running_var,
            num_features: config.num_features,
            eps: config.eps,
            momentum: config.momentum,
            training: false,
        })
    }

    pub fn display(&self) -> String {
        format!("num_features={}, eps={}, momentum={}", self.num_features, self.eps, self.momentum)
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    /// BatchNorm1d forward
    ///
    /// ## Argument
    ///
    /// * `input`: (N, C, L) or (N, L, C)
    pub fn forward(&self, input: &Tensor<Dev>) -> NnResult<Tensor<Dev>> {
        match input.rank() {
            2 => self.forward_impl(input),
            3 => {
                let input_permuted = input.permute((0, 2, 1))?;
                let (n, l, c) = input_permuted.dims3()?;
                let input_flattened = input_permuted.reshape((n * l, c))?;

                let out_flattened = self.forward_impl(&input_flattened)?;

                let out = out_flattened.reshape((n, l, c))?.permute((0, 2, 1))?;

                Ok(out)
            }
            _ => Err(NnError::BatchNorm1dUnsupportShape(input.shape().clone()))?,
        }
    }

    fn forward_impl(&self, x: &Tensor<Dev>) -> NnResult<Tensor<Dev>> {
        assert_eq!(x.rank(), 2);

        let x_normalized = if self.training {
            let batch_mean = x.mean_keepdim(0)?;
            let batch_var = x.var_keepdim(0)?;

            let x_normalized = x.broadcast_sub(&batch_mean)?.broadcast_div(&(&batch_var + self.eps).sqrt()?)?;

            self.running_mean.mul_scalar_(self.momentum)?;
            self.running_mean.add_(&((1.0 - self.momentum) * batch_mean))?;

            self.running_var.mul_scalar_(self.momentum)?;
            self.running_var.add_(&((1.0 - self.momentum) * batch_var))?;

            x_normalized
        } else {
            x.broadcast_sub(&self.running_mean)?.broadcast_div(&self.running_var.add_scalar(self.eps)?.sqrt()?)?
        };

        let out = self.gamma.broadcast_mul(&x_normalized)?.broadcast_add(&self.beta)?;
        Ok(out)
    }
}
