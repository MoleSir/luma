use luma_macros::Module;
use luma_tensor::ops::construct::TensorCreationOptions;
use luma_tensor::{Device, Tensor};

use crate::init::{Init, NonLinearity};
use crate::{NnResult, Parameter};

// ============================================================================
//   LinearConfig
// ============================================================================

/// Configuration for [`Linear`] layers.
///
/// Use [`LinearConfig::new`] for sensible defaults; override individual fields
/// with struct-update syntax when you need custom initialisation.
#[derive(Debug, Clone)]
pub struct LinearConfig {
    pub in_features: usize,
    pub out_features: usize,
    pub bias: bool,
    pub weight_init: Init,
    pub bias_init: Init,
}

impl LinearConfig {
    /// Reasonable defaults for an `(in_features, out_features)` linear layer.
    ///
    /// - `bias = true`
    /// - `weight_init` = Kaiming uniform (fan-in mode, ReLU gain)
    /// - `bias_init`   = uniform ± 1/√(in_features)
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let bound = 1.0 / (in_features as f64).sqrt();
        Self {
            in_features,
            out_features,
            bias: true,
            weight_init: Init::kaiming_uniform(NonLinearity::Relu, false),
            bias_init: Init::uniform(-bound, bound),
        }
    }
}

// ============================================================================
//   Linear
// ============================================================================

#[derive(Module, Clone)]
#[module(display = "display")]
pub struct Linear<D: Device> {
    pub weight: Parameter<D>,       // (out_features, in_features)
    pub bias: Option<Parameter<D>>, // (out_features)

    #[module(skip)]
    pub in_features: usize,
    #[module(skip)]
    pub out_features: usize,
}

impl<D: Device> Linear<D> {
    /// Shortcut constructor with default initialisation.
    ///
    /// Equivalent to building a [`LinearConfig`] through
    /// [`LinearConfig::new`] and calling [`from_config`](Self::from_config).
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
        options: impl Into<TensorCreationOptions<D, luma_tensor::Float>>,
    ) -> NnResult<Self> {
        let config = LinearConfig { in_features, out_features, bias, ..LinearConfig::new(in_features, out_features) };
        Self::from_config(&config, options)
    }

    /// Full-control constructor from a [`LinearConfig`].
    pub fn from_config(config: &LinearConfig, options: impl Into<TensorCreationOptions<D, luma_tensor::Float>>) -> NnResult<Self> {
        let options: TensorCreationOptions<D, luma_tensor::Float> = options.into();
        let opts = (&options.device, options.dtype);

        let weight =
            config
                .weight_init
                .init_param_with((config.out_features, config.in_features), config.in_features, config.out_features, opts)?;
        let bias = if config.bias {
            Some(config.bias_init.init_param_with((config.out_features,), config.in_features, config.out_features, opts)?)
        } else {
            None
        };

        Ok(Self { weight, bias, in_features: config.in_features, out_features: config.out_features })
    }

    /// Custom display — called by `Module::extra_display` via `#[module(display = "display")]`.
    pub fn display(&self) -> String {
        format!("in={}, out={}", self.in_features, self.out_features)
    }

    /// Forward pass: `input @ weight^T + bias`.
    pub fn forward(&self, input: &Tensor<D, luma_tensor::Float>) -> NnResult<Tensor<D, luma_tensor::Float>> {
        crate::functional::linear(input, &self.weight, self.bias.as_deref())
    }
}

// ============================================================================
//   Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Module;
    use luma_tensor::{Cpu, Tensor};

    #[test]
    fn test_linear_with_skip_and_display() {
        let w = Parameter::new(Tensor::zeros(&[2, 3], Cpu::default()).unwrap());
        let b = Parameter::new(Tensor::zeros(&[2], Cpu::default()).unwrap());
        let linear = Linear { weight: w, bias: Some(b), in_features: 3, out_features: 2 };

        // 1. Clone works (combined derive)
        let _cloned = linear.clone();

        // 2. Tree display via UFCS (inherent method shadows trait)
        let tree = format!("{}", Module::<Cpu>::display(&linear));
        assert!(tree.contains("Linear"), "tree: {tree}");
        assert!(tree.contains("in=3, out=2"), "tree: {tree}");

        // 3. Direct display returns the custom String (inherent method)
        let direct = linear.display();
        assert_eq!(direct, "in=3, out=2");
    }
}
