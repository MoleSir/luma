// ============================================================================================ //
//                        NonLinearity — auto-calculate gain
// ============================================================================================ //

/// Activation function for automatic gain computation in Kaiming initialisation.
#[derive(Debug, Clone, Copy)]
pub enum NonLinearity {
    /// Linear / identity (`gain = 1.0`)
    Linear,
    Sigmoid,
    Tanh,
    Relu,
    /// Leaky ReLU with the given negative slope α.
    LeakyRelu {
        negative_slope: f64,
    },
    /// Gaussian error linear unit.
    Gelu,
    /// Sigmoid linear unit.
    Silu,
}

impl NonLinearity {
    /// Compute the recommended gain for Kaiming initialisation.
    ///
    /// Values follow PyTorch's `torch.nn.init.calculate_gain`.
    pub fn calculate_gain(&self) -> f64 {
        match self {
            Self::Linear | Self::Sigmoid => 1.0,
            Self::Tanh => 5.0 / 3.0,
            Self::Relu | Self::Gelu | Self::Silu => 2.0f64.sqrt(),
            Self::LeakyRelu { negative_slope: a } => (2.0 / (1.0 + a * a)).sqrt(),
        }
    }
}

// ============================================================================================ //
//                        Init
// ============================================================================================ //

#[derive(Debug, Clone, Copy)]
pub enum Init {
    Constant {
        value: f64,
    },
    /// Fills tensor with 1s everywhere
    Ones,
    /// Fills tensor with 0s everywhere
    Zeros,
    /// Fills tensor with values drawn uniformly between specified values
    Uniform {
        min: f64,
        max: f64,
    },
    /// Fills tensor with values drawn from normal distribution with specified mean and std
    Normal {
        mean: f64,
        std: f64,
    },
    /// Fills tensor with values according to the uniform version of Kaiming initialization
    KaimingUniform {
        gain: f64,
        fan_out_only: bool,
    },
    /// Fills tensor with values according to the uniform version of Kaiming initialization
    KaimingNormal {
        gain: f64,
        fan_out_only: bool,
    },
    /// Fills tensor with values according to the uniform version of Xavier Glorot initialization
    XavierUniform {
        gain: f64,
    },
    /// Fills tensor with values according to the normal version of Xavier Glorot initialization
    XavierNormal {
        gain: f64,
    },
}

impl Init {
    pub fn zeros() -> Self {
        Self::Zeros
    }

    pub fn ones() -> Self {
        Self::Ones
    }

    pub fn constant(value: f64) -> Self {
        Self::Constant { value }
    }

    /// Returns an init that generates values from a standard normal distribution
    pub fn standard_normal() -> Self {
        Self::Normal { mean: 0.0, std: 1.0 }
    }

    /// Returns an init that generates values from a normal distribution.
    pub fn normal(mean: f64, std: f64) -> Self {
        Self::Normal { mean, std }
    }

    /// Returns an init that generates values from a uniform distribution between 0 and 1.
    pub fn standard_uniform() -> Self {
        Self::Uniform { min: 0.0, max: 1.0 }
    }

    /// Returns an init that generates values from a uniform distribution between min and max.
    pub fn uniform(min: f64, max: f64) -> Self {
        Self::Uniform { min, max }
    }

    /// Kaiming (He) Uniform Initialisation — auto-computes `gain`.
    ///
    /// For manual gain control use [`kaiming_uniform_with_gain`](Self::kaiming_uniform_with_gain).
    pub fn kaiming_uniform(nonlinearity: NonLinearity, fan_out_only: bool) -> Self {
        Self::KaimingUniform { gain: nonlinearity.calculate_gain(), fan_out_only }
    }

    /// Kaiming (He) Uniform Initialisation with an explicit gain value.
    pub fn kaiming_uniform_with_gain(gain: f64, fan_out_only: bool) -> Self {
        Self::KaimingUniform { gain, fan_out_only }
    }

    /// Kaiming (He) Normal Initialisation — auto-computes `gain`.
    ///
    /// For manual gain control use [`kaiming_normal_with_gain`](Self::kaiming_normal_with_gain).
    pub fn kaiming_normal(nonlinearity: NonLinearity, fan_out_only: bool) -> Self {
        Self::KaimingNormal { gain: nonlinearity.calculate_gain(), fan_out_only }
    }

    /// Kaiming (He) Normal Initialisation with an explicit gain value.
    pub fn kaiming_normal_with_gain(gain: f64, fan_out_only: bool) -> Self {
        Self::KaimingNormal { gain, fan_out_only }
    }

    /// Xavier (Glorot) Uniform Initialization.
    /// Recommended for layers with Sigmoid/Tanh activations.
    pub fn xavier_uniform(gain: f64) -> Self {
        Self::XavierUniform { gain }
    }

    /// Xavier (Glorot) Normal Initialization.
    /// Recommended for layers with Sigmoid/Tanh activations.
    pub fn xavier_normal(gain: f64) -> Self {
        Self::XavierNormal { gain }
    }
}

// ============================================================================================ //
//                        Apply — materialize a tensor from the init
// ============================================================================================ //

use luma_tensor::ops::construct::TensorCreationOptions;
use luma_tensor::{Device, Float, Shape, Tensor};

use crate::{Buffer, NnResult, Parameter};
use std::cell::Cell;

impl Init {
    /// Create a tensor with the given shape using this initialisation.
    ///
    /// Use [`init_with`](Self::init_with) when you need Kaiming / Xavier
    /// initialisation (which depends on fan-in / fan-out).
    #[inline]
    pub fn init<D: Device>(
        &self,
        shape: impl Into<Shape>,
        options: impl Into<TensorCreationOptions<D, Float>>,
    ) -> NnResult<Tensor<D, Float>> {
        self.apply(shape, None, None, options)
    }

    /// Create a tensor, providing fan-in / fan-out for Kaiming / Xavier.
    #[inline]
    pub fn init_with<D: Device>(
        &self,
        shape: impl Into<Shape>,
        fan_in: usize,
        fan_out: usize,
        options: impl Into<TensorCreationOptions<D, Float>>,
    ) -> NnResult<Tensor<D, Float>> {
        self.apply(shape, Some(fan_in), Some(fan_out), options)
    }

    /// Create a [`Parameter`] with the given shape.
    #[inline]
    pub fn init_param<D: Device>(
        &self,
        shape: impl Into<Shape>,
        options: impl Into<TensorCreationOptions<D, Float>>,
    ) -> NnResult<Parameter<D>> {
        self.init(shape, options).map(Parameter::new)
    }

    /// Create a [`Parameter`] with the given shape.
    #[inline]
    pub fn init_param_with<D: Device>(
        &self,
        shape: impl Into<Shape>,
        fan_in: usize,
        fan_out: usize,
        options: impl Into<TensorCreationOptions<D, Float>>,
    ) -> NnResult<Parameter<D>> {
        self.init_with(shape, fan_in, fan_out, options).map(Parameter::new)
    }

    /// Create a [`Buffer`] with the given shape.
    #[inline]
    pub fn init_buffer<D: Device>(
        &self,
        shape: impl Into<Shape>,
        options: impl Into<TensorCreationOptions<D, Float>>,
    ) -> NnResult<Buffer<D>> {
        self.init(shape, options).map(Buffer::<D, Float>::new)
    }

    /// Create a [`Buffer`] with the given shape.
    #[inline]
    pub fn init_buffer_with<D: Device>(
        &self,
        shape: impl Into<Shape>,
        fan_in: usize,
        fan_out: usize,
        options: impl Into<TensorCreationOptions<D, Float>>,
    ) -> NnResult<Buffer<D>> {
        self.init_with(shape, fan_in, fan_out, options).map(Buffer::<D, Float>::new)
    }

    /// Core implementation.
    pub fn apply<D: Device>(
        &self,
        shape: impl Into<Shape>,
        fan_in: Option<usize>,
        fan_out: Option<usize>,
        options: impl Into<TensorCreationOptions<D, Float>>,
    ) -> NnResult<Tensor<D, Float>> {
        let shape = shape.into();
        let opts = options.into();

        if is_meta_init() {
            // TODO: Tensor::meta(shape) constructor not yet public.
            // For now, return zeros in meta-init mode.
            return Ok(Tensor::<D, Float>::zeros(shape, opts)?);
        }

        let tensor = match self {
            Init::Zeros => Tensor::zeros(shape, opts),
            Init::Ones => Tensor::ones(shape, opts),
            Init::Constant { value } => Tensor::full(shape, *value, opts),
            Init::Uniform { min, max } => Tensor::rand(*min, *max, shape, opts),
            Init::Normal { mean, std } => Tensor::randn(*mean, *std, shape, opts),
            Init::KaimingUniform { gain, fan_out_only } => {
                let bound = 3.0f64.sqrt() * gain * kaiming_std(*fan_out_only, fan_in, fan_out);
                Tensor::rand(-bound, bound, shape, opts)
            }
            Init::KaimingNormal { gain, fan_out_only } => {
                let std = gain * kaiming_std(*fan_out_only, fan_in, fan_out);
                Tensor::randn(0.0, std, shape, opts)
            }
            Init::XavierUniform { gain } => {
                let a = 3.0f64.sqrt() * gain * xavier_std(fan_in, fan_out);
                Tensor::rand(-a, a, shape, opts)
            }
            Init::XavierNormal { gain } => {
                let std = gain * xavier_std(fan_in, fan_out);
                Tensor::randn(0.0, std, shape, opts)
            }
        };

        Ok(tensor?)
    }
}

/// √(1 / fan).  Panics if neither fan_in nor fan_out is provided.
fn kaiming_std(fan_out_only: bool, fan_in: Option<usize>, fan_out: Option<usize>) -> f64 {
    let fan = if fan_out_only { fan_out } else { fan_in };
    let fan = fan.expect("Kaiming init requires fan_in or fan_out — use init_with()");
    (fan as f64).sqrt().recip()
}

/// √(2 / (fan_in + fan_out)).  Panics if fan_in or fan_out is missing.
fn xavier_std(fan_in: Option<usize>, fan_out: Option<usize>) -> f64 {
    let fi = fan_in.expect("Xavier init requires fan_in — use init_with()");
    let fo = fan_out.expect("Xavier init requires fan_out — use init_with()");
    (2.0 / (fi + fo) as f64).sqrt()
}

// ============================================================================================ //
//                        Meta-init guard (deferred weight allocation)
// ============================================================================================ //

thread_local! {
    static META_INIT: Cell<bool> = Cell::new(false);
}

fn is_meta_init() -> bool {
    META_INIT.with(|c| c.get())
}

fn set_meta_init(enabled: bool) {
    META_INIT.with(|c| c.set(enabled));
}

/// RAII guard that enables *meta-init* mode.
///
/// While active, calls to [`Init::apply`] return meta tensors instead of
/// allocating real storage.  This lets you construct a module skeleton
/// (e.g. when loading weights from a file) without the cost of allocating
/// then immediately overwriting the parameter buffers.
///
/// ```ignore
/// let _guard = MetaInitGuard::new();
/// let mut model = Linear::init_default(&config)?;
/// model.load_safetensors("weights.safetensors", &device, true)?;
/// ```
///
/// TODO: currently returns zeros instead of true meta tensors because
/// `Tensor::meta(shape)` is not yet a public constructor.
pub struct MetaInitGuard;

impl MetaInitGuard {
    pub fn new() -> Self {
        set_meta_init(true);
        Self
    }
}

impl Drop for MetaInitGuard {
    fn drop(&mut self) {
        set_meta_init(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use luma_tensor::Cpu;

    #[test]
    fn test_init_zeros() {
        let t = Init::zeros().init((2, 3), &Cpu).unwrap();
        assert_eq!(t.dims(), &[2, 3]);
        assert!(t.to_vec().unwrap().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_init_ones() {
        let t = Init::ones().init((2, 2), &Cpu).unwrap();
        assert!(t.to_vec().unwrap().iter().all(|&x| (x - 1.0).abs() < 1e-5));
    }

    #[test]
    fn test_init_constant() {
        let t = Init::constant(3.14).init((1,), &Cpu).unwrap();
        assert!((t.to_vec().unwrap()[0] - 3.14).abs() < 1e-5);
    }

    #[test]
    fn test_init_normal() {
        let t = Init::normal(0.0, 1.0).init((100,), &Cpu).unwrap();
        assert_eq!(t.dims(), &[100]);
        // values should not all be equal
        let v = t.to_vec().unwrap();
        assert!(v.iter().any(|&x| (x - v[0]).abs() > 1e-9), "normal init should produce variation");
    }

    #[test]
    fn test_init_uniform() {
        let t = Init::uniform(-1.0, 1.0).init((50,), &Cpu).unwrap();
        let v = t.to_vec().unwrap();
        assert!(v.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_init_kaiming_uniform() {
        let t = Init::kaiming_uniform(NonLinearity::Relu, false).init_with((64, 128), 64, 128, &Cpu).unwrap();
        assert_eq!(t.dims(), &[64, 128]);
    }

    #[test]
    fn test_init_kaiming_uniform_with_gain() {
        let t = Init::kaiming_uniform_with_gain(2.0, true).init_with((32, 64), 32, 64, &Cpu).unwrap();
        assert_eq!(t.dims(), &[32, 64]);
    }

    #[test]
    fn test_calculate_gain() {
        assert!((NonLinearity::Relu.calculate_gain() - 2.0f64.sqrt()).abs() < 1e-10);
        assert!((NonLinearity::Tanh.calculate_gain() - 5.0 / 3.0).abs() < 1e-10);
        assert!((NonLinearity::Linear.calculate_gain() - 1.0).abs() < 1e-10);
        assert!((NonLinearity::LeakyRelu { negative_slope: 0.01 }.calculate_gain() - (2.0 / 1.0001f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_init_xavier_uniform() {
        let t = Init::xavier_uniform(1.0).init_with((32, 64), 32, 64, &Cpu).unwrap();
        assert_eq!(t.dims(), &[32, 64]);
    }

    #[test]
    fn test_init_param() {
        let p = Init::ones().init_param((4,), &Cpu).unwrap();
        assert!(p.requires_grad());
        assert_eq!(p.0.dims(), &[4]);
    }

    #[test]
    fn test_init_buffer() {
        let b = Init::zeros().init_buffer((3, 3), &Cpu).unwrap();
        let v = b.0.to_vec().unwrap();
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_meta_init_guard() {
        // Guard enables meta mode; currently falls back to zeros.
        let _guard = MetaInitGuard::new();
        let t = Init::ones().init((2,), &Cpu).unwrap();
        // TODO: once Tensor::meta() is public, this should be is_meta() == true
        assert!(!t.is_meta());
    }
}
