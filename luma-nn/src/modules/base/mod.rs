mod buffer;
mod containers;
mod forward;
mod module;
mod param;
mod to_device;
mod visitor;

pub use buffer::*;
pub use forward::*;
pub use module::*;
pub use param::*;
pub use to_device::*;
pub use visitor::*;

// ============================================================================================ //
//                        Tests
// ============================================================================================ //

#[cfg(test)]
mod tests {
    use super::*;
    use luma_macros::Module;
    use luma_tensor::{Cpu, Device, Float, Int, Tensor};
    use std::marker::PhantomData;

    /// A minimal module that exercises the derive macro on a struct.
    #[derive(Module)]
    struct Linear<D: Device> {
        weight: Parameter<D>,
        bias: Option<Parameter<D>>,
    }

    /// An activation enum exercising the derive macro on an enum.
    #[derive(Module)]
    enum Activation<D: Device> {
        Relu(ReluModule<D>),
    }

    #[derive(Module)]
    struct ReluModule<D: Device> {
        #[module(skip)]
        _marker: PhantomData<D>,
    }

    #[test]
    fn test_linear_display() {
        let w = Parameter::new(Tensor::zeros(&[2, 3], Cpu::default()).unwrap());
        let b = Parameter::new(Tensor::zeros(&[2], Cpu::default()).unwrap());
        let linear = Linear { weight: w, bias: Some(b) };

        let s = format!("{}", linear.display());
        assert!(s.contains("Linear"), "display should contain module name, got: '{s}'");
    }

    #[test]
    fn test_linear_fields() {
        let w = Parameter::new(Tensor::zeros(&[2, 3], Cpu::default()).unwrap());
        let b = Parameter::new(Tensor::zeros(&[2], Cpu::default()).unwrap());
        let linear = Linear { weight: w, bias: Some(b) };

        assert!(linear.weight.requires_grad());
        assert!(linear.bias.as_ref().unwrap().requires_grad());
    }

    #[test]
    fn test_activation_enum_display() {
        let relu = ReluModule { _marker: PhantomData::<Cpu> };
        let act = Activation::Relu(relu);

        let s = format!("{}", act.display());
        assert!(s.contains("Activation"), "display should contain enum name");
        assert!(s.contains("Relu"), "display should contain variant name");
    }

    #[test]
    fn test_module_skip_attribute() {
        let relu = ReluModule { _marker: PhantomData::<Cpu> };
        let act = Activation::Relu(relu);

        let s = format!("{}", act.display());
        assert!(!s.contains("_marker"), "skipped field should not appear");
    }

    /// Test `#[module(display = "...")]` custom display override.
    #[derive(Module)]
    #[module(display = "my_extra")]
    struct CustomDisplay<D: Device> {
        #[module(skip)]
        value: i32,
        _device: PhantomData<D>,
    }

    impl<D: Device> CustomDisplay<D> {
        fn my_extra(&self) -> String {
            format!("value={}", self.value)
        }
    }

    #[test]
    fn test_custom_display_override() {
        let m = CustomDisplay::<Cpu> { value: 42, _device: PhantomData };
        let s = format!("{}", m.display());
        assert!(s.contains("CustomDisplay"), "should contain module name");
        assert!(s.contains("value=42"), "should contain custom display info");
    }

    /// Buffer with an explicit kind param (Int) should still work with the
    /// derive macro.
    #[derive(Module)]
    struct BatchNormLike<D: Device> {
        running_mean: Buffer<D>,
        num_batches: Buffer<D, Int>,
    }

    #[test]
    fn test_buffer_with_int_kind() {
        let mean = Buffer::<Cpu, Float>::new(Tensor::zeros(&[4], Cpu::default()).unwrap());
        let count = Buffer::<Cpu, Int>::new(Tensor::<Cpu, Int>::zeros(&[1], Cpu::default()).unwrap());
        let bn = BatchNormLike { running_mean: mean, num_batches: count };

        let s = format!("{}", bn.display());
        assert!(s.contains("BatchNormLike"), "display should work with int buffers, got: '{s}'");
    }

    #[test]
    fn test_state_dict_roundtrip() {
        let cpu = Cpu::default();
        // Create a Linear module with known weights
        let w = Parameter::new(
            Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), luma_tensor::dtype::FloatDType::F32).unwrap(),
        );
        let b = Parameter::new(Tensor::<Cpu>::from_slice(&[0.1, 0.2], (2,), luma_tensor::dtype::FloatDType::F32).unwrap());
        let original = Linear { weight: w, bias: Some(b) };

        // Extract state dict
        let states = original.named_states();
        assert_eq!(states.len(), 2);
        assert!(states.contains_key("weight"));
        assert!(states.contains_key("bias"));

        // Create a fresh Linear with different weights
        let w2 = Parameter::new(Tensor::zeros(&[2, 3], cpu.clone()).unwrap());
        let b2 = Parameter::new(Tensor::zeros(&[2], cpu.clone()).unwrap());
        let mut loaded = Linear { weight: w2, bias: Some(b2) };

        // Load state dict
        loaded.load_state_dict(&states, true).unwrap();

        // Verify
        let w_vals = loaded.weight.0.to_vec().unwrap();
        assert!((w_vals[0] - 1.0).abs() < 1e-5);
        assert!((w_vals[5] - 6.0).abs() < 1e-5);

        let b_vals = loaded.bias.as_ref().unwrap().0.to_vec().unwrap();
        assert!((b_vals[0] - 0.1).abs() < 1e-5);
        assert!((b_vals[1] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_save_load_safetensors_roundtrip() {
        let cpu = Cpu::default();
        // Create Linear with known weights
        let w = Parameter::new(
            Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), luma_tensor::dtype::FloatDType::F32).unwrap(),
        );
        let b = Parameter::new(Tensor::<Cpu>::from_slice(&[0.1, 0.2], (2,), luma_tensor::dtype::FloatDType::F32).unwrap());
        let original = Linear { weight: w, bias: Some(b) };

        // Save to file
        let path = std::env::temp_dir().join("luma_nn_test.safetensors");
        original.save_safetensors(&path).unwrap();

        // Load into fresh module
        let w2 = Parameter::new(Tensor::zeros(&[2, 3], cpu.clone()).unwrap());
        let b2 = Parameter::new(Tensor::zeros(&[2], cpu.clone()).unwrap());
        let mut loaded = Linear { weight: w2, bias: Some(b2) };
        loaded.load_safetensors(&path, &cpu, true).unwrap();

        let _ = std::fs::remove_file(&path);

        // Verify
        let w_vals = loaded.weight.0.to_vec().unwrap();
        assert!((w_vals[0] - 1.0).abs() < 1e-5);
        assert!((w_vals[5] - 6.0).abs() < 1e-5);

        let b_vals = loaded.bias.as_ref().unwrap().0.to_vec().unwrap();
        assert!((b_vals[0] - 0.1).abs() < 1e-5);
    }
}
