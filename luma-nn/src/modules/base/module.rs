use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::path::Path;
use std::{any::type_name, convert::Infallible};

use luma_io::lpk::LumaPack;
use luma_tensor::{Device, DynTensor, Float, Tensor};

use super::visitor::{ModuleVisitor, ModuleVisitorMut, TensorVisitor, TensorVisitorMut};
use crate::NnError;

// ============================================================================================ //
//                        Module trait
// ============================================================================================ //

/// The central abstraction for neural-network building blocks.
///
/// Every module exposes its parameters, buffers, persisted state, and
/// sub-modules through a uniform set of visitor methods.  Derive
/// `#[derive(Module)]` (from `luma-macros`) to auto-generate those methods;
/// hand-implement them only for leaf types like [`Parameter`] and [`Buffer`].
pub trait Module<D: Device>: Sized {
    // ================================================================= //
    //                     Tensor visitor methods
    // ================================================================= //

    #[allow(unused_variables)]
    fn visit_param<Visitor: TensorVisitor<D>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn visit_param_mut<Visitor: TensorVisitorMut<D>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn visit_buffer<Visitor: TensorVisitor<D>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn visit_buffer_mut<Visitor: TensorVisitorMut<D>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn visit_state<Visitor: TensorVisitor<D>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn visit_state_mut<Visitor: TensorVisitorMut<D>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    // ================================================================= //
    //                     Module visitor methods
    // ================================================================= //

    #[allow(unused_variables)]
    fn visit_module<Visitor: ModuleVisitor<D>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn visit_module_mut<Visitor: ModuleVisitorMut<D>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    // ================================================================= //
    //                     Convenience methods
    // ================================================================= //

    fn module_name() -> &'static str {
        let full_name = type_name::<Self>();
        // The type name may include generic parameters such as
        // `Linear<luma_tensor::Cpu>`. Strip them before extracting
        // the last path segment.
        let base = match full_name.find('<') {
            Some(i) => &full_name[..i],
            None => full_name,
        };
        base.split("::").last().unwrap_or(base)
    }

    fn display(&self) -> ModuleDisplayer<'_, Self, D> {
        ModuleDisplayer { module: self, indent: 0, _marker: PhantomData }
    }

    fn train(&mut self, mode: bool) {
        let mut visitor = TrainModeVisitor::new(mode);
        self.visit_module_mut(&mut visitor).unwrap();
    }

    fn eval(&mut self) {
        self.train(false);
    }

    // ================================================================= //
    //                     State-dict / serialization
    // ================================================================= //

    /// Returns parameters as an ordered `Vec`, matching field-declaration order.
    ///
    /// This is the preferred method for passing parameters to an optimiser.
    fn params(&self) -> Vec<Tensor<D, Float>> {
        let mut visitor = ParametersVisitor::new();
        self.visit_param(&mut visitor).unwrap();
        visitor.params
    }

    fn named_params(&self) -> HashMap<String, Tensor<D, Float>> {
        let mut visitor = NamedFloatTensorsVisitor::new();
        self.visit_param(&mut visitor).unwrap();
        visitor.map
    }

    fn named_buffers(&self) -> HashMap<String, Tensor<D, Float>> {
        let mut visitor = NamedFloatTensorsVisitor::new();
        self.visit_buffer(&mut visitor).unwrap();
        visitor.map
    }

    /// Collect params + buffers (everything serializable).
    fn named_states(&self) -> HashMap<String, DynTensor<D>> {
        let mut map = self.named_dyn_params();
        map.extend(self.named_dyn_buffers());
        map
    }

    fn named_dyn_params(&self) -> HashMap<String, DynTensor<D>> {
        let mut visitor = NamedDynTensorsVisitor::new();
        self.visit_param(&mut visitor).unwrap();
        visitor.map
    }

    fn named_dyn_buffers(&self) -> HashMap<String, DynTensor<D>> {
        let mut visitor = NamedDynTensorsVisitor::new();
        self.visit_buffer(&mut visitor).unwrap();
        visitor.map
    }

    /// Load parameters and buffers from a state dictionary.
    fn load_state_dict(&mut self, dict: &HashMap<String, DynTensor<D>>, strict: bool) -> Result<(), NnError> {
        let mut visitor = LoadStateDictVisitor::new(dict, strict);
        self.visit_state_mut(&mut visitor)
    }

    fn save_lpk(&self, path: impl AsRef<Path>) -> Result<(), NnError> {
        let states = self.named_states();
        let mut pack = LumaPack::new();
        pack.tensors = states;
        luma_io::lpk::save_file(&pack, path)?;
        Ok(())
    }

    fn load_lpk(&mut self, path: impl AsRef<Path>, device: &D, strict: bool) -> Result<(), NnError> {
        let pack = luma_io::lpk::load_file(path, device)?;
        self.load_state_dict(&pack.tensors, strict)
    }

    /// Save all state to a safetensors file.
    fn save_safetensors(&self, path: impl AsRef<Path>) -> Result<(), NnError> {
        let states = self.named_states();
        luma_io::safetensors::save_file(&states, None, path)?;
        Ok(())
    }

    /// Load all state from a safetensors file onto the given device, then copy into this module.
    fn load_safetensors(&mut self, path: impl AsRef<Path>, device: &D, strict: bool) -> Result<(), NnError> {
        let content = luma_io::safetensors::load_file(path, device)?;
        self.load_state_dict(&content.tensors, strict)
    }

    // ================================================================= //
    //                     Override methods
    // ================================================================= //

    /// Re-initialise parameters using the module's stored init configuration.
    ///
    /// Default is a no-op. Modules that carry parameters override this (or use
    /// `#[module(reset = "fn_name")]` in the derive macro) to fill weights
    /// and biases with fresh random values according to their chosen init
    /// strategy.  This is useful after loading a state dict when you want to
    /// re-randomise the parameters while keeping the architecture identical.
    #[allow(unused_variables)]
    fn reset_parameters(&mut self) -> Result<(), NnError> {
        Ok(())
    }

    fn extra_display(&self) -> String {
        String::new()
    }

    #[allow(unused_variables)]
    fn set_train(&mut self, mode: bool) {}
}

// ============================================================================================ //
//                        Module display infrastructure
// ============================================================================================ //

pub struct ModuleDisplayer<'a, M, D> {
    pub module: &'a M,
    pub indent: usize,
    pub _marker: PhantomData<D>,
}

impl<M, D> fmt::Display for ModuleDisplayer<'_, M, D>
where
    M: Module<D>,
    D: Device,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut visitor = DisplayVisitor { f, indent: self.indent, child_count_stack: vec![], name: String::new() };
        self.module.visit_module(&mut visitor)
    }
}

struct DisplayVisitor<'a, 'b> {
    f: &'a mut fmt::Formatter<'b>,
    indent: usize,
    child_count_stack: Vec<usize>,
    name: String,
}

impl<D: Device> ModuleVisitor<D> for DisplayVisitor<'_, '_> {
    type Error = fmt::Error;

    fn visit_module<M: Module<D>>(&mut self, module: &M) -> Result<(), Self::Error> {
        if !self.name.is_empty() {
            let parent_child_count = self.child_count_stack.last_mut().unwrap();
            *parent_child_count += 1;
            write!(self.f, "\n{:indent$}({}): ", "", self.name, indent = self.indent)?;
        }

        write!(self.f, "{}", M::module_name())?;

        let extra = module.extra_display();
        if !extra.is_empty() {
            write!(self.f, "({}", extra)?;
        } else {
            write!(self.f, "(")?;
        }

        self.child_count_stack.push(0);
        Ok(())
    }

    fn enter_submodule<M: Module<D>>(&mut self, name: &str, _submodule: &M) -> Result<(), Self::Error> {
        self.indent += 2;
        self.name = name.to_string();
        Ok(())
    }

    fn exit_submodule<M: Module<D>>(&mut self, _name: &str, _submodule: &M) -> Result<(), Self::Error> {
        self.indent -= 2;
        Ok(())
    }

    fn visit_module_end<M: Module<D>>(&mut self, _module: &M) -> Result<(), Self::Error> {
        let child_count = self.child_count_stack.pop().unwrap_or(0);
        if child_count > 0 {
            write!(self.f, "\n{:indent$})", "", indent = self.indent)?;
        } else {
            write!(self.f, ")")?;
        }
        Ok(())
    }
}

// ============================================================================================ //
//                        TrainModeVisitor
// ============================================================================================ //

struct TrainModeVisitor {
    mode: bool,
}

impl TrainModeVisitor {
    fn new(mode: bool) -> Self {
        Self { mode }
    }
}

impl<D: Device> ModuleVisitorMut<D> for TrainModeVisitor {
    type Error = Infallible;

    fn visit_module_mut<M: Module<D>>(&mut self, module: &mut M) -> Result<(), Self::Error> {
        module.set_train(self.mode);
        Ok(())
    }
}

// ============================================================================================ //
//                        NamedDynTensorsVisitor — collect state_dict
// ============================================================================================ //

struct NamedDynTensorsVisitor<D: Device> {
    map: HashMap<String, DynTensor<D>>,
    path: Vec<String>,
}

impl<D: Device> NamedDynTensorsVisitor<D> {
    fn new() -> Self {
        Self { map: HashMap::new(), path: vec![] }
    }
}

impl<D: Device> TensorVisitor<D> for NamedDynTensorsVisitor<D> {
    type Error = Infallible;

    fn visit_float(&mut self, tensor: &luma_tensor::Tensor<D, luma_tensor::Float>) -> Result<(), Self::Error> {
        self.map.insert(self.path.join("."), DynTensor::Float(tensor.clone()));
        Ok(())
    }

    fn visit_int(&mut self, tensor: &luma_tensor::Tensor<D, luma_tensor::Int>) -> Result<(), Self::Error> {
        self.map.insert(self.path.join("."), DynTensor::Int(tensor.clone()));
        Ok(())
    }

    fn visit_bool(&mut self, tensor: &luma_tensor::Tensor<D, luma_tensor::Bool>) -> Result<(), Self::Error> {
        self.map.insert(self.path.join("."), DynTensor::Bool(tensor.clone()));
        Ok(())
    }

    fn enter_submodule<M: Module<D>>(&mut self, name: &str, _m: &M) {
        self.path.push(name.to_string());
    }

    fn exit_submodule<M: Module<D>>(&mut self, _name: &str, _m: &M) {
        self.path.pop();
    }
}

// ============================================================================================ //
//                        NamedFloatTensorsVisitor — collect only float tensors
// ============================================================================================ //

struct NamedFloatTensorsVisitor<D: Device> {
    map: HashMap<String, Tensor<D, Float>>,
    path: Vec<String>,
}

impl<D: Device> NamedFloatTensorsVisitor<D> {
    fn new() -> Self {
        Self { map: HashMap::new(), path: vec![] }
    }
}

impl<D: Device> TensorVisitor<D> for NamedFloatTensorsVisitor<D> {
    type Error = Infallible;

    fn visit_float(&mut self, tensor: &Tensor<D, Float>) -> Result<(), Self::Error> {
        self.map.insert(self.path.join("."), tensor.clone());
        Ok(())
    }

    fn visit_int(&mut self, _tensor: &Tensor<D, luma_tensor::Int>) -> Result<(), Self::Error> {
        Ok(())
    }

    fn visit_bool(&mut self, _tensor: &Tensor<D, luma_tensor::Bool>) -> Result<(), Self::Error> {
        Ok(())
    }

    fn enter_submodule<M: Module<D>>(&mut self, name: &str, _m: &M) {
        self.path.push(name.to_string());
    }

    fn exit_submodule<M: Module<D>>(&mut self, _name: &str, _m: &M) {
        self.path.pop();
    }
}

// ============================================================================================ //
//                        ParametersVisitor — collect float params into a Vec
// ============================================================================================ //

struct ParametersVisitor<D: Device> {
    params: Vec<Tensor<D, Float>>,
}

impl<D: Device> ParametersVisitor<D> {
    fn new() -> Self {
        Self { params: vec![] }
    }
}

impl<D: Device> TensorVisitor<D> for ParametersVisitor<D> {
    type Error = Infallible;

    fn visit_float(&mut self, tensor: &Tensor<D, Float>) -> Result<(), Self::Error> {
        self.params.push(tensor.clone());
        Ok(())
    }

    fn visit_int(&mut self, _tensor: &Tensor<D, luma_tensor::Int>) -> Result<(), Self::Error> {
        Ok(())
    }

    fn visit_bool(&mut self, _tensor: &Tensor<D, luma_tensor::Bool>) -> Result<(), Self::Error> {
        Ok(())
    }

    fn enter_submodule<M: Module<D>>(&mut self, _name: &str, _m: &M) {}

    fn exit_submodule<M: Module<D>>(&mut self, _name: &str, _m: &M) {}
}

// ============================================================================================ //
//                        LoadStateDictVisitor — load state_dict into params
// ============================================================================================ //

struct LoadStateDictVisitor<'a, D: Device> {
    params: &'a HashMap<String, DynTensor<D>>,
    path: Vec<String>,
    strict: bool,
}

impl<'a, D: Device> LoadStateDictVisitor<'a, D> {
    fn new(params: &'a HashMap<String, DynTensor<D>>, strict: bool) -> Self {
        Self { params, path: Vec::new(), strict }
    }
}

impl<D: Device> TensorVisitorMut<D> for LoadStateDictVisitor<'_, D> {
    type Error = NnError;

    fn visit_float_mut(&mut self, tensor: &mut Tensor<D, Float>) -> Result<(), Self::Error> {
        let name = self.path.join(".");
        match self.params.get(&name) {
            Some(src) => {
                let src = src.as_float().ok_or_else(|| NnError::ShapeUnmatchWhenLoadParam(tensor.shape().clone(), src.shape().clone()))?;
                if tensor.shape() != src.shape() {
                    return Err(NnError::ShapeUnmatchWhenLoadParam(tensor.shape().clone(), src.shape().clone()));
                }
                let requires_grad = tensor.requires_grad();
                tensor.copy_(src)?;
                tensor.set_requires_grad(requires_grad);
                Ok(())
            }
            None => {
                if self.strict {
                    Err(NnError::ParamNotFound(name, "load_state_dict"))
                } else {
                    Ok(())
                }
            }
        }
    }

    fn visit_int_mut(&mut self, tensor: &mut luma_tensor::Tensor<D, luma_tensor::Int>) -> Result<(), Self::Error> {
        let name = self.path.join(".");
        match self.params.get(&name) {
            Some(src) => {
                let src = src.as_int().ok_or_else(|| NnError::ShapeUnmatchWhenLoadParam(tensor.shape().clone(), src.shape().clone()))?;
                if tensor.shape() != src.shape() {
                    return Err(NnError::ShapeUnmatchWhenLoadParam(tensor.shape().clone(), src.shape().clone()));
                }
                tensor.copy_(src)?;
                Ok(())
            }
            None => {
                if self.strict {
                    Err(NnError::ParamNotFound(name, "load_state_dict"))
                } else {
                    Ok(())
                }
            }
        }
    }

    fn visit_bool_mut(&mut self, tensor: &mut luma_tensor::Tensor<D, luma_tensor::Bool>) -> Result<(), Self::Error> {
        let name = self.path.join(".");
        match self.params.get(&name) {
            Some(src) => {
                let src = src.as_bool().ok_or_else(|| NnError::ShapeUnmatchWhenLoadParam(tensor.shape().clone(), src.shape().clone()))?;
                if tensor.shape() != src.shape() {
                    return Err(NnError::ShapeUnmatchWhenLoadParam(tensor.shape().clone(), src.shape().clone()));
                }
                tensor.copy_(src)?;
                Ok(())
            }
            None => {
                if self.strict {
                    Err(NnError::ParamNotFound(name, "load_state_dict"))
                } else {
                    Ok(())
                }
            }
        }
    }

    fn enter_submodule<M: Module<D>>(&mut self, name: &str, _m: &mut M) {
        self.path.push(name.to_string());
    }

    fn exit_submodule<M: Module<D>>(&mut self, _name: &str, _m: &mut M) {
        self.path.pop();
    }
}
