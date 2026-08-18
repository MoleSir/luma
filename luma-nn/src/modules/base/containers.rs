use std::marker::PhantomData;

use luma_tensor::Device;

use super::module::Module;
use super::visitor::{ModuleVisitor, ModuleVisitorMut, TensorVisitor, TensorVisitorMut};

// ============================================================================================ //
//                        impl Module for Option<M>
// ============================================================================================ //

impl<D: Device, M: Module<D>> Module<D> for Option<M> {
    fn visit_param<Visitor: TensorVisitor<D>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        if let Some(m) = self {
            m.visit_param(visitor)?;
        }
        Ok(())
    }
    fn visit_param_mut<Visitor: TensorVisitorMut<D>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        if let Some(m) = self {
            m.visit_param_mut(visitor)?;
        }
        Ok(())
    }
    fn visit_buffer<Visitor: TensorVisitor<D>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        if let Some(m) = self {
            m.visit_buffer(visitor)?;
        }
        Ok(())
    }
    fn visit_buffer_mut<Visitor: TensorVisitorMut<D>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        if let Some(m) = self {
            m.visit_buffer_mut(visitor)?;
        }
        Ok(())
    }
    fn visit_state<Visitor: TensorVisitor<D>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        if let Some(m) = self {
            m.visit_state(visitor)?;
        }
        Ok(())
    }
    fn visit_state_mut<Visitor: TensorVisitorMut<D>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        if let Some(m) = self {
            m.visit_state_mut(visitor)?;
        }
        Ok(())
    }
    fn visit_module<Visitor: ModuleVisitor<D>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        if let Some(m) = self {
            m.visit_module(visitor)?;
        }
        Ok(())
    }
    fn visit_module_mut<Visitor: ModuleVisitorMut<D>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        if let Some(m) = self {
            m.visit_module_mut(visitor)?;
        }
        Ok(())
    }
}

// ============================================================================================ //
//                        impl Module for Vec<M>
// ============================================================================================ //

impl<D: Device, M: Module<D>> Module<D> for Vec<M> {
    fn visit_param<Visitor: TensorVisitor<D>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        for (i, m) in self.iter().enumerate() {
            visitor.enter_submodule(&i.to_string(), m);
            m.visit_param(visitor)?;
            visitor.exit_submodule(&i.to_string(), m);
        }
        Ok(())
    }
    fn visit_param_mut<Visitor: TensorVisitorMut<D>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        for (i, m) in self.iter_mut().enumerate() {
            visitor.enter_submodule(&i.to_string(), m);
            m.visit_param_mut(visitor)?;
            visitor.exit_submodule(&i.to_string(), m);
        }
        Ok(())
    }
    fn visit_buffer<Visitor: TensorVisitor<D>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        for (i, m) in self.iter().enumerate() {
            visitor.enter_submodule(&i.to_string(), m);
            m.visit_buffer(visitor)?;
            visitor.exit_submodule(&i.to_string(), m);
        }
        Ok(())
    }
    fn visit_buffer_mut<Visitor: TensorVisitorMut<D>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        for (i, m) in self.iter_mut().enumerate() {
            visitor.enter_submodule(&i.to_string(), m);
            m.visit_buffer_mut(visitor)?;
            visitor.exit_submodule(&i.to_string(), m);
        }
        Ok(())
    }
    fn visit_state<Visitor: TensorVisitor<D>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        for (i, m) in self.iter().enumerate() {
            visitor.enter_submodule(&i.to_string(), m);
            m.visit_state(visitor)?;
            visitor.exit_submodule(&i.to_string(), m);
        }
        Ok(())
    }
    fn visit_state_mut<Visitor: TensorVisitorMut<D>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        for (i, m) in self.iter_mut().enumerate() {
            visitor.enter_submodule(&i.to_string(), m);
            m.visit_state_mut(visitor)?;
            visitor.exit_submodule(&i.to_string(), m);
        }
        Ok(())
    }
    fn visit_module<Visitor: ModuleVisitor<D>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        for (i, m) in self.iter().enumerate() {
            visitor.enter_submodule(&i.to_string(), m)?;
            m.visit_module(visitor)?;
            visitor.exit_submodule(&i.to_string(), m)?;
        }
        Ok(())
    }
    fn visit_module_mut<Visitor: ModuleVisitorMut<D>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        for (i, m) in self.iter_mut().enumerate() {
            visitor.enter_submodule(&i.to_string(), m)?;
            m.visit_module_mut(visitor)?;
            visitor.exit_submodule(&i.to_string(), m)?;
        }
        Ok(())
    }
}

// ============================================================================================ //
//                        impl Module for PhantomData<D>
// ============================================================================================ //

/// `PhantomData<D>` is a no-op module — it carries no tensors or sub-modules.
impl<D: Device> Module<D> for PhantomData<D> {}
