use luma_tensor::{Bool, Device, Float, Int, Tensor};

use super::module::Module;

// ============================================================================================ //
//                        Tensor visitors (leaf nodes)
// ============================================================================================ //

/// Read-only visitor for tensor leaf nodes (params, buffers, state).
///
/// Every method has a default no-op — override only the kind(s) you care about.
pub trait TensorVisitor<D: Device> {
    type Error;

    #[allow(unused_variables)]
    fn visit_float(&mut self, _tensor: &Tensor<D, Float>) -> Result<(), Self::Error> {
        Ok(())
    }
    #[allow(unused_variables)]
    fn visit_int(&mut self, _tensor: &Tensor<D, Int>) -> Result<(), Self::Error> {
        Ok(())
    }
    #[allow(unused_variables)]
    fn visit_bool(&mut self, _tensor: &Tensor<D, Bool>) -> Result<(), Self::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn enter_submodule<M: Module<D>>(&mut self, _name: &str, _m: &M) {}
    #[allow(unused_variables)]
    fn exit_submodule<M: Module<D>>(&mut self, _name: &str, _m: &M) {}
}

/// Mutable visitor for tensor leaf nodes.
pub trait TensorVisitorMut<D: Device> {
    type Error;

    #[allow(unused_variables)]
    fn visit_float_mut(&mut self, _tensor: &mut Tensor<D, Float>) -> Result<(), Self::Error> {
        Ok(())
    }
    #[allow(unused_variables)]
    fn visit_int_mut(&mut self, _tensor: &mut Tensor<D, Int>) -> Result<(), Self::Error> {
        Ok(())
    }
    #[allow(unused_variables)]
    fn visit_bool_mut(&mut self, _tensor: &mut Tensor<D, Bool>) -> Result<(), Self::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn enter_submodule<M: Module<D>>(&mut self, _name: &str, _m: &mut M) {}
    #[allow(unused_variables)]
    fn exit_submodule<M: Module<D>>(&mut self, _name: &str, _m: &mut M) {}
}

// ============================================================================================ //
//                        Module visitors (tree nodes)
// ============================================================================================ //

/// Read-only visitor for module tree nodes.
pub trait ModuleVisitor<D: Device> {
    type Error;

    #[allow(unused_variables)]
    fn visit_module<M: Module<D>>(&mut self, _module: &M) -> Result<(), Self::Error> {
        Ok(())
    }
    #[allow(unused_variables)]
    fn visit_module_end<M: Module<D>>(&mut self, _module: &M) -> Result<(), Self::Error> {
        Ok(())
    }
    #[allow(unused_variables)]
    fn enter_submodule<M: Module<D>>(&mut self, _name: &str, _submodule: &M) -> Result<(), Self::Error> {
        Ok(())
    }
    #[allow(unused_variables)]
    fn exit_submodule<M: Module<D>>(&mut self, _name: &str, _submodule: &M) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Mutable visitor for module tree nodes.
pub trait ModuleVisitorMut<D: Device> {
    type Error;

    #[allow(unused_variables)]
    fn visit_module_mut<M: Module<D>>(&mut self, _module: &mut M) -> Result<(), Self::Error> {
        Ok(())
    }
    #[allow(unused_variables)]
    fn visit_module_mut_end<M: Module<D>>(&mut self, _module: &mut M) -> Result<(), Self::Error> {
        Ok(())
    }
    #[allow(unused_variables)]
    fn enter_submodule<M: Module<D>>(&mut self, _name: &str, _submodule: &mut M) -> Result<(), Self::Error> {
        Ok(())
    }
    #[allow(unused_variables)]
    fn exit_submodule<M: Module<D>>(&mut self, _name: &str, _submodule: &mut M) -> Result<(), Self::Error> {
        Ok(())
    }
}
