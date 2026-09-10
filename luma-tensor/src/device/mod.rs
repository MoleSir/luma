pub mod bool_ops;
pub mod float_ops;
pub mod int_ops;
pub use bool_ops::BoolOps;
pub use float_ops::FloatOps;
pub use int_ops::IntOps;
use crate::{Bool, Float, Int, dtype::Storage};

pub trait Device: 'static + Clone + Send + Sync + Default + FloatOps<Self> + IntOps<Self> + BoolOps<Self> {
    type FloatStorage: Storage<Self, Float>;
    type IntStorage: Storage<Self, Int>;
    type BoolStorage: Storage<Self, Bool>;

    fn name(&self) -> String;

    /// Whether two handles refer to the same underlying device.
    ///
    /// Defaults to `true` — correct for stateless devices that are value
    /// types. Devices with distinct instances (e.g. a GPU backend with
    /// multiple ordinals) must override this.
    fn same_device(&self, _other: &Self) -> bool {
        true
    }
}
