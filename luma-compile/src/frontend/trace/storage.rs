//! Symbolic storages: a tracing tensor stores a graph value id instead of data.

use luma_tensor::dtype::{BoolDType, FloatDType, IntDType};
use luma_tensor::{Bool, Float, Int, Storage};

use super::Trace;
use crate::graph::ValueId;

/// Access to the graph value id behind a traced storage — the kind-erased
/// view used by [`Traced`](super::Traced).
pub trait TraceValueId {
    fn value_id(&self) -> ValueId;
}

#[derive(Clone)]
pub struct TraceFloatStorage {
    pub(crate) value: ValueId,
    pub(crate) dtype: FloatDType,
    pub(crate) device: Trace,
}

impl TraceValueId for TraceFloatStorage {
    fn value_id(&self) -> ValueId {
        self.value
    }
}

impl Storage<Trace, Float> for TraceFloatStorage {
    fn dtype(&self) -> FloatDType {
        self.dtype
    }
    fn device(&self) -> &Trace {
        &self.device
    }
}

#[derive(Clone)]
pub struct TraceIntStorage {
    pub(crate) value: ValueId,
    pub(crate) dtype: IntDType,
    pub(crate) device: Trace,
}

impl TraceValueId for TraceIntStorage {
    fn value_id(&self) -> ValueId {
        self.value
    }
}

impl Storage<Trace, Int> for TraceIntStorage {
    fn dtype(&self) -> IntDType {
        self.dtype
    }
    fn device(&self) -> &Trace {
        &self.device
    }
}

#[derive(Clone)]
pub struct TraceBoolStorage {
    pub(crate) value: ValueId,
    pub(crate) dtype: BoolDType,
    pub(crate) device: Trace,
}

impl TraceValueId for TraceBoolStorage {
    fn value_id(&self) -> ValueId {
        self.value
    }
}

impl Storage<Trace, Bool> for TraceBoolStorage {
    fn dtype(&self) -> BoolDType {
        self.dtype
    }
    fn device(&self) -> &Trace {
        &self.device
    }
}
