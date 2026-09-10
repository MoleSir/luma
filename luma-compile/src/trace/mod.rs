//! A [`Device`] that records operations into a [`Graph`] instead of computing.
//!
//! Because every tensor op in `luma-tensor` is generic over `D: Device`, passing
//! a [`Trace`] as the device makes a whole module's `forward` emit graph nodes
//! without running any kernel. Data-movement ops are captured through the
//! `FloatOps`/`IntOps`/`BoolOps` seams; *view* ops (transpose/reshape/…), which
//! bypass those seams, are captured through the same seams via `f_view`/`i_view`/
//! `b_view`.

mod bool;
mod entry;
mod float;
mod int;
mod storage;

pub use entry::{TraceInput, trace};
pub use storage::{TraceBoolStorage, TraceFloatStorage, TraceIntStorage, TraceValueId};

use std::sync::{Arc, Mutex};

use luma_tensor::dtype::{BoolDType, FloatDType, IntDType};
use luma_tensor::{DType, DTypeKind, Device, Layout, Result, Shape, Tensor, ViewOp};

use crate::graph::{Graph, NodeOp, ValueId};

/// A tracing device. Cheap to clone; every clone shares the same underlying graph.
#[derive(Clone, Default)]
pub struct Trace {
    graph: Arc<Mutex<Graph>>,
}

impl Trace {
    pub fn new() -> Self {
        Self::default()
    }

    /// Access the recorded graph.
    pub fn graph(&self) -> Arc<Mutex<Graph>> {
        self.graph.clone()
    }

    fn emit(&self, op: NodeOp, inputs: Vec<ValueId>, dtype: DType, shape: Shape) -> Result<ValueId> {
        Ok(self.graph.lock().unwrap().add_node(op, inputs, dtype, shape))
    }

    /// Record a view node for `src` under `dst_l`, returning the new value id.
    fn emit_view(&self, src: ValueId, dst_l: &Layout, view: ViewOp) -> ValueId {
        let mut g = self.graph.lock().unwrap();
        let dtype = g.values[src].dtype;
        g.add_node(map_view(view), vec![src], dtype, dst_l.shape().clone())
    }

    fn float_leaf(&self, dtype: FloatDType, shape: &Shape) -> TraceFloatStorage {
        let id = self.graph.lock().unwrap().add_value(dtype.into(), shape.clone());
        TraceFloatStorage { value: id, dtype, device: self.clone() }
    }

    fn int_leaf(&self, dtype: IntDType, shape: &Shape) -> TraceIntStorage {
        let id = self.graph.lock().unwrap().add_value(dtype.into(), shape.clone());
        TraceIntStorage { value: id, dtype, device: self.clone() }
    }

    fn bool_leaf(&self, dtype: BoolDType, shape: &Shape) -> TraceBoolStorage {
        let id = self.graph.lock().unwrap().add_value(dtype.into(), shape.clone());
        TraceBoolStorage { value: id, dtype, device: self.clone() }
    }

    /// A constant leaf carrying real data (captured by `Tensor::to_device`
    /// through `f_from_bytes`/`i_from_bytes`/`b_from_bytes`).
    fn float_const(&self, dtype: FloatDType, shape: &Shape, data: Vec<u8>) -> TraceFloatStorage {
        let id = self.graph.lock().unwrap().add_constant(dtype.into(), shape.clone(), data);
        TraceFloatStorage { value: id, dtype, device: self.clone() }
    }

    fn int_const(&self, dtype: IntDType, shape: &Shape, data: Vec<u8>) -> TraceIntStorage {
        let id = self.graph.lock().unwrap().add_constant(dtype.into(), shape.clone(), data);
        TraceIntStorage { value: id, dtype, device: self.clone() }
    }

    fn bool_const(&self, dtype: BoolDType, shape: &Shape, data: Vec<u8>) -> TraceBoolStorage {
        let id = self.graph.lock().unwrap().add_constant(dtype.into(), shape.clone(), data);
        TraceBoolStorage { value: id, dtype, device: self.clone() }
    }
}

/// In-place mutation has no meaning in a functional SSA graph.
fn inplace_unsupported<T>(op: &'static str) -> Result<T> {
    Err(crate::error::TraceError::InplaceUnsupported(op).into())
}

/// Read-back needs concrete data, which a tracing device never stores.
fn readback_unsupported<T>(op: &'static str) -> Result<T> {
    Err(crate::error::TraceError::ReadbackUnsupported(op).into())
}

fn map_view(view: ViewOp) -> NodeOp {
    match view {
        ViewOp::Reshape => NodeOp::Reshape,
        ViewOp::Transpose(a, b) => NodeOp::Transpose(a, b),
        ViewOp::Permute(d) => NodeOp::Permute(d),
        ViewOp::Narrow(d, s, l) => NodeOp::Narrow(d, s, l),
        ViewOp::Slice(d, s, e, st) => NodeOp::Slice(d, s, e, st),
        ViewOp::Broadcast => NodeOp::Broadcast,
        ViewOp::Squeeze(d) => NodeOp::Squeeze(d),
        ViewOp::Unsqueeze(d) => NodeOp::Unsqueeze(d),
    }
}

impl Device for Trace {
    type FloatStorage = TraceFloatStorage;
    type IntStorage = TraceIntStorage;
    type BoolStorage = TraceBoolStorage;

    fn name(&self) -> String {
        "trace".to_string()
    }
}

// ---- graph value id accessors (concrete-typed, no `Option`) ----

/// A tensor produced by the [`Trace`] device, exposing its graph value id.
pub trait Traced {
    /// The graph value id this traced tensor refers to.
    fn trace_id(&self) -> usize;
}

impl<K> Traced for Tensor<Trace, K>
where
    K: DTypeKind<Trace>,
    K::Storage: TraceValueId,
{
    fn trace_id(&self) -> usize {
        self.storage().expect("meta tensor has no trace value").read().expect("storage read lock").value_id()
    }
}

// ---- shape helpers shared by the three kind impls ----

fn arange_len(start: i64, end: i64, step: i64) -> usize {
    if step == 0 {
        return 0;
    }
    let n = if step > 0 {
        if end <= start { 0 } else { ((end - start) + step - 1) / step }
    } else {
        let s = -step;
        if end >= start { 0 } else { ((start - end) + s - 1) / s }
    };
    n as usize
}
