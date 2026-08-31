//! `torch.jit.trace`-style entry point: run a module against an example input
//! on the [`Trace`] device and return the recorded [`Graph`].
//!
//! State capture works through `Tensor::to_device`: transferring the module to
//! the trace device walks every field (the derived `ToDevice` impl), and each
//! parameter/buffer lands in [`Trace`]'s `*_from_bytes` seam, which records it
//! as a *constant* leaf carrying its data. The example input is recorded as a
//! data-less *input* leaf. The module's forward then appends nodes for every
//! op. The resulting graph is self-contained (constants included) and can be
//! executed later on any concrete device.

use std::sync::{Arc, Mutex};

use luma_nn::{ModuleForward, ToDevice};
use luma_tensor::dtype::{BoolDType, FloatDType, IntDType};
use luma_tensor::{Bool, DTypeKind, Device, Float, Int, Result, Shape, Tensor};

use super::{Trace, TraceValueId, Traced};
use crate::CompileResult;
use crate::graph::Graph;

/// Kind-level constructor for input placeholders on the trace device.
/// Hand-written per kind so that `K::DType` stays a concrete type
/// (`FloatDType`/…) and each kind can use its own shape-aware factory
/// (`zeros` for `Float`/`Int`, `falses` for `Bool` — the values of an input
/// placeholder are never read).
#[doc(hidden)]
pub trait InputCreate: DTypeKind<Trace> {
    fn create_input(trace: &Trace, shape: Shape, dtype: Self::DType) -> Result<Tensor<Trace, Self>>;
}

impl InputCreate for Float {
    fn create_input(trace: &Trace, shape: Shape, dtype: FloatDType) -> Result<Tensor<Trace, Float>> {
        Tensor::<Trace, Float>::zeros(shape, (trace, dtype))
    }
}

impl InputCreate for Int {
    fn create_input(trace: &Trace, shape: Shape, dtype: IntDType) -> Result<Tensor<Trace, Int>> {
        Tensor::<Trace, Int>::zeros(shape, (trace, dtype))
    }
}

impl InputCreate for Bool {
    fn create_input(trace: &Trace, shape: Shape, dtype: BoolDType) -> Result<Tensor<Trace, Bool>> {
        Tensor::<Trace, Bool>::falses(shape, (trace, dtype))
    }
}

impl Trace {
    /// Create an *input* placeholder: a data-less leaf registered as a graph
    /// input. Its values are never read (the trace device stores no data) —
    /// only the shape and dtype enter the graph.
    pub fn input<K, S: Into<Shape>>(&self, shape: S, dtype: K::DType) -> Result<Tensor<Trace, K>>
    where
        K: InputCreate,
        K::Storage: TraceValueId,
    {
        let t = K::create_input(self, shape.into(), dtype)?;
        self.graph.lock().unwrap().mark_input(t.trace_id());
        Ok(t)
    }
}

/// Types that can be turned into trace-device *input* placeholders.
///
/// Implemented for tensors and tuples of tensors (multi-input modules such as
/// losses take a tuple).
pub trait TraceInput<D: Device>: Sized {
    /// The same value instantiated on the [`Trace`] device.
    type Traced;

    fn to_trace_input(&self, trace: &Trace) -> Result<Self::Traced>;
}

/// Kind-level dispatch for input creation. Hand-written per kind so that
/// `K::DType` stays a concrete type (`FloatDType`/…) on both sides of the
/// `D` → `Trace` crossing — the compiler otherwise cannot equate the two
/// associated types.
trait InputDispatch<D: Device>: DTypeKind<D> + InputCreate {
    fn input_dispatch(t: &Tensor<D, Self>, trace: &Trace) -> Result<Tensor<Trace, Self>>;
}

impl<D: Device> InputDispatch<D> for Float {
    fn input_dispatch(t: &Tensor<D, Float>, trace: &Trace) -> Result<Tensor<Trace, Float>> {
        trace.input(t.shape().clone(), t.dtype())
    }
}

impl<D: Device> InputDispatch<D> for Int {
    fn input_dispatch(t: &Tensor<D, Int>, trace: &Trace) -> Result<Tensor<Trace, Int>> {
        trace.input(t.shape().clone(), t.dtype())
    }
}

impl<D: Device> InputDispatch<D> for Bool {
    fn input_dispatch(t: &Tensor<D, Bool>, trace: &Trace) -> Result<Tensor<Trace, Bool>> {
        trace.input(t.shape().clone(), t.dtype())
    }
}

impl<D: Device, K: InputDispatch<D>> TraceInput<D> for Tensor<D, K> {
    type Traced = Tensor<Trace, K>;

    fn to_trace_input(&self, trace: &Trace) -> Result<Self::Traced> {
        K::input_dispatch(self, trace)
    }
}

impl<D: Device, A: TraceInput<D>, B: TraceInput<D>> TraceInput<D> for (A, B) {
    type Traced = (A::Traced, B::Traced);

    fn to_trace_input(&self, trace: &Trace) -> Result<Self::Traced> {
        Ok((self.0.to_trace_input(trace)?, self.1.to_trace_input(trace)?))
    }
}

/// Trace a module: run it symbolically against an example input and return the
/// recorded [`Graph`].
///
/// The module is transferred to the [`Trace`] device, capturing every
/// parameter and buffer as a graph constant; the example input becomes a graph
/// input; the forward's output is marked as the graph output. The graph is
/// self-contained — no reference to the original module is needed afterwards.
pub fn trace<D, M>(module: &M, example: &M::Input) -> CompileResult<Arc<Mutex<Graph>>>
where
    D: Device,
    M: ModuleForward<D> + ToDevice<Trace>,
    M::Input: TraceInput<D>,
    <M as ToDevice<Trace>>::Output: ModuleForward<Trace, Input = <M::Input as TraceInput<D>>::Traced>,
    <<M as ToDevice<Trace>>::Output as ModuleForward<Trace>>::Output: Traced,
{
    let trace_dev = Trace::new();
    let traced_module = module.to_device(&trace_dev)?;
    let traced_input = example.to_trace_input(&trace_dev)?;
    let traced_output = traced_module.forward(&traced_input)?;
    trace_dev.graph().lock().unwrap().mark_output(traced_output.trace_id());
    Ok(trace_dev.graph())
}
