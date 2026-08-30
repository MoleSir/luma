//! luma-jit's unified error type, mirroring `luma_nn::error`:
//! `#[thiserrorctx::context_error]` generates the `JitResult<T>` alias and
//! `.context()` helpers, `#[error(transparent)] + #[from]` passes underlying
//! tensor/nn errors through, and domain errors are grouped into per-category
//! sub-enums ([`TraceError`], [`ExecuteError`], [`VerifyError`]).

use crate::graph::Scalar;
use crate::opt::verify::VerifyError;
use luma_tensor::{DType, KindTag, Shape};

#[thiserrorctx::context_error]
pub enum JitError {
    /// Tensor-layer errors (kernels, shapes, dtypes) — shared by tracing,
    /// executing, and the optimization passes.
    #[error(transparent)]
    Tensor(#[from] luma_tensor::Error),

    /// Errors from a traced module's `forward` (luma-nn).
    #[error(transparent)]
    Nn(#[from] luma_nn::NnError),

    #[error(transparent)]
    Trace(#[from] TraceError),

    #[error(transparent)]
    Execute(#[from] ExecuteError),

    #[error(transparent)]
    Verify(#[from] VerifyError),
}

/// Errors specific to tracing (symbolic execution) — the [`Trace`](crate::Trace)
/// device never stores data, so read-back and in-place ops are rejected.
#[derive(thiserror::Error, Debug)]
pub enum TraceError {
    #[error("trace: cannot materialize '{0}' (no data is stored during tracing)")]
    ReadbackUnsupported(&'static str),

    #[error("trace: in-place op '{0}' has no meaning in a functional graph")]
    InplaceUnsupported(&'static str),
}

/// `TraceError` surfaces through the `FloatOps`/`IntOps`/`BoolOps` seams, which
/// are trait-bound to return `luma_tensor::Result`.
impl From<TraceError> for luma_tensor::Error {
    fn from(e: TraceError) -> Self {
        luma_tensor::Error::Msg(e.to_string())
    }
}

/// Errors specific to executing a compiled graph on a concrete device.
#[derive(thiserror::Error, Debug)]
pub enum ExecuteError {
    // ---- user-triggerable: `GraphExecutor::run` input validation ----
    #[error("executor: expected {expected} inputs, got {got}")]
    InputCountMismatch { expected: usize, got: usize },

    #[error("executor: input {idx} mismatch — graph expects {expected_dtype:?} {expected_shape:?}, got {got_dtype:?} {got_shape:?}")]
    InputMismatch { idx: usize, expected_dtype: DType, expected_shape: Shape, got_dtype: DType, got_shape: Shape },

    // ---- internal invariants (verify() should have caught these) ----
    #[error("executor: {op} expects more than {got} inputs")]
    ExpectMoreInputs { op: String, got: usize },

    #[error("executor: unknown value %{0}")]
    UnknownValue(usize),

    #[error("executor: dangling value %{0} — no constant, input, or producing node")]
    DanglingValue(usize),

    #[error("executor: node with no output: {0}")]
    NodeWithoutOutput(String),

    #[error("executor: unexpected Constant node (constants are data-carrying leaves)")]
    UnexpectedConstantNode,

    #[error("executor: {0}")]
    UnsupportedOp(String),

    #[error("executor: {what}: expected {expected:?}, got {got:?}")]
    KindMismatch { what: &'static str, expected: KindTag, got: KindTag },

    #[error("executor: scalar {scalar} does not match operand kind {kind:?}")]
    ScalarKindMismatch { scalar: Scalar, kind: KindTag },

    #[error("executor: {op} needs two operands of the same numeric kind, got {a:?}/{b:?}")]
    PairKindMismatch { op: String, a: KindTag, b: KindTag },

    #[error("executor: Cat inputs must share one kind, got {expected:?}/{got:?}")]
    CatKindMismatch { expected: KindTag, got: KindTag },

    #[error("executor: Pick branches must share one kind, got {a:?}/{b:?}")]
    BranchKindMismatch { a: KindTag, b: KindTag },

    #[error("executor: scalar Pick is not yet supported — the IR does not record the scalar operand")]
    PickScalarUnsupported,
}
