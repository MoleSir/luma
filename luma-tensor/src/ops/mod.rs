//! The autograd computation-graph node type [`Op`] and its op-kind enums.
//!
//! Note on scalars: in the old design op scalars were typed `T` (tied to the
//! tensor's element type). Here a `Float` tensor's precision is a runtime
//! [`DType`](crate::DType), so all op scalars are stored as `f64` — the widest
//! float — and cast back to the tensor's actual precision when the op runs or
//! its gradient is computed.

mod arith;
mod boolean;
mod cast;
pub mod construct;
mod display;
mod indexer;
mod matmul;
mod nn;
mod numeric;
mod reduce;
mod shape;
mod shape_infer;
mod to;
mod transfer;

use boolean::PickDTypeKind;
use cast::CastDTypeKind;
use construct::{BytesDTypeKind, ConstructDTypeKind};
use indexer::IndexingAddDTypeKind;
use indexer::IndexingDTypeKind;
use matmul::MatmulDTypeKind;
use numeric::NumericDTypeKind;
use reduce::ReduceDTypeKind;
use shape::ShapeDTypeKind;

pub use indexer::{IndexOp, Indexer, Slice};
pub use transfer::TransferDTypeKind;

use crate::{Bool, DTypeKind, Device, Float, Int, Tensor};

/// A node in the (implicit) computation graph: the operation that produced a
/// `Float` tensor, holding `Arc` references to its inputs. Only `Float` tensors
/// record ops — `Int`/`Bool` tensors are never differentiated.
///
/// Inputs are typed by kind: differentiable inputs are `Tensor<D, Float>`,
/// index inputs are `Tensor<D, Int>`, and masks are `Tensor<D, Bool>`.
pub enum Op<D: Device> {
    Binary(Tensor<D, Float>, Tensor<D, Float>, BinaryOp),
    BinaryScalarRhs(Tensor<D, Float>, f64, BinaryOp),
    BinaryScalarLhs(f64, Tensor<D, Float>, BinaryOp),
    FloatUnary(Tensor<D, Float>, FloatUnaryOp),
    Unary(Tensor<D, Float>, UnaryOp<f64>),
    Reduce(Tensor<D, Float>, ReduceOp, Vec<usize>),
    Matmul(Tensor<D, Float>, Tensor<D, Float>),
    Broadcast(Tensor<D, Float>),
    Narrow(Tensor<D, Float>, usize, usize, usize),
    Slice(Tensor<D, Float>, usize, usize, usize, usize),
    IndexSelect(Tensor<D, Float>, Tensor<D, Int>, usize),
    IndexAdd(Tensor<D, Float>, Tensor<D, Int>, Tensor<D, Float>, usize),
    ScatterAdd(Tensor<D, Float>, Tensor<D, Int>, Tensor<D, Float>, usize),
    Gather(Tensor<D, Float>, Tensor<D, Int>, usize),
    Reshape(Tensor<D, Float>),
    Transpose(Tensor<D, Float>, usize, usize),
    Permute(Tensor<D, Float>, Vec<usize>),
    Cat(Vec<Tensor<D, Float>>, usize),
    Pick(Tensor<D, Bool>, Option<Tensor<D, Float>>, Option<Tensor<D, Float>>),
    Copy(Tensor<D, Float>),
    RmsNorm(Tensor<D, Float>, Tensor<D, Float>, f64),
    Softmax(Tensor<D, Float>, usize),
    /// Precision cast within the float kind (e.g. f32 -> f64). Records the input
    /// so the gradient can be cast back — the key capability the old design lacked.
    Cast(Tensor<D, Float>),
}

/// A *view* operation: it shares the source tensor's storage under a new
/// [`Layout`] without touching device kernels.
///
/// Views bypass the `Device` kernel seam entirely (`Tensor::share_storage`), so
/// tracing devices learn about them through [`Device::on_view`](crate::Device::on_view)
/// rather than through `FloatOps`/`IntOps`/`BoolOps`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViewOp {
    Reshape,
    Transpose(usize, usize),
    Permute(Vec<usize>),
    Narrow(usize, usize, usize),
    Slice(usize, usize, usize, usize),
    Broadcast,
    Squeeze(usize),
    Unsqueeze(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Mul,
    Sub,
    Div,
    Maximum,
    Minimum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Min,
    Max,
    Mean,
    Prod,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FloatUnaryOp {
    Exp,
    Ln,
    Sin,
    Cos,
    Tanh,
    Sqr,
    Sqrt,
    Recip,
    Gelu,
    GeluErf,
    Erf,
    Relu,
    LeakyRelu(f64),
    Silu,
    Sigmoid,
    Floor,
    Ceil,
    Round,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp<S> {
    Neg,
    Abs,
    Sign,
    Affine(S, S),
    Pow(S),
    Clamp(Option<S>, Option<S>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Le,
    Ge,
    Lt,
    Gt,
}

pub trait BaseOpsDTypeKind<D: Device>:
    DTypeKind<D>
    + PickDTypeKind<D>
    + CastDTypeKind<D>
    + IndexingDTypeKind<D>
    + ShapeDTypeKind<D>
    + ConstructDTypeKind<D>
    + BytesDTypeKind<D>
    + Sized
{
}

impl<D: Device> BaseOpsDTypeKind<D> for Float {}
impl<D: Device> BaseOpsDTypeKind<D> for Int {}
impl<D: Device> BaseOpsDTypeKind<D> for Bool {}

pub trait NumOpsDTypeKind<D: Device>:
    BaseOpsDTypeKind<D> + IndexingAddDTypeKind<D> + MatmulDTypeKind<D> + NumericDTypeKind<D> + ReduceDTypeKind<D> + Sized
{
}

impl<D: Device> NumOpsDTypeKind<D> for Float {}
impl<D: Device> NumOpsDTypeKind<D> for Int {}
