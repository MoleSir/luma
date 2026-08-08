//! The autograd computation-graph node type [`Op`] and its op-kind enums.
//!
//! Note on scalars: in the old design op scalars were typed `T` (tied to the
//! tensor's element type). Here a `Float` tensor's precision is a runtime
//! [`DType`](crate::DType), so all op scalars are stored as `f64` — the widest
//! float — and cast back to the tensor's actual precision when the op runs or
//! its gradient is computed.

mod boolean;
mod construct;
mod cast;
mod display;
pub mod indexer;
mod matmul;
mod nn;
mod numeric;
mod readback;
mod reduce;
mod shape;

use construct::ConstructDTypeKind;
pub use construct::{DEFAULT_FLOAT, DEFAULT_INT};
use indexer::IndexingDTypeKind;
use matmul::MatmulDTypeKind;
use numeric::NumericDTypeKind;
use reduce::ReduceDTypeKind;
pub use shape::ShapeDTypeKind;

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
    Unary(Tensor<D, Float>, UnaryOp),
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
    // ---- elementwise ops shared by Float and Int ----
    Neg(Tensor<D, Float>),
    Abs(Tensor<D, Float>),
    Sign(Tensor<D, Float>),
    Pow(Tensor<D, Float>, f64),
    Affine(Tensor<D, Float>, f64, f64),
    Clamp(Tensor<D, Float>, Option<f64>, Option<f64>),
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
pub enum UnaryOp {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Le,
    Ge,
    Lt,
    Gt,
}

pub trait FloatDTypeKind<D: Device>: 
    DTypeKind<D> + 
    ConstructDTypeKind<D> + 
    IndexingDTypeKind<D> + 
    MatmulDTypeKind<D> + 
    NumericDTypeKind<D> + 
    ReduceDTypeKind<D> + 
    ShapeDTypeKind<D> 
{
}

impl<D: Device> FloatDTypeKind<D> for Float {}