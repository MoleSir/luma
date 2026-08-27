//! Runtime slots and type-specialised execution steps.

use luma_tensor::dtype::DType;
use luma_tensor::{BinaryOp, CmpOp, FloatUnaryOp, ReduceOp, Shape, UnaryOp};

// ============================================================================
//    Slots and steps
// ============================================================================

/// Runtime reference to a slot: kind + index into the per-kind array.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Slot {
    F(usize),
    I(usize),
    B(usize),
}

impl Slot {
    pub(crate) fn f(&self) -> usize {
        match self {
            Slot::F(i) => *i,
            _ => panic!("slot kind mismatch — executor bug"),
        }
    }
    pub(crate) fn i(&self) -> usize {
        match self {
            Slot::I(i) => *i,
            _ => panic!("slot kind mismatch — executor bug"),
        }
    }
    pub(crate) fn b(&self) -> usize {
        match self {
            Slot::B(i) => *i,
            _ => panic!("slot kind mismatch — executor bug"),
        }
    }
}

/// A view operation with its operands filled in from the recorded graph
/// (views are kind-erased — they share storage and preserve kind).
#[derive(Clone, Debug)]
pub(crate) enum ViewStep {
    Reshape(Shape),
    Transpose(usize, usize),
    Permute(Vec<usize>),
    Narrow(usize, usize, usize),
    Slice(usize, usize, usize, usize),
    Broadcast(Shape),
    Squeeze(usize),
    Unsqueeze(usize),
}

/// A type-specialised execution step produced by lowering a graph node.
/// Every variant already knows the kinds of its operands.
#[derive(Clone, Debug)]
pub(crate) enum Step {
    // ---- elementwise / arithmetic ----
    BinaryF(BinaryOp, Slot, Slot, Slot),
    BinaryI(BinaryOp, Slot, Slot, Slot),
    BinaryScalarRhsF(BinaryOp, f64, Slot, Slot),
    BinaryScalarRhsI(BinaryOp, i64, Slot, Slot),
    BinaryScalarLhsF(BinaryOp, f64, Slot, Slot),
    BinaryScalarLhsI(BinaryOp, i64, Slot, Slot),
    UnaryF(UnaryOp<f64>, Slot, Slot),
    UnaryI(UnaryOp<i64>, Slot, Slot),
    FloatUnaryF(FloatUnaryOp, Slot, Slot),
    CmpF(CmpOp, Slot, Slot, Slot),
    CmpI(CmpOp, Slot, Slot, Slot),
    CmpScalarF(CmpOp, f64, Slot, Slot),
    CmpScalarI(CmpOp, i64, Slot, Slot),
    And(Slot, Slot, Slot),
    Or(Slot, Slot, Slot),
    Xor(Slot, Slot, Slot),
    Not(Slot, Slot),
    CastFromF(DType, Slot, Slot),
    CastFromI(DType, Slot, Slot),
    CastFromB(DType, Slot, Slot),
    // ---- reductions / matrix ----
    ReduceF(ReduceOp, Vec<usize>, bool, Slot, Slot),
    ReduceI(ReduceOp, Vec<usize>, bool, Slot, Slot),
    ArgReduceF(usize, bool, bool, Slot, Slot), // dim, take_max, keepdim
    ArgReduceI(usize, bool, bool, Slot, Slot),
    MatmulF(Slot, Slot, Slot),
    MatmulI(Slot, Slot, Slot),
    // ---- indexing ----
    IndexSelectF(usize, Slot, Slot, Slot),
    IndexSelectI(usize, Slot, Slot, Slot),
    GatherF(usize, Slot, Slot, Slot),
    GatherI(usize, Slot, Slot, Slot),
    IndexAddF(usize, Slot, Slot, Slot, Slot), // init, idx, src
    IndexAddI(usize, Slot, Slot, Slot, Slot),
    ScatterAddF(usize, Slot, Slot, Slot, Slot),
    ScatterAddI(usize, Slot, Slot, Slot, Slot),
    // ---- shape / nn ----
    CatF(usize, Vec<Slot>, Slot),
    CatI(usize, Vec<Slot>, Slot),
    CatB(usize, Vec<Slot>, Slot),
    Softmax(usize, Slot, Slot),
    RmsNorm(f64, Slot, Slot, Slot),
    PickF(Slot, Slot, Slot, Slot), // mask, on_true, on_false
    PickI(Slot, Slot, Slot, Slot),
    PickB(Slot, Slot, Slot, Slot),
    PickTrueF(f64, Slot, Slot, Slot), // value, mask, on_false
    PickTrueI(i64, Slot, Slot, Slot),
    PickTrueB(bool, Slot, Slot, Slot),
    PickFalseF(f64, Slot, Slot, Slot), // value, mask, on_true
    PickFalseI(i64, Slot, Slot, Slot),
    PickFalseB(bool, Slot, Slot, Slot),
    Arange(i64, i64, i64, Slot),
    // ---- views (kind-erased) ----
    View(ViewStep, Slot, Slot),
}
