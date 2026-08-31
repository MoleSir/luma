//! A device/kind-erased computation graph IR produced by [`Trace`](crate::Trace).
//!
//! Unlike [`luma_tensor::Op`], this graph is:
//! - **device-erased**: no `D: Device` generic anywhere, so one graph can be
//!   lowered to any backend or serialized to ONNX/TorchScript;
//! - **kind-erased**: values carry a runtime [`DType`] instead of a compile-time
//!   `Float`/`Int`/`Bool` marker;
//! - **SSA**: every value is produced by exactly one node (leaves have none).

use std::fmt;

use luma_tensor::{BinaryOp, CmpOp, DType, FloatUnaryOp, ReduceOp, Shape, UnaryOp};

pub type ValueId = usize;

/// Raw little-endian bytes of a constant leaf (canonical logical order, the
/// form produced by `Tensor::to_bytes` and consumed by `Tensor::from_bytes`).
///
/// `Debug` prints only the length — the payload can be a whole weight tensor.
#[derive(Clone, PartialEq)]
pub struct ConstData(pub Vec<u8>);

impl fmt::Debug for ConstData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ConstData(len={})", self.0.len())
    }
}

#[derive(Debug, Clone)]
pub struct Value {
    pub id: ValueId,
    pub dtype: DType,
    pub shape: Shape,
    /// `Some` on constant leaves; `None` on graph inputs and op outputs.
    pub data: Option<ConstData>,
}

/// A scalar attribute carried by an op (float / int / bool), type-erased for the IR.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    F64(f64),
    I64(i64),
    Bool(bool),
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::F64(v) => write!(f, "{v}"),
            Scalar::I64(v) => write!(f, "{v}"),
            Scalar::Bool(v) => write!(f, "{v}"),
        }
    }
}

impl Scalar {
    /// `0` in either numeric form (float or int).
    pub fn is_zero(&self) -> bool {
        matches!(self, Scalar::F64(0.0) | Scalar::I64(0))
    }

    /// `1` in either numeric form (float or int).
    pub fn is_one(&self) -> bool {
        matches!(self, Scalar::F64(1.0) | Scalar::I64(1))
    }
}

#[derive(Debug, Clone)]
pub enum NodeOp {
    // ---- leaves (constant / graph input) ----
    Constant,

    // ---- elementwise / arithmetic ----
    Binary(BinaryOp),
    BinaryScalarRhs(Scalar, BinaryOp),
    BinaryScalarLhs(Scalar, BinaryOp),
    Unary(UnaryOp<f64>),
    UnaryI(UnaryOp<i64>),
    FloatUnary(FloatUnaryOp),
    Cmp(CmpOp),
    CmpScalar(Scalar, CmpOp),
    Cast(DType),

    // ---- logical (bool) ----
    And,
    Or,
    Xor,
    Not,

    // ---- reductions / matrix ----
    Reduce(ReduceOp, Vec<usize>),
    ReduceAll(Vec<usize>),
    ReduceAny(Vec<usize>),
    ArgReduce(usize, bool),
    Matmul,

    // ---- indexing ----
    IndexSelect(usize),
    Gather(usize),
    IndexAdd(usize),
    ScatterAdd(usize),

    // ---- shape / nn ----
    Cat(usize),
    Softmax(usize),
    RmsNorm(f64),
    Pick,
    PickTrue(Scalar),
    PickFalse(Scalar),
    Arange(i64, i64, i64),

    // ---- views ----
    Reshape,
    Transpose(usize, usize),
    Permute(Vec<usize>),
    Narrow(usize, usize, usize),
    Slice(usize, usize, usize, usize),
    Broadcast,
    Squeeze(usize),
    Unsqueeze(usize),
}

#[derive(Debug, Clone)]
pub struct Node {
    pub op: NodeOp,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
}

#[derive(Debug, Clone, Default)]
pub struct Graph {
    pub values: Vec<Value>,
    pub nodes: Vec<Node>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
}

impl Graph {
    /// Add a leaf value (no producing node). Ids are dense and never reused.
    ///
    /// Data-less leaves are *graph inputs* (or floating placeholders); see
    /// [`Graph::add_constant`] for leaves that carry data.
    pub fn add_value(&mut self, dtype: DType, shape: Shape) -> ValueId {
        let id = self.values.len();
        self.values.push(Value { id, dtype, shape, data: None });
        id
    }

    /// Add a constant leaf carrying raw little-endian data (as produced by
    /// `Tensor::to_bytes` / consumed by `Tensor::from_bytes`). Constants are
    /// leaves with no producing node, distinguished from graph inputs by
    /// their `data` payload.
    pub fn add_constant(&mut self, dtype: DType, shape: Shape, data: Vec<u8>) -> ValueId {
        let id = self.values.len();
        self.values.push(Value { id, dtype, shape, data: Some(ConstData(data)) });
        id
    }

    /// Register a leaf value as a graph input.
    pub fn mark_input(&mut self, id: ValueId) {
        self.inputs.push(id);
    }

    /// Register a value as a graph output.
    pub fn mark_output(&mut self, id: ValueId) {
        self.outputs.push(id);
    }

    /// Add a node with a single output value; returns the output value id.
    pub fn add_node(&mut self, op: NodeOp, inputs: Vec<ValueId>, dtype: DType, shape: Shape) -> ValueId {
        let out = self.add_value(dtype, shape);
        self.nodes.push(Node { op, inputs, outputs: vec![out] });
        out
    }

    /// Insert a node at position `idx` (its output value id is still appended
    /// at the end). Rewriting rules that *create* nodes must place them before
    /// their consumers — the executor runs nodes in array order, so appending
    /// would make a mid-graph consumer read an empty slot.
    pub fn insert_node(&mut self, idx: usize, op: NodeOp, inputs: Vec<ValueId>, dtype: DType, shape: Shape) -> ValueId {
        let out = self.add_value(dtype, shape);
        self.nodes.insert(idx, Node { op, inputs, outputs: vec![out] });
        out
    }

    pub fn value(&self, id: ValueId) -> &Value {
        &self.values[id]
    }

    /// Run the front-end optimization pipeline in place
    /// (simplify → fold → simplify → cse → dce → verify).
    pub fn optimize(&mut self) -> crate::JitResult<()> {
        let g = std::mem::take(self);
        *self = crate::frontend::opt::optimize(g)?;
        Ok(())
    }
}

impl fmt::Display for NodeOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeOp::Constant => write!(f, "const"),
            NodeOp::Binary(op) => write!(f, "{op:?}"),
            NodeOp::BinaryScalarRhs(c, op) => write!(f, "{op:?}(scalar {c})"),
            NodeOp::BinaryScalarLhs(c, op) => write!(f, "{op:?}({c}, scalar)"),
            NodeOp::Unary(op) => write!(f, "{op:?}"),
            NodeOp::UnaryI(op) => write!(f, "{op:?}"),
            NodeOp::FloatUnary(op) => write!(f, "{op:?}"),
            NodeOp::Cmp(op) => write!(f, "{op:?}"),
            NodeOp::CmpScalar(c, op) => write!(f, "{op:?}(scalar {c})"),
            NodeOp::Cast(dt) => write!(f, "cast({dt:?})"),
            NodeOp::And => write!(f, "and"),
            NodeOp::Or => write!(f, "or"),
            NodeOp::Xor => write!(f, "xor"),
            NodeOp::Not => write!(f, "not"),
            NodeOp::Reduce(op, dims) => write!(f, "{op:?}{dims:?}"),
            NodeOp::ReduceAll(dims) => write!(f, "all{dims:?}"),
            NodeOp::ReduceAny(dims) => write!(f, "any{dims:?}"),
            NodeOp::ArgReduce(d, take_max) => write!(f, "arg{}max({d})", if *take_max { "" } else { "min " }),
            NodeOp::Matmul => write!(f, "matmul"),
            NodeOp::IndexSelect(d) => write!(f, "index_select({d})"),
            NodeOp::Gather(d) => write!(f, "gather({d})"),
            NodeOp::IndexAdd(d) => write!(f, "index_add({d})"),
            NodeOp::ScatterAdd(d) => write!(f, "scatter_add({d})"),
            NodeOp::Cat(d) => write!(f, "cat({d})"),
            NodeOp::Softmax(d) => write!(f, "softmax({d})"),
            NodeOp::RmsNorm(eps) => write!(f, "rms_norm({eps})"),
            NodeOp::Pick => write!(f, "pick"),
            NodeOp::PickTrue(v) => write!(f, "pick_true({v})"),
            NodeOp::PickFalse(v) => write!(f, "pick_false({v})"),
            NodeOp::Arange(s, e, st) => write!(f, "arange({s}, {e}, {st})"),
            NodeOp::Reshape => write!(f, "reshape"),
            NodeOp::Transpose(d1, d2) => write!(f, "transpose({d1}, {d2})"),
            NodeOp::Permute(dims) => write!(f, "permute({dims:?})"),
            NodeOp::Narrow(d, s, l) => write!(f, "narrow({d}, {s}, {l})"),
            NodeOp::Slice(d, s, e, st) => write!(f, "slice({d}, {s}..{e}:{st})"),
            NodeOp::Broadcast => write!(f, "broadcast"),
            NodeOp::Squeeze(d) => write!(f, "squeeze({d})"),
            NodeOp::Unsqueeze(d) => write!(f, "unsqueeze({d})"),
        }
    }
}

fn ids(ids: &[ValueId]) -> String {
    ids.iter().map(|&i| format!("%{i}")).collect::<Vec<_>>().join(", ")
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "inputs:  {}", ids(&self.inputs))?;
        for node in &self.nodes {
            let out = node.outputs.first().copied().unwrap_or(0);
            let v = &self.values[out];
            writeln!(f, "  %{out:<2} = {:<20} ({})  {:?}  {}", node.op, ids(&node.inputs), v.dtype, v.shape)?;
        }
        writeln!(f, "outputs: {}", ids(&self.outputs))
    }
}
