//! Execute a traced [`Graph`] on a concrete device.
//!
//! [`Graph::compile`] is a one-time, per-device step that:
//! 1. **verifies** the graph's structural invariants ([`verify`](crate::frontend::opt::verify::verify));
//! 2. **infers the kind** (`Float`/`Int`/`Bool`) of every value (constants and
//!    inputs from their dtype, op outputs from their semantics) — rejecting
//!    malformed graphs, e.g. a `Bool` fed into `Matmul`;
//! 3. **materialises constants once** (`DynTensor::from_bytes`) and allocates
//!    one slot per value, partitioned into three typed arrays;
//! 4. **lowers** each node into a type-specialised [`Step`], so [`GraphExecutor::run`]
//!    contains no kind dispatch.
//!
//! `run` validates the inputs against the recorded shapes/dtypes, stores them
//! in the input slots, and executes the steps in order. Nodes are appended in
//! SSA emission order (every input precedes its consumers), so a straight
//! forward pass always finds complete inputs — no reordering needed.

mod infer;
mod lower;
mod ops;
mod step;

use infer::infer_kinds;
use lower::lower_steps;
use ops::{apply_view, binary_result, cmp_result, f_reduce, float_unary_result, i_reduce, unary_result};
use step::Slot;
pub use step::Step;

use luma_tensor::dtype::{BoolDType, DType};
use luma_tensor::{BinaryOp, Bool, CmpOp, Device, DynTensor, Float, Int, KindTag, Shape, Tensor};

use crate::graph::Graph;
use crate::{ExecuteError, JitResult};

// ============================================================================
//    GraphExecutor
// ============================================================================

/// A [`Graph`] compiled for a concrete device `D`: constants materialised
/// once, intermediate values slotted into three per-kind arrays, nodes lowered
/// into a type-specialised step list.
pub struct GraphExecutor<D: Device> {
    device: D,
    floats: Vec<Option<Tensor<D, Float>>>,
    ints: Vec<Option<Tensor<D, Int>>>,
    bools: Vec<Option<Tensor<D, Bool>>>,
    steps: Vec<Step>,
    inputs: Vec<(Slot, DType, Shape)>,
    outputs: Vec<Slot>,
}

impl<D: Device> GraphExecutor<D> {
    /// Compile a graph: kind inference + validation, slot allocation, one-time
    /// constant materialisation, and lowering into typed steps.
    pub fn compile(graph: &Graph, device: &D) -> JitResult<Self> {
        crate::frontend::opt::verify::verify(graph)?;
        let kinds = infer_kinds(graph)?;

        // Allocate one slot per value, dense within its kind array.
        let mut nf = 0usize;
        let mut ni = 0usize;
        let mut nb = 0usize;
        let mut slots: Vec<Slot> = Vec::with_capacity(graph.values.len());
        for v in &graph.values {
            let slot = match kinds[v.id] {
                KindTag::Float => {
                    let s = Slot::F(nf);
                    nf += 1;
                    s
                }
                KindTag::Int => {
                    let s = Slot::I(ni);
                    ni += 1;
                    s
                }
                KindTag::Bool => {
                    let s = Slot::B(nb);
                    nb += 1;
                    s
                }
            };
            slots.push(slot);
        }

        // Materialise constants once (the only from_bytes in the whole pipeline).
        let mut floats = vec![None; nf];
        let mut ints = vec![None; ni];
        let mut bools = vec![None; nb];
        for v in graph.values.iter().filter(|v| v.data.is_some()) {
            let t = DynTensor::from_bytes(&v.data.as_ref().expect("filtered").0, v.dtype, v.shape.clone(), device)?;
            match slots[v.id] {
                Slot::F(i) => floats[i] = Some(t.into_float()?),
                Slot::I(i) => ints[i] = Some(t.into_int()?),
                Slot::B(i) => bools[i] = Some(t.into_bool()?),
            }
        }

        // Input metadata (slot + recorded dtype/shape) for run-time validation.
        let inputs = graph
            .inputs
            .iter()
            .map(|&id| {
                let v = &graph.values[id];
                (slots[id], v.dtype, v.shape.clone())
            })
            .collect();
        let outputs = graph.outputs.iter().map(|&id| slots[id]).collect();
        let steps = lower_steps(graph, &kinds, &slots)?;

        Ok(Self { device: device.clone(), floats, ints, bools, steps, inputs, outputs })
    }

    /// The device this executor runs on.
    pub fn device(&self) -> &D {
        &self.device
    }

    /// Run the graph against concrete inputs (one per recorded graph input,
    /// in order) and return the graph outputs.
    pub fn run(&mut self, inputs: &[DynTensor<D>]) -> JitResult<Vec<DynTensor<D>>> {
        if inputs.len() != self.inputs.len() {
            return Err(ExecuteError::InputCountMismatch { expected: self.inputs.len(), got: inputs.len() }.into());
        }

        for (idx, ((slot, dtype, shape), t)) in self.inputs.iter().zip(inputs).enumerate() {
            if t.dtype() != *dtype || t.dims() != shape.dims() {
                return Err(ExecuteError::InputMismatch {
                    idx,
                    expected_dtype: *dtype,
                    expected_shape: shape.clone(),
                    got_dtype: t.dtype(),
                    got_shape: t.shape().clone(),
                }
                .into());
            }
            match slot {
                Slot::F(i) => self.floats[*i] = Some(t.as_float().expect("dtype checked").clone()),
                Slot::I(i) => self.ints[*i] = Some(t.as_int().expect("dtype checked").clone()),
                Slot::B(i) => self.bools[*i] = Some(t.as_bool().expect("dtype checked").clone()),
            }
        }

        for step in self.steps.clone() {
            self.exec_step(&step)?;
        }

        Ok(self
            .outputs
            .iter()
            .map(|s| match s {
                Slot::F(i) => self.floats[*i].as_ref().expect("output computed").clone().into(),
                Slot::I(i) => self.ints[*i].as_ref().expect("output computed").clone().into(),
                Slot::B(i) => self.bools[*i].as_ref().expect("output computed").clone().into(),
            })
            .collect())
    }

    fn exec_step(&mut self, step: &Step) -> JitResult<()> {
        match step {
            Step::BinaryF(op, a, b, o) => {
                let r = binary_result(self.floats[a.f()].as_ref().unwrap(), self.floats[b.f()].as_ref().unwrap(), *op)?;
                self.floats[o.f()] = Some(r);
            }
            Step::BinaryI(op, a, b, o) => {
                let r = binary_result(self.ints[a.i()].as_ref().unwrap(), self.ints[b.i()].as_ref().unwrap(), *op)?;
                self.ints[o.i()] = Some(r);
            }
            Step::BinaryScalarRhsF(op, s, a, o) => {
                let x = self.floats[a.f()].as_ref().unwrap().clone();
                let r = match op {
                    BinaryOp::Add => x.add_scalar(*s),
                    BinaryOp::Sub => x.sub_scalar(*s),
                    BinaryOp::Mul => x.mul_scalar(*s),
                    BinaryOp::Div => x.div_scalar(*s),
                    BinaryOp::Maximum => x.maximum_scalar(*s),
                    BinaryOp::Minimum => x.minimum_scalar(*s),
                }?;
                self.floats[o.f()] = Some(r);
            }
            Step::BinaryScalarRhsI(op, s, a, o) => {
                let x = self.ints[a.i()].as_ref().unwrap().clone();
                let r = match op {
                    BinaryOp::Add => x.add_scalar(*s),
                    BinaryOp::Sub => x.sub_scalar(*s),
                    BinaryOp::Mul => x.mul_scalar(*s),
                    BinaryOp::Div => x.div_scalar(*s),
                    BinaryOp::Maximum => x.maximum_scalar(*s),
                    BinaryOp::Minimum => x.minimum_scalar(*s),
                }?;
                self.ints[o.i()] = Some(r);
            }
            Step::BinaryScalarLhsF(op, s, a, o) => {
                let x = self.floats[a.f()].as_ref().unwrap().clone();
                let r = match op {
                    // Commutative ops: the scalar side is irrelevant.
                    BinaryOp::Add => x.add_scalar(*s),
                    BinaryOp::Mul => x.mul_scalar(*s),
                    BinaryOp::Maximum => x.maximum_scalar(*s),
                    BinaryOp::Minimum => x.minimum_scalar(*s),
                    BinaryOp::Sub => x.sub_scalar_lhs(*s),
                    BinaryOp::Div => x.div_scalar_lhs(*s),
                }?;
                self.floats[o.f()] = Some(r);
            }
            Step::BinaryScalarLhsI(op, s, a, o) => {
                let x = self.ints[a.i()].as_ref().unwrap().clone();
                let r = match op {
                    BinaryOp::Add => x.add_scalar(*s),
                    BinaryOp::Mul => x.mul_scalar(*s),
                    BinaryOp::Maximum => x.maximum_scalar(*s),
                    BinaryOp::Minimum => x.minimum_scalar(*s),
                    BinaryOp::Sub => x.sub_scalar_lhs(*s),
                    BinaryOp::Div => x.div_scalar_lhs(*s),
                }?;
                self.ints[o.i()] = Some(r);
            }
            Step::UnaryF(op, a, o) => {
                let r = unary_result(self.floats[a.f()].as_ref().unwrap(), op.clone())?;
                self.floats[o.f()] = Some(r);
            }
            Step::UnaryI(op, a, o) => {
                let r = unary_result(self.ints[a.i()].as_ref().unwrap(), op.clone())?;
                self.ints[o.i()] = Some(r);
            }
            Step::FloatUnaryF(op, a, o) => {
                let r = float_unary_result(self.floats[a.f()].as_ref().unwrap(), *op)?;
                self.floats[o.f()] = Some(r);
            }
            Step::CmpF(op, a, b, o) => {
                let r = cmp_result(self.floats[a.f()].as_ref().unwrap(), self.floats[b.f()].as_ref().unwrap(), *op)?;
                self.bools[o.b()] = Some(r);
            }
            Step::CmpI(op, a, b, o) => {
                let r = cmp_result(self.ints[a.i()].as_ref().unwrap(), self.ints[b.i()].as_ref().unwrap(), *op)?;
                self.bools[o.b()] = Some(r);
            }
            Step::CmpScalarF(op, s, a, o) => {
                let x = self.floats[a.f()].as_ref().unwrap().clone();
                let r = match op {
                    CmpOp::Eq => x.eq_scalar(*s),
                    CmpOp::Ne => x.ne_scalar(*s),
                    CmpOp::Le => x.le_scalar(*s),
                    CmpOp::Ge => x.ge_scalar(*s),
                    CmpOp::Lt => x.lt_scalar(*s),
                    CmpOp::Gt => x.gt_scalar(*s),
                }?;
                self.bools[o.b()] = Some(r);
            }
            Step::CmpScalarI(op, s, a, o) => {
                let x = self.ints[a.i()].as_ref().unwrap().clone();
                let r = match op {
                    CmpOp::Eq => x.eq_scalar(*s),
                    CmpOp::Ne => x.ne_scalar(*s),
                    CmpOp::Le => x.le_scalar(*s),
                    CmpOp::Ge => x.ge_scalar(*s),
                    CmpOp::Lt => x.lt_scalar(*s),
                    CmpOp::Gt => x.gt_scalar(*s),
                }?;
                self.bools[o.b()] = Some(r);
            }
            Step::And(a, b, o) => {
                let r = self.bools[a.b()].as_ref().unwrap().and(self.bools[b.b()].as_ref().unwrap())?;
                self.bools[o.b()] = Some(r);
            }
            Step::Or(a, b, o) => {
                let r = self.bools[a.b()].as_ref().unwrap().or(self.bools[b.b()].as_ref().unwrap())?;
                self.bools[o.b()] = Some(r);
            }
            Step::Xor(a, b, o) => {
                let r = self.bools[a.b()].as_ref().unwrap().xor(self.bools[b.b()].as_ref().unwrap())?;
                self.bools[o.b()] = Some(r);
            }
            Step::Not(a, o) => {
                let r = self.bools[a.b()].as_ref().unwrap().not()?;
                self.bools[o.b()] = Some(r);
            }
            Step::CastFromF(dt, a, o) => {
                let t = self.floats[a.f()].as_ref().unwrap().clone();
                match dt.kind() {
                    KindTag::Float => self.floats[o.f()] = Some(t.cast(dt.as_float())?),
                    KindTag::Int => self.ints[o.i()] = Some(t.cast(dt.as_int())?),
                    KindTag::Bool => self.bools[o.b()] = Some(t.cast(BoolDType::Bool)?),
                }
            }
            Step::CastFromI(dt, a, o) => {
                let t = self.ints[a.i()].as_ref().unwrap().clone();
                match dt.kind() {
                    KindTag::Float => self.floats[o.f()] = Some(t.cast(dt.as_float())?),
                    KindTag::Int => self.ints[o.i()] = Some(t.cast(dt.as_int())?),
                    KindTag::Bool => self.bools[o.b()] = Some(t.cast(BoolDType::Bool)?),
                }
            }
            Step::CastFromB(dt, a, o) => {
                let t = self.bools[a.b()].as_ref().unwrap().clone();
                match dt.kind() {
                    KindTag::Float => self.floats[o.f()] = Some(t.cast(dt.as_float())?),
                    KindTag::Int => self.ints[o.i()] = Some(t.cast(dt.as_int())?),
                    KindTag::Bool => self.bools[o.b()] = Some(t.cast(BoolDType::Bool)?),
                }
            }
            Step::ReduceF(op, dims, keepdim, a, o) => {
                let r = f_reduce(self.floats[a.f()].as_ref().unwrap(), *op, dims, *keepdim)?;
                self.floats[o.f()] = Some(r);
            }
            Step::ReduceI(op, dims, keepdim, a, o) => {
                let r = i_reduce(self.ints[a.i()].as_ref().unwrap(), *op, dims, *keepdim)?;
                self.ints[o.i()] = Some(r);
            }
            Step::ArgReduceF(dim, take_max, keepdim, a, o) => {
                let t = self.floats[a.f()].as_ref().unwrap().clone();
                let r = if *take_max {
                    if *keepdim { t.argmax_keepdim(*dim)? } else { t.argmax(*dim)? }
                } else if *keepdim {
                    t.argmin_keepdim(*dim)?
                } else {
                    t.argmin(*dim)?
                };
                self.ints[o.i()] = Some(r);
            }
            Step::ArgReduceI(dim, take_max, keepdim, a, o) => {
                let t = self.ints[a.i()].as_ref().unwrap().clone();
                let r = if *take_max {
                    if *keepdim { t.argmax_keepdim(*dim)? } else { t.argmax(*dim)? }
                } else if *keepdim {
                    t.argmin_keepdim(*dim)?
                } else {
                    t.argmin(*dim)?
                };
                self.ints[o.i()] = Some(r);
            }
            Step::MatmulF(a, b, o) => {
                let r = self.floats[a.f()].as_ref().unwrap().matmul(self.floats[b.f()].as_ref().unwrap())?;
                self.floats[o.f()] = Some(r);
            }
            Step::MatmulI(a, b, o) => {
                let r = self.ints[a.i()].as_ref().unwrap().matmul(self.ints[b.i()].as_ref().unwrap())?;
                self.ints[o.i()] = Some(r);
            }
            Step::IndexSelectF(dim, a, b, o) => {
                let r = self.floats[a.f()].as_ref().unwrap().index_select(self.ints[b.i()].as_ref().unwrap(), *dim)?;
                self.floats[o.f()] = Some(r);
            }
            Step::IndexSelectI(dim, a, b, o) => {
                let r = self.ints[a.i()].as_ref().unwrap().index_select(self.ints[b.i()].as_ref().unwrap(), *dim)?;
                self.ints[o.i()] = Some(r);
            }
            Step::GatherF(dim, a, b, o) => {
                let r = self.floats[a.f()].as_ref().unwrap().gather(self.ints[b.i()].as_ref().unwrap(), *dim)?;
                self.floats[o.f()] = Some(r);
            }
            Step::GatherI(dim, a, b, o) => {
                let r = self.ints[a.i()].as_ref().unwrap().gather(self.ints[b.i()].as_ref().unwrap(), *dim)?;
                self.ints[o.i()] = Some(r);
            }
            Step::IndexAddF(dim, init, idx, src, o) => {
                let r = self.floats[init.f()].as_ref().unwrap().index_add(
                    self.ints[idx.i()].as_ref().unwrap(),
                    self.floats[src.f()].as_ref().unwrap(),
                    *dim,
                )?;
                self.floats[o.f()] = Some(r);
            }
            Step::IndexAddI(dim, init, idx, src, o) => {
                let r = self.ints[init.i()].as_ref().unwrap().index_add(
                    self.ints[idx.i()].as_ref().unwrap(),
                    self.ints[src.i()].as_ref().unwrap(),
                    *dim,
                )?;
                self.ints[o.i()] = Some(r);
            }
            Step::ScatterAddF(dim, init, idx, src, o) => {
                let r = self.floats[init.f()].as_ref().unwrap().scatter_add(
                    self.ints[idx.i()].as_ref().unwrap(),
                    self.floats[src.f()].as_ref().unwrap(),
                    *dim,
                )?;
                self.floats[o.f()] = Some(r);
            }
            Step::ScatterAddI(dim, init, idx, src, o) => {
                let r = self.ints[init.i()].as_ref().unwrap().scatter_add(
                    self.ints[idx.i()].as_ref().unwrap(),
                    self.ints[src.i()].as_ref().unwrap(),
                    *dim,
                )?;
                self.ints[o.i()] = Some(r);
            }
            Step::CatF(dim, ins, o) => {
                let arrs: Vec<&Tensor<D, Float>> = ins.iter().map(|s| self.floats[s.f()].as_ref().unwrap()).collect();
                self.floats[o.f()] = Some(Tensor::cat(&arrs, *dim)?);
            }
            Step::CatI(dim, ins, o) => {
                let arrs: Vec<&Tensor<D, Int>> = ins.iter().map(|s| self.ints[s.i()].as_ref().unwrap()).collect();
                self.ints[o.i()] = Some(Tensor::cat(&arrs, *dim)?);
            }
            Step::CatB(dim, ins, o) => {
                let arrs: Vec<&Tensor<D, Bool>> = ins.iter().map(|s| self.bools[s.b()].as_ref().unwrap()).collect();
                self.bools[o.b()] = Some(Tensor::cat(&arrs, *dim)?);
            }
            Step::Softmax(dim, a, o) => {
                let r = self.floats[a.f()].as_ref().unwrap().softmax(*dim)?;
                self.floats[o.f()] = Some(r);
            }
            Step::RmsNorm(eps, a, b, o) => {
                let r = self.floats[a.f()].as_ref().unwrap().rms_norm(self.floats[b.f()].as_ref().unwrap(), *eps)?;
                self.floats[o.f()] = Some(r);
            }
            Step::PickF(m, tv, fv, o) => {
                let r = self.bools[m.b()]
                    .as_ref()
                    .unwrap()
                    .pick(self.floats[tv.f()].as_ref().unwrap(), self.floats[fv.f()].as_ref().unwrap())?;
                self.floats[o.f()] = Some(r);
            }
            Step::PickI(m, tv, fv, o) => {
                let r =
                    self.bools[m.b()].as_ref().unwrap().pick(self.ints[tv.i()].as_ref().unwrap(), self.ints[fv.i()].as_ref().unwrap())?;
                self.ints[o.i()] = Some(r);
            }
            Step::PickB(m, tv, fv, o) => {
                let r =
                    self.bools[m.b()].as_ref().unwrap().pick(self.bools[tv.b()].as_ref().unwrap(), self.bools[fv.b()].as_ref().unwrap())?;
                self.bools[o.b()] = Some(r);
            }
            Step::PickTrueF(v, m, f, o) => {
                let r = self.bools[m.b()].as_ref().unwrap().pick_true(*v, self.floats[f.f()].as_ref().unwrap())?;
                self.floats[o.f()] = Some(r);
            }
            Step::PickTrueI(v, m, f, o) => {
                let r = self.bools[m.b()].as_ref().unwrap().pick_true(*v, self.ints[f.i()].as_ref().unwrap())?;
                self.ints[o.i()] = Some(r);
            }
            Step::PickTrueB(v, m, f, o) => {
                let r = self.bools[m.b()].as_ref().unwrap().pick_true(*v, self.bools[f.b()].as_ref().unwrap())?;
                self.bools[o.b()] = Some(r);
            }
            Step::PickFalseF(v, m, t, o) => {
                let r = self.bools[m.b()].as_ref().unwrap().pick_false(self.floats[t.f()].as_ref().unwrap(), *v)?;
                self.floats[o.f()] = Some(r);
            }
            Step::PickFalseI(v, m, t, o) => {
                let r = self.bools[m.b()].as_ref().unwrap().pick_false(self.ints[t.i()].as_ref().unwrap(), *v)?;
                self.ints[o.i()] = Some(r);
            }
            Step::PickFalseB(v, m, t, o) => {
                let r = self.bools[m.b()].as_ref().unwrap().pick_false(self.bools[t.b()].as_ref().unwrap(), *v)?;
                self.bools[o.b()] = Some(r);
            }
            Step::Arange(start, end, step, o) => {
                let r = Tensor::<D, Int>::arange(*start, *end, *step, self.device.clone())?;
                self.ints[o.i()] = Some(r);
            }
            Step::View(v, a, o) => match (a, o) {
                (Slot::F(ai), Slot::F(oi)) => {
                    let r = apply_view(self.floats[*ai].as_ref().unwrap(), v)?;
                    self.floats[*oi] = Some(r);
                }
                (Slot::I(ai), Slot::I(oi)) => {
                    let r = apply_view(self.ints[*ai].as_ref().unwrap(), v)?;
                    self.ints[*oi] = Some(r);
                }
                (Slot::B(ai), Slot::B(oi)) => {
                    let r = apply_view(self.bools[*ai].as_ref().unwrap(), v)?;
                    self.bools[*oi] = Some(r);
                }
                _ => unreachable!("compile validated kinds"),
            },
        }
        Ok(())
    }
}

impl Graph {
    /// Compile the graph for execution on `device`.
    ///
    /// A one-time, per-device step: kind inference (with validation),
    /// constant materialisation, and lowering into a type-specialised plan.
    /// Run the result with [`GraphExecutor::run`].
    pub fn compile<D: Device>(&self, device: &D) -> JitResult<GraphExecutor<D>> {
        GraphExecutor::compile(self, device)
    }
}
