use bytemuck::Zeroable;
use luma_tensor::{BinaryOp, CmpOp, FloatUnaryOp, ReduceOp, UnaryOp};

use crate::Wgpu;

mod bool_ops;
pub(crate) mod float_ops;
mod int_ops;
mod nn_provide;

mod binary_op {
    pub const ADD: u32 = 0;
    pub const SUB: u32 = 1;
    pub const MUL: u32 = 2;
    pub const DIV: u32 = 3;
    pub const MAX: u32 = 4;
    pub const MIN: u32 = 5;
}

pub(crate) fn binary_op_code(op: BinaryOp) -> u32 {
    match op {
        BinaryOp::Add => binary_op::ADD,
        BinaryOp::Sub => binary_op::SUB,
        BinaryOp::Div => binary_op::DIV,
        BinaryOp::Mul => binary_op::MUL,
        BinaryOp::Maximum => binary_op::MAX,
        BinaryOp::Minimum => binary_op::MIN,
    }
}

mod reduce_op {
    pub const SUM: u32 = 0;
    pub const MIN: u32 = 1;
    pub const MAX: u32 = 2;
    pub const MEAN: u32 = 3;
    pub const PROD: u32 = 4;
}

pub(crate) fn reduce_op_code(op: ReduceOp) -> u32 {
    match op {
        ReduceOp::Sum => reduce_op::SUM,
        ReduceOp::Min => reduce_op::MIN,
        ReduceOp::Max => reduce_op::MAX,
        ReduceOp::Mean => reduce_op::MEAN,
        ReduceOp::Prod => reduce_op::PROD,
    }
}

mod float_unary_op {
    pub const EXP: u32 = 0;
    pub const LN: u32 = 1;
    pub const SIN: u32 = 2;
    pub const COS: u32 = 3;
    pub const TANH: u32 = 4;
    pub const SQR: u32 = 5;
    pub const SQRT: u32 = 6;
    pub const RECIP: u32 = 7;
    pub const GELU: u32 = 8;
    pub const GELU_ERF: u32 = 9;
    pub const ERF: u32 = 10;
    pub const RELU: u32 = 11;
    pub const LEAKY_RELU: u32 = 12;
    pub const SILU: u32 = 13;
    pub const SIGMOID: u32 = 14;
    pub const FLOOR: u32 = 15;
    pub const CEIL: u32 = 16;
    pub const ROUND: u32 = 17;
}

pub(crate) fn float_unary_op_code(op: FloatUnaryOp) -> u32 {
    match op {
        FloatUnaryOp::Exp => float_unary_op::EXP,
        FloatUnaryOp::Ln => float_unary_op::LN,
        FloatUnaryOp::Sin => float_unary_op::SIN,
        FloatUnaryOp::Cos => float_unary_op::COS,
        FloatUnaryOp::Tanh => float_unary_op::TANH,
        FloatUnaryOp::Sqr => float_unary_op::SQR,
        FloatUnaryOp::Sqrt => float_unary_op::SQRT,
        FloatUnaryOp::Recip => float_unary_op::RECIP,
        FloatUnaryOp::Gelu => float_unary_op::GELU,
        FloatUnaryOp::GeluErf => float_unary_op::GELU_ERF,
        FloatUnaryOp::Erf => float_unary_op::ERF,
        FloatUnaryOp::Relu => float_unary_op::RELU,
        FloatUnaryOp::LeakyRelu(_) => float_unary_op::LEAKY_RELU,
        FloatUnaryOp::Silu => float_unary_op::SILU,
        FloatUnaryOp::Sigmoid => float_unary_op::SIGMOID,
        FloatUnaryOp::Floor => float_unary_op::FLOOR,
        FloatUnaryOp::Ceil => float_unary_op::CEIL,
        FloatUnaryOp::Round => float_unary_op::ROUND,
    }
}

mod unary_op {
    pub const NEG: u32 = 0;
    pub const ABS: u32 = 1;
    pub const SIGN: u32 = 2;
    pub const AFFINE: u32 = 3;
    pub const POW: u32 = 4;
    pub const CLAMP: u32 = 5;
}

pub(crate) fn unary_op_code<S>(op: UnaryOp<S>) -> u32 {
    match op {
        UnaryOp::Neg => unary_op::NEG,
        UnaryOp::Abs => unary_op::ABS,
        UnaryOp::Sign => unary_op::SIGN,
        UnaryOp::Affine(_, _) => unary_op::AFFINE,
        UnaryOp::Pow(_) => unary_op::POW,
        UnaryOp::Clamp(_, _) => unary_op::CLAMP,
    }
}

pub(crate) fn unary_op_args<S: Zeroable + Copy>(op: UnaryOp<S>) -> (S, S, u32, u32) {
    match op {
        UnaryOp::Affine(arg0, arg1) => (arg0, arg1, 0, 0),
        UnaryOp::Pow(arg0) => (arg0, S::zeroed(), 0, 0),
        UnaryOp::Clamp(arg0, arg1) => (
            arg0.unwrap_or(S::zeroed()),
            arg1.unwrap_or(S::zeroed()),
            if arg0.is_some() { 1 } else { 0 },
            if arg1.is_some() { 1 } else { 0 },
        ),
        _ => (S::zeroed(), S::zeroed(), 0, 0),
    }
}

mod cmp_op {
    pub const EQ: u32 = 0;
    pub const NE: u32 = 1;
    pub const LE: u32 = 2;
    pub const GE: u32 = 3;
    pub const LT: u32 = 4;
    pub const GT: u32 = 5;
}

pub(crate) fn cmp_op_code(op: CmpOp) -> u32 {
    match op {
        CmpOp::Eq => cmp_op::EQ,
        CmpOp::Ne => cmp_op::NE,
        CmpOp::Le => cmp_op::LE,
        CmpOp::Ge => cmp_op::GE,
        CmpOp::Lt => cmp_op::LT,
        CmpOp::Gt => cmp_op::GT,
    }
}

pub struct BindGroupBuilder<'a> {
    entries: Vec<wgpu::BindGroupEntry<'a>>,
}

impl<'a> BindGroupBuilder<'a> {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn storage_read(mut self, binding: u32, buffer: &'a wgpu::Buffer) -> Self {
        self.entries.push(wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() });
        self
    }

    pub fn storage_read_at(mut self, binding: u32, buffer: &'a wgpu::Buffer, offset: u64) -> Self {
        let resource = wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer, offset, size: None });
        self.entries.push(wgpu::BindGroupEntry { binding, resource });
        self
    }

    pub fn storage_rw(mut self, binding: u32, buffer: &'a wgpu::Buffer) -> Self {
        self.entries.push(wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() });
        self
    }

    pub fn storage_rw_at(mut self, binding: u32, buffer: &'a wgpu::Buffer, offset: u64) -> Self {
        let resource = wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer, offset, size: None });
        self.entries.push(wgpu::BindGroupEntry { binding, resource });
        self
    }

    pub fn uniform(mut self, binding: u32, buffer: &'a wgpu::Buffer) -> Self {
        self.entries.push(wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() });
        self
    }

    pub fn build(self, device: &Wgpu, layout: &wgpu::BindGroupLayout) -> wgpu::BindGroup {
        device.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("luma-wgpu bind group"),
            layout,
            entries: &self.entries,
        })
    }
}

impl Default for BindGroupBuilder<'_> {
    fn default() -> Self {
        Self::new()
    }
}
