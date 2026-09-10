//! NodeOp → ONNX `(op_type, attributes)` 映射。
//!
//! v1 范围只做**直接映射**：一个 NodeOp 对应一个 ONNX 节点、参数全部走 attribute、
//! 输入输出与 graph 里的 value 一一对应。其余 NodeOp（需要合成 initializer 输入、
//! 或展开成多个节点的：RmsNorm、Pick/PickTrue/PickFalse、BinaryScalar*、Affine、
//! Pow(scalar)、Clamp、Sqr、Gelu/GeluErf、Silu、Ne、ReduceAll/Any、IndexAdd/
//! ScatterAdd、Reshape、Narrow/Slice、Broadcast、Arange）统一返回 `Err`，留待下一步。
//!
//! opset 13 下的「非直接映射」（对照 onnx schema 确认过，不是想当然的 attributes）：
//! - ReduceSum-13 起 axes 是**输入**（int64 tensor），ReduceMean/Min/Max/Prod 仍是 attribute
//! - Squeeze/Unsqueeze-13 起 axes 是**输入**
//! - Clip-6 起 min/max 是**输入**（Clamp 需合成标量输入）
//!
//! 已知偏差：ArgReduce 的 graph 输出 dtype 是 I32，ONNX ArgMax/ArgMin 输出固定 int64，
//! 需要时补 Cast —— 留待分解阶段。

use luma_compile::graph::{Node, NodeOp, Value};
use luma_tensor::{BinaryOp, CmpOp, FloatUnaryOp, ReduceOp, UnaryOp};

use crate::export::to_onnx_data_type;
use crate::types::attribute_proto::AttributeType;
use crate::types::AttributeProto;

pub(super) fn to_onnx_op(op: &NodeOp, node: &Node, values: &[Value]) -> Result<(String, Vec<AttributeProto>), String> {
    use NodeOp::*;

    let unsupported = || format!("unsupported op for onnx export: {op}");

    // NodeOp 不记 keepdim，keepdims 由已记录的输入/输出 rank 反推：
    // 输出保留原 rank ⇒ 1；维度被 squeeze 掉 ⇒ 0。
    let input_rank = node.inputs.first().and_then(|&i| values.get(i)).map(|v| v.shape.rank());
    let output_rank = node.outputs.first().and_then(|&i| values.get(i)).map(|v| v.shape.rank());
    let keepdims = |name: &str| -> Result<i64, String> {
        match (input_rank, output_rank) {
            (Some(i), Some(o)) => Ok((i == o) as i64),
            _ => Err(format!("{name}: cannot infer keepdims — missing value shape")),
        }
    };

    let mut attrs: Vec<AttributeProto> = Vec::new();
    match op {
        Constant | BinaryScalarRhs(..) | BinaryScalarLhs(..) | CmpScalar(..) => Err(unsupported()),

        Binary(b) => Ok((binary_name(*b).into(), vec![])),

        Unary(u) => unary_name(u).map(|n| (n.into(), vec![])).ok_or_else(unsupported),
        UnaryI(u) => unary_name(u).map(|n| (n.into(), vec![])).ok_or_else(unsupported),

        FloatUnary(f) => {
            let (name, alpha) = float_unary_name(f).ok_or_else(unsupported)?;
            if let Some(a) = alpha {
                attrs.push(attr_f("alpha", a));
            }
            Ok((name.into(), attrs))
        }

        Cmp(c) => cmp_name(*c).map(|n| (n.into(), vec![])).ok_or_else(unsupported),

        Cast(dt) => {
            let to: i32 = to_onnx_data_type(*dt).into();
            attrs.push(attr_i("to", to as i64));
            Ok(("Cast".into(), attrs))
        }

        And => Ok(("And".into(), vec![])),
        Or => Ok(("Or".into(), vec![])),
        Xor => Ok(("Xor".into(), vec![])),
        Not => Ok(("Not".into(), vec![])),

        Reduce(reduce, dims) => {
            let (name, axes_is_input) = match reduce {
                ReduceOp::Sum => ("ReduceSum", true),
                ReduceOp::Mean => ("ReduceMean", false),
                ReduceOp::Min => ("ReduceMin", false),
                ReduceOp::Max => ("ReduceMax", false),
                ReduceOp::Prod => ("ReduceProd", false),
            };
            if axes_is_input && !dims.is_empty() {
                // ReduceSum-13 的 axes 是输入而不是属性
                return Err(unsupported());
            }
            attrs.push(attr_i("keepdims", keepdims(name)?));
            if !dims.is_empty() {
                attrs.push(attr_ints("axes", dims));
            }
            Ok((name.into(), attrs))
        }

        ReduceAll(..) | ReduceAny(..) => Err(unsupported()),

        ArgReduce(d, take_max) => {
            let name = if *take_max { "ArgMax" } else { "ArgMin" };
            attrs.push(attr_i("axis", *d as i64));
            attrs.push(attr_i("keepdims", keepdims(name)?));
            attrs.push(attr_i("select_last_index", 0));
            Ok((name.into(), attrs))
        }

        Matmul => Ok(("MatMul".into(), vec![])),

        IndexSelect(d) => Ok(("Gather".into(), vec![attr_i("axis", *d as i64)])),
        Gather(d) => Ok(("GatherElements".into(), vec![attr_i("axis", *d as i64)])),
        IndexAdd(..) | ScatterAdd(..) => Err(unsupported()),

        Cat(d) => Ok(("Concat".into(), vec![attr_i("axis", *d as i64)])),
        Softmax(d) => Ok(("Softmax".into(), vec![attr_i("axis", *d as i64)])),
        RmsNorm(..) => Err(unsupported()),
        Pick | PickTrue(..) | PickFalse(..) => Err(unsupported()),
        Arange(..) => Err(unsupported()),

        Reshape => Err(unsupported()),
        Transpose(d1, d2) => {
            let rank = input_rank.ok_or_else(|| format!("Transpose: missing input shape"))?;
            let mut perm: Vec<usize> = (0..rank).collect();
            perm.swap(*d1, *d2);
            attrs.push(attr_ints("perm", &perm));
            Ok(("Transpose".into(), attrs))
        }
        Permute(dims) => {
            attrs.push(attr_ints("perm", dims));
            Ok(("Transpose".into(), attrs))
        }
        Narrow(..) | Slice(..) | Broadcast => Err(unsupported()),
        // Squeeze/Unsqueeze-13 起 axes 是输入，不是属性
        Squeeze(..) | Unsqueeze(..) => Err(unsupported()),
    }
}

// ============================================================================
//    op_type / attribute 辅助
// ============================================================================

fn binary_name(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "Add",
        BinaryOp::Sub => "Sub",
        BinaryOp::Mul => "Mul",
        BinaryOp::Div => "Div",
        BinaryOp::Maximum => "Max",
        BinaryOp::Minimum => "Min",
    }
}

/// Unary/UnaryI 中能 1:1 映射的（Neg/Abs/Sign）；Affine/Pow/Clamp 返回 None。
fn unary_name<S>(op: &UnaryOp<S>) -> Option<&'static str> {
    match op {
        UnaryOp::Neg => Some("Neg"),
        UnaryOp::Abs => Some("Abs"),
        UnaryOp::Sign => Some("Sign"),
        _ => None,
    }
}

/// FloatUnary 中能 1:1 映射的；Sqr/Gelu/GeluErf/Silu 返回 None。
/// LeakyRelu 需要 alpha attribute。
fn float_unary_name(op: &FloatUnaryOp) -> Option<(&'static str, Option<f32>)> {
    match op {
        FloatUnaryOp::Exp => Some(("Exp", None)),
        FloatUnaryOp::Ln => Some(("Log", None)),
        FloatUnaryOp::Sin => Some(("Sin", None)),
        FloatUnaryOp::Cos => Some(("Cos", None)),
        FloatUnaryOp::Tanh => Some(("Tanh", None)),
        FloatUnaryOp::Sqrt => Some(("Sqrt", None)),
        FloatUnaryOp::Recip => Some(("Reciprocal", None)),
        FloatUnaryOp::Erf => Some(("Erf", None)),
        FloatUnaryOp::Relu => Some(("Relu", None)),
        FloatUnaryOp::LeakyRelu(a) => Some(("LeakyRelu", Some(*a as f32))),
        FloatUnaryOp::Sigmoid => Some(("Sigmoid", None)),
        FloatUnaryOp::Floor => Some(("Floor", None)),
        FloatUnaryOp::Ceil => Some(("Ceil", None)),
        FloatUnaryOp::Round => Some(("Round", None)),
        FloatUnaryOp::Sqr | FloatUnaryOp::Gelu | FloatUnaryOp::GeluErf | FloatUnaryOp::Silu => None,
    }
}

/// Cmp 中能 1:1 映射的（Ne 需要 Not(Equal) 组合）。
fn cmp_name(op: CmpOp) -> Option<&'static str> {
    match op {
        CmpOp::Eq => Some("Equal"),
        CmpOp::Le => Some("LessOrEqual"),
        CmpOp::Ge => Some("GreaterOrEqual"),
        CmpOp::Lt => Some("Less"),
        CmpOp::Gt => Some("Greater"),
        CmpOp::Ne => None,
    }
}

fn attr_i(name: &str, val: i64) -> AttributeProto {
    AttributeProto { name: name.into(), r#type: AttributeType::Int as i32, i: val, ..Default::default() }
}

fn attr_f(name: &str, val: f32) -> AttributeProto {
    AttributeProto { name: name.into(), r#type: AttributeType::Float as i32, f: val, ..Default::default() }
}

fn attr_ints(name: &str, vals: &[usize]) -> AttributeProto {
    AttributeProto {
        name: name.into(),
        r#type: AttributeType::Ints as i32,
        ints: vals.iter().map(|&v| v as i64).collect(),
        ..Default::default()
    }
}
