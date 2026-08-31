use super::{
    dce::dce,
    util::{self, build_producer, build_use_counts, topo_sort},
};
use crate::{Graph, JitResult, NodeOp, Scalar, ValueId};
use luma_tensor::{BinaryOp, UnaryOp};

pub fn simplify(graph: Graph) -> JitResult<Graph> {
    let mut graph = graph;

    loop {
        let producer = build_producer(&graph);
        let use_counts = build_use_counts(&graph);
        let is_output: Vec<bool> = {
            let mut v = vec![false; graph.values.len()];
            for &o in &graph.outputs {
                v[o] = true;
            }
            v
        };
        let mut changed = false;
        for node_idx in 0..graph.nodes.len() {
            let out = graph.nodes[node_idx].outputs[0];
            // 尸体节点：输出无人引用且不是图输出。跳过——否则新建节点的规则
            // （标量合并）会反复匹配旧链，每轮都 add_node，永不收敛。
            if use_counts[out] == 0 && !is_output[out] {
                continue;
            }
            if let Some(to) = match_node(&mut graph, node_idx, &producer) {
                let from = graph.nodes[node_idx].outputs[0];
                if from != to {
                    util::replace_uses(&mut graph, from, to);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
        // 新节点是 append 的，可能晚于它的消费者执行——重排恢复 SSA 执行序。
        topo_sort(&mut graph);
    }

    dce(graph)
}

macro_rules! scalar_merge {
    ($graph:ident, $producer:ident, $node:ident, $rhs_b:ident,
        $( [$op:ident, $new_op:ident, $a:ident, $b:ident, $merge_expr:expr] ),* $(,)?) => {{
        let inner = $producer.get($node.inputs[0]).copied().flatten()?;
        let inner_node = &$graph.nodes[inner];
        let NodeOp::BinaryScalarRhs(rhs_a, inner_op) = &inner_node.op else {
            return None;
        };
        $(
            if matches!(inner_op, BinaryOp::$op) {
                match (rhs_a, $rhs_b) {
                    (Scalar::F64($a), Scalar::F64($b)) => {
                        let xx = inner_node.inputs[0];
                        let dtype = $graph.values[$node.outputs[0]].dtype;
                        let shape = $graph.values[$node.outputs[0]].shape.clone();
                        return Some($graph.add_node(
                            NodeOp::BinaryScalarRhs(Scalar::F64($merge_expr), BinaryOp::$new_op),
                            vec![xx], dtype, shape,
                        ));
                    }
                    (Scalar::I64($a), Scalar::I64($b)) => {
                        let xx = inner_node.inputs[0];
                        let dtype = $graph.values[$node.outputs[0]].dtype;
                        let shape = $graph.values[$node.outputs[0]].shape.clone();
                        return Some($graph.add_node(
                            NodeOp::BinaryScalarRhs(Scalar::I64($merge_expr), BinaryOp::$new_op),
                            vec![xx], dtype, shape,
                        ));
                    }
                    _ => {}
                }
            }
        )*
        None
    }};
}

/// 单节点窥孔匹配：判定"节点的输出恒等于某个已有值"，返回该替代值。
///
/// 大多数字规则产出已有值（通常是节点自身的输入）；标量合并等规则需要
/// 新建节点（`add_node`），所以这里拿 `&mut Graph`。
fn match_node(graph: &mut Graph, node_idx: usize, producer: &[Option<usize>]) -> Option<ValueId> {
    let node = &graph.nodes[node_idx];
    let x = node.inputs[0];
    let in_shape = |i: usize| &graph.values[node.inputs[i]].shape;
    let out_shape = || &graph.values[node.outputs[0]].shape;

    match &node.op {
        // ---- 标量恒等元: t + 0 == t | t - 0 == t | t * 1 == t | t / 1 == t ----
        NodeOp::BinaryScalarRhs(rhs, BinaryOp::Add | BinaryOp::Sub) if rhs.is_zero() => Some(x),
        NodeOp::BinaryScalarRhs(rhs, BinaryOp::Mul | BinaryOp::Div) if rhs.is_one() => Some(x),
        // 0 + t == t | 1 * t == t
        NodeOp::BinaryScalarLhs(rhs, BinaryOp::Add) if rhs.is_zero() => Some(x),
        NodeOp::BinaryScalarLhs(rhs, BinaryOp::Mul) if rhs.is_one() => Some(x),

        // ---- 同输入二元: max(x,x) == x | min(x,x) == x ----
        NodeOp::Binary(op) if matches!(op, BinaryOp::Maximum | BinaryOp::Minimum) && node.inputs[0] == node.inputs[1] => Some(x),

        // ---- 无参一元恒等（float + int 两套）----
        NodeOp::Unary(UnaryOp::Affine(1.0, 0.0)) => Some(x),
        NodeOp::Unary(UnaryOp::Pow(1.0)) => Some(x),
        NodeOp::Unary(UnaryOp::Clamp(None, None)) => Some(x),
        NodeOp::UnaryI(UnaryOp::Affine(1, 0)) => Some(x),
        NodeOp::UnaryI(UnaryOp::Pow(1)) => Some(x),
        NodeOp::UnaryI(UnaryOp::Clamp(None, None)) => Some(x),

        // ---- bool 逻辑同输入: x & x == x | x | x == x ----
        NodeOp::And | NodeOp::Or if node.inputs[0] == node.inputs[1] => Some(x),

        // ---- 视图恒等（shape 记录在 value 上，单节点即可判定）----
        NodeOp::Reshape if in_shape(0) == out_shape() => Some(x),
        NodeOp::Broadcast if in_shape(0) == out_shape() => Some(x),
        NodeOp::Transpose(a, b) if a == b => Some(x),
        NodeOp::Permute(dims) if dims.iter().enumerate().all(|(i, &d)| d == i) => Some(x),
        NodeOp::Slice(d, s, e, st) if *s == 0 && *st == 1 && in_shape(0).dims().get(*d) == Some(e) => Some(x),
        NodeOp::Squeeze(_) | NodeOp::Unsqueeze(_) if in_shape(0) == out_shape() => Some(x),

        // ---- 空转 cast: dtype 相同 ----
        NodeOp::Cast(dt) if graph.values[node.inputs[0]].dtype == *dt => Some(x),

        // x = neg(neg(x))
        NodeOp::Unary(UnaryOp::Neg) => {
            // 查看这个输入是由哪个 node 输出的
            // （get 安全：遍历中替换引入的新节点不在 producer 快照里，本轮不合并）
            let last_node = producer.get(node.inputs[0]).copied().flatten()?;
            // input => last =neg=> node =neg=> output
            if matches!(&graph.nodes[last_node].op, NodeOp::Unary(UnaryOp::Neg)) {
                Some(graph.nodes[last_node].inputs[0])
            } else {
                None
            }
        }

        // x = transpose(transpose(x, a, b), a, b) —— 转置自逆，两次任意参数对调 = 恒等
        NodeOp::Transpose(dim1, dim2) => {
            let last_node = producer.get(node.inputs[0]).copied().flatten()?;
            if let NodeOp::Transpose(ldim1, ldim2) = &graph.nodes[last_node].op {
                // 两组参数无序相等（(a,b)∘(a,b) 与 (a,b)∘(b,a) 都是两次交换）
                if (dim1 == ldim1 && dim2 == ldim2) || (dim1 == ldim2 && dim2 == ldim1) {
                    Some(graph.nodes[last_node].inputs[0])
                } else {
                    None
                }
            } else {
                None
            }
        }

        // x = squeeze(unsqueeze(x, d), d)
        // x = unsqueeze(squeeze(x, d), d)
        NodeOp::Unsqueeze(dim) => {
            let last_node = producer.get(node.inputs[0]).copied().flatten()?;
            if let NodeOp::Squeeze(ldim) = &graph.nodes[last_node].op
                && dim == ldim
            {
                Some(graph.nodes[last_node].inputs[0])
            } else {
                None
            }
        }
        NodeOp::Squeeze(dim) => {
            let last_node = producer.get(node.inputs[0]).copied().flatten()?;
            if let NodeOp::Unsqueeze(ldim) = &graph.nodes[last_node].op
                && dim == ldim
            {
                Some(graph.nodes[last_node].inputs[0])
            } else {
                None
            }
        }

        NodeOp::BinaryScalarRhs(rhs_b, BinaryOp::Add) => scalar_merge!(
            graph,
            producer,
            node,
            rhs_b,
            // (x + a) + b == x + (a + b)
            [Add, Add, a, b, a + b],
            // (x - a) + b == x + (b - a)
            [Sub, Add, a, b, b - a],
        ),
        NodeOp::BinaryScalarRhs(rhs_b, BinaryOp::Sub) => scalar_merge!(
            graph,
            producer,
            node,
            rhs_b,
            // (x + a) - b == x + (a - b)
            [Add, Add, a, b, a - b],
            // (x - a) - b == x - (a + b)
            [Sub, Add, a, b, a + b],
        ),
        NodeOp::BinaryScalarRhs(rhs_b, BinaryOp::Mul) => scalar_merge!(
            graph,
            producer,
            node,
            rhs_b,
            // (x * a) * b == x * (a * b)
            [Mul, Mul, a, b, a * b],
            // (x / a) * b == x * (b / a)
            [Div, Mul, a, b, b / a],
        ),
        NodeOp::BinaryScalarRhs(rhs_b, BinaryOp::Div) => scalar_merge!(
            graph,
            producer,
            node,
            rhs_b,
            // (x * a) / b == x * (a / b)
            [Mul, Mul, a, b, a / b],
            // (x / a) / b == x / (a * b)
            [Div, Mul, a, b, a * b],
        ),

        _ => None,
    }
}
