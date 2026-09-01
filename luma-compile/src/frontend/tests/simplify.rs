use crate::frontend::simplify::simplify;
use crate::frontend::verify::verify;
use crate::{Graph, NodeOp, Scalar};
use luma_tensor::dtype::FloatDType;
use luma_tensor::{BinaryOp, Cpu, DType, FloatUnaryOp, Shape, Tensor, UnaryOp};

/// x + 0 → x（float 与 int 两种标量形式）。
fn identity_graph(scalar: Scalar, op: BinaryOp) -> Graph {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let y = g.add_node(NodeOp::BinaryScalarRhs(scalar, op), vec![x], DType::F32, Shape::from((2, 2)));
    g.mark_output(y);
    g
}

#[test]
fn add_zero_f64() {
    let g2 = simplify(identity_graph(Scalar::F64(0.0), BinaryOp::Add)).unwrap();
    verify(&g2).unwrap();
    assert!(g2.nodes.is_empty(), "x + 0 节点应被消除");
    assert_eq!(g2.values.len(), 1);
    assert_eq!(g2.outputs, vec![0]);
}

#[test]
fn sub_zero_mul_one_div_one() {
    for (s, op) in [(Scalar::F64(0.0), BinaryOp::Sub), (Scalar::F64(1.0), BinaryOp::Mul), (Scalar::F64(1.0), BinaryOp::Div)] {
        let g2 = simplify(identity_graph(s, op)).unwrap();
        verify(&g2).unwrap();
        assert!(g2.nodes.is_empty(), "{op:?} 恒等元应被消除");
        assert_eq!(g2.outputs, vec![0]);
    }
}

#[test]
fn add_zero_i64() {
    let mut g = Graph::default();
    let x = g.add_value(DType::I32, Shape::from((2, 2)));
    g.mark_input(x);
    let y = g.add_node(NodeOp::BinaryScalarRhs(Scalar::I64(0), BinaryOp::Add), vec![x], DType::I32, Shape::from((2, 2)));
    g.mark_output(y);

    let g2 = simplify(g).unwrap();
    verify(&g2).unwrap();
    assert!(g2.nodes.is_empty(), "int x + 0 应被消除");
    assert_eq!(g2.outputs, vec![0]);
}

/// 链式 (x + 0) + 0 必须单遍消干净。
#[test]
fn chained_identity_fully_removed() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let a = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(0.0), BinaryOp::Add), vec![x], DType::F32, Shape::from((2, 2)));
    let b = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(0.0), BinaryOp::Add), vec![a], DType::F32, Shape::from((2, 2)));
    g.mark_output(b);

    let g2 = simplify(g).unwrap();
    verify(&g2).unwrap();
    assert!(g2.nodes.is_empty(), "链式恒等元应单遍消干净");
    assert_eq!(g2.outputs, vec![0]);
}

/// 非零标量不能消。
#[test]
fn nonzero_scalar_kept() {
    let g2 = simplify(identity_graph(Scalar::F64(1.0), BinaryOp::Add)).unwrap();
    verify(&g2).unwrap();
    assert_eq!(g2.nodes.len(), 1, "x + 1 不是恒等元");
}

/// 空转 cast（目标 dtype == 输入 dtype）应消除。
#[test]
fn empty_cast_removed() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let y = g.add_node(NodeOp::Cast(DType::F32), vec![x], DType::F32, Shape::from((2, 2)));
    g.mark_output(y);

    let g2 = simplify(g).unwrap();
    verify(&g2).unwrap();
    assert!(g2.nodes.is_empty(), "空转 cast 应消除");
    assert_eq!(g2.outputs, vec![0]);
}

/// 有实际转换的 cast 不能消。
#[test]
fn real_cast_kept() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let y = g.add_node(NodeOp::Cast(DType::F64), vec![x], DType::F64, Shape::from((2, 2)));
    g.mark_output(y);

    let g2 = simplify(g).unwrap();
    verify(&g2).unwrap();
    assert_eq!(g2.nodes.len(), 1, "f32 -> f64 是真实 cast");
}

/// 端到端：simplify 后的图执行结果必须与简化前一致。
#[test]
fn simplified_graph_executes_same() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let a = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(0.0), BinaryOp::Add), vec![x], DType::F32, Shape::from((2, 2)));
    let y = g.add_node(NodeOp::FloatUnary(FloatUnaryOp::Sqr), vec![a], DType::F32, Shape::from((2, 2)));
    g.mark_output(y);

    let g2 = simplify(g).unwrap();
    verify(&g2).unwrap();
    assert_eq!(g2.nodes.len(), 1, "只剩 sqr");

    let input = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), FloatDType::F32).unwrap();
    let expected = input.sqr().unwrap().to_vec().unwrap();

    let mut exec = g2.compile(&Cpu::default()).unwrap();
    let out = exec.run(&[input.clone().into()]).unwrap();
    assert_eq!(out[0].as_float().unwrap().to_vec().unwrap(), expected);
}

/// max(x, x) / min(x, x) 同输入 → 恒等。
#[test]
fn max_min_same_input_removed() {
    for op in [BinaryOp::Maximum, BinaryOp::Minimum] {
        let mut g = Graph::default();
        let x = g.add_value(DType::F32, Shape::from((2, 2)));
        g.mark_input(x);
        let y = g.add_node(NodeOp::Binary(op), vec![x, x], DType::F32, Shape::from((2, 2)));
        g.mark_output(y);

        let g2 = simplify(g).unwrap();
        verify(&g2).unwrap();
        assert!(g2.nodes.is_empty(), "{op:?}(x, x) 应消除");
        assert_eq!(g2.outputs, vec![0]);
    }
}

/// max(x, y) 不同输入不能消。
#[test]
fn max_different_inputs_kept() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    let y = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    g.mark_input(y);
    let z = g.add_node(NodeOp::Binary(BinaryOp::Maximum), vec![x, y], DType::F32, Shape::from((2, 2)));
    g.mark_output(z);

    let g2 = simplify(g).unwrap();
    verify(&g2).unwrap();
    assert_eq!(g2.nodes.len(), 1, "max(x, y) 不是恒等");
}

/// 无参一元恒等（float + int 两套）。
#[test]
fn unary_identity_removed() {
    let cases: [NodeOp; 6] = [
        NodeOp::Unary(UnaryOp::Affine(1.0, 0.0)),
        NodeOp::Unary(UnaryOp::Pow(1.0)),
        NodeOp::Unary(UnaryOp::Clamp(None, None)),
        NodeOp::UnaryI(UnaryOp::Affine(1, 0)),
        NodeOp::UnaryI(UnaryOp::Pow(1)),
        NodeOp::UnaryI(UnaryOp::Clamp(None, None)),
    ];
    for op in &cases {
        let mut g = Graph::default();
        let x = g.add_value(DType::F32, Shape::from((2, 2)));
        g.mark_input(x);
        let y = g.add_node(op.clone(), vec![x], DType::F32, Shape::from((2, 2)));
        g.mark_output(y);

        let g2 = simplify(g).unwrap();
        verify(&g2).unwrap();
        assert!(g2.nodes.is_empty(), "{op:?} 恒等应消除");
        assert_eq!(g2.outputs, vec![0]);
    }
}

/// 非恒等 unary 必须保留。
#[test]
fn non_identity_unary_kept() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let y = g.add_node(NodeOp::Unary(UnaryOp::Affine(2.0, 1.0)), vec![x], DType::F32, Shape::from((2, 2)));
    g.mark_output(y);

    let g2 = simplify(g).unwrap();
    assert_eq!(g2.nodes.len(), 1, "affine(2, 1) 不是恒等");
}

/// x & x / x | x → x。
#[test]
fn and_or_same_input_removed() {
    for op in [NodeOp::And, NodeOp::Or] {
        let mut g = Graph::default();
        let x = g.add_value(DType::Bool, Shape::from((2, 2)));
        g.mark_input(x);
        let y = g.add_node(op.clone(), vec![x, x], DType::Bool, Shape::from((2, 2)));
        g.mark_output(y);

        let g2 = simplify(g).unwrap();
        verify(&g2).unwrap();
        assert!(g2.nodes.is_empty(), "{op:?}(x, x) 应消除");
        assert_eq!(g2.outputs, vec![0]);
    }
}

/// 视图恒等：transpose(a,a)、恒等 permute、同 shape reshape/broadcast、全切片。
#[test]
fn view_identity_removed() {
    let build = |op: NodeOp, in_shape: Shape, out_shape: Shape| -> Graph {
        let mut g = Graph::default();
        let x = g.add_value(DType::F32, in_shape);
        g.mark_input(x);
        let y = g.add_node(op, vec![x], DType::F32, out_shape);
        g.mark_output(y);
        g
    };

    for g in [
        build(NodeOp::Transpose(1, 1), Shape::from((2, 3)), Shape::from((2, 3))),
        build(NodeOp::Permute(vec![0, 1, 2]), Shape::from((2, 3, 4)), Shape::from((2, 3, 4))),
        build(NodeOp::Reshape, Shape::from((2, 3)), Shape::from((2, 3))),
        build(NodeOp::Broadcast, Shape::from((2, 3)), Shape::from((2, 3))),
        build(NodeOp::Slice(1, 0, 3, 1), Shape::from((2, 3)), Shape::from((2, 3))),
    ] {
        let g2 = simplify(g).unwrap();
        verify(&g2).unwrap();
        assert!(g2.nodes.is_empty(), "视图恒等应消除");
        assert_eq!(g2.outputs, vec![0]);
    }
}

/// 非恒等视图必须保留。
#[test]
fn non_identity_views_kept() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 3)));
    g.mark_input(x);
    let y = g.add_node(NodeOp::Transpose(0, 1), vec![x], DType::F32, Shape::from((3, 2)));
    g.mark_output(y);
    let g2 = simplify(g).unwrap();
    assert_eq!(g2.nodes.len(), 1, "transpose(0, 1) 是真实转置");

    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 3)));
    g.mark_input(x);
    let y = g.add_node(NodeOp::Slice(1, 0, 2, 1), vec![x], DType::F32, Shape::from((2, 2)));
    g.mark_output(y);
    let g2 = simplify(g).unwrap();
    assert_eq!(g2.nodes.len(), 1, "非全切片不能消");
}

/// trace 一个真实模块，simplify 不破坏任何东西。
#[test]
fn traced_linear_survives_simplify() {
    let linear = luma_nn::Linear::new(3, 4, true, Cpu::default()).unwrap();
    let x = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0], (1, 3), FloatDType::F32).unwrap();
    let g = crate::trace(&linear, &x).unwrap();
    let g = g.lock().unwrap().clone();
    let n_nodes = g.nodes.len();

    let g2 = simplify(g).unwrap();
    verify(&g2).unwrap();
    // Linear 里有一个真实的同 shape broadcast（broadcast_add 对已匹配的 lhs
    // 也记录了恒等 broadcast 节点），应被消掉，其余保留。
    assert_eq!(g2.nodes.len(), n_nodes - 1, "同 shape broadcast 应被消掉");

    let expected = linear.forward(&x).unwrap().to_vec().unwrap();
    let mut exec = g2.compile(&Cpu::default()).unwrap();
    let out = exec.run(&[x.into()]).unwrap();
    assert_eq!(out[0].as_float().unwrap().to_vec().unwrap(), expected);
}

/// neg(neg(x)) == x。
#[test]
fn neg_neg_removed() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let a = g.add_node(NodeOp::Unary(UnaryOp::Neg), vec![x], DType::F32, Shape::from((2, 2)));
    let b = g.add_node(NodeOp::Unary(UnaryOp::Neg), vec![a], DType::F32, Shape::from((2, 2)));
    g.mark_output(b);

    let g2 = simplify(g).unwrap();
    verify(&g2).unwrap();
    assert!(g2.nodes.is_empty(), "neg(neg(x)) 应消除");
    assert_eq!(g2.outputs, vec![0]);
}

/// 单个 neg 不能消。
#[test]
fn single_neg_kept() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let a = g.add_node(NodeOp::Unary(UnaryOp::Neg), vec![x], DType::F32, Shape::from((2, 2)));
    g.mark_output(a);

    let g2 = simplify(g).unwrap();
    assert_eq!(g2.nodes.len(), 1);
}

/// transpose 两次（同参数或互换参数）都恒等。
#[test]
fn transpose_twice_removed() {
    // transpose(0,1) ∘ transpose(0,1)
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 3)));
    g.mark_input(x);
    let a = g.add_node(NodeOp::Transpose(0, 1), vec![x], DType::F32, Shape::from((3, 2)));
    let b = g.add_node(NodeOp::Transpose(0, 1), vec![a], DType::F32, Shape::from((2, 3)));
    g.mark_output(b);

    let g2 = simplify(g).unwrap();
    verify(&g2).unwrap();
    assert!(g2.nodes.is_empty(), "transpose(a,b)∘transpose(a,b) 应消除");
    assert_eq!(g2.outputs, vec![0]);

    // transpose(0,1) ∘ transpose(1,0)
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 3)));
    g.mark_input(x);
    let a = g.add_node(NodeOp::Transpose(0, 1), vec![x], DType::F32, Shape::from((3, 2)));
    let b = g.add_node(NodeOp::Transpose(1, 0), vec![a], DType::F32, Shape::from((2, 3)));
    g.mark_output(b);

    let g2 = simplify(g).unwrap();
    verify(&g2).unwrap();
    assert!(g2.nodes.is_empty(), "transpose(a,b)∘transpose(b,a) 应消除");
    assert_eq!(g2.outputs, vec![0]);
}

/// 不同 dim 对的两次转置不能消。
#[test]
fn transpose_non_inverse_kept() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 3, 4)));
    g.mark_input(x);
    let a = g.add_node(NodeOp::Transpose(0, 1), vec![x], DType::F32, Shape::from((3, 2, 4)));
    let b = g.add_node(NodeOp::Transpose(0, 2), vec![a], DType::F32, Shape::from((4, 2, 3)));
    g.mark_output(b);

    let g2 = simplify(g).unwrap();
    verify(&g2).unwrap();
    assert_eq!(g2.nodes.len(), 2, "transpose(0,1)∘transpose(0,2) 不是恒等");
}

/// squeeze∘unsqueeze / unsqueeze∘squeeze（dim 相同）都恒等。
#[test]
fn squeeze_unsqueeze_removed() {
    // unsqueeze(1) ∘ squeeze(1)
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 3)));
    g.mark_input(x);
    let a = g.add_node(NodeOp::Unsqueeze(1), vec![x], DType::F32, Shape::from((2, 1, 3)));
    let b = g.add_node(NodeOp::Squeeze(1), vec![a], DType::F32, Shape::from((2, 3)));
    g.mark_output(b);

    let g2 = simplify(g).unwrap();
    verify(&g2).unwrap();
    assert!(g2.nodes.is_empty(), "squeeze(unsqueeze(x)) 应消除");
    assert_eq!(g2.outputs, vec![0]);

    // squeeze(1) ∘ unsqueeze(1)
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 1, 3)));
    g.mark_input(x);
    let a = g.add_node(NodeOp::Squeeze(1), vec![x], DType::F32, Shape::from((2, 3)));
    let b = g.add_node(NodeOp::Unsqueeze(1), vec![a], DType::F32, Shape::from((2, 1, 3)));
    g.mark_output(b);

    let g2 = simplify(g).unwrap();
    verify(&g2).unwrap();
    assert!(g2.nodes.is_empty(), "unsqueeze(squeeze(x)) 应消除");
    assert_eq!(g2.outputs, vec![0]);
}

/// dim 不匹配的 squeeze∘unsqueeze 不能消。
#[test]
fn squeeze_unsqueeze_dim_mismatch_kept() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((1, 3)));
    g.mark_input(x);
    let a = g.add_node(NodeOp::Unsqueeze(1), vec![x], DType::F32, Shape::from((1, 1, 3)));
    let b = g.add_node(NodeOp::Squeeze(0), vec![a], DType::F32, Shape::from((1, 3)));
    g.mark_output(b);

    let g2 = simplify(g).unwrap();
    verify(&g2).unwrap();
    assert_eq!(g2.nodes.len(), 2, "squeeze(0)∘unsqueeze(1) dim 不同，不能消");
}
/// (x + 1) + 2 == x + 3：两个节点合成一个（float）。
#[test]
fn scalar_add_merge_f64() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let a = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(1.0), BinaryOp::Add), vec![x], DType::F32, Shape::from((2, 2)));
    let b = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(2.0), BinaryOp::Add), vec![a], DType::F32, Shape::from((2, 2)));
    g.mark_output(b);

    let g2 = simplify(g).unwrap();
    verify(&g2).unwrap();
    assert_eq!(g2.nodes.len(), 1, "两个 add 合成一个");
    assert!(matches!(&g2.nodes[0].op, NodeOp::BinaryScalarRhs(Scalar::F64(3.0), BinaryOp::Add)), "标量应合并为 3");
    assert_eq!(g2.nodes[0].inputs, vec![0], "输入应为 x");
    assert_eq!(g2.outputs, vec![1]);
}

/// (x + 1) + 2 == x + 3（int）。
#[test]
fn scalar_add_merge_i64() {
    let mut g = Graph::default();
    let x = g.add_value(DType::I32, Shape::from((2, 2)));
    g.mark_input(x);
    let a = g.add_node(NodeOp::BinaryScalarRhs(Scalar::I64(1), BinaryOp::Add), vec![x], DType::I32, Shape::from((2, 2)));
    let b = g.add_node(NodeOp::BinaryScalarRhs(Scalar::I64(2), BinaryOp::Add), vec![a], DType::I32, Shape::from((2, 2)));
    g.mark_output(b);

    let g2 = simplify(g).unwrap();
    verify(&g2).unwrap();
    assert_eq!(g2.nodes.len(), 1);
    assert!(matches!(&g2.nodes[0].op, NodeOp::BinaryScalarRhs(Scalar::I64(3), BinaryOp::Add)));
}

/// 链式 (x + 1) + 2 + 3 必须全部合并成 x + 6（验证不动点循环收敛）。
#[test]
fn scalar_add_merge_chain() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let a = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(1.0), BinaryOp::Add), vec![x], DType::F32, Shape::from((2, 2)));
    let b = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(2.0), BinaryOp::Add), vec![a], DType::F32, Shape::from((2, 2)));
    let c = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(3.0), BinaryOp::Add), vec![b], DType::F32, Shape::from((2, 2)));
    g.mark_output(c);

    let g2 = simplify(g).unwrap();
    verify(&g2).unwrap();
    assert_eq!(g2.nodes.len(), 1, "三个 add 应合并成一个");
    assert!(matches!(&g2.nodes[0].op, NodeOp::BinaryScalarRhs(Scalar::F64(6.0), BinaryOp::Add)));
}

/// 混合 op（(x + 1) * 2）不能合并（不是同 op 链）。
#[test]
fn scalar_mixed_op_kept() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let a = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(1.0), BinaryOp::Add), vec![x], DType::F32, Shape::from((2, 2)));
    let b = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(2.0), BinaryOp::Mul), vec![a], DType::F32, Shape::from((2, 2)));
    g.mark_output(b);

    let g2 = simplify(g).unwrap();
    verify(&g2).unwrap();
    assert_eq!(g2.nodes.len(), 2, "add 后接 mul 不能合并");
}

/// 四组同 op 标量链都合并成一个节点（合并后加减族统一为 Add、乘除族统一为 Mul）：
/// (x+1)+2=x+3 | (x-1)-2=x-3 | (x*2)*3=x*6 | (x/6)/2=x/12
#[test]
fn scalar_merge_all_ops() {
    // (内层op, 外层op, a, b, 期望合并op, 期望标量)
    let cases = [
        (BinaryOp::Add, BinaryOp::Add, 1.0, 2.0, BinaryOp::Add, 3.0),
        (BinaryOp::Sub, BinaryOp::Sub, 1.0, 2.0, BinaryOp::Add, 3.0),
        (BinaryOp::Mul, BinaryOp::Mul, 2.0, 3.0, BinaryOp::Mul, 6.0),
        (BinaryOp::Div, BinaryOp::Div, 6.0, 2.0, BinaryOp::Mul, 12.0),
        // 交叉组合
        (BinaryOp::Add, BinaryOp::Sub, 1.0, 2.0, BinaryOp::Add, -1.0), // (x+1)-2 = x-1
        (BinaryOp::Sub, BinaryOp::Add, 1.0, 2.0, BinaryOp::Add, 1.0),  // (x-1)+2 = x+1
        (BinaryOp::Mul, BinaryOp::Div, 2.0, 4.0, BinaryOp::Mul, 0.5),  // (x*2)/4 = x*0.5
        (BinaryOp::Div, BinaryOp::Mul, 2.0, 4.0, BinaryOp::Mul, 2.0),  // (x/2)*4 = x*2
    ];
    for (inner_op, outer_op, a, b, merged_op, merged) in cases {
        let mut g = Graph::default();
        let x = g.add_value(DType::F32, Shape::from((2, 2)));
        g.mark_input(x);
        let n1 = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(a), inner_op), vec![x], DType::F32, Shape::from((2, 2)));
        let n2 = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(b), outer_op), vec![n1], DType::F32, Shape::from((2, 2)));
        g.mark_output(n2);

        let g2 = simplify(g).unwrap();
        verify(&g2).unwrap();
        assert_eq!(g2.nodes.len(), 1, "{inner_op:?}→{outer_op:?} 链应合并成一个节点");
        assert!(
            matches!(&g2.nodes[0].op, NodeOp::BinaryScalarRhs(Scalar::F64(v), o) if *v == merged && *o == merged_op),
            "{inner_op:?}→{outer_op:?} 合并结果应为 {merged_op:?} {merged}"
        );
        assert_eq!(g2.nodes[0].inputs, vec![0], "输入应为 x");
    }
}

/// 交叉 op（(x * 2) + 3）不能合并（宏要求内层与外层同 op）。
#[test]
fn scalar_cross_op_kept() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let n1 = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(2.0), BinaryOp::Mul), vec![x], DType::F32, Shape::from((2, 2)));
    let n2 = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(3.0), BinaryOp::Add), vec![n1], DType::F32, Shape::from((2, 2)));
    g.mark_output(n2);

    let g2 = simplify(g).unwrap();
    verify(&g2).unwrap();
    assert_eq!(g2.nodes.len(), 2, "(x*2)+3 跨 op 不能合并");
}

/// 共享消费者：合并后的新节点必须插入到内层位置，执行序正确。
/// 这是"append 到末尾"bug 的回归测试——append 会让中间的消费者
/// 先于生产者执行，读空 slot panic。
#[test]
fn scalar_merge_shared_consumer_executes() {
    let mut g = Graph::default();
    let x = g.add_value(DType::F32, Shape::from((2, 2)));
    g.mark_input(x);
    let a = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(1.0), BinaryOp::Add), vec![x], DType::F32, Shape::from((2, 2)));
    let b = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(2.0), BinaryOp::Add), vec![a], DType::F32, Shape::from((2, 2)));
    let c = g.add_node(NodeOp::BinaryScalarRhs(Scalar::F64(3.0), BinaryOp::Add), vec![b], DType::F32, Shape::from((2, 2)));
    g.mark_output(b); // b 是输出，且被 c 消费 → 合并 b 后 c 是"中间消费者"
    g.mark_output(c);

    let g2 = simplify(g).unwrap();
    verify(&g2).unwrap();

    let input = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), FloatDType::F32).unwrap();
    let mut exec = g2.compile(&Cpu::default()).unwrap();
    let out = exec.run(&[input.clone().into()]).unwrap();
    // out[0] = x + 3（b 被合并），out[1] = x + 6
    let expected0 = input.clone().add_scalar(3.0).unwrap().to_vec().unwrap();
    let expected1 = input.add_scalar(6.0).unwrap().to_vec().unwrap();
    assert_eq!(out[0].as_float().unwrap().to_vec().unwrap(), expected0);
    assert_eq!(out[1].as_float().unwrap().to_vec().unwrap(), expected1);
}
