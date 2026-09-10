use luma_compile::Graph;
use luma_tensor::DType;
use crate::types::{tensor_proto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto, ValueInfoProto};
mod ops;

/// 导出 compile graph 为 ONNX ModelProto（target opset 13）。
/// 返回 `Err(String)`：遇到 v1 未覆盖的 NodeOp（见 ops 模块注释）。
pub fn export_graph(graph: &Graph) -> Result<ModelProto, String> {
    let input_graph = graph;
    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();

    model.ir_version = 8;
    model.producer_name = "luma".to_string();
    model.opset_import = vec![OperatorSetIdProto { domain: String::new(), version: 13 }];

    // 操作结点
    let mut nodes = vec![];
    for input_node in input_graph.nodes.iter() {
        let mut node = NodeProto::default();
        node.input = input_node.inputs.iter().cloned().map(|i| value_name(i)).collect();
        node.output = input_node.outputs.iter().cloned().map(|i| value_name(i)).collect();
        let (op_type, attribute) = ops::to_onnx_op(&input_node.op, input_node, &input_graph.values)?;
        node.op_type = op_type;
        node.attribute = attribute;
        nodes.push(node);
    }
    graph.node = nodes;

    // 初始化，const tensor
    let mut initializers = vec![];
    for value in input_graph.values.iter() {
        if let Some(data) = &value.data {
            let mut tensor = TensorProto::default();
            tensor.name = value_name(value.id);
            tensor.dims = value.shape.dims().iter().map(|i| *i as i64).collect();
            tensor.data_type = to_onnx_data_type(value.dtype).into();
            tensor.raw_data = data.0.clone();
            initializers.push(tensor);
        }
    }
    graph.initializer = initializers;

    // 输入输出
    let mut inputs = vec![]; 
    for &input_id in input_graph.inputs.iter() {
        let mut input = ValueInfoProto::default();
        input.name = value_name(input_id);
        inputs.push(input);
    }
    graph.input = inputs;

    let mut outputs = vec![]; 
    for &output_id in input_graph.outputs.iter() {
        let mut output = ValueInfoProto::default();
        output.name = value_name(output_id);
        outputs.push(output);
    }
    graph.output = outputs;

    model.graph = Some(graph);
    Ok(model)
}

fn value_name(id: usize) -> String {
    format!("%{}", id)
}

pub(crate) fn to_onnx_data_type(dtype: DType) -> tensor_proto::DataType {
    match dtype {
        DType::F32 => tensor_proto::DataType::Float,
        DType::F64 => tensor_proto::DataType::Double,
        DType::U32 => tensor_proto::DataType::Uint32,
        DType::I32 => tensor_proto::DataType::Int32,
        DType::U8 => tensor_proto::DataType::Uint8,
        DType::Bool => tensor_proto::DataType::Bool,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AttributeProto;
    use luma_compile::{Graph, NodeOp};
    use luma_tensor::{FloatUnaryOp, ReduceOp, Shape};

    fn attr<'a>(node: &'a NodeProto, name: &str) -> &'a AttributeProto {
        node.attribute.iter().find(|a| a.name == name).unwrap()
    }

    fn graph_nodes(model: &ModelProto) -> &[NodeProto] {
        &model.graph.as_ref().unwrap().node
    }

    /// matmul → relu → softmax(1) → reduce_mean([1])，外加 keepdim 分支
    fn sample_graph() -> Graph {
        let mut g = Graph::default();
        let a = g.add_value(DType::F32, Shape::from((2, 3)));
        let b = g.add_value(DType::F32, Shape::from((3, 4)));
        let m = g.add_node(NodeOp::Matmul, vec![a, b], DType::F32, Shape::from((2, 4)));
        let r = g.add_node(NodeOp::FloatUnary(FloatUnaryOp::Relu), vec![m], DType::F32, Shape::from((2, 4)));
        let s = g.add_node(NodeOp::Softmax(1), vec![r], DType::F32, Shape::from((2, 4)));
        let red = g.add_node(NodeOp::Reduce(ReduceOp::Mean, vec![1]), vec![s], DType::F32, Shape::from((2,)));
        g.mark_input(a);
        g.mark_input(b);
        g.mark_output(red);
        g
    }

    #[test]
    fn maps_direct_ops() {
        let model = export_graph(&sample_graph()).unwrap();
        let nodes = graph_nodes(&model);
        let op_types: Vec<&str> = nodes.iter().map(|n| n.op_type.as_str()).collect();
        assert_eq!(op_types, ["MatMul", "Relu", "Softmax", "ReduceMean"]);

        // Softmax(1) → axis 属性
        assert_eq!(attr(&nodes[2], "axis").i, 1);
        // Reduce(Mean, [1])，输出 rank 2 → 1 ⇒ keepdims=0
        let rm = &nodes[3];
        assert_eq!(attr(rm, "keepdims").i, 0);
        assert_eq!(attr(rm, "axes").ints, [1]);
        // 输入输出命名（value id → %id）
        assert_eq!(nodes[0].input, ["%0", "%1"]);
        assert_eq!(nodes[0].output, ["%2"]);
    }

    #[test]
    fn sets_model_header() {
        let model = export_graph(&sample_graph()).unwrap();
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_import.len(), 1);
        assert_eq!(model.opset_import[0].domain, "");
        assert_eq!(model.opset_import[0].version, 13);
    }

    #[test]
    fn keepdims_inferred_true_when_rank_kept() {
        let mut g = Graph::default();
        let x = g.add_value(DType::F32, Shape::from((2, 4)));
        // keepdim 的 reduce：输出保留原 rank（形状 (2,1)）
        let red = g.add_node(NodeOp::Reduce(ReduceOp::Mean, vec![1]), vec![x], DType::F32, Shape::from((2, 1)));
        g.mark_output(red);
        let model = export_graph(&g).unwrap();
        assert_eq!(attr(&graph_nodes(&model)[0], "keepdims").i, 1);
    }

    #[test]
    fn maps_leakyrelu_argmax_cast() {
        let mut g = Graph::default();
        let x = g.add_value(DType::F32, Shape::from((2, 4)));
        let l = g.add_node(NodeOp::FloatUnary(FloatUnaryOp::LeakyRelu(0.02)), vec![x], DType::F32, Shape::from((2, 4)));
        let a = g.add_node(NodeOp::ArgReduce(0, true), vec![l], DType::I32, Shape::from((4,)));
        let c = g.add_node(NodeOp::Cast(DType::F64), vec![a], DType::F64, Shape::from((4,)));
        g.mark_output(c);
        let model = export_graph(&g).unwrap();
        let nodes = graph_nodes(&model);
        assert_eq!(nodes[0].op_type, "LeakyRelu");
        assert_eq!(attr(&nodes[0], "alpha").f, 0.02);
        // ArgReduce(d, take_max) → ArgMax(axis=d, keepdims=0, select_last_index=0)
        assert_eq!(nodes[1].op_type, "ArgMax");
        assert_eq!(attr(&nodes[1], "axis").i, 0);
        assert_eq!(attr(&nodes[1], "keepdims").i, 0);
        assert_eq!(attr(&nodes[1], "select_last_index").i, 0);
        // Cast(F64) → to=DOUBLE(11)
        assert_eq!(nodes[2].op_type, "Cast");
        assert_eq!(attr(&nodes[2], "to").i, 11);
    }

    #[test]
    fn maps_transpose_perm() {
        let mut g = Graph::default();
        let x = g.add_value(DType::F32, Shape::from((2, 3)));
        let t = g.add_node(NodeOp::Transpose(0, 1), vec![x], DType::F32, Shape::from((3, 2)));
        g.mark_output(t);
        let model = export_graph(&g).unwrap();
        assert_eq!(attr(&graph_nodes(&model)[0], "perm").ints, [1, 0]);
    }

    #[test]
    fn unsupported_ops_error() {
        // RmsNorm 等非直接映射 op 返回 Err
        let mut g = Graph::default();
        let x = g.add_value(DType::F32, Shape::from((2, 4)));
        let w = g.add_value(DType::F32, Shape::from((4,)));
        let y = g.add_node(NodeOp::RmsNorm(1e-5), vec![x, w], DType::F32, Shape::from((2, 4)));
        g.mark_output(y);
        let err = export_graph(&g).unwrap_err();
        assert!(err.contains("unsupported op for onnx export: rms_norm"), "{err}");

        // Constant 永不出现（verify 拒绝），撞上也要报错而不是 panic
        let mut g2 = Graph::default();
        g2.add_value(DType::F32, Shape::from((2, 4)));
        let v = g2.add_value(DType::F32, Shape::from((2, 4)));
        g2.nodes.push(luma_compile::Node { op: NodeOp::Constant, inputs: vec![], outputs: vec![v] });
        g2.mark_output(v);
        // Constant 的 Display 是 "const"
        assert!(export_graph(&g2).unwrap_err().contains("const"), "{}", export_graph(&g2).unwrap_err());
    }
}