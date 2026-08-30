use super::{dce::dce, util};
use crate::{Graph, GraphExecutor, JitResult};
use luma_tensor::{Cpu, DynTensor};
use std::collections::HashMap;

pub fn fold(graph: Graph) -> JitResult<Graph> {
    let mut graph = graph;

    // 标记哪些 values 是 const
    let mut is_const = vec![false; graph.values.len()];
    for v in graph.values.iter() {
        if v.data.is_some() {
            is_const[v.id] = true;
        }
    }

    // 哪些结点的输入是 const
    let mut foldable = vec![false; graph.nodes.len()];
    for (i, node) in graph.nodes.iter().enumerate() {
        if node.inputs.iter().all(|&inp| is_const[inp]) {
            is_const[node.outputs[0]] = true;
            foldable[i] = true;
        }
    }

    for i in 0..graph.nodes.len() {
        if !foldable[i] {
            continue;
        }

        // 收集这个结点的信息
        let op = graph.nodes[i].op.clone();
        let inputs = graph.nodes[i].inputs.clone();
        let out_dtype = graph.values[graph.nodes[i].outputs[0]].dtype;
        let out_shape = graph.values[graph.nodes[i].outputs[0]].shape.clone();

        // 构造一个图执行这个结点
        let mut sub = Graph::default();
        // map 保存输入 index -> 新图的 index
        let mut map = HashMap::new();

        for &inp in &inputs {
            let v = &graph.values[inp];
            map.entry(inp)
                .or_insert_with(|| sub.add_constant(v.dtype, v.shape.clone(), v.data.as_ref().expect("foldable 输入必有 data").0.clone()));
        }
        // 输入转为新图的 index
        let tin = sub.add_node(op, inputs.iter().map(|inp| map[inp]).collect(), out_dtype, out_shape.clone());
        sub.mark_output(tin);

        // 用 Cpu executor 执行临时图（无输入）
        let mut exec = GraphExecutor::<Cpu>::compile(&sub, &Cpu)?;
        let out = exec.run(&[])?;
        let bytes = match &out[0] {
            DynTensor::Float(t) => t.to_bytes()?,
            DynTensor::Int(t) => t.to_bytes()?,
            DynTensor::Bool(t) => t.to_bytes()?,
        };

        // 对原图，相当于原来 output 直接替换为了这个常量 c
        let c = graph.add_constant(out_dtype, out_shape, bytes);
        let from = graph.nodes[i].outputs[0];
        util::replace_uses(&mut graph, from, c);
    }

    dce(graph)
}
