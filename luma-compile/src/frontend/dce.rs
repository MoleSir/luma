use crate::{CompileResult, Graph};
use std::collections::{HashMap, HashSet, VecDeque};

use super::util::build_producer;

pub fn dce(graph: Graph) -> CompileResult<Graph> {
    // 生产映射：value -> 生产它的 node 索引（叶子为 None），供反向可达标记用
    let producers = build_producer(&graph);

    // 初始化所有位置都没有使用
    let mut used = vec![false; graph.values.len()];

    // 输入/输出不可以消除
    for &input in graph.inputs.iter() {
        used[input] = true;
    }

    let mut queue = VecDeque::new();
    for &output in graph.outputs.iter() {
        used[output] = true;
        queue.push_back(output);
    }

    let mut used_nodes = HashSet::new();
    while !queue.is_empty() {
        let output = queue.pop_front().unwrap();
        // 叶子（graph input / 常量）没有生产者，直接跳过
        let Some(node_idx) = producers.get(output).copied().flatten() else { continue };
        used_nodes.insert(node_idx);
        // 取出一个值，找到生产这个值需要的 value，说明这些 input 是需要的！
        for &input in &graph.nodes[node_idx].inputs {
            if used[input] {
                // 避免重复标记
                continue;
            }
            // 标记这个 input tensor 必须使用，同时插入队列
            used[input] = true;
            queue.push_back(input);
        }
    }

    // 1. used: 表示原来这些 values 不需要
    // 2. used_nodes：表示哪些 node 需要保存

    // 创建新的 tensor index
    let mut value_idx_map = HashMap::new();
    for (i, u) in used.iter().enumerate() {
        // values[i] 是否需要保存
        if *u {
            value_idx_map.insert(i, value_idx_map.len());
        }
    }

    let trans_value_idxs = |idxs: &mut Vec<usize>| {
        for idx in idxs.iter_mut() {
            *idx = value_idx_map[idx];
        }
    };

    // 新建 values
    let mut values = vec![];
    for (old_value_idx, mut old_values) in graph.values.into_iter().enumerate() {
        if used[old_value_idx] {
            old_values.id = value_idx_map[&old_value_idx];
            values.push(old_values);
        }
    }

    // 新建 nodes
    let mut nodes = vec![];
    for (old_node_idx, mut node) in graph.nodes.into_iter().enumerate() {
        if used_nodes.contains(&old_node_idx) {
            trans_value_idxs(&mut node.inputs);
            trans_value_idxs(&mut node.outputs);
            nodes.push(node);
        }
    }

    let mut inputs = graph.inputs;
    trans_value_idxs(&mut inputs);
    let mut outputs = graph.outputs;
    trans_value_idxs(&mut outputs);

    Ok(Graph { values, nodes, inputs, outputs })
}
