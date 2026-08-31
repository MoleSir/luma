use super::dce::dce;
use super::util::replace_uses;
use crate::{Graph, JitResult};
use std::collections::HashMap;

/// 公共子表达式消除：op 相同且输入相同的节点合并成一个。
pub fn cse(graph: Graph) -> JitResult<Graph> {
    let mut graph = graph;
    // 每个 node 的 key：(op, 输入，输出)
    let mut seen: HashMap<(String, Vec<usize>), usize> = HashMap::new();

    for i in 0..graph.nodes.len() {
        let out = graph.nodes[i].outputs[0];
        let key = (graph.nodes[i].op.to_string(), graph.nodes[i].inputs.clone());
        if let Some(&first) = seen.get(&key) {
            replace_uses(&mut graph, out, first);
        } else {
            seen.insert(key, out);
        }
    }

    dce(graph)
}
