//! Shared helpers for the optimization passes.

use crate::{Graph, ValueId};

/// `producer[v]` = index of the node that produces value `v` (`None` for
/// leaves: graph inputs and constants). Used by `dce` (reverse reachability)
/// and by the multi-node peephole rules in `simplify`.
pub(crate) fn build_producer(graph: &Graph) -> Vec<Option<usize>> {
    let mut producer = vec![None; graph.values.len()];
    for (i, node) in graph.nodes.iter().enumerate() {
        for &out in &node.outputs {
            producer[out] = Some(i);
        }
    }
    producer
}

/// `use_counts[v]` = how many nodes consume value `v` (graph outputs are *not*
/// counted — check them separately). A node whose output has zero users and is
/// not a graph output is a *corpse*: its producer chain is stale, so rewriting
/// rules that create new nodes must skip it or they loop forever.
pub(crate) fn build_use_counts(graph: &Graph) -> Vec<usize> {
    let mut counts = vec![0; graph.values.len()];
    for node in &graph.nodes {
        for &input in &node.inputs {
            counts[input] += 1;
        }
    }
    counts
}

/// Restore SSA execution order after a rewrite pass appended new nodes.
///
/// Rewriting rules `add_node` (append) so mid-graph consumers keep their
/// indices stable; but a new node consumed by an existing mid-graph node then
/// *executes after* its consumer, and the executor would read an empty slot.
/// This moves every node to just after its last input producer, so the array
/// order is a valid topological order again. Iterates until stable; graph sizes
/// here are small, so the quadratic worst case is fine.
pub(crate) fn topo_sort(graph: &mut Graph) {
    loop {
        let producer = build_producer(graph);
        let mut moved = false;
        for i in 0..graph.nodes.len() {
            // 输入生产者的最大位置
            let mut last_p = 0usize;
            for &input in &graph.nodes[i].inputs {
                if let Some(p) = producer[input] {
                    last_p = last_p.max(p);
                }
            }
            if last_p > i {
                let node = graph.nodes.remove(i);
                graph.nodes.insert(last_p, node);
                moved = true;
                break; // 重排后重新扫描
            }
        }
        if !moved {
            break;
        }
    }
}

/// 将 graph 中所有结点的 from 替换为 to，具体表现在对 node 的输入，如果是 from 替换为 to
pub(crate) fn replace_uses(graph: &mut Graph, from: ValueId, to: ValueId) {
    for node in graph.nodes.iter_mut() {
        for input in node.inputs.iter_mut() {
            if *input == from {
                *input = to;
            }
        }
    }

    for out in graph.outputs.iter_mut() {
        if *out == from {
            *out = to;
        }
    }
}
