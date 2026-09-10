use crate::{CompileResult, Graph, ValueId};
use luma_tensor::DType;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// 一个 (dtype, 元素数) 组的内存规划结果。
///
/// 同组 = 同 dtype 同元素数 = 字节数相同（定宽 dtype），block 可以互换。
/// 按 dtype 分组正好对应 executor 按 kind 分区的 slot 数组——f32/i32 跨 kind
/// 共享会破坏 typed slot 模型，也没有意义。
#[derive(Debug)]
pub struct MemoryGroup {
    pub dtype: DType,
    pub element_count: usize,
    /// value id -> block id（组内编号，从 0 起）
    pub tensor_map: HashMap<ValueId, usize>,
    /// 该组需要的 block 总数
    pub block_count: usize,
}

/// 每个 value 的最后一次使用（node 下标）。纯图分析，executor 的死槽清理和
/// plan_memory 共用——按 node 执行顺序遍历，覆盖语义 = 最后一次使用。
pub(crate) fn last_use(graph: &Graph) -> Vec<Option<usize>> {
    let mut last_use: Vec<Option<usize>> = vec![None; graph.values.len()];
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        for &input_tensor_id in node.inputs.iter() {
            last_use[input_tensor_id] = Some(node_idx);
        }
    }
    last_use
}

/// 对整张图做内存规划：每个中间值收集 (创建, 最后使用) 区间，按
/// (dtype, 元素数) 分组，组内贪心复用 block。
///
/// 输入/输出/常量排除在外：常量要活过所有 run，输出要活到 run 结束，
/// 输入由调用方持有——都不能参与复用。
pub fn plan_memory(graph: &Graph) -> CompileResult<Vec<MemoryGroup>> {
    // value id 是稠密索引，用 Vec 而不是 HashMap（和 opt/util.rs 的惯例一致）。
    let mut first_create: Vec<Option<usize>> = vec![None; graph.values.len()];
    let last_use = last_use(graph);

    for (node_idx, node) in graph.nodes.iter().enumerate() {
        // 如果图合法，每个 tensor 作为 node 输出只有一次，表示第一次创建时间
        for &output_tensor_id in node.outputs.iter() {
            first_create[output_tensor_id] = Some(node_idx);
        }
    }

    let mut groups: HashMap<(DType, usize), Vec<(ValueId, usize, usize)>> = HashMap::new();
    for value in graph.values.iter() {
        let value_idx = value.id;
        // 排除输入/输出/常量，只优化中间变量
        if value.data.is_some() || graph.inputs.contains(&value_idx) || graph.outputs.contains(&value_idx) {
            continue;
        }
        // 尸体/悬空值（compile 前不保证 dce 过）：缺创建或使用记录就跳过，
        // 分析不能对奇怪的图 panic。
        let Some(created) = first_create.get(value_idx).copied().flatten() else { continue };
        let Some(dead) = last_use.get(value_idx).copied().flatten() else { continue };
        // dtype / 元素数量相同的内存占用相同！
        let key = (value.dtype, value.shape.element_count());
        groups.entry(key).or_default().push((value_idx, created, dead));
    }

    // 对每个组做块复用规划
    let mut result = Vec::new();
    for ((dtype, element_count), tensor_lives) in groups {
        let (tensor_map, block_count) = pack_blocks(tensor_lives)?;
        result.push(MemoryGroup { dtype, element_count, tensor_map, block_count });
    }

    Ok(result)
}

/// 对一组 dtype、size 相同的 value 做块复用规划（interval packing）。
///
/// 每个值表示 `(value_id, birth, death)`：由 birth 节点创建，在 death 节点
/// 最后一次被使用。按 birth 排序后贪心：death 时释放 block，可复用给
/// birth >= death 的新值。
///
/// **边界共享（death == birth）依赖一个 executor 不变量**：每个 Step 先读完
/// 所有输入、再给输出槽赋值（函数式内核、无 in-place op；view 共享 Arc 存储，
/// 槽位替换也安全）。它让顺序链 x→a→b→c 只占 1 个 block。如果将来出现
/// 流式/原地写内核，这里要收紧成 `death < birth`。
pub(crate) fn pack_blocks(tensor_lives: Vec<(ValueId, usize, usize)>) -> CompileResult<(HashMap<ValueId, usize>, usize)> {
    // 按照 birth 进行排序
    let mut tensor_lives = tensor_lives;
    tensor_lives.sort_by_key(|&(_, birth, _)| birth);

    type BlockId = usize;
    type Death = usize;

    let mut free_blocks: Vec<BlockId> = vec![]; // 空闲的 block
    let mut alloc_block_count = 0; // 申请的 block 总数
    let mut tensor_map: HashMap<ValueId, BlockId> = HashMap::new(); // tensor -> block
    // 存活 tensor 的 (death, block)：BinaryHeap 是最大堆，Reverse 包一层变最小堆。
    // 堆顶即最小的 death——谁死得早谁先释放。O(n log n) vs 原版每轮全扫 O(n²)。
    let mut livings: BinaryHeap<Reverse<(Death, BlockId)>> = BinaryHeap::new();

    // 按创建顺序处理
    for (tensor_id, birth, death) in tensor_lives {
        // 时间来到 birth：把 death <= birth 的 block 全部释放
        while let Some(Reverse((d, block_id))) = livings.peek() {
            if *d <= birth {
                free_blocks.push(*block_id);
                livings.pop();
            } else {
                break; // 堆顶都还没死，后面的更不可能死
            }
        }

        // 为 tensor 分配一个 block：优先复用空闲的（LIFO，刚释放的还在缓存里），否则新开
        let block_id = match free_blocks.pop() {
            Some(block_id) => block_id,
            None => {
                let new_block_id = alloc_block_count;
                alloc_block_count += 1;
                new_block_id
            }
        };

        tensor_map.insert(tensor_id, block_id);
        livings.push(Reverse((death, block_id)));
    }

    Ok((tensor_map, alloc_block_count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NodeOp;
    use luma_tensor::{BinaryOp, FloatUnaryOp, Shape};

    fn sqr(g: &mut Graph, input: ValueId) -> ValueId {
        g.add_node(NodeOp::FloatUnary(FloatUnaryOp::Sqr), vec![input], DType::F32, Shape::from((2, 2)))
    }

    fn add2(g: &mut Graph, input: ValueId) -> ValueId {
        g.add_node(NodeOp::Binary(BinaryOp::Add), vec![input, input], DType::F32, Shape::from((2, 2)))
    }

    fn group<'a>(plan: &'a [MemoryGroup], dtype: DType, count: usize) -> &'a MemoryGroup {
        plan.iter().find(|g| g.dtype == dtype && g.element_count == count).expect("group exists")
    }

    /// 两条顺序链：a[0,1] 与 b[2,3] 活跃区间不相交，共享 1 个 block。
    #[test]
    fn disjoint_chains_share_block() {
        let mut g = Graph::default();
        let x = g.add_value(DType::F32, Shape::from((2, 2)));
        g.mark_input(x);

        let a = sqr(&mut g, x); // step 0, death 1
        let o1 = add2(&mut g, a); // step 1
        g.mark_output(o1);
        let b = sqr(&mut g, x); // step 2, death 3
        let o2 = add2(&mut g, b); // step 3
        g.mark_output(o2);

        let plan = plan_memory(&g).unwrap();
        let grp = group(&plan, DType::F32, 4);
        assert_eq!(grp.tensor_map.len(), 2, "只有 a、b 是中间变量");
        assert_eq!(grp.block_count, 1, "a 与 b 不相交，应共享一个 block");
        assert_eq!(grp.tensor_map[&a], grp.tensor_map[&b]);
    }

    /// 顺序链 x→a→b→c→d→out：每个中间值死在下一个诞生那步（death == birth），
    /// 边界共享让整条链只占 1 个 block。
    #[test]
    fn chain_shares_one_block() {
        let mut g = Graph::default();
        let x = g.add_value(DType::F32, Shape::from((2, 2)));
        g.mark_input(x);

        let a = sqr(&mut g, x); // step 0, death 1
        let b = sqr(&mut g, a); // step 1, death 2
        let c = sqr(&mut g, b); // step 2, death 3
        let d = sqr(&mut g, c); // step 3, death 4
        let o = add2(&mut g, d); // step 4
        g.mark_output(o);

        let plan = plan_memory(&g).unwrap();
        let grp = group(&plan, DType::F32, 4);
        assert_eq!(grp.block_count, 1, "death == birth 边界共享让链只占一个 block");
    }

    /// 同时存活的中间值不能共享：a[0,2] 与 b[1,3] 重叠 → 2 个 block。
    #[test]
    fn overlapping_values_get_separate_blocks() {
        let mut g = Graph::default();
        let x = g.add_value(DType::F32, Shape::from((2, 2)));
        g.mark_input(x);

        let a = sqr(&mut g, x); // step 0, death 2
        let b = sqr(&mut g, x); // step 1, death 3
        let o1 = add2(&mut g, a); // step 2
        let o2 = add2(&mut g, b); // step 3
        g.mark_output(o1);
        g.mark_output(o2);

        let plan = plan_memory(&g).unwrap();
        let grp = group(&plan, DType::F32, 4);
        assert_eq!(grp.block_count, 2, "活跃区间重叠，不能共享");
    }

    /// 输入/输出/常量排除在规划外，只有中间变量进组。
    #[test]
    fn excludes_inputs_outputs_constants() {
        let mut g = Graph::default();
        let x = g.add_value(DType::F32, Shape::from((2, 2)));
        g.mark_input(x);
        let k = g.add_constant(DType::F32, Shape::from((2, 2)), vec![0; 16]);
        let a = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![x, k], DType::F32, Shape::from((2, 2)));
        let o = add2(&mut g, a);
        g.mark_output(o);

        let plan = plan_memory(&g).unwrap();
        let grp = group(&plan, DType::F32, 4);
        assert_eq!(grp.tensor_map.len(), 1, "只有 a 是中间变量");
        assert!(!grp.tensor_map.contains_key(&x), "输入不参与");
        assert!(!grp.tensor_map.contains_key(&k), "常量不参与");
        assert!(!grp.tensor_map.contains_key(&o), "输出不参与");
    }

    /// 尸体（输出从未被消费）不 panic、不进组。
    #[test]
    fn corpse_values_skipped() {
        let mut g = Graph::default();
        let x = g.add_value(DType::F32, Shape::from((2, 2)));
        g.mark_input(x);

        let dead = sqr(&mut g, x); // 从未被使用：有创建无使用
        let a = sqr(&mut g, x);
        let o = add2(&mut g, a);
        g.mark_output(o);

        let plan = plan_memory(&g).unwrap();
        let grp = group(&plan, DType::F32, 4);
        assert!(!grp.tensor_map.contains_key(&dead), "尸体跳过");
        assert_eq!(grp.tensor_map.len(), 1);
    }

    /// (dtype, 元素数) 都相同才共享内存：三种组合分成三组。
    #[test]
    fn groups_by_dtype_and_element_count() {
        let mut g = Graph::default();
        let x = g.add_value(DType::F32, Shape::from((2, 2)));
        let y = g.add_value(DType::I32, Shape::from((3,)));
        let z = g.add_value(DType::F32, Shape::from((1, 2)));
        g.mark_input(x);
        g.mark_input(y);
        g.mark_input(z);

        let a = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![x, x], DType::F32, Shape::from((2, 2))); // (F32, 4)
        let oa = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![a, a], DType::F32, Shape::from((2, 2)));
        g.mark_output(oa);
        let b = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![y, y], DType::I32, Shape::from((3,))); // (I32, 3)
        let ob = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![b, b], DType::I32, Shape::from((3,)));
        g.mark_output(ob);
        let c = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![z, z], DType::F32, Shape::from((1, 2))); // (F32, 2)
        let oc = g.add_node(NodeOp::Binary(BinaryOp::Add), vec![c, c], DType::F32, Shape::from((1, 2)));
        g.mark_output(oc);

        let plan = plan_memory(&g).unwrap();
        assert_eq!(plan.len(), 3, "(F32,4) (I32,3) (F32,2) 互不相同");
        for grp in &plan {
            assert_eq!(grp.block_count, 1);
        }
    }
}
