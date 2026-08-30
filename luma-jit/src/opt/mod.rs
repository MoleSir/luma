pub mod cse;
pub mod dce;
pub mod fold;
pub mod simplify;

#[cfg(test)]
mod tests;
pub mod util;
pub mod verify;

use crate::{Graph, JitResult};

/// 前端优化流水线：simplify → fold → simplify → cse → dce → verify。
///
/// 每个 pass 内联 dce 收尸（simplify/fold/cse 只做替换，尸体统一清理）。
/// fold 后可能暴露新的恒等模式，所以 simplify 跑两遍。
pub fn optimize(graph: Graph) -> JitResult<Graph> {
    let graph = simplify::simplify(graph)?;
    let graph = fold::fold(graph)?;
    let graph = simplify::simplify(graph)?;
    let graph = cse::cse(graph)?;
    let graph = dce::dce(graph)?;
    verify::verify(&graph)?;
    Ok(graph)
}

/*

因果链
    
    simplify → fold → simplify → cse → dce → verify

1. simplify 第一：先缩小图
    本地化简（恒等元、空转 cast、对消）把图变小——后面所有 pass 都少干活。更关键：fold 不会白算。如果 fold 在前，(1+0) + (2+3) 里 fold
    会执行整个常量子图（包括 1+0 这个本可消除的冗余计算）；simplify 先消掉它，fold 只算必要的。

2. fold 第二：折叠常量子图
    simplify 后的图做常量传播 + 执行折叠，产出新常量叶子。

3. simplify 第三（关键）：fold 暴露的新机会
    fold 产出的常量会开启新的恒等模式：

    x * (2 / 2)    → fold 折叠 2/2 → 常量 1 → simplify 的 x * 1 → x
    x + (3 - 3)    → fold 折叠 → 常量 0 → simplify 的 x + 0 → x
    (x + 1) + 2    → fold 无操作，但标量合并是 simplify 自己的事

    没有这第二轮，图里就永远留着 x * 1。fold 不产生恒等，但它产生"触发恒等规则的常量"——所以 fold 后面必须跟一轮 simplify。

4. cse 第四：规范化 + 折叠让重复"可见"
    CSE 的 key 是 (op, inputs)——前面的 pass 让原本 key 不同的重复变成 key 相同：

    simplify 的标量合并: (x+1)+2 → x+3 与 (x+3)+0 → x+3 → 两个 key 相同的节点 → cse 合并
    fold 后:            两个结构不同但结果相同的常量子图 → 折叠成相同常量 → cse 合并
    
    如果 cse 在前，这些"后生成的重复"就漏掉了。而且图越小 key 空间越小。注意 cse 不创造新机会（只合并、不产生新恒等）——所以 cse 之后不需要再跑 simplify。

5. dce 最后：统一收尸
    所有 pass 只做替换、制造尸体（simplify/fold/cse 内部都内联了一次 dce，但那是各自图的收尾）。pipeline 末尾再清一次保证输出干净。为什么不在中间：中间跑
    dce 的话，后面 pass 又制造新尸体，还得再跑——收尾一次足够。而且尸体跳过逻辑（use_counts）让后续 pass 对尸体免疫，不跑 dce 也正确。
    
6. verify 最后：安全网
    确认优化后的图满足所有不变量（SSA 位置拓扑、叶子三分法）——pipeline 输出保证合法。

换顺序会怎样？

- fold 放第一：正确但更差——fold 执行包含冗余的常量子图（多算），且简化创造的机会（x*1）要等 fold 后那轮才触发……其实也收敛，但白算。
- cse 放第一：正确但漏机会——简化/折叠"创造"的重复合并不到。
- 多跑几轮：fold → simplify 是唯一需要"接力"的对；第三轮 simplify 无新机会（cse 不创造恒等），所以三轮是"够用"的平衡点。

一句话总结：①② 是"先小后算"，②③ 是"折叠喂化简"的接力，③④ 是"规范化让重复可见"，⑤⑥ 是收尾。

  ▎ "这个顺序不是拍脑袋定的，核心思路是：每个 pass 的产出，要为下一个 pass 创造机会，并且让下一个 pass 更便宜。
  ▎
  ▎ 首先 simplify 先行——它是本地化简，把图缩小，这样后面所有 pass 都在更小的图上工作；更重要的是避免 fold 白算，比如 (1+0)+(2+3)，如果先 fold，它会执行 
  ▎ 1+0 这个本可消除的冗余计算。
  ▎
  ▎ 然后 fold 折叠常量子图。关键在第三步：fold 之后再跑一轮 simplify——因为 fold 会产生新常量，而这些常量会触发恒等规则。最经典的例子：x * (2/2)，fold 把 
  ▎ 2/2 算成常量 1，第二轮 simplify 才能把 x*1 消成 x。如果没有这轮接力，图里就永远留着一个 x*1。
  ▎
  ▎ CSE 放在第四——它的 key 是 (op, inputs)，前面的规范化让原本看起来不同的重复变成相同的 key：比如标量合并把 (x+1)+2 和 (x+3)+0 都变成 x+3，这时 CSE 
  ▎ 才能合并它们。反过来，CSE 不产生新的恒等机会，所以它后面不需要再跟 simplify。
  ▎
  ▎ 最后 DCE 统一收尸——前面所有 pass 只做替换、制造死节点，收尾一次清干净；verify 做最终合法性检查。
  ▎
  ▎ 一句话：先小后算，折叠喂化简，规范化让重复可见，最后统一收尾。"

*/