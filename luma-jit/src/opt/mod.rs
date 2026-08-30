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
