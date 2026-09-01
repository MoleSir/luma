//! 内存规划演示：trace 一个 3 层 MLP → 前端优化 → `plan_memory`，
//! 对比"每个中间变量独占一块"（naive）与"活跃区间不重叠则共享一块"（planned）。
//!
//! Run with: `cargo run -p luma-compile --release --example mem`

use luma_compile::backend::mem::plan_memory;
use luma_compile::trace;
use luma_macros::Module;
use luma_nn::{Linear, ModuleForward, NnResult};
use luma_tensor::dtype::FloatDType;
use luma_tensor::{Cpu, DType, Device, Float, Tensor};

#[derive(Module)]
struct MLP<D: Device> {
    l1: Linear<D>,
    l2: Linear<D>,
    l3: Linear<D>,
}

impl<D: Device> ModuleForward<D> for MLP<D> {
    type Input = Tensor<D, Float>;
    type Output = Tensor<D, Float>;

    fn forward(&self, x: &Tensor<D, Float>) -> NnResult<Tensor<D, Float>> {
        let h1 = self.l1.forward(x)?.relu()?;
        let h2 = self.l2.forward(&h1)?.relu()?;
        self.l3.forward(&h2)
    }
}

fn elem_bytes(dt: DType) -> usize {
    match dt {
        DType::F32 | DType::I32 | DType::U32 => 4,
        DType::F64 => 8,
        DType::U8 | DType::Bool => 1,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mlp = MLP {
        l1: Linear::new(128, 256, true, Cpu::default())?,
        l2: Linear::new(256, 256, true, Cpu::default())?,
        l3: Linear::new(256, 64, true, Cpu::default())?,
    };
    let x = Tensor::<Cpu>::from_slice(&[0.5f64; 128], (1, 128), FloatDType::F32)?;

    let graph = trace(&mlp, &x)?;
    let mut g = graph.lock().unwrap().clone();
    println!("traced: {} nodes, {} values", g.nodes.len(), g.values.len());
    g.optimize()?;
    println!("optimized: {} nodes, {} values", g.nodes.len(), g.values.len());

    let plan = plan_memory(&g)?;

    // naive：中间变量各占一块
    let naive: usize = g
        .values
        .iter()
        .filter(|v| v.data.is_none() && !g.inputs.contains(&v.id) && !g.outputs.contains(&v.id))
        .map(|v| elem_bytes(v.dtype) * v.shape.element_count())
        .sum();

    // planned：每组 block_count 块
    let planned: usize = plan.iter().map(|grp| grp.block_count * elem_bytes(grp.dtype) * grp.element_count).sum();

    println!("\nintermediate values → {} groups:", plan.len());
    for grp in &plan {
        println!(
            "  {:?}[{}]: {:>2} values → {:>2} blocks  ({:>8} bytes)",
            grp.dtype,
            grp.element_count,
            grp.tensor_map.len(),
            grp.block_count,
            grp.block_count * elem_bytes(grp.dtype) * grp.element_count,
        );
    }
    println!("\nnaive:   {:>10} bytes", naive);
    println!("planned: {:>10} bytes", planned);
    println!("reduction: {:.2}x", naive as f64 / planned.max(1) as f64);
    Ok(())
}
