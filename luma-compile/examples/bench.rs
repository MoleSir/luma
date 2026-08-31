//! Baseline benchmark: direct module forward vs. traced graph execution.
//!
//! Compares three paths for a Linear(512, 512):
//!   1. direct forward on the module (the op layer, no graph);
//!   2. traced graph **without** front-end optimization;
//!   3. traced graph **with** `Graph::optimize`.
//!
//! Run with: `cargo run -p luma-compile --release --example bench`

use std::time::{Duration, Instant};

use luma_compile::trace;
use luma_nn::Linear;
use luma_tensor::dtype::FloatDType;
use luma_tensor::{Cpu, Tensor};

const ITERS: usize = 100;

fn time<F: FnMut()>(mut f: F) -> Duration {
    f(); // warmup: page faults, lazy constant materialisation, gemm init
    let start = Instant::now();
    for _ in 0..ITERS {
        f();
    }
    start.elapsed() / ITERS as u32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let linear = Linear::new(512, 1024, true, Cpu)?;
    let x = Tensor::<Cpu>::from_slice(&[0.5f64; 512 * 16], (16, 512), FloatDType::F32)?;

    // 1. direct forward
    let direct = time(|| {
        let _ = linear.forward(&x).unwrap();
    });

    // 2/3. trace once, then compile with/without optimization
    let graph = trace(&linear, &x)?;
    let mut g = graph.lock().unwrap().clone();
    println!("traced nodes: {}", g.nodes.len());

    let mut exec_raw = g.clone().compile(&Cpu)?;
    let raw_run = time(|| {
        let _ = exec_raw.run(&[x.clone().into()]).unwrap();
    });

    g.optimize()?;
    println!("optimized nodes: {}", g.nodes.len());
    let mut exec_opt = g.compile(&Cpu)?;
    let opt_run = time(|| {
        let _ = exec_opt.run(&[x.clone().into()]).unwrap();
    });

    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    println!("\n{ITERS} iterations (warmup + avg):");
    println!("  direct forward   {:>8.3} ms/iter", ms(direct));
    println!("  graph (raw)      {:>8.3} ms/iter", ms(raw_run));
    println!("  graph (optimized){:>8.3} ms/iter", ms(opt_run));
    println!("\n  optimize speedup: {:.2}x   (graph vs direct: {:.2}x)", ms(raw_run) / ms(opt_run), ms(opt_run) / ms(direct));
    Ok(())
}
