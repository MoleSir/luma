//! Trace a `Linear + ReLU` forward pass into a graph without running any kernel.
//!
//! Run with: `cargo run -p luma-compile --example linear`

use luma_compile::{Trace, Traced};
use luma_nn::functional::linear;
use luma_tensor::dtype::FloatDType;
use luma_tensor::{Float, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // One tracing device; every clone shares the same graph.
    let trace = Trace::new();
    let opts = (&trace, FloatDType::F32);

    // Build example inputs/parameters on the *trace* device. These are recorded
    // as leaves (no data is ever stored or computed).
    let x = Tensor::<Trace, Float>::full(&[2, 3], 1.0, opts)?;
    let w = Tensor::<Trace, Float>::full(&[4, 3], 1.0, opts)?;
    let b = Tensor::<Trace, Float>::full(&[4], 0.0, opts)?;

    // The same `luma_nn::functional::linear` used for real training, now tracing.
    let y = linear(&x, &w, Some(&b))?;
    let y = y.relu()?;

    // Mark graph inputs/outputs and print the recorded graph.
    let graph = trace.graph();
    {
        let mut g = graph.lock().unwrap();
        g.inputs = vec![x.trace_id(), w.trace_id(), b.trace_id()];
        g.outputs = vec![y.trace_id()];
        println!("{g}");
    }

    // Sanity check: shapes were inferred symbolically, no kernel ran.
    assert_eq!(y.shape().dims(), &[2, 4]);
    println!("\nOK: traced linear+relu produced a graph (output shape {:?})", y.shape());
    Ok(())
}
