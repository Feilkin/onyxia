//! Whole-model lowering gate (plan step B5).
//!
//! Run in a worktree that has the model files:
//!
//! ```sh
//! cargo run -p onyxia-lower --example lower-stats -- \
//!     models/gemma-3-270m-it-ONNX/onnx/model.onnx
//! ```
//!
//! Success criteria: lowering completes with zero unresolved ops, every
//! value has an inferred (possibly symbolic) shape, and the stats show the
//! ONNX shape plumbing folded away (ir_nodes well below onnx_nodes).

use onyxia_lower::{lower_with_stats, standard_registry};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: lower-stats <model.onnx> [--dot <out.dot>]")?;
    let dot_out = {
        let args: Vec<String> = std::env::args().collect();
        args.iter()
            .position(|a| a == "--dot")
            .and_then(|i| args.get(i + 1).cloned())
    };

    eprintln!("parsing {path} …");
    let graph = onyxia_onnx::load_and_parse_model(&path)?;

    eprintln!("lowering …");
    let started = std::time::Instant::now();
    let (module, stats) = lower_with_stats(graph, &standard_registry())?;
    let elapsed = started.elapsed();

    println!("lowering ok in {elapsed:.2?}");
    println!("  onnx nodes:      {}", stats.onnx_nodes);
    println!(
        "  ir nodes:        {} ({} composites, {} primitives)",
        stats.ir_nodes,
        stats.composite_nodes,
        stats.ir_nodes - stats.composite_nodes
    );
    println!(
        "  folded away:     {} nodes ({:.1}%)",
        stats.onnx_nodes.saturating_sub(stats.ir_nodes),
        100.0 * stats.onnx_nodes.saturating_sub(stats.ir_nodes) as f64
            / stats.onnx_nodes.max(1) as f64
    );
    println!(
        "  consts:          {} entries, {:.1} MiB",
        stats.consts.0,
        stats.consts.1 as f64 / (1024.0 * 1024.0)
    );
    println!("  dim symbols:     {}", stats.symbols);
    println!("  inputs:          {}", module.inputs.len());
    println!("  outputs:         {}", module.outputs.len());

    // Shape coverage: how many values still have fully-unresolved dims.
    let mut symbolic = 0usize;
    for id in module.value_ids() {
        if !module.value(id).ty.shape.is_static() {
            symbolic += 1;
        }
    }
    println!(
        "  values:          {} total, {symbolic} with symbolic shapes",
        module.values.len()
    );

    // Per-op histogram: what actually remains in the graph.
    let mut histogram: std::collections::HashMap<String, usize> = Default::default();
    for node in &module.nodes {
        let key = match &node.kind {
            onyxia_ir::NodeKind::Prim(p) => p.name().to_string(),
            onyxia_ir::NodeKind::Composite(c) => format!("[{}]", c.name),
        };
        *histogram.entry(key).or_default() += 1;
    }
    let mut entries: Vec<(String, usize)> = histogram.into_iter().collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    println!("  node histogram (composites in brackets):");
    for (name, count) in entries {
        println!("    {count:>5}  {name}");
    }

    if let Some(dot_path) = dot_out {
        std::fs::write(&dot_path, onyxia_ir::dot::to_dot(&module))?;
        println!("  dot written to:  {dot_path}");
    }
    Ok(())
}
