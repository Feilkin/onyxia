//! Model validation: parse and lower to IR, without touching a GPU.
//!
//! Lowering is the real gate — it resolves every ONNX op against the
//! lowering registry, folds shape subgraphs, runs symbolic shape inference,
//! and validates the resulting module. A model that lowers cleanly will
//! prepare on any backend that has kernels/decompositions for its nodes.

use anyhow::{Context, Result};
use onyxia_ir::NodeKind;
use onyxia_onnx::Graph;
use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::path::Path;

/// Validate a model and report statistics.
pub fn validate_model(model_path: &Path, verbose: bool) -> Result<()> {
    println!("Validating model: {}\n", model_path.display());

    // Load model
    print_step("Loading model", verbose);
    let model =
        onyxia_onnx::load_and_parse_model(model_path).context("Failed to load and parse model")?;
    print_success();
    print_model_stats(&model);

    if verbose {
        print_operator_summary(&model);
    }

    // Lower to IR: registry resolution, attribute normalization, shape
    // subgraph folding, symbolic shape inference, module validation.
    print_step("Lowering to IR", verbose);
    let onnx_nodes = model.nodes.len();
    let module = onyxia_lower::lower(model, &onyxia_lower::standard_registry())
        .context("Lowering failed")?;
    print_success();

    // Report what came out.
    let mut prims = 0usize;
    let mut composites: BTreeMap<&str, usize> = BTreeMap::new();
    for node in &module.nodes {
        match &node.kind {
            NodeKind::Prim(_) => prims += 1,
            NodeKind::Composite(c) => *composites.entry(c.name.as_str()).or_insert(0) += 1,
        }
    }
    let composite_total: usize = composites.values().sum();

    println!("\nLowered module:");
    println!(
        "  - {} ONNX nodes → {} IR nodes ({} primitives + {} composites)",
        onnx_nodes,
        module.nodes.len(),
        prims,
        composite_total
    );
    println!("  - {} values, {} constants", module.values.len(), module.consts.len());
    if !module.symbols.is_empty() {
        let names: Vec<&str> = module.symbols.names().collect();
        println!("  - dim symbols: {}", names.join(", "));
    }
    if verbose && !composites.is_empty() {
        println!("  - composites:");
        for (name, count) in &composites {
            println!("      {} × {}", count, name);
        }
    }

    print_final_success();
    Ok(())
}

fn print_step(name: &str, verbose: bool) {
    if verbose {
        print!("  {}... ", name);
        std::io::stdout().flush().ok();
    }
}

fn print_success() {
    println!("✓");
}

fn print_final_success() {
    println!("\nValidation passed! Model lowers cleanly.");
}

fn print_model_stats(model: &Graph) {
    println!("✓ Model loaded successfully");
    println!("  - {} nodes", model.nodes.len());
    println!("  - {} tensors", model.tensor_info.len());

    // Count unique operator types
    let op_types: HashSet<_> = model.nodes.iter().map(|n| &n.op_type).collect();
    println!("  - {} operator types\n", op_types.len());
}

fn print_operator_summary(model: &Graph) {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for node in &model.nodes {
        *counts.entry(node.op_type.clone()).or_insert(0) += 1;
    }

    let mut sorted: Vec<_> = counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));

    println!("  Top operators:");
    for (op_type, count) in sorted.iter().take(10) {
        println!("    {} × {}", count, op_type);
    }
}
