//! Model validation without full compilation.

use anyhow::{Context, Result};
use onyxia_compiler::CompilerPipeline;
use onyxia_core::{IrGraph, OperatorRegistry, Stage, TensorShape};
use onyxia_onnx::{AttributeValue, Graph};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::Path;

/// Validate a model and report any issues.
pub fn validate_model(
    model_path: &Path,
    dynamic_dims: HashMap<String, usize>,
    verbose: bool,
    until_stage: Option<Stage>,
) -> Result<()> {
    println!("Validating model: {}\n", model_path.display());

    // Load model
    print_step("Loading model", verbose);
    let model =
        onyxia_onnx::load_and_parse_model(model_path).context("Failed to load and parse model")?;

    print_success();
    print_model_stats(&model);

    // Check operator support
    print_step("Checking operator support", verbose);
    let registry = onyxia_operators::core_operator_registry();
    check_operator_support(&model, &registry)?;
    print_success();

    if verbose {
        print_operator_summary(&model);
    }

    // Convert to IR
    print_step("Converting to IR", verbose);
    let ir_graph = IrGraph::from_onnx(&model).context("Failed to convert to IR")?;
    print_success();

    // Compile model (simplified pipeline, no staged validation)
    print_step("Compiling model", verbose);
    let mut pipeline = CompilerPipeline::new();

    // Note: Dynamic dimensions no longer passed at compile time in simplified pipeline
    let _ = (dynamic_dims, until_stage); // Suppress unused warnings
    // TODO: Re-evaluate validation workflow when dispatch model is complete

    let _compiled = pipeline
        .compile(&model, &registry)
        .context("Failed to compile model")?;
    print_success();

    // Check for warnings
    let warnings = check_for_warnings(&ir_graph)?;
    if !warnings.is_empty() {
        println!("\n⚠ Warnings:\n");
        for warning in &warnings {
            println!("  {}\n", warning);
        }
        println!("Validation passed with {} warning(s).", warnings.len());
    } else {
        print_final_success();
    }

    Ok(())

    /* Original staged validation - disabled pending dispatch model completion
    // Run compilation stages
    let mut pipeline = CompilerPipeline::new(dynamic_dims.clone());
    let target_stage = until_stage.unwrap_or(Stage::Inference);

    // Resolution
    print_step("Running symbolic resolution", verbose);
    run_stage(&mut pipeline, &mut ir_graph, &registry, Stage::Resolution)?;
    print_success();

    if verbose && !dynamic_dims.is_empty() {
        println!("  Resolved dimensions:");
        for (name, value) in &dynamic_dims {
            println!("    {} = {}", name, value);
        }
    }

    if target_stage == Stage::Resolution {
        print_final_success();
        return Ok(());
    }

    // Folding
    print_step("Running constant folding", verbose);
    run_stage(&mut pipeline, &mut ir_graph, &registry, Stage::Folding)?;
    print_success();

    if target_stage == Stage::Folding {
        print_final_success();
        return Ok(());
    }

    // Inference
    print_step("Running shape inference", verbose);
    match run_stage(&mut pipeline, &mut ir_graph, &registry, Stage::Inference) {
        Ok(_) => {
            print_success();

            // Check for warnings
            let warnings = check_for_warnings(&ir_graph)?;
            if !warnings.is_empty() {
                println!("\n⚠ Warnings:\n");
                for warning in &warnings {
                    println!("  {}\n", warning);
                }
                println!("Validation passed with {} warning(s).", warnings.len());
            } else {
                print_final_success();
            }
        }
        Err(e) => {
            println!("✗ Shape inference failed\n");
            println!("Error: {}\n", e);

            // Try to provide helpful suggestions
            suggest_fixes(&e, &dynamic_dims);

            println!("Validation failed.");
            return Err(e);
        }
    }

    Ok(())
    */
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
    println!("\nValidation passed! Model is ready for compilation.");
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

fn check_operator_support(model: &Graph, registry: &OperatorRegistry) -> Result<()> {
    let mut unsupported = Vec::new();

    for node in &model.nodes {
        if registry.get(&node.op_type).is_none() {
            unsupported.push(&node.op_type);
        }
    }

    if !unsupported.is_empty() {
        unsupported.sort();
        unsupported.dedup();

        println!("✗ Unsupported operators:\n");
        for op_type in unsupported {
            println!("    {}", op_type);
        }

        anyhow::bail!("Model contains unsupported operators");
    }

    Ok(())
}

/* Original helper function - disabled pending dispatch model completion
fn run_stage(
    pipeline: &mut CompilerPipeline,
    graph: &mut IrGraph,
    registry: &OperatorRegistry,
    stage: Stage,
) -> Result<()> {
    pipeline
        .run_until_stage(graph, registry, stage)
        .with_context(|| format!("Failed at stage {:?}", stage))
}
*/

fn check_for_warnings(graph: &IrGraph) -> Result<Vec<String>> {
    let mut warnings = Vec::new();

    // Check for large operations
    for (node_id, node) in graph.nodes() {
        // Check for large matrix multiplications
        if node.op_type == "MatMul"
            && let Some(warning) = check_large_matmul(graph, node_id)?
        {
            warnings.push(warning);
        }

        // Check for many-input concats
        if node.op_type == "Concat"
            && let Some(warning) = check_large_concat(graph, node_id)?
        {
            warnings.push(warning);
        }
    }

    Ok(warnings)
}

fn check_large_matmul(graph: &IrGraph, node_id: onyxia_core::IrNodeId) -> Result<Option<String>> {
    let node = graph.node(node_id)?;

    // Get input shapes
    if node.inputs.len() < 2 {
        return Ok(None);
    }

    let input0 = graph.edge(node.inputs[0])?;
    let input1 = graph.edge(node.inputs[1])?;

    // Check if shapes are static
    let shape0 = match &input0.shape {
        TensorShape::Static(s) => s,
        _ => return Ok(None),
    };

    let shape1 = match &input1.shape {
        TensorShape::Static(s) => s,
        _ => return Ok(None),
    };

    // For matrix multiplication, calculate approximate memory usage
    if shape0.len() >= 2 && shape1.len() >= 2 {
        let m = shape0[shape0.len() - 2];
        let k = shape0[shape0.len() - 1];
        let n = shape1[shape1.len() - 1];

        // Calculate total elements (including batch dims)
        let batch_size0: usize = shape0.iter().take(shape0.len() - 2).product();
        let batch_size1: usize = shape1.iter().take(shape1.len() - 2).product();
        let batch_size = batch_size0.max(batch_size1);

        let output_elements = batch_size * m * n;
        let input_elements = batch_size * (m * k + k * n);

        // Estimate memory in MB (assuming f32 = 4 bytes)
        let memory_mb = (output_elements + input_elements) * 4 / 1_000_000;

        // Warn if memory usage is large
        if memory_mb >= 256 {
            return Ok(Some(format!(
                "Node '{}' (MatMul):\n    Large matrix multiplication detected: {:?} × {:?}\n    May require significant GPU memory (~{} MB for this operation)",
                node.name, shape0, shape1, memory_mb
            )));
        }
    }

    Ok(None)
}

fn check_large_concat(graph: &IrGraph, node_id: onyxia_core::IrNodeId) -> Result<Option<String>> {
    let node = graph.node(node_id)?;

    // Check number of inputs
    let num_inputs = node.inputs.len();

    // Get concatenation axis from attributes
    let axis = node
        .attributes
        .get("axis")
        .and_then(|v| {
            if let AttributeValue::Int(i) = v {
                Some(*i)
            } else {
                None
            }
        })
        .unwrap_or(0);

    // Warn if many inputs on non-zero axis
    if num_inputs >= 64 && axis != 0 {
        return Ok(Some(format!(
            "Node '{}' (Concat):\n    Concatenating {} inputs on non-zero axis (axis={})\n    Consider using fewer concat operations for better performance",
            node.name, num_inputs, axis
        )));
    }

    Ok(None)
}

#[allow(dead_code)] // May be used in future validation improvements
fn suggest_fixes(error: &anyhow::Error, dynamic_dims: &HashMap<String, usize>) {
    let error_str = format!("{:?}", error);

    // Detect common patterns and suggest fixes

    if error_str.contains("Expand") && error_str.contains("dimension") {
        println!("This appears to be a model export issue:");
        println!("  The model has hardcoded constant shapes that conflict with");
        println!("  your dynamic dimensions.");
        println!();
        println!("Suggestions:");
        println!("  - The model may have been exported with a fixed sequence length");

        if let Some(&seq_len) = dynamic_dims.get("sequence_length") {
            // Suggest doubling the sequence length
            println!("  - Try --dynamic-dim sequence_length={}", seq_len * 2);
        }

        println!("  - Use a model export with proper dynamic shapes");
        return;
    }

    if error_str.contains("unsupported operator") || error_str.contains("Unsupported operator") {
        println!("Suggestions:");
        println!("  - Check if there's a newer version of Onyxia with this operator");
        println!("  - File an issue to request operator implementation");
        return;
    }

    if error_str.contains("shape") || error_str.contains("Shape") {
        println!("Suggestions:");
        println!("  - Try different dynamic dimension values");
        println!("  - Use --verbose flag for more details");
        println!("  - Check the model export configuration");
    }
}
