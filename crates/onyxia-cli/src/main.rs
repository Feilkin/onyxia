//! Onyxia CLI - test ONNX models, generate dot graphs, benchmark execution.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use onyxia_onnx::TensorShape;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "onyxia")]
#[command(about = "GPU compute shader runtime for ONNX models", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a Graphviz DOT file from an ONNX model
    Dot {
        /// Path to the ONNX model file
        #[arg(value_name = "MODEL")]
        model: PathBuf,

        /// Output file path (defaults to stdout)
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Simplification level: full, layers, summary
        #[arg(short, long, default_value = "full")]
        simplify: String,
    },
    /// Inspect an ONNX model's structure
    Inspect {
        /// Path to the ONNX model file
        #[arg(value_name = "MODEL")]
        model: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Dot {
            model,
            output,
            simplify,
        } => {
            cmd_dot(model, output, &simplify)?;
        }
        Commands::Inspect { model } => {
            cmd_inspect(model)?;
        }
    }

    Ok(())
}

/// Generate DOT format from ONNX model.
fn cmd_dot(model_path: PathBuf, output_path: Option<PathBuf>, simplify: &str) -> Result<()> {
    // Load the ONNX model
    let model = onyxia_onnx::load_model(&model_path)
        .with_context(|| format!("Failed to load model from {}", model_path.display()))?;

    // Convert to DOT format
    let simplify_level = match simplify {
        "layers" => onyxia_onnx::DotSimplification::Layers,
        "summary" => onyxia_onnx::DotSimplification::Summary,
        _ => onyxia_onnx::DotSimplification::Full,
    };
    let dot = onyxia_onnx::to_dot_with_options(&model, simplify_level);

    // Write to output (file or stdout)
    if let Some(output_path) = output_path {
        std::fs::write(&output_path, dot)
            .with_context(|| format!("Failed to write DOT output to {}", output_path.display()))?;
        eprintln!("Wrote DOT output to {}", output_path.display());
    } else {
        print!("{}", dot);
    }

    Ok(())
}

/// Inspect an ONNX model's structure.
fn cmd_inspect(model_path: PathBuf) -> Result<()> {
    // Load the ONNX model
    let model_proto = onyxia_onnx::load_model(&model_path)
        .with_context(|| format!("Failed to load model from {}", model_path.display()))?;

    // Parse into graph structure
    let mut model =
        onyxia_onnx::parse_model(&model_proto).with_context(|| "Failed to parse model")?;

    // Infer shapes for analysis using planner's kernel-based inference
    let registry = onyxia_planner::KernelRegistry::with_defaults();
    let dynamic_dims = std::collections::HashMap::new(); // Empty for inspection
    onyxia_planner::infer_shapes(&mut model, &registry, &dynamic_dims)
        .with_context(|| "Failed to infer shapes")?;

    println!("Model: {}", model.metadata.name);
    println!("  IR version: {}", model.metadata.ir_version);
    println!("  Producer: {}", model.metadata.producer_name);
    println!("  Nodes: {}", model.nodes.len());
    println!("  Tensors: {}", model.tensor_info.len());
    println!();

    println!("Inputs ({}):", model.inputs.len());
    for input_name in &model.inputs {
        if let Some(&tensor_id) = model.tensors.get(input_name) {
            let info = &model.tensor_info[tensor_id];
            println!("  {} - {:?} {:?}", info.name, info.dtype, info.shape);
        }
    }
    println!();

    println!("Outputs ({}):", model.outputs.len());
    for output_name in &model.outputs {
        if let Some(&tensor_id) = model.tensors.get(output_name) {
            let info = &model.tensor_info[tensor_id];
            println!("  {} - {:?} {:?}", info.name, info.dtype, info.shape);
        }
    }
    println!();

    // Count tensors by shape type
    let mut unknown_count = 0;
    let mut known_count = 0;
    let mut unknown_names = Vec::new();

    for info in &model.tensor_info {
        match &info.shape {
            TensorShape::Unknown | TensorShape::Absent => {
                unknown_count += 1;
                if unknown_names.len() < 5 {
                    unknown_names.push(info.name.clone());
                }
            }
            _ => known_count += 1,
        }
    }

    println!("Shape statistics:");
    println!("  Known shapes: {}", known_count);
    println!("  Unknown shapes: {}", unknown_count);
    if !unknown_names.is_empty() {
        println!(
            "  First unknown: {:?}",
            &unknown_names[..unknown_names.len().min(5)]
        );
    }
    println!();

    // Show first few operations
    println!("First 20 operations:");
    for (i, node) in model.nodes.iter().take(20).enumerate() {
        println!("  {}. {} ({})", i + 1, node.name, node.op_type);
        println!("     Inputs: {:?}", node.inputs);
        println!("     Outputs: {:?}", node.outputs);
    }
    println!();

    // Find operations that use embedding table
    println!("Operations using 'embed_tokens':");
    for (i, node) in model.nodes.iter().enumerate() {
        if node.inputs.iter().any(|inp| inp.contains("embed_tokens"))
            || node.outputs.iter().any(|out| out.contains("embed_tokens"))
        {
            println!("  {}. {} ({})", i + 1, node.name, node.op_type);
            println!("     Inputs: {:?}", node.inputs);
            println!("     Outputs: {:?}", node.outputs);
        }
    }

    Ok(())
}
