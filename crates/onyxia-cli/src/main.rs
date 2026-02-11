//! Onyxia CLI - test ONNX models, generate dot graphs, benchmark execution.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
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
