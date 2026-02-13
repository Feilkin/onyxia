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

        /// Filter nodes by name prefix (e.g., "/model/pos_ids_reformat")
        #[arg(long, value_name = "PREFIX")]
        filter: Option<String>,

        /// Maximum depth from filtered nodes to include (0 = only matched nodes)
        #[arg(long, default_value = "0")]
        depth: usize,
    },
    /// Inspect an ONNX model's structure
    Inspect {
        /// Path to the ONNX model file
        #[arg(value_name = "MODEL")]
        model: PathBuf,

        /// Dynamic dimension values (format: name=value, can be repeated)
        #[arg(short = 'd', long = "dynamic-dim")]
        dynamic_dims: Vec<String>,
    },
    /// Run an ONNX model for text generation
    RunModel {
        /// Path to the ONNX model file
        #[arg(value_name = "MODEL")]
        model: PathBuf,

        /// Path to the tokenizer directory (containing tokenizer.json)
        #[arg(short, long, value_name = "TOKENIZER")]
        tokenizer: PathBuf,

        /// Text prompt for generation
        #[arg(short, long)]
        prompt: String,

        /// Maximum number of tokens to generate
        #[arg(long, default_value = "100")]
        max_tokens: usize,

        /// Temperature for sampling (0.0 = greedy, 1.0 = no scaling, >1.0 = more random)
        #[arg(long, default_value = "0.7")]
        temperature: f32,

        /// Top-K sampling: keep only top K tokens (0 = disabled)
        #[arg(long, default_value = "0")]
        top_k: usize,

        /// Top-P (nucleus) sampling: cumulative probability threshold (0.0 = disabled, 1.0 = all)
        #[arg(long, default_value = "0.0")]
        top_p: f32,

        /// Random seed for reproducible generation (omit for non-deterministic)
        #[arg(long)]
        seed: Option<u64>,

        /// Maximum sequence length for KV cache allocation
        #[arg(long, default_value = "2048")]
        max_seq_len: usize,

        /// Number of transformer layers (for KV cache discovery)
        #[arg(long, default_value = "26")]
        num_layers: usize,

        /// Disable streaming output (print all at once instead of token-by-token)
        #[arg(long)]
        no_stream: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Dot {
            model,
            output,
            simplify,
            filter,
            depth,
        } => {
            cmd_dot(model, output, &simplify, filter, depth)?;
        }
        Commands::Inspect {
            model,
            dynamic_dims,
        } => {
            cmd_inspect(model, dynamic_dims)?;
        }
        Commands::RunModel {
            model,
            tokenizer,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            max_seq_len,
            num_layers,
            no_stream,
        } => {
            pollster::block_on(cmd_run_model(
                model,
                tokenizer,
                prompt,
                max_tokens,
                temperature,
                top_k,
                top_p,
                seed,
                max_seq_len,
                num_layers,
                !no_stream,
            ))?;
        }
    }

    Ok(())
}

/// Generate DOT format from ONNX model.
fn cmd_dot(
    model_path: PathBuf,
    output_path: Option<PathBuf>,
    simplify: &str,
    filter: Option<String>,
    depth: usize,
) -> Result<()> {
    // Load the ONNX model
    let model = onyxia_onnx::load_model(&model_path)
        .with_context(|| format!("Failed to load model from {}", model_path.display()))?;

    // Parse model if we need to filter
    let dot = if let Some(filter_prefix) = filter {
        let base_dir = model_path.parent();
        let graph =
            onyxia_onnx::parse_model(&model, base_dir).with_context(|| "Failed to parse model")?;

        // Find nodes matching the filter
        let mut selected_nodes: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for node in &graph.nodes {
            if node.name.starts_with(&filter_prefix) {
                selected_nodes.insert(&node.name);
            }
        }

        if selected_nodes.is_empty() {
            anyhow::bail!("No nodes found matching filter '{}'", filter_prefix);
        }

        // Expand by depth (include connected nodes)
        if depth > 0 {
            for _ in 0..depth {
                let mut to_add = Vec::new();
                for node in &graph.nodes {
                    // Check if this node connects to any selected node
                    let has_selected_input = node.inputs.iter().any(|inp| {
                        graph.nodes.iter().any(|n| {
                            selected_nodes.contains(n.name.as_str()) && n.outputs.contains(inp)
                        })
                    });
                    let has_selected_output = node.outputs.iter().any(|out| {
                        graph.nodes.iter().any(|n| {
                            selected_nodes.contains(n.name.as_str()) && n.inputs.contains(out)
                        })
                    });
                    if has_selected_input || has_selected_output {
                        to_add.push(node.name.as_str());
                    }
                }
                for name in to_add {
                    selected_nodes.insert(name);
                }
            }
        }

        eprintln!(
            "Selected {} nodes matching filter '{}' (depth={})",
            selected_nodes.len(),
            filter_prefix,
            depth
        );

        // Generate DOT only for selected nodes
        generate_filtered_dot(&graph, &selected_nodes)
    } else {
        // Convert to DOT format
        let simplify_level = match simplify {
            "layers" => onyxia_onnx::DotSimplification::Layers,
            "summary" => onyxia_onnx::DotSimplification::Summary,
            _ => onyxia_onnx::DotSimplification::Full,
        };
        onyxia_onnx::to_dot_with_options(&model, simplify_level)
    };

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

/// Generate DOT format for a filtered subset of nodes.
fn generate_filtered_dot(
    graph: &onyxia_onnx::Graph,
    selected_nodes: &std::collections::HashSet<&str>,
) -> String {
    let mut dot = String::from("digraph model {\n");
    dot.push_str("  rankdir=TB;\n");
    dot.push_str("  node [shape=box, style=rounded];\n\n");

    // Track which tensors are used
    let mut used_tensors: std::collections::HashSet<&str> = std::collections::HashSet::new();

    // Add nodes
    for node in &graph.nodes {
        if !selected_nodes.contains(node.name.as_str()) {
            continue;
        }

        let node_id = node
            .name
            .replace('/', "_")
            .replace('.', "_")
            .replace('[', "_")
            .replace(']', "_");
        dot.push_str(&format!(
            "  {} [label=\"{}\\n({})\"];\n",
            node_id,
            escape_dot_string(&node.name),
            node.op_type
        ));

        // Track tensors
        for inp in &node.inputs {
            used_tensors.insert(inp);
        }
        for out in &node.outputs {
            used_tensors.insert(out);
        }
    }

    dot.push_str("\n");

    // Add tensor nodes (inputs/outputs/intermediates)
    for tensor_name in &used_tensors {
        let tensor_id = tensor_name
            .replace('/', "_")
            .replace('.', "_")
            .replace('[', "_")
            .replace(']', "_");

        // Determine tensor style
        let (shape, color) = if let Some(&tid) = graph.tensors.get(*tensor_name) {
            let info = &graph.tensor_info[tid];
            let shape_str = match &info.shape {
                TensorShape::Static(dims) => format!("{:?}", dims),
                TensorShape::Unknown => "?".to_string(),
                TensorShape::Absent => "absent".to_string(),
                TensorShape::Dynamic(d) => format!("dyn:{:?}", d),
            };

            let color = if graph.inputs.contains(&tensor_name.to_string()) {
                "lightblue"
            } else if graph.outputs.contains(&tensor_name.to_string()) {
                "lightgreen"
            } else if info.initializer.is_some() {
                "lightyellow"
            } else {
                "white"
            };

            (shape_str, color)
        } else {
            ("?".to_string(), "white")
        };

        dot.push_str(&format!(
            "  {} [label=\"{}\\n{}\", shape=ellipse, style=filled, fillcolor={}];\n",
            tensor_id,
            escape_dot_string(tensor_name),
            escape_dot_string(&shape),
            color
        ));
    }

    dot.push_str("\n");

    // Add edges
    for node in &graph.nodes {
        if !selected_nodes.contains(node.name.as_str()) {
            continue;
        }

        let node_id = node
            .name
            .replace('/', "_")
            .replace('.', "_")
            .replace('[', "_")
            .replace(']', "_");

        for inp in &node.inputs {
            let inp_id = inp
                .replace('/', "_")
                .replace('.', "_")
                .replace('[', "_")
                .replace(']', "_");
            dot.push_str(&format!("  {} -> {};\n", inp_id, node_id));
        }

        for out in &node.outputs {
            let out_id = out
                .replace('/', "_")
                .replace('.', "_")
                .replace('[', "_")
                .replace(']', "_");
            dot.push_str(&format!("  {} -> {};\n", node_id, out_id));
        }
    }

    dot.push_str("}\n");
    dot
}

/// Escape special characters for DOT format.
fn escape_dot_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\"', "\\\"")
        .replace('\n', "\\n")
}

/// Inspect an ONNX model's structure.
fn cmd_inspect(model_path: PathBuf, dynamic_dim_args: Vec<String>) -> Result<()> {
    // Load and parse the ONNX model (handles external data automatically)
    let mut model = onyxia_onnx::load_and_parse_model(&model_path).with_context(|| {
        format!(
            "Failed to load and parse model from {}",
            model_path.display()
        )
    })?;

    // Parse dynamic dimensions from arguments
    let mut dynamic_dims = std::collections::HashMap::new();
    for arg in dynamic_dim_args {
        let parts: Vec<&str> = arg.split('=').collect();
        if parts.len() != 2 {
            anyhow::bail!(
                "Invalid dynamic dimension format '{}'. Expected format: name=value",
                arg
            );
        }
        let name = parts[0].to_string();
        let value: usize = parts[1].parse().with_context(|| {
            format!(
                "Invalid value '{}' for dynamic dimension '{}'",
                parts[1], name
            )
        })?;
        dynamic_dims.insert(name, value);
    }

    // Infer shapes for analysis using planner's kernel-based inference
    let registry = onyxia_planner::KernelRegistry::with_defaults();
    onyxia_planner::resolve_dynamic_dimensions(&mut model, &dynamic_dims)
        .with_context(|| "Failed to resolve dynamic dimensions")?;
    onyxia_planner::infer_shapes(&mut model, &registry)
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

/// Run an ONNX model for text generation.
#[allow(clippy::too_many_arguments)]
async fn cmd_run_model(
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: Option<u64>,
    max_seq_len: usize,
    num_layers: usize,
    stream: bool,
) -> Result<()> {
    use onyxia_cli::generate::{generate, print_stats};
    use onyxia_cli::llm::{LlmConfig, LlmSession};
    use onyxia_cli::sampling::SamplingConfig;
    use onyxia_cli::tokenizer::Tokenizer;

    println!("Loading model from {}...", model_path.display());

    // Load and parse ONNX model
    let mut model = onyxia_onnx::load_and_parse_model(&model_path)
        .with_context(|| format!("Failed to load model from {}", model_path.display()))?;

    // Set up dynamic dimensions - use max sequence length so buffers can handle variable inputs
    let mut dynamic_dims = std::collections::HashMap::new();
    dynamic_dims.insert("batch_size".to_string(), 1);
    dynamic_dims.insert("sequence_length".to_string(), max_seq_len); // Max length for buffer allocation
    dynamic_dims.insert("total_sequence_length".to_string(), max_seq_len);
    dynamic_dims.insert("past_sequence_length".to_string(), 0);
    dynamic_dims.insert("num_attention_heads".to_string(), 4);
    dynamic_dims.insert("num_key_value_heads".to_string(), 1);
    dynamic_dims.insert("head_dim".to_string(), 256);

    // Resolve dynamic dimensions and infer shapes
    let registry = onyxia_planner::KernelRegistry::with_defaults();
    onyxia_planner::resolve_dynamic_dimensions(&mut model, &dynamic_dims)
        .with_context(|| "Failed to resolve dynamic dimensions")?;
    onyxia_planner::infer_shapes(&mut model, &registry)
        .with_context(|| "Failed to infer shapes")?;

    println!("Compiling execution plan...");

    // Compile model to execution plan
    let plan = onyxia_planner::compile(&model, &registry, &dynamic_dims)
        .with_context(|| "Failed to compile model")?;

    println!("Initializing GPU runtime...");

    // Create runtime and load plan
    let runtime = onyxia_runtime::Runtime::new()
        .await
        .with_context(|| "Failed to create GPU runtime")?;
    let executor = runtime
        .load_model(plan)
        .await
        .with_context(|| "Failed to load execution plan")?;

    // Create LLM session
    let llm_config = LlmConfig {
        max_seq_len,
        num_layers,
    };
    let mut session = LlmSession::new(executor, &llm_config);

    println!("Loading tokenizer from {}...", tokenizer_path.display());

    // Load tokenizer (expects path to directory containing tokenizer.json)
    let tokenizer_file = tokenizer_path.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_file)
        .with_context(|| format!("Failed to load tokenizer from {}", tokenizer_file.display()))?;

    // Get EOS token ID (default to 1 for Gemma)
    let eos_token_id = tokenizer.eos_token_id() as u32;

    // Set up sampling config
    let sampling_config = SamplingConfig {
        temperature,
        top_k,
        top_p,
        seed,
    };

    println!("\n{}", "=".repeat(50));
    println!("Prompt: {}", prompt);
    println!("{}", "=".repeat(50));
    println!("Generating...\n");

    // Generate text
    let (generated_text, stats) = generate(
        &mut session,
        &tokenizer,
        &prompt,
        max_tokens,
        &sampling_config,
        stream,
        eos_token_id,
    )
    .with_context(|| "Generation failed")?;

    // Print output if not streaming
    if !stream {
        println!("{}", generated_text);
    }

    // Print statistics
    print_stats(&stats);

    Ok(())
}
