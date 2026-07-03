//! Onyxia CLI - test ONNX models, generate dot graphs, benchmark execution.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use onyxia_cli::{TraceDirection, TraceFormat};
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
    /// Generate a DOT file of the lowered IR module (primitives + composites)
    IrDot {
        /// Path to the ONNX model file
        #[arg(value_name = "MODEL")]
        model: PathBuf,

        /// Output file path (defaults to stdout)
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
    },
    /// Inspect an ONNX model's structure
    Inspect {
        /// Path to the ONNX model file
        #[arg(value_name = "MODEL")]
        model: PathBuf,
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

        /// Maximum sequence length (bound on prompt + generated tokens)
        #[arg(long, default_value = "2048")]
        max_seq_len: usize,

        /// Execution backend
        #[arg(long, value_enum, default_value_t = BackendKind::Wgpu)]
        backend: BackendKind,

        /// Disable streaming output (print all at once instead of token-by-token)
        #[arg(long)]
        no_stream: bool,
    },
    /// Run a scripted multi-turn chat (for testing multi-turn decode).
    ///
    /// Each `--message` is a user turn, processed in order. The full
    /// conversation is re-prefilled every turn (KV cache cleared via
    /// reset_full), mirroring the gemma-chat demo exactly.
    Chat {
        /// Path to the ONNX model file
        #[arg(value_name = "MODEL")]
        model: PathBuf,

        /// Path to the tokenizer directory (containing tokenizer.json)
        #[arg(short, long, value_name = "TOKENIZER")]
        tokenizer: PathBuf,

        /// User message(s), one per turn, in order. Repeat the flag.
        #[arg(short, long = "message", required = true)]
        messages: Vec<String>,

        /// Maximum number of tokens to generate per turn
        #[arg(long, default_value = "200")]
        max_tokens: usize,

        /// Temperature (0.0 = greedy/deterministic)
        #[arg(long, default_value = "0.0")]
        temperature: f32,

        /// Top-K sampling: keep only top K tokens (0 = disabled)
        #[arg(long, default_value = "0")]
        top_k: usize,

        /// Top-P (nucleus) sampling threshold (0.0 = disabled)
        #[arg(long, default_value = "0.0")]
        top_p: f32,

        /// Random seed for reproducible sampling
        #[arg(long)]
        seed: Option<u64>,

        /// Maximum sequence length (bound on prompt + generated tokens)
        #[arg(long, default_value = "2048")]
        max_seq_len: usize,

        /// Execution backend
        #[arg(long, value_enum, default_value_t = BackendKind::Wgpu)]
        backend: BackendKind,

        /// Print the exact rendered prompt + token count fed to the model each turn
        #[arg(long)]
        print_prompt: bool,

        /// Keep the KV cache between turns instead of clearing it before
        /// each re-prefill. Diagnostic only: the stale cache corrupts output.
        #[arg(long)]
        buggy_reset: bool,
    },
    /// Inspect a specific node in the model
    InspectNode {
        /// Path to the ONNX model file
        #[arg(value_name = "MODEL")]
        model: PathBuf,

        /// Node name(s) to inspect
        #[arg(long = "name", required = true)]
        names: Vec<String>,

        /// Show full tensor values (may be large)
        #[arg(long)]
        full_values: bool,
    },
    /// List nodes in the model, optionally filtered
    ListNodes {
        /// Path to the ONNX model file
        #[arg(value_name = "MODEL")]
        model: PathBuf,

        /// Filter by operator type(s)
        #[arg(long = "op-type")]
        op_types: Vec<String>,

        /// Filter by name pattern (regex)
        #[arg(long)]
        name_pattern: Option<String>,

        /// Show input/output shapes
        #[arg(long)]
        show_shapes: bool,

        /// Show summary statistics instead of listing
        #[arg(long)]
        summary: bool,
    },
    /// Inspect tensor(s) by name
    InspectTensor {
        /// Path to the ONNX model file
        #[arg(value_name = "MODEL")]
        model: PathBuf,

        /// Tensor name(s) to inspect
        #[arg(long = "name")]
        names: Vec<String>,

        /// List all constant tensors
        #[arg(long)]
        list_constants: bool,

        /// Show full tensor values
        #[arg(long)]
        full: bool,
    },
    /// Trace data flow around a specific node
    TraceNode {
        /// Path to the ONNX model file
        #[arg(value_name = "MODEL")]
        model: PathBuf,

        /// Node name to trace
        #[arg(long)]
        name: String,

        /// Number of hops to trace (0 = just the node)
        #[arg(long, default_value = "1")]
        depth: usize,

        /// Direction to trace: both, upstream, downstream
        #[arg(long, default_value = "both")]
        direction: TraceDirection,

        /// Output format: text, dot
        #[arg(long, default_value = "text")]
        format: TraceFormat,

        /// Output file (for dot format)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Validate a model: parse, lower to IR, infer shapes — no GPU needed
    Validate {
        /// Path to the ONNX model file
        #[arg(value_name = "MODEL")]
        model: PathBuf,

        /// Show detailed progress
        #[arg(short, long)]
        verbose: bool,
    },
}

/// Which execution backend runs the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum BackendKind {
    /// Hand-written wgpu backend (generated WGSL + fused composite kernels).
    Wgpu,
    /// CubeCL backend (primitives only; composites run via decompositions).
    Cubecl,
}

/// A prepared LLM session on either backend. `Session` has an associated
/// tensor type, so this can't be a trait object — dispatch once here and
/// stay generic downstream.
enum AnySession {
    Wgpu(onyxia_cli::llm::LlmSession<onyxia_backend_wgpu::WgpuSession>),
    Cubecl(onyxia_cli::llm::LlmSession<onyxia_backend_cubecl::CubeclSession>),
}

async fn prepare_session(
    kind: BackendKind,
    module: onyxia_ir::Module,
    max_seq_len: usize,
) -> Result<AnySession> {
    use onyxia_cli::llm::LlmSession;
    Ok(match kind {
        BackendKind::Wgpu => {
            println!("Initializing GPU (wgpu backend)...");
            let ctx = onyxia_backend_wgpu::GpuContext::new()
                .await
                .with_context(|| "Failed to create GPU context")?;
            let backend = onyxia_backend_wgpu::WgpuBackend::new(ctx);
            println!("Preparing session (uploading weights)...");
            AnySession::Wgpu(LlmSession::new(&backend, module, max_seq_len)?)
        }
        BackendKind::Cubecl => {
            println!("Initializing GPU (cubecl backend, primitives only)...");
            let backend = onyxia_backend_cubecl::CubeclBackend::new();
            println!("Preparing session (uploading weights)...");
            AnySession::Cubecl(LlmSession::new(&backend, module, max_seq_len)?)
        }
    })
}

fn main() -> Result<()> {
    // Initialize tracing with Tracy profiler support
    #[cfg(feature = "tracy")]
    {
        use tracing_subscriber::layer::SubscriberExt;
        tracing::subscriber::set_global_default(
            tracing_subscriber::registry().with(tracing_tracy::TracyLayer::default()),
        )
        .expect("Failed to set tracing subscriber");
    }

    #[cfg(not(feature = "tracy"))]
    {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .init();
    }

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
        Commands::IrDot { model, output } => {
            cmd_ir_dot(model, output)?;
        }
        Commands::Inspect { model } => {
            cmd_inspect(model)?;
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
            backend,
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
                !no_stream,
                backend,
            ))?;
        }
        Commands::Chat {
            model,
            tokenizer,
            messages,
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            max_seq_len,
            backend,
            print_prompt,
            buggy_reset,
        } => {
            pollster::block_on(cmd_chat(
                model,
                tokenizer,
                messages,
                max_tokens,
                temperature,
                top_k,
                top_p,
                seed,
                max_seq_len,
                print_prompt,
                buggy_reset,
                backend,
            ))?;
        }
        Commands::InspectNode {
            model,
            names,
            full_values,
        } => {
            cmd_inspect_node(model, names, full_values)?;
        }
        Commands::ListNodes {
            model,
            op_types,
            name_pattern,
            show_shapes,
            summary,
        } => {
            cmd_list_nodes(model, op_types, name_pattern, show_shapes, summary)?;
        }
        Commands::InspectTensor {
            model,
            names,
            list_constants,
            full,
        } => {
            cmd_inspect_tensor(model, names, list_constants, full)?;
        }
        Commands::TraceNode {
            model,
            name,
            depth,
            direction,
            format,
            output,
        } => {
            cmd_trace_node(model, name, depth, direction, format, output)?;
        }
        Commands::Validate { model, verbose } => {
            onyxia_cli::validate::validate_model(&model, verbose)?;
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

/// Generate DOT for the lowered IR module.
fn cmd_ir_dot(model_path: PathBuf, output_path: Option<PathBuf>) -> Result<()> {
    eprintln!("Loading model from {}...", model_path.display());
    let graph = onyxia_onnx::load_and_parse_model(&model_path)
        .with_context(|| format!("Failed to load model from {}", model_path.display()))?;

    eprintln!("Lowering to IR...");
    let module = onyxia_lower::lower(graph, &onyxia_lower::standard_registry())
        .with_context(|| "Failed to lower model")?;

    eprintln!(
        "Lowered: {} nodes, {} values",
        module.nodes.len(),
        module.values.len()
    );

    let dot = onyxia_ir::dot::to_dot(&module);

    // Write to output (file or stdout)
    if let Some(output_path) = output_path {
        std::fs::write(&output_path, &dot)
            .with_context(|| format!("Failed to write DOT output to {}", output_path.display()))?;
        eprintln!("Wrote dispatch graph to {}", output_path.display());
        eprintln!(
            "Render with: dot -Tpng {} -o dispatch.png",
            output_path.display()
        );
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

        let node_id = node.name.replace(['/', '.', '[', ']'], "_");
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

    dot.push('\n');

    // Add tensor nodes (inputs/outputs/intermediates)
    for tensor_name in &used_tensors {
        let tensor_id = tensor_name.replace(['/', '.', '[', ']'], "_");

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

    dot.push('\n');

    // Add edges
    for node in &graph.nodes {
        if !selected_nodes.contains(node.name.as_str()) {
            continue;
        }

        let node_id = node.name.replace(['/', '.', '[', ']'], "_");

        for inp in &node.inputs {
            let inp_id = inp.replace(['/', '.', '[', ']'], "_");
            dot.push_str(&format!("  {} -> {};\n", inp_id, node_id));
        }

        for out in &node.outputs {
            let out_id = out.replace(['/', '.', '[', ']'], "_");
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
fn cmd_inspect(model_path: PathBuf) -> Result<()> {
    // Load and parse the ONNX model (handles external data automatically)
    let model = onyxia_onnx::load_and_parse_model(&model_path).with_context(|| {
        format!(
            "Failed to load and parse model from {}",
            model_path.display()
        )
    })?;

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

    // Collect unique operation types with counts
    let mut op_type_counts: std::collections::HashMap<&str, usize> =
        std::collections::HashMap::new();
    for node in &model.nodes {
        *op_type_counts.entry(&node.op_type).or_insert(0) += 1;
    }

    // Sort by operation name for consistent output
    let mut op_types: Vec<_> = op_type_counts.iter().collect();
    op_types.sort_by_key(|(name, _)| *name);

    println!(
        "Operation types ({} unique, {} total):",
        op_types.len(),
        model.nodes.len()
    );
    for (op_type, count) in op_types {
        println!(
            "  {} - {} instance{}",
            op_type,
            count,
            if *count == 1 { "" } else { "s" }
        );
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
    stream: bool,
    backend: BackendKind,
) -> Result<()> {
    println!("Loading model from {}...", model_path.display());
    let graph = onyxia_onnx::load_and_parse_model(&model_path)
        .with_context(|| format!("Failed to load model from {}", model_path.display()))?;

    println!("Lowering to IR...");
    let module = onyxia_lower::lower(graph, &onyxia_lower::standard_registry())
        .with_context(|| "Failed to lower model")?;

    let session = prepare_session(backend, module, max_seq_len).await?;
    match session {
        AnySession::Wgpu(session) => run_model_with_session(
            session,
            tokenizer_path,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            stream,
        ),
        AnySession::Cubecl(session) => run_model_with_session(
            session,
            tokenizer_path,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            stream,
        ),
    }
}

/// The generation half of `run-model`, generic over the backend session.
#[allow(clippy::too_many_arguments)]
fn run_model_with_session<S: onyxia_ir::Session>(
    mut session: onyxia_cli::llm::LlmSession<S>,
    tokenizer_path: PathBuf,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: Option<u64>,
    stream: bool,
) -> Result<()> {
    use onyxia_cli::generate::{generate, print_stats};
    use onyxia_cli::sampling::SamplingConfig;
    use onyxia_cli::tokenizer::{ChatMessage, Tokenizer};

    println!("Loading tokenizer from {}...", tokenizer_path.display());

    // Load tokenizer (expects path to directory containing tokenizer.json)
    let tokenizer_file = tokenizer_path.join("tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(&tokenizer_file)
        .with_context(|| format!("Failed to load tokenizer from {}", tokenizer_file.display()))?;

    // Auto-load chat template when available (e.g., Gemma instruct models).
    let chat_template_file = tokenizer_path.join("chat_template.jinja");
    if chat_template_file.exists() {
        tokenizer = tokenizer
            .with_chat_template_file(&chat_template_file)
            .with_context(|| {
                format!(
                    "Failed to load chat template from {}",
                    chat_template_file.display()
                )
            })?;
    }

    let prompt_for_model = if chat_template_file.exists() {
        tokenizer
            .apply_chat_template(
                &[ChatMessage {
                    role: "user".to_string(),
                    content: prompt.clone(),
                }],
                true,
            )
            .with_context(|| "Failed to apply chat template")?
    } else {
        prompt.clone()
    };

    // Build stop-token set (EOS always, plus end-of-turn for chat templates when available)
    let mut stop_token_ids = vec![tokenizer.eos_token_id() as u32];
    if chat_template_file.exists()
        && let Ok(end_of_turn_ids) = tokenizer.encode("<end_of_turn>", false)
        && end_of_turn_ids.len() == 1
    {
        stop_token_ids.push(end_of_turn_ids[0] as u32);
    }

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
        &prompt_for_model,
        max_tokens,
        &sampling_config,
        stream,
        &stop_token_ids,
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

/// Run a scripted multi-turn chat, re-prefilling the full conversation each
/// turn (mirrors the gemma-chat demo). Used to test multi-turn decode.
#[allow(clippy::too_many_arguments)]
async fn cmd_chat(
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    messages: Vec<String>,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: Option<u64>,
    max_seq_len: usize,
    print_prompt: bool,
    buggy_reset: bool,
    backend: BackendKind,
) -> Result<()> {
    println!("Loading model from {}...", model_path.display());
    let graph = onyxia_onnx::load_and_parse_model(&model_path)
        .with_context(|| format!("Failed to load model from {}", model_path.display()))?;

    println!("Lowering to IR...");
    let module = onyxia_lower::lower(graph, &onyxia_lower::standard_registry())
        .with_context(|| "Failed to lower model")?;

    let session = prepare_session(backend, module, max_seq_len).await?;
    match session {
        AnySession::Wgpu(session) => chat_with_session(
            session,
            tokenizer_path,
            messages,
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            print_prompt,
            buggy_reset,
        ),
        AnySession::Cubecl(session) => chat_with_session(
            session,
            tokenizer_path,
            messages,
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            print_prompt,
            buggy_reset,
        ),
    }
}

/// The conversation half of `chat`, generic over the backend session.
#[allow(clippy::too_many_arguments)]
fn chat_with_session<S: onyxia_ir::Session>(
    mut session: onyxia_cli::llm::LlmSession<S>,
    tokenizer_path: PathBuf,
    messages: Vec<String>,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: Option<u64>,
    print_prompt: bool,
    buggy_reset: bool,
) -> Result<()> {
    use onyxia_cli::generate::generate;
    use onyxia_cli::sampling::SamplingConfig;
    use onyxia_cli::tokenizer::{ChatMessage, Tokenizer};

    let tokenizer_file = tokenizer_path.join("tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(&tokenizer_file)
        .with_context(|| format!("Failed to load tokenizer from {}", tokenizer_file.display()))?;

    let chat_template_file = tokenizer_path.join("chat_template.jinja");
    if !chat_template_file.exists() {
        anyhow::bail!(
            "chat requires a chat_template.jinja in the tokenizer directory ({})",
            tokenizer_path.display()
        );
    }
    tokenizer = tokenizer
        .with_chat_template_file(&chat_template_file)
        .with_context(|| "Failed to load chat template")?;

    // Stop on EOS and Gemma's <end_of_turn>.
    let mut stop_token_ids = vec![tokenizer.eos_token_id() as u32];
    if let Ok(ids) = tokenizer.encode("<end_of_turn>", false)
        && ids.len() == 1
    {
        stop_token_ids.push(ids[0] as u32);
    }

    let sampling = SamplingConfig {
        temperature,
        top_k,
        top_p,
        seed,
    };

    println!(
        "\nReset mode: {}\n",
        if buggy_reset {
            "reset() [buggy: keeps stale KV cache]"
        } else {
            "reset_full() [clears KV cache]"
        }
    );

    let mut conversation: Vec<ChatMessage> = Vec::new();
    for (i, user_msg) in messages.into_iter().enumerate() {
        conversation.push(ChatMessage {
            role: "user".to_string(),
            content: user_msg.clone(),
        });

        let prompt = tokenizer
            .apply_chat_template(&conversation, true)
            .with_context(|| "Failed to apply chat template")?;

        if print_prompt {
            let ids = tokenizer.encode(&prompt, false)?;
            println!(
                "\n----- turn {} prefill prompt ({} tokens) -----\n{}\n----- end prompt -----",
                i + 1,
                ids.len(),
                prompt
            );
        }

        // Re-prefill the full conversation each turn.
        if buggy_reset {
            session.reset();
        } else {
            session.reset_full();
        }

        println!("\n[user]  {user_msg}");
        print!("[model] ");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        let (text, _stats) = generate(
            &mut session,
            &tokenizer,
            &prompt,
            max_tokens,
            &sampling,
            true,
            &stop_token_ids,
        )
        .with_context(|| format!("Generation failed on turn {}", i + 1))?;

        // Strip stop-token text before storing so re-templating stays clean.
        let cleaned = text
            .replace("<end_of_turn>", "")
            .replace("<eos>", "")
            .trim()
            .to_string();
        conversation.push(ChatMessage {
            role: "model".to_string(),
            content: cleaned,
        });
    }

    println!();
    Ok(())
}

/// Inspect specific nodes in an ONNX model.
fn cmd_inspect_node(model_path: PathBuf, node_names: Vec<String>, full_values: bool) -> Result<()> {
    let model = onyxia_onnx::load_and_parse_model(&model_path).with_context(|| {
        format!(
            "Failed to load and parse model from {}",
            model_path.display()
        )
    })?;

    onyxia_cli::inspect::inspect_nodes(&model, &node_names, full_values)
}

/// List nodes in an ONNX model with optional filtering.
fn cmd_list_nodes(
    model_path: PathBuf,
    op_types: Vec<String>,
    name_pattern: Option<String>,
    show_shapes: bool,
    summary: bool,
) -> Result<()> {
    let model = onyxia_onnx::load_and_parse_model(&model_path).with_context(|| {
        format!(
            "Failed to load and parse model from {}",
            model_path.display()
        )
    })?;

    onyxia_cli::inspect::list_nodes(
        &model,
        &op_types,
        name_pattern.as_deref(),
        show_shapes,
        summary,
    )
}

/// Inspect tensor(s) in an ONNX model.
fn cmd_inspect_tensor(
    model_path: PathBuf,
    names: Vec<String>,
    list_constants: bool,
    full: bool,
) -> Result<()> {
    // Load and parse the ONNX model
    let model = onyxia_onnx::load_and_parse_model(&model_path).with_context(|| {
        format!(
            "Failed to load and parse model from {}",
            model_path.display()
        )
    })?;

    // Inspect the tensors
    onyxia_cli::inspect::inspect_tensor(&model, &names, list_constants, full)
}

/// Trace data flow around a specific node.
fn cmd_trace_node(
    model_path: PathBuf,
    name: String,
    depth: usize,
    direction: TraceDirection,
    format: TraceFormat,
    output: Option<PathBuf>,
) -> Result<()> {
    // Load and parse the ONNX model
    let model = onyxia_onnx::load_and_parse_model(&model_path).with_context(|| {
        format!(
            "Failed to load and parse model from {}",
            model_path.display()
        )
    })?;

    // Trace the node
    onyxia_cli::inspect::trace_node(&model, &name, depth, direction, format, output.as_deref())
}

