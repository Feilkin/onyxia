//! Integration tests for generation workflow.
//!
//! These tests require GPU and a model file, so they are marked with `#[ignore]`.
//! Run with: `cargo nextest run -p onyxia-cli --run-ignored all`

use onyxia_cli::generate::generate;
use onyxia_cli::llm::{LlmConfig, LlmSession};
use onyxia_cli::sampling::SamplingConfig;
use onyxia_cli::tokenizer::Tokenizer;
use std::path::PathBuf;

/// Helper to get the workspace root directory.
fn workspace_root() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    PathBuf::from(manifest_dir)
        .parent()
        .expect("No parent directory")
        .parent()
        .expect("No workspace root")
        .to_path_buf()
}

/// Test end-to-end generation with Gemma model.
///
/// This test loads the full Gemma 270M model, runs tokenization, prefill, decode,
/// and sampling to generate 10 tokens. It verifies:
/// - The model loads successfully
/// - The tokenizer works
/// - Generation produces valid text (not empty or garbage)
/// - Statistics are reasonable
#[test]
#[ignore] // Requires GPU and model file
fn test_generate_gemma() {
    // Get model and tokenizer paths
    let workspace = workspace_root();
    let model_path = workspace
        .join("models")
        .join("gemma-3-270m-it-ONNX")
        .join("onnx")
        .join("model_q4.onnx");
    let tokenizer_dir = workspace.join("models").join("gemma-3-270m-it-ONNX");

    if !model_path.exists() {
        eprintln!(
            "Skipping test_generate_gemma: model not found at {}",
            model_path.display()
        );
        return;
    }

    // Run async test with pollster
    pollster::block_on(async {
        test_generate_gemma_impl(model_path, tokenizer_dir)
            .await
            .expect("Generation test failed");
    });
}

async fn test_generate_gemma_impl(
    model_path: PathBuf,
    tokenizer_dir: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading model from {}...", model_path.display());

    // Load and parse ONNX model
    let mut model = onyxia_onnx::load_and_parse_model(&model_path)?;

    // Set up dynamic dimensions - use max sequence length so buffers can handle variable inputs
    let mut dynamic_dims = std::collections::HashMap::new();
    dynamic_dims.insert("batch_size".to_string(), 1);
    dynamic_dims.insert("sequence_length".to_string(), 128); // Max length for buffer allocation
    dynamic_dims.insert("total_sequence_length".to_string(), 128);
    dynamic_dims.insert("past_sequence_length".to_string(), 0);
    dynamic_dims.insert("num_attention_heads".to_string(), 4);
    dynamic_dims.insert("num_key_value_heads".to_string(), 1);
    dynamic_dims.insert("head_dim".to_string(), 256);

    // Resolve dynamic dimensions and infer shapes
    let registry = onyxia_planner::KernelRegistry::with_defaults();
    onyxia_planner::resolve_dynamic_dimensions(&mut model, &dynamic_dims)?;
    onyxia_planner::infer_shapes(&mut model, &registry)?;

    println!("Compiling execution plan...");

    // Compile model
    let plan = onyxia_planner::compile(&model, &registry, &dynamic_dims)?;

    println!("Plan has {} operations", plan.operations.len());
    if plan.operations.is_empty() {
        eprintln!(
            "WARNING: Execution plan is empty! Planner may not be generating operations yet."
        );
    }

    println!("Initializing GPU runtime...");

    // Create runtime and load plan
    let runtime = onyxia_runtime::Runtime::new().await?;
    let executor = runtime.load_model(plan).await?;

    // Create LLM session
    let llm_config = LlmConfig {
        max_seq_len: 128,
        num_layers: 26,
    };
    let mut session = LlmSession::new(executor, &llm_config);

    println!("Loading tokenizer...");

    // Load tokenizer
    let tokenizer_file = tokenizer_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_file)?;
    let eos_token_id = tokenizer.eos_token_id() as u32;

    // Test prompt
    let prompt = "What is Rust?";
    let max_tokens = 10;

    // Sampling config (greedy for determinism)
    let sampling_config = SamplingConfig {
        temperature: 0.0, // Greedy sampling
        top_k: 0,
        top_p: 0.0,
        seed: Some(42),
    };

    println!("Generating from prompt: '{}'", prompt);

    // Generate text
    let (generated_text, stats) = generate(
        &mut session,
        &tokenizer,
        prompt,
        max_tokens,
        &sampling_config,
        false, // No streaming in test
        eos_token_id,
    )?;

    println!("Generated text: '{}'", generated_text);
    println!("Stats: {:?}", stats);

    // Verify results
    assert!(
        !generated_text.is_empty(),
        "Generated text should not be empty"
    );
    assert!(
        stats.tokens_generated > 0,
        "Should have generated at least 1 token"
    );
    assert!(
        stats.tokens_generated <= max_tokens,
        "Should not exceed max_tokens"
    );
    assert!(stats.prefill_time > 0.0, "Prefill time should be positive");
    assert!(stats.total_time > 0.0, "Total time should be positive");

    // Check that generated text is not just special tokens or garbage
    let printable_chars = generated_text
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .count();
    assert!(
        printable_chars > 0,
        "Generated text should contain printable characters"
    );

    println!("✓ Generation test passed");

    Ok(())
}

/// Test that different sampling configurations produce different outputs.
#[test]
#[ignore] // Requires GPU and model file
fn test_sampling_variance() {
    let workspace = workspace_root();
    let model_path = workspace
        .join("models")
        .join("gemma-3-270m-it-ONNX")
        .join("onnx")
        .join("model_q4.onnx");
    let tokenizer_dir = workspace.join("models").join("gemma-3-270m-it-ONNX");

    if !model_path.exists() {
        eprintln!(
            "Skipping test_sampling_variance: model not found at {}",
            model_path.display()
        );
        return;
    }

    pollster::block_on(async {
        test_sampling_variance_impl(model_path, tokenizer_dir)
            .await
            .expect("Sampling variance test failed");
    });
}

async fn test_sampling_variance_impl(
    model_path: PathBuf,
    tokenizer_dir: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load model (same as previous test)
    let mut model = onyxia_onnx::load_and_parse_model(&model_path)?;
    let mut dynamic_dims = std::collections::HashMap::new();
    dynamic_dims.insert("batch_size".to_string(), 1);
    dynamic_dims.insert("sequence_length".to_string(), 128); // Max length for buffer allocation
    dynamic_dims.insert("total_sequence_length".to_string(), 128);
    dynamic_dims.insert("past_sequence_length".to_string(), 0);
    dynamic_dims.insert("num_attention_heads".to_string(), 4);
    dynamic_dims.insert("num_key_value_heads".to_string(), 1);
    dynamic_dims.insert("head_dim".to_string(), 256);

    let registry = onyxia_planner::KernelRegistry::with_defaults();
    onyxia_planner::resolve_dynamic_dimensions(&mut model, &dynamic_dims)?;
    onyxia_planner::infer_shapes(&mut model, &registry)?;

    let plan = onyxia_planner::compile(&model, &registry, &dynamic_dims)?;
    let runtime = onyxia_runtime::Runtime::new().await?;
    let executor = runtime.load_model(plan).await?;

    let llm_config = LlmConfig {
        max_seq_len: 128,
        num_layers: 26,
    };
    let mut session = LlmSession::new(executor, &llm_config);

    let tokenizer_file = tokenizer_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_file)?;
    let eos_token_id = tokenizer.eos_token_id() as u32;

    let prompt = "Hello";

    // Generate with two different seeds
    let config1 = SamplingConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 0.0,
        seed: Some(123),
    };

    let (text1, _) = generate(
        &mut session,
        &tokenizer,
        prompt,
        5,
        &config1,
        false,
        eos_token_id,
    )?;

    // Reset session for second generation
    session.reset();

    let config2 = SamplingConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 0.0,
        seed: Some(456),
    };

    let (text2, _) = generate(
        &mut session,
        &tokenizer,
        prompt,
        5,
        &config2,
        false,
        eos_token_id,
    )?;

    println!("Text with seed 123: '{}'", text1);
    println!("Text with seed 456: '{}'", text2);

    // With different seeds and temperature=1.0, we expect different outputs
    // (Note: there's a small chance they could be the same, but very unlikely)
    assert_ne!(
        text1, text2,
        "Different seeds should produce different outputs (with high probability)"
    );

    println!("✓ Sampling variance test passed");

    Ok(())
}

/// Test that generation with the same seed produces deterministic output.
#[test]
#[ignore] // Requires GPU and model file
fn test_deterministic_generation() {
    let workspace = workspace_root();
    let model_path = workspace
        .join("models")
        .join("gemma-3-270m-it-ONNX")
        .join("onnx")
        .join("model_q4.onnx");
    let tokenizer_dir = workspace.join("models").join("gemma-3-270m-it-ONNX");

    if !model_path.exists() {
        eprintln!(
            "Skipping test_deterministic_generation: model not found at {}",
            model_path.display()
        );
        return;
    }

    pollster::block_on(async {
        test_deterministic_generation_impl(model_path, tokenizer_dir)
            .await
            .expect("Deterministic generation test failed");
    });
}

async fn test_deterministic_generation_impl(
    model_path: PathBuf,
    tokenizer_dir: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load model
    let mut model = onyxia_onnx::load_and_parse_model(&model_path)?;
    let mut dynamic_dims = std::collections::HashMap::new();
    dynamic_dims.insert("batch_size".to_string(), 1);
    dynamic_dims.insert("sequence_length".to_string(), 128); // Max length for buffer allocation
    dynamic_dims.insert("total_sequence_length".to_string(), 128);
    dynamic_dims.insert("past_sequence_length".to_string(), 0);
    dynamic_dims.insert("num_attention_heads".to_string(), 4);
    dynamic_dims.insert("num_key_value_heads".to_string(), 1);
    dynamic_dims.insert("head_dim".to_string(), 256);

    let registry = onyxia_planner::KernelRegistry::with_defaults();
    onyxia_planner::resolve_dynamic_dimensions(&mut model, &dynamic_dims)?;
    onyxia_planner::infer_shapes(&mut model, &registry)?;

    let plan = onyxia_planner::compile(&model, &registry, &dynamic_dims)?;
    let runtime = onyxia_runtime::Runtime::new().await?;
    let executor = runtime.load_model(plan).await?;

    let llm_config = LlmConfig {
        max_seq_len: 128,
        num_layers: 26,
    };
    let mut session = LlmSession::new(executor, &llm_config);

    let tokenizer_file = tokenizer_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_file)?;
    let eos_token_id = tokenizer.eos_token_id() as u32;

    let prompt = "The capital of France is";

    // Same config (including seed) for both generations
    let config = SamplingConfig {
        temperature: 0.7,
        top_k: 50,
        top_p: 0.0,
        seed: Some(999),
    };

    // First generation
    let (text1, _) = generate(
        &mut session,
        &tokenizer,
        prompt,
        8,
        &config,
        false,
        eos_token_id,
    )?;

    // Reset session for second generation
    session.reset();

    // Second generation with same config
    let (text2, _) = generate(
        &mut session,
        &tokenizer,
        prompt,
        8,
        &config,
        false,
        eos_token_id,
    )?;

    println!("First generation:  '{}'", text1);
    println!("Second generation: '{}'", text2);

    // Same seed should produce identical output
    assert_eq!(text1, text2, "Same seed should produce identical output");

    println!("✓ Deterministic generation test passed");

    Ok(())
}
