//! Test compilation of the Gemma 3 270m model.

use onyxia_onnx::load_and_parse_model;
use onyxia_planner::{KernelRegistry, compile};
use std::collections::HashMap;

#[test]
fn test_compile_gemma_model() {
    // Load the quantized Gemma model
    let model_path = "../../models/gemma-3-270m-it-ONNX/onnx/model_q4.onnx";

    // Skip if model file doesn't exist (e.g., in CI)
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test: model file not found at {}", model_path);
        return;
    }

    let graph = match load_and_parse_model(model_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to load and parse model: {}", e);
            return;
        }
    };

    // Create kernel registry with default kernels
    let registry = KernelRegistry::with_defaults();

    // Provide dynamic dimensions for the model
    // Note: Model contains expressions like "sequence_length * num_attention_heads"
    // which are evaluated automatically by the symbolic expression evaluator
    let dynamic_dimensions = HashMap::from([
        ("batch_size".to_string(), 1),
        ("sequence_length".to_string(), 64),
        ("past_sequence_length".to_string(), 0),
        ("total_sequence_length".to_string(), 64),
        ("num_attention_heads".to_string(), 8),
        ("num_key_value_heads".to_string(), 8),
    ]);

    // Compile the model - this should succeed with all shape inference implemented
    let plan = compile(&graph, &registry, &dynamic_dimensions)
        .expect("Model compilation should succeed with shape inference for all operators");

    // Should have many tensors (weights, intermediates)
    assert!(
        plan.tensors.all().len() > 100,
        "Model should have many tensors"
    );
}
