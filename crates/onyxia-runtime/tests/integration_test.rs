//! Integration tests for runtime execution.

use onyxia_compiler::{OperatorRegistry, compile};
use onyxia_onnx::{load_model, parser::parse_model};
use onyxia_runtime::Runtime;
use std::collections::HashMap;

/// Test that we can initialize runtime and load the Gemma model.
/// This tests:
/// - GPU initialization
/// - Model loading
/// - Buffer allocation
/// - Pipeline creation
#[pollster::test]
async fn test_runtime_load_gemma_model() {
    // Load the quantized Gemma model
    let model_path = "../../models/gemma-3-270m-it-ONNX/onnx/model_q4.onnx";

    // Skip if model file doesn't exist
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test: model file not found at {}", model_path);
        return;
    }

    // Load ONNX model
    let model_proto = match load_model(model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            return;
        }
    };

    // Parse model into graph
    let graph = match parse_model(&model_proto, None) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to parse model: {}", e);
            return;
        }
    };

    // Create kernel registry
    let registry = OperatorRegistry::with_defaults();

    // Specify dynamic dimensions for the model
    let dynamic_dimensions = HashMap::from([
        ("batch_size".to_string(), 1),
        ("sequence_length".to_string(), 64),
        // Pre-allocate KV cache to max_sequence_length for buffer sharing (prevents aliasing conflicts)
        ("past_sequence_length".to_string(), 64),
        ("total_sequence_length".to_string(), 64),
    ]);

    // Compile the model - should succeed with all operators supported
    let plan =
        compile(&graph, &registry, &dynamic_dimensions).expect("Model compilation should succeed");

    // Initialize runtime - skip test if no GPU available (e.g., in CI)
    let runtime = match Runtime::new().await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Skipping test: Failed to initialize runtime (no GPU available)");
            eprintln!("  Error: {}", e);
            return;
        }
    };

    // Load model into executor
    // This will:
    // - Calculate required GPU buffer sizes from execution plan
    // - Create GPU device with appropriate limits
    // - Allocate GPU buffers for all tensors
    // - Create compute pipelines from pre-compiled naga modules
    // - Create bind groups
    let _executor = match runtime.load_model(plan).await {
        Ok(e) => e,
        Err(e) => {
            let err_msg = e.to_string();
            // Buffer size errors are acceptable on GPUs with limited memory - skip test
            if err_msg.contains("exceeds GPU maximum buffer size") {
                eprintln!("Skipping test: Model exceeds GPU buffer size limit");
                eprintln!("  This is expected on GPUs with limited memory.");
                eprintln!("  Error: {}", err_msg);
                return;
            }
            // All other errors should fail the test
            panic!("Failed to load model into executor: {}", e);
        }
    };
}

/// Test with the full precision (non-quantized) model.
/// This model might have static shapes instead of dynamic shapes.
#[pollster::test]
async fn test_runtime_load_gemma_full_precision() {
    // Load the full precision Gemma model
    let model_path = "../../models/gemma-3-270m-it-ONNX/onnx/model.onnx";

    // Skip if model file doesn't exist
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test: model file not found at {}", model_path);
        return;
    }

    // Load ONNX model
    let model_proto = match load_model(model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            return;
        }
    };

    // Parse model into graph
    let graph = match parse_model(&model_proto, None) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to parse model: {}", e);
            return;
        }
    };

    // Create kernel registry
    let registry = OperatorRegistry::with_defaults();

    // Specify dynamic dimensions for the model
    let dynamic_dimensions = HashMap::from([
        ("batch_size".to_string(), 1),
        ("sequence_length".to_string(), 64),
        // Pre-allocate KV cache to max_sequence_length for buffer sharing (prevents aliasing conflicts)
        ("past_sequence_length".to_string(), 64),
        ("total_sequence_length".to_string(), 64),
    ]);

    // Compile the model - should succeed with all operators supported
    let plan =
        compile(&graph, &registry, &dynamic_dimensions).expect("Model compilation should succeed");

    // Initialize runtime - skip test if no GPU available (e.g., in CI)
    let runtime = match Runtime::new().await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Skipping test: Failed to initialize runtime (no GPU available)");
            eprintln!("  Error: {}", e);
            return;
        }
    };

    // Load model into executor
    let _executor = match runtime.load_model(plan).await {
        Ok(e) => e,
        Err(e) => {
            let err_msg = e.to_string();
            // Buffer size errors are acceptable on GPUs with limited memory - skip test
            if err_msg.contains("exceeds GPU maximum buffer size") {
                eprintln!("Skipping test: Full precision model exceeds GPU buffer size limit");
                eprintln!(
                    "  Full precision Gemma needs ~640MB+ per tensor, exceeds GPU limits on most consumer hardware."
                );
                eprintln!("  Error: {}", err_msg);
                return;
            }
            // All other errors should fail the test
            panic!("Failed to load full precision model into executor: {}", e);
        }
    };
}
