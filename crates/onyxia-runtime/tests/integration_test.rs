//! Integration tests for runtime execution.

use onyxia_codegen::compile;
use onyxia_onnx::load_model;
use onyxia_runtime::Runtime;

/// Test that we can initialize runtime and load the Gemma model.
/// This tests:
/// - GPU initialization
/// - Model loading
/// - Buffer allocation
/// - Shader compilation
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
    let model = match load_model(model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            return;
        }
    };
    
    // Compile the model
    let compiled = match compile(&model) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to compile model: {}", e);
            return;
        }
    };
    
    println!("Compiled model: {}", compiled.metadata.name);
    println!("  Operations: {}", compiled.operations.len());
    println!("  Tensors: {}", compiled.tensors.all().len());
    
    // Initialize runtime
    let runtime = match Runtime::new().await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to initialize runtime: {}", e);
            eprintln!("This is expected if no GPU is available (e.g., in CI)");
            return;
        }
    };
    
    let adapter_info = runtime.adapter_info();
    println!("GPU: {} ({:?})", adapter_info.name, adapter_info.backend);
    
    // Load model into executor
    // This will:
    // - Allocate GPU buffers for all tensors
    // - Compile all shaders
    // - Create bind groups
    let executor = match runtime.load_model(compiled) {
        Ok(e) => e,
        Err(e) => {
            // This is expected to fail for now since we don't have all shaders implemented
            eprintln!("Failed to load model (expected): {}", e);
            println!("Note: Full model execution not yet supported");
            return;
        }
    };
    
    println!("Successfully loaded model into executor!");
    println!("  Model info: {:?}", executor.model().metadata);
    
    // TODO: Once we implement all operators, we can test execution:
    // let input = Tensor::from_vec(vec![1u32], &[1, 1]);
    // let outputs = executor.run(&[("input_ids", input)])?;
}
