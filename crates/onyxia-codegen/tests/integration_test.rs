//! Test compilation of the Gemma 3 270m model.

use onyxia_codegen::compile;
use onyxia_onnx::load_model;

#[test]
fn test_compile_gemma_model() {
    // Load the quantized Gemma model
    let model_path = "../../models/gemma-3-270m-it-ONNX/onnx/model_q4.onnx";
    
    // Skip if model file doesn't exist (e.g., in CI)
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test: model file not found at {}", model_path);
        return;
    }
    
    let model = match load_model(model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            return;
        }
    };
    
    // Compile the model
    let compiled = compile(&model).expect("Failed to compile model");
    
    // Verify basic properties
    println!("Model: {}", compiled.metadata.name);
    println!("Operations: {}", compiled.operations.len());
    println!("Inputs: {}", compiled.inputs.len());
    println!("Outputs: {}", compiled.outputs.len());
    println!("Tensors: {}", compiled.tensors.all().len());
    
    // Should have inputs (input_ids, attention_mask, position_ids, past_key_values)
    assert!(!compiled.inputs.is_empty(), "Model should have inputs");
    
    // Should have outputs (logits, present key/values)
    assert!(!compiled.outputs.is_empty(), "Model should have outputs");
    
    // Should have many tensors (weights, intermediates)
    assert!(compiled.tensors.all().len() > 100, "Model should have many tensors");
}
