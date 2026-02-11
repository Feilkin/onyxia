//! WGSL shader compiler and execution graph builder for Onyxia.
//!
//! This crate takes ONNX models and generates WGSL compute shaders,
//! producing an executable graph that `onyxia-runtime` can execute on the GPU.
//!
//! # Example
//!
//! ```no_run
//! use onyxia_codegen::compile;
//! use onyxia_onnx::load_model;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load ONNX model
//! let model = load_model("model.onnx")?;
//!
//! // Compile to executable graph
//! let compiled = compile(&model)?;
//!
//! println!("Compiled {} operations", compiled.operations.len());
//! # Ok(())
//! # }
//! ```

pub mod compiled;
pub mod error;
pub mod scheduler;
pub mod shaders;

pub use compiled::{CompiledModel, ModelMetadata, Operation, OpType, TensorRegistry};
pub use error::{CodegenError, Result};
pub use shaders::{ShaderDefValue, ShaderDefs};

use onyxia_onnx::{parse_model as parse_onnx, Graph, ModelProto};
use scheduler::Scheduler;

/// Compile an ONNX model into an executable graph.
///
/// This is the main entry point for the codegen crate. It takes an ONNX
/// `ModelProto` and produces a `CompiledModel` ready for execution.
///
/// # Arguments
///
/// * `model` - The ONNX model to compile
///
/// # Returns
///
/// Returns a `CompiledModel` or an error if compilation fails.
///
/// # Example
///
/// ```no_run
/// use onyxia_codegen::compile;
/// use onyxia_onnx::load_model;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model = load_model("model.onnx")?;
/// let compiled = compile(&model)?;
/// # Ok(())
/// # }
/// ```
pub fn compile(model: &ModelProto) -> Result<CompiledModel> {
    // Step 1: Parse ONNX model into internal graph
    let graph = parse_onnx(model)?;
    
    // Step 2: Schedule operations (topological sort)
    let scheduler = Scheduler::new(graph);
    let ordered_nodes = scheduler.schedule()?;
    
    // Step 3: Build compiled model
    let compiled = build_compiled_model(scheduler.graph(), &ordered_nodes)?;
    
    Ok(compiled)
}

/// Build a CompiledModel from a scheduled graph.
fn build_compiled_model(graph: &Graph, _ordered_nodes: &[usize]) -> Result<CompiledModel> {
    let mut registry = TensorRegistry::new();
    let mut tensor_id_map = std::collections::HashMap::new();
    
    // Add all tensors to registry
    for (orig_id, info) in graph.tensor_info.iter().enumerate() {
        let new_id = registry.add(info.clone());
        tensor_id_map.insert(orig_id, new_id);
    }
    
    // Build operations (placeholder - no shader generation yet)
    let operations = Vec::new(); // TODO: Generate operations with shaders
    
    // Map input/output names to IDs
    let inputs: Vec<_> = graph
        .inputs
        .iter()
        .filter_map(|name| graph.tensors.get(name).and_then(|&id| tensor_id_map.get(&id).copied()))
        .collect();
    
    let outputs: Vec<_> = graph
        .outputs
        .iter()
        .filter_map(|name| graph.tensors.get(name).and_then(|&id| tensor_id_map.get(&id).copied()))
        .collect();
    
    Ok(CompiledModel {
        operations,
        tensors: registry,
        inputs,
        outputs,
        metadata: ModelMetadata {
            name: graph.metadata.name.clone(),
            version: graph.metadata.model_version,
            ir_version: graph.metadata.ir_version,
            producer: graph.metadata.producer_name.clone(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use onyxia_onnx::onnx::{GraphProto, ModelProto};
    
    #[test]
    fn test_compile_empty_model() {
        let model = ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                name: "test".to_string(),
                ..Default::default()
            }),
            ..Default::default()
        };
        
        let result = compile(&model);
        assert!(result.is_ok());
        
        let compiled = result.unwrap();
        assert_eq!(compiled.metadata.name, "test");
        assert_eq!(compiled.metadata.ir_version, 8);
    }
}
