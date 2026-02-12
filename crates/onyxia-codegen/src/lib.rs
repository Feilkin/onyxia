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
pub mod kernel;
pub mod kernels;
pub mod plan;
pub mod scheduler;
pub mod shaders;

pub use compiled::{
    CompiledModel, ModelMetadata, OpParams, OpType, Operation, ShaderHandle, TensorRegistry,
};
pub use error::{CodegenError, Result};
pub use kernel::{KernelRegistry, OpKernel, PlanContext};
pub use plan::{
    BindingDesc, BufferRef, CompiledShader, ExecutionPlan, PlannedOp, ScratchBufferDesc,
    ShaderIndex, Step,
};
pub use shaders::{ShaderDefValue, ShaderDefs};

use onyxia_onnx::{Graph, ModelProto, OnnxError, parse_model as parse_onnx};
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
        .filter_map(|name| {
            graph
                .tensors
                .get(name)
                .and_then(|&id| tensor_id_map.get(&id).copied())
        })
        .collect();

    let outputs: Vec<_> = graph
        .outputs
        .iter()
        .filter_map(|name| {
            graph
                .tensors
                .get(name)
                .and_then(|&id| tensor_id_map.get(&id).copied())
        })
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

/// Compile an ONNX graph into an execution plan.
///
/// Uses the provided kernel registry to map ONNX operations to GPU steps.
/// `dynamic_dimensions` provides concrete values for symbolic dimensions
/// (e.g., {"batch": 1, "sequence": 512}), allowing all shader defs to be
/// fully resolved at plan time.
///
/// # Arguments
///
/// * `graph` - The ONNX graph to compile
/// * `registry` - Kernel registry mapping op_types to implementations
/// * `dynamic_dimensions` - Concrete values for symbolic dimensions
///
/// # Returns
///
/// Returns an `ExecutionPlan` or an error if compilation fails.
///
/// # Errors
///
/// - `CodegenError::UnsupportedOp` if an operation has no registered kernel
/// - `CodegenError::InvalidShape` if a dynamic dimension is not provided
/// - `CodegenError::InvalidShape` if any tensor has Unknown shape
///
/// # Example
///
/// ```rust
/// use onyxia_codegen::{compile_to_plan, KernelRegistry};
/// use onyxia_onnx::Graph;
/// use std::collections::HashMap;
///
/// # fn example(graph: &Graph) -> Result<(), Box<dyn std::error::Error>> {
/// let registry = KernelRegistry::with_defaults();
/// let dynamic_dimensions = HashMap::new();
/// let plan = compile_to_plan(graph, &registry, &dynamic_dimensions)?;
/// println!("Compiled {} operations", plan.operations.len());
/// # Ok(())
/// # }
/// ```
pub fn compile_to_plan(
    graph: &Graph,
    registry: &KernelRegistry,
    dynamic_dimensions: &std::collections::HashMap<String, usize>,
) -> Result<ExecutionPlan> {
    use onyxia_onnx::{Dimension, TensorShape};

    // Step 1: Run scheduler to get topologically ordered node IDs
    let scheduler = Scheduler::new(graph.clone());
    let ordered_nodes = scheduler.schedule()?;

    // Step 2: Create shared Vec<CompiledShader> for deduplication
    let mut shaders = Vec::new();

    // Step 3: Resolve all tensor shapes
    let mut resolved_tensors = TensorRegistry::new();
    let mut tensor_id_map = std::collections::HashMap::new();

    for (orig_id, info) in graph.tensor_info.iter().enumerate() {
        let resolved_shape = match &info.shape {
            TensorShape::Static(dims) => TensorShape::Static(dims.clone()),
            TensorShape::Dynamic(dims) => {
                let mut resolved_dims = Vec::with_capacity(dims.len());
                for dim in dims {
                    match dim {
                        Dimension::Static(size) => resolved_dims.push(*size),
                        Dimension::Named(name) => {
                            let size = dynamic_dimensions.get(name).ok_or_else(|| {
                                CodegenError::InvalidShape(format!(
                                    "Dynamic dimension '{}' not provided in dynamic_dimensions",
                                    name
                                ))
                            })?;
                            resolved_dims.push(*size);
                        }
                    }
                }
                TensorShape::Static(resolved_dims)
            }
            TensorShape::Unknown => {
                return Err(CodegenError::InvalidShape(format!(
                    "Tensor '{}' has unknown shape, cannot compile",
                    info.name
                )));
            }
        };

        let resolved_info = onyxia_onnx::TensorInfo {
            name: info.name.clone(),
            dtype: info.dtype,
            shape: resolved_shape,
            kind: info.kind,
            initializer: info.initializer.clone(),
        };

        let new_id = resolved_tensors.add(resolved_info);
        tensor_id_map.insert(orig_id, new_id);
    }

    // Step 4: For each node in scheduled order, plan operations
    let mut operations = Vec::new();

    for &node_id in &ordered_nodes {
        let node = &graph.nodes[node_id];

        // Look up kernel by op_type
        let kernel = registry
            .get(&node.op_type)
            .ok_or_else(|| CodegenError::UnsupportedOp(node.op_type.clone()))?;

        // Resolve input tensor names → IDs
        let input_ids: Vec<_> = node
            .inputs
            .iter()
            .filter(|name| !name.is_empty()) // Skip empty inputs (optional inputs in ONNX)
            .map(|name| {
                let orig_id = graph.tensor_id(name)?;
                tensor_id_map
                    .get(&orig_id)
                    .copied()
                    .ok_or_else(|| OnnxError::MissingTensor(name.clone()).into())
            })
            .collect::<Result<_>>()?;

        // Resolve output tensor names → IDs
        let output_ids: Vec<_> = node
            .outputs
            .iter()
            .filter(|name| !name.is_empty()) // Skip empty outputs
            .map(|name| {
                let orig_id = graph.tensor_id(name)?;
                tensor_id_map
                    .get(&orig_id)
                    .copied()
                    .ok_or_else(|| OnnxError::MissingTensor(name.clone()).into())
            })
            .collect::<Result<_>>()?;

        // Construct PlanContext
        let mut ctx = PlanContext::new(
            node,
            graph,
            &input_ids,
            &output_ids,
            dynamic_dimensions,
            &mut shaders,
        );

        // Call kernel.plan() → get Vec<Step>
        let steps = kernel.plan(&mut ctx)?;

        // Build PlannedOp from the node + steps + scratch buffers
        let planned_op = PlannedOp {
            name: if node.name.is_empty() {
                format!("{}_{}", node.op_type, node_id)
            } else {
                node.name.clone()
            },
            op_type: node.op_type.clone(),
            inputs: input_ids.clone(),
            outputs: output_ids.clone(),
            steps,
            scratch_buffers: ctx.scratch_buffers,
        };

        operations.push(planned_op);
    }

    // Step 5: Map input/output names to resolved IDs
    let inputs: Vec<_> = graph
        .inputs
        .iter()
        .map(|name| {
            let orig_id = graph.tensor_id(name)?;
            tensor_id_map
                .get(&orig_id)
                .copied()
                .ok_or_else(|| OnnxError::MissingTensor(name.clone()).into())
        })
        .collect::<Result<_>>()?;

    let outputs: Vec<_> = graph
        .outputs
        .iter()
        .map(|name| {
            let orig_id = graph.tensor_id(name)?;
            tensor_id_map
                .get(&orig_id)
                .copied()
                .ok_or_else(|| OnnxError::MissingTensor(name.clone()).into())
        })
        .collect::<Result<_>>()?;

    // Step 6: Build ExecutionPlan
    let plan = ExecutionPlan {
        operations,
        shaders,
        tensors: resolved_tensors,
        inputs,
        outputs,
        metadata: ModelMetadata {
            name: graph.metadata.name.clone(),
            version: graph.metadata.model_version,
            ir_version: graph.metadata.ir_version,
            producer: graph.metadata.producer_name.clone(),
        },
    };

    Ok(plan)
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

    // Helper function to create a simple Add graph
    fn make_add_graph() -> Graph {
        use onyxia_onnx::{DataType, Node, TensorInfo, TensorKind, TensorShape};

        let mut graph = Graph::new();

        // 3 tensors: two inputs, one output
        let _a = graph.add_tensor(TensorInfo {
            name: "a".into(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Input,
            initializer: None,
        });
        let _b = graph.add_tensor(TensorInfo {
            name: "b".into(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Input,
            initializer: None,
        });
        let _c = graph.add_tensor(TensorInfo {
            name: "c".into(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Output,
            initializer: None,
        });

        // 1 node: Add(a, b) -> c
        let mut node = Node::new("Add");
        node.name = "add_0".into();
        node.inputs = vec!["a".into(), "b".into()];
        node.outputs = vec!["c".into()];
        graph.add_node(node);

        graph.inputs = vec!["a".into(), "b".into()];
        graph.outputs = vec!["c".into()];

        graph
    }

    #[test]
    fn test_compile_to_plan_basic() {
        let graph = make_add_graph();
        let registry = KernelRegistry::with_defaults();
        let dynamic_dimensions = std::collections::HashMap::new();

        let result = compile_to_plan(&graph, &registry, &dynamic_dimensions);
        assert!(result.is_ok(), "compile_to_plan should succeed");

        let plan = result.unwrap();

        // Assert result has 1 operation with 1 step
        assert_eq!(plan.operations.len(), 1, "Should have 1 operation");
        let op = &plan.operations[0];
        assert_eq!(op.op_type, "Add");
        assert_eq!(op.steps.len(), 1, "Should have 1 step");

        // Assert step.shader_index == 0
        if let Step::Dispatch { shader_index, .. } = &op.steps[0] {
            assert_eq!(*shader_index, 0, "Shader index should be 0");
        } else {
            panic!("Expected Dispatch step");
        }

        // Assert plan.shaders.len() == 1
        assert_eq!(plan.shaders.len(), 1, "Should have 1 compiled shader");

        // Assert plan.inputs and plan.outputs are correctly mapped
        assert_eq!(plan.inputs.len(), 2, "Should have 2 inputs");
        assert_eq!(plan.outputs.len(), 1, "Should have 1 output");

        // Assert all tensor shapes in plan.tensors are TensorShape::Static
        for info in plan.tensors.all() {
            assert!(
                info.shape.is_static(),
                "All shapes should be static, but {} is {:?}",
                info.name,
                info.shape
            );
        }
    }

    #[test]
    fn test_compile_to_plan_unsupported_op() {
        use onyxia_onnx::{DataType, Node, TensorInfo, TensorKind, TensorShape};

        let mut graph = Graph::new();

        // Add tensors
        graph.add_tensor(TensorInfo {
            name: "input".into(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Input,
            initializer: None,
        });
        graph.add_tensor(TensorInfo {
            name: "output".into(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Output,
            initializer: None,
        });

        // Add node with unsupported operation
        let mut node = Node::new("UnsupportedOp");
        node.inputs = vec!["input".into()];
        node.outputs = vec!["output".into()];
        graph.add_node(node);

        graph.inputs = vec!["input".into()];
        graph.outputs = vec!["output".into()];

        let registry = KernelRegistry::with_defaults();
        let dynamic_dimensions = std::collections::HashMap::new();

        let result = compile_to_plan(&graph, &registry, &dynamic_dimensions);
        assert!(result.is_err(), "Should fail with unsupported op");

        match result.unwrap_err() {
            CodegenError::UnsupportedOp(op) => {
                assert_eq!(op, "UnsupportedOp");
            }
            e => panic!("Expected UnsupportedOp error, got {:?}", e),
        }
    }

    #[test]
    fn test_compile_to_plan_unresolved_dynamic_dim() {
        use onyxia_onnx::{DataType, Dimension, Node, TensorInfo, TensorKind, TensorShape};

        let mut graph = Graph::new();

        // Add tensors with dynamic dimensions
        graph.add_tensor(TensorInfo {
            name: "a".into(),
            dtype: DataType::F32,
            shape: TensorShape::Dynamic(vec![
                Dimension::Named("batch".into()),
                Dimension::Static(4),
            ]),
            kind: TensorKind::Input,
            initializer: None,
        });
        graph.add_tensor(TensorInfo {
            name: "b".into(),
            dtype: DataType::F32,
            shape: TensorShape::Dynamic(vec![
                Dimension::Named("batch".into()),
                Dimension::Static(4),
            ]),
            kind: TensorKind::Input,
            initializer: None,
        });
        graph.add_tensor(TensorInfo {
            name: "c".into(),
            dtype: DataType::F32,
            shape: TensorShape::Dynamic(vec![
                Dimension::Named("batch".into()),
                Dimension::Static(4),
            ]),
            kind: TensorKind::Output,
            initializer: None,
        });

        let mut node = Node::new("Add");
        node.inputs = vec!["a".into(), "b".into()];
        node.outputs = vec!["c".into()];
        graph.add_node(node);

        graph.inputs = vec!["a".into(), "b".into()];
        graph.outputs = vec!["c".into()];

        let registry = KernelRegistry::with_defaults();
        // Empty dynamic_dimensions - missing "batch"
        let dynamic_dimensions = std::collections::HashMap::new();

        let result = compile_to_plan(&graph, &registry, &dynamic_dimensions);
        assert!(
            result.is_err(),
            "Should fail with unresolved dynamic dimension"
        );

        match result.unwrap_err() {
            CodegenError::InvalidShape(msg) => {
                assert!(
                    msg.contains("batch"),
                    "Error should mention 'batch' dimension"
                );
            }
            e => panic!("Expected InvalidShape error, got {:?}", e),
        }
    }

    #[test]
    fn test_compile_to_plan_with_dynamic_dims_resolved() {
        use onyxia_onnx::{DataType, Dimension, Node, TensorInfo, TensorKind, TensorShape};

        let mut graph = Graph::new();

        // Add tensors with dynamic dimensions
        graph.add_tensor(TensorInfo {
            name: "a".into(),
            dtype: DataType::F32,
            shape: TensorShape::Dynamic(vec![
                Dimension::Named("batch".into()),
                Dimension::Static(4),
            ]),
            kind: TensorKind::Input,
            initializer: None,
        });
        graph.add_tensor(TensorInfo {
            name: "b".into(),
            dtype: DataType::F32,
            shape: TensorShape::Dynamic(vec![
                Dimension::Named("batch".into()),
                Dimension::Static(4),
            ]),
            kind: TensorKind::Input,
            initializer: None,
        });
        graph.add_tensor(TensorInfo {
            name: "c".into(),
            dtype: DataType::F32,
            shape: TensorShape::Dynamic(vec![
                Dimension::Named("batch".into()),
                Dimension::Static(4),
            ]),
            kind: TensorKind::Output,
            initializer: None,
        });

        let mut node = Node::new("Add");
        node.inputs = vec!["a".into(), "b".into()];
        node.outputs = vec!["c".into()];
        graph.add_node(node);

        graph.inputs = vec!["a".into(), "b".into()];
        graph.outputs = vec!["c".into()];

        let registry = KernelRegistry::with_defaults();
        let mut dynamic_dimensions = std::collections::HashMap::new();
        dynamic_dimensions.insert("batch".to_string(), 2);

        let result = compile_to_plan(&graph, &registry, &dynamic_dimensions);
        assert!(
            result.is_ok(),
            "Should succeed with provided dynamic dimensions"
        );

        let plan = result.unwrap();

        // Verify all shapes are resolved to Static
        for info in plan.tensors.all() {
            if let TensorShape::Static(dims) = &info.shape {
                assert_eq!(dims[0], 2, "Batch dimension should be resolved to 2");
                assert_eq!(dims[1], 4, "Second dimension should be 4");
            } else {
                panic!("Expected Static shape after resolution");
            }
        }
    }

    #[test]
    fn test_compile_to_plan_unknown_shape_error() {
        use onyxia_onnx::{DataType, Node, TensorInfo, TensorKind, TensorShape};

        let mut graph = Graph::new();

        // Add tensor with unknown shape
        graph.add_tensor(TensorInfo {
            name: "input".into(),
            dtype: DataType::F32,
            shape: TensorShape::Unknown,
            kind: TensorKind::Input,
            initializer: None,
        });
        graph.add_tensor(TensorInfo {
            name: "output".into(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Output,
            initializer: None,
        });

        let mut node = Node::new("Add");
        node.inputs = vec!["input".into()];
        node.outputs = vec!["output".into()];
        graph.add_node(node);

        graph.inputs = vec!["input".into()];
        graph.outputs = vec!["output".into()];

        let registry = KernelRegistry::with_defaults();
        let dynamic_dimensions = std::collections::HashMap::new();

        let result = compile_to_plan(&graph, &registry, &dynamic_dimensions);
        assert!(result.is_err(), "Should fail with unknown shape");

        match result.unwrap_err() {
            CodegenError::InvalidShape(msg) => {
                assert!(
                    msg.contains("unknown"),
                    "Error should mention unknown shape"
                );
            }
            e => panic!("Expected InvalidShape error, got {:?}", e),
        }
    }
}
