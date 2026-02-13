//! Execution plan compiler for Onyxia.
//!
//! This crate takes ONNX graphs and compiles them into execution plans with
//! pre-compiled WGSL shaders that `onyxia-runtime` can execute on the GPU.
//!
//! # Example
//!
//! ```no_run
//! use onyxia_planner::{compile, KernelRegistry};
//! use onyxia_onnx::Graph;
//! use std::collections::HashMap;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Parse ONNX model to graph
//! # let graph = onyxia_onnx::Graph::new();
//!
//! // Compile to execution plan
//! let registry = KernelRegistry::with_defaults();
//! let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
//! let plan = compile(&graph, &registry, &dynamic_dimensions)?;
//!
//! println!("Compiled {} operations", plan.operations.len());
//! # Ok(())
//! # }
//! ```

pub mod error;
pub mod inference;
pub mod kernel;
pub mod kernels;
pub mod plan;
pub mod scheduler;
pub mod shape_inference;
pub mod symbolic_expr;

pub use error::{CodegenError, Result};
pub use kernel::{KernelRegistry, OpKernel, PlanContext};
pub use plan::{
    BindingDesc, BufferRef, CompiledShader, ExecutionPlan, ModelMetadata, PlannedOp,
    ScratchBufferDesc, ShaderIndex, Step, TensorRegistry,
};
pub use shape_inference::{infer_shapes, resolve_dynamic_dimensions};

use onyxia_onnx::{Graph, OnnxError};
use scheduler::Scheduler;

/// Compile an ONNX graph into an execution plan.
///
/// This is the main entry point for the planner crate. Uses the provided kernel
/// registry to map ONNX operations to GPU steps. `dynamic_dimensions` provides
/// concrete values for symbolic dimensions (e.g., {"batch": 1, "sequence": 512}),
/// allowing all shader defs to be fully resolved at plan time.
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
/// use onyxia_planner::{compile, KernelRegistry};
/// use onyxia_onnx::Graph;
/// use std::collections::HashMap;
///
/// # fn example(graph: &Graph) -> Result<(), Box<dyn std::error::Error>> {
/// let registry = KernelRegistry::with_defaults();
/// let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
/// let plan = compile(graph, &registry, &dynamic_dimensions)?;
/// println!("Compiled {} operations", plan.operations.len());
/// # Ok(())
/// # }
/// ```
pub fn compile(
    graph: &Graph,
    registry: &KernelRegistry,
    dynamic_dimensions: &std::collections::HashMap<String, usize>,
) -> Result<ExecutionPlan> {
    use onyxia_onnx::TensorShape;

    // Phase 1: Substitute all Dynamic(Named(...)) → Static using dynamic_dimensions
    let mut graph = graph.clone();
    shape_inference::resolve_dynamic_dimensions(&mut graph, dynamic_dimensions)?;

    // Phase 2: Iterative forward shape inference using kernel-defined rules
    shape_inference::infer_shapes(&mut graph, registry)?;

    // Step 1: Run scheduler to get topologically ordered node IDs
    let scheduler = Scheduler::new(graph.clone());
    let ordered_nodes = scheduler.schedule()?;

    // Step 2: Create shared Vec<CompiledShader> for deduplication
    let mut shaders = Vec::new();

    // Step 3: Validate all tensor shapes are Static, build resolved tensor registry
    let mut resolved_tensors = TensorRegistry::new();
    let mut tensor_id_map = std::collections::HashMap::new();

    for (orig_id, info) in graph.tensor_info.iter().enumerate() {
        match &info.shape {
            TensorShape::Static(_) => {}
            TensorShape::Absent => {
                // Optional input not provided — skip, it doesn't actually exist
                continue;
            }
            TensorShape::Unknown => {
                return Err(CodegenError::InvalidShape(format!(
                    "Tensor '{}' has unknown shape — shape inference failed for this operation",
                    info.name
                )));
            }
            TensorShape::Dynamic(_) => {
                return Err(CodegenError::InvalidShape(format!(
                    "Tensor '{}' still has Dynamic shape after Phase 1 — this is a bug",
                    info.name
                )));
            }
        }

        let resolved_info = onyxia_onnx::TensorInfo {
            name: info.name.clone(),
            dtype: info.dtype,
            shape: info.shape.clone(),
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
            &graph,
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
    fn test_compile_basic() {
        let graph = make_add_graph();
        let registry = KernelRegistry::with_defaults();
        let dynamic_dimensions = std::collections::HashMap::new();

        let result = compile(&graph, &registry, &dynamic_dimensions);
        assert!(result.is_ok(), "compile should succeed");

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
    fn test_compile_unsupported_op() {
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

        let result = compile(&graph, &registry, &dynamic_dimensions);
        assert!(result.is_err(), "Should fail with unsupported op");

        match result.unwrap_err() {
            CodegenError::UnsupportedOp(op) => {
                assert_eq!(op, "UnsupportedOp");
            }
            e => panic!("Expected UnsupportedOp error, got {:?}", e),
        }
    }

    #[test]
    fn test_compile_unresolved_dynamic_dim() {
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

        let result = compile(&graph, &registry, &dynamic_dimensions);
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
    fn test_compile_with_dynamic_dims_resolved() {
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

        let result = compile(&graph, &registry, &dynamic_dimensions);
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
    fn test_compile_unknown_shape_error() {
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

        let result = compile(&graph, &registry, &dynamic_dimensions);
        assert!(result.is_err(), "Should fail with unknown shape");

        match result.unwrap_err() {
            CodegenError::InvalidShape(msg) => {
                assert!(
                    msg.contains("nknown"),
                    "Error should mention unknown shape, got: {}",
                    msg
                );
            }
            e => panic!("Expected InvalidShape error, got {:?}", e),
        }
    }
}
