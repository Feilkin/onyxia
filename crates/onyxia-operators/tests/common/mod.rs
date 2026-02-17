//! Common test utilities for operator tests.
//!
//! This module provides shared helper functions and graph builders used across
//! both unit tests (shape inference, planning) and GPU integration tests.

use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use std::collections::HashMap;

// Re-export types commonly used in tests
pub use onyxia_compiler::CompilerPipeline;
pub use onyxia_core::Operator;
pub use onyxia_runtime::{Runtime, Tensor};

/// Create a two-input, one-output graph for binary elementwise operations.
///
/// Graph structure:
/// - Inputs: a:[dtype;shape], b:[dtype;shape]
/// - Operation: OpType(a, b) -> c
/// - Output: c:[dtype;shape]
///
/// # Arguments
/// * `op_type` - ONNX operator type (e.g., "Add", "Sub", "Mul")
/// * `node_name` - Name for the operation node
/// * `dtype` - Data type for tensors
/// * `shape` - Shape for input and output tensors
pub fn make_binary_elementwise_graph(
    op_type: &str,
    node_name: &str,
    dtype: DataType,
    shape: &[usize],
) -> Graph {
    let mut graph = Graph::new();

    // Add input tensor 'a'
    graph.add_tensor(TensorInfo {
        name: "a".to_string(),
        dtype,
        shape: TensorShape::Static(shape.to_vec()),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add input tensor 'b'
    graph.add_tensor(TensorInfo {
        name: "b".to_string(),
        dtype,
        shape: TensorShape::Static(shape.to_vec()),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add output tensor 'c'
    graph.add_tensor(TensorInfo {
        name: "c".to_string(),
        dtype,
        shape: TensorShape::Static(shape.to_vec()),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create operation node
    let mut node = Node::new(op_type);
    node.name = node_name.to_string();
    node.inputs = vec!["a".to_string(), "b".to_string()];
    node.outputs = vec!["c".to_string()];
    graph.add_node(node);

    // Set graph inputs and outputs
    graph.inputs = vec!["a".to_string(), "b".to_string()];
    graph.outputs = vec!["c".to_string()];

    // Set metadata
    graph.metadata.name = format!("test_{}_graph", op_type.to_lowercase());
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

/// Create a single-input, single-output graph for unary operations.
///
/// Graph structure:
/// - Input: input:[dtype;shape]
/// - Operation: OpType(input) -> output
/// - Output: output:[dtype;shape]
///
/// # Arguments
/// * `op_type` - ONNX operator type (e.g., "Gelu", "Tanh")
/// * `node_name` - Name for the operation node
/// * `dtype` - Data type for tensors
/// * `input_shape` - Shape for input tensor
/// * `output_shape` - Shape for output tensor
pub fn make_unary_graph(
    op_type: &str,
    node_name: &str,
    dtype: DataType,
    input_shape: &[usize],
    output_shape: &[usize],
) -> Graph {
    let mut graph = Graph::new();

    // Add input tensor
    graph.add_tensor(TensorInfo {
        name: "input".to_string(),
        dtype,
        shape: TensorShape::Static(input_shape.to_vec()),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add output tensor
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype,
        shape: TensorShape::Static(output_shape.to_vec()),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create operation node
    let mut node = Node::new(op_type);
    node.name = node_name.to_string();
    node.inputs = vec!["input".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    // Set graph inputs and outputs
    graph.inputs = vec!["input".to_string()];
    graph.outputs = vec!["output".to_string()];

    // Set metadata
    graph.metadata.name = format!("test_{}_graph", op_type.to_lowercase());
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

/// Compile and run a graph on GPU, returning output tensors.
///
/// This is a convenience helper that:
/// 1. Compiles the graph using the core operator registry
/// 2. Initializes a GPU runtime
/// 3. Loads the compiled model
/// 4. Executes with the provided inputs
/// 5. Returns outputs as a HashMap
///
/// # Arguments
/// * `graph` - The ONNX graph to execute
/// * `inputs` - Named input tensors (name, data)
///
/// # Returns
/// HashMap of output tensor names to their data as Vec<f32>
pub async fn compile_and_run_f32(
    graph: &Graph,
    inputs: &[(&str, Tensor)],
) -> HashMap<String, Vec<f32>> {
    // Compile graph
    let registry = onyxia_operators::core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline
        .compile(graph, &registry)
        .expect("Compilation should succeed");

    // Initialize runtime
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(model)
        .await
        .expect("Model loading should succeed");

    // Run with inputs
    let outputs = executor.run(inputs).expect("Execution should succeed");

    // Convert outputs to Vec<f32>
    outputs
        .into_iter()
        .map(|(name, tensor)| {
            let data = tensor.to_vec::<f32>().expect("Should convert to f32");
            (name, data)
        })
        .collect()
}
