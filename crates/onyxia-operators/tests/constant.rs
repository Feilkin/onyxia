//! Tests for the Constant operator.

mod common;

use common::{CompilerPipeline, Runtime};
use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_operators::core_operator_registry;

/// Helper to create a Constant graph returning a constant tensor.
fn make_constant_graph(data: Vec<u8>, dtype: DataType, shape: &[usize]) -> Graph {
    let mut graph = Graph::new();

    // Add constant output tensor with initializer data
    graph.add_tensor(TensorInfo {
        name: "constant_out".to_string(),
        dtype,
        shape: TensorShape::Static(shape.to_vec()),
        kind: TensorKind::Output,
        initializer: Some(data),
    });

    // Create Constant node
    let mut node = Node::new("Constant");
    node.name = "constant_node".to_string();
    node.inputs = vec![]; // No inputs
    node.outputs = vec!["constant_out".to_string()];
    graph.add_node(node);

    // Set graph inputs and outputs
    graph.inputs = vec![]; // No inputs
    graph.outputs = vec!["constant_out".to_string()];

    // Set metadata
    graph.metadata.name = "test_constant_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

#[pollster::test]
#[ignore] // Requires GPU
async fn test_constant_f32() {
    // Create constant data: [1.0, 2.0, 3.0, 4.0]
    let constant_data = vec![
        0x00, 0x00, 0x80, 0x3f, // 1.0f32
        0x00, 0x00, 0x00, 0x40, // 2.0f32
        0x00, 0x00, 0x40, 0x40, // 3.0f32
        0x00, 0x00, 0x80, 0x40, // 4.0f32
    ];

    let graph = make_constant_graph(constant_data, DataType::F32, &[4]);

    // Compile
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    // Execute (no inputs needed)
    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_model(model)
        .await
        .expect("Model loading should succeed");

    let outputs = executor.run(&[]).expect("Execution should succeed");

    // Verify constant values
    let output = outputs.get("constant_out").expect("Output should exist");
    let data = output.to_vec::<f32>().expect("Should convert to f32");

    assert_eq!(data.len(), 4);
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);

    println!("✓ Constant f32 test passed!");
}

#[pollster::test]
#[ignore] // Requires GPU
async fn test_constant_i64() {
    // Create constant data: [10, 20, 30]
    let constant_data: Vec<u8> = vec![10i64, 20, 30]
        .iter()
        .flat_map(|&v| v.to_le_bytes())
        .collect();

    let graph = make_constant_graph(constant_data, DataType::I64, &[3]);

    // Compile
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    // Execute
    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_model(model)
        .await
        .expect("Model loading should succeed");

    let outputs = executor.run(&[]).expect("Execution should succeed");

    // Verify constant values
    let output = outputs.get("constant_out").expect("Output should exist");
    let data = output.to_vec::<i64>().expect("Should convert to i64");

    assert_eq!(data.len(), 3);
    assert_eq!(data, vec![10, 20, 30]);

    println!("✓ Constant i64 test passed!");
}

#[pollster::test]
#[ignore] // Requires GPU
async fn test_constant_scalar() {
    // Create constant scalar: 42.0f32
    let constant_data = 42.0f32.to_le_bytes().to_vec();

    let graph = make_constant_graph(constant_data, DataType::F32, &[]); // Empty shape = scalar

    // Compile
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    // Execute
    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_model(model)
        .await
        .expect("Model loading should succeed");

    let outputs = executor.run(&[]).expect("Execution should succeed");

    // Verify constant value
    let output = outputs.get("constant_out").expect("Output should exist");
    let data = output.to_vec::<f32>().expect("Should convert to f32");

    assert_eq!(data.len(), 1);
    assert_eq!(data[0], 42.0);

    println!("✓ Constant scalar test passed!");
}

#[pollster::test]
#[ignore] // Requires GPU
async fn test_constant_multidimensional() {
    // Create constant 2x3 matrix: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    let constant_data: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .flat_map(|&v| v.to_le_bytes())
        .collect();

    let graph = make_constant_graph(constant_data, DataType::F32, &[2, 3]);

    // Compile
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    // Execute
    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_model(model)
        .await
        .expect("Model loading should succeed");

    let outputs = executor.run(&[]).expect("Execution should succeed");

    // Verify constant values
    let output = outputs.get("constant_out").expect("Output should exist");
    let data = output.to_vec::<f32>().expect("Should convert to f32");

    assert_eq!(data.len(), 6);
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    println!("✓ Constant multidimensional test passed!");
}
