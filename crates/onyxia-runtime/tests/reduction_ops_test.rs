//! End-to-end tests for reduction operations.
//!
//! Tests: ReduceSum

use onyxia_onnx::{AttributeValue, DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_planner::{KernelRegistry, compile};
use onyxia_runtime::{Runtime, Tensor};
use std::collections::HashMap;

/// Helper function to create a ReduceSum graph.
fn make_reducesum_graph() -> Graph {
    let mut graph = Graph::new();

    // Add input tensor with 4 elements
    graph.add_tensor(TensorInfo {
        name: "input".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add output tensor with 1 element
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create ReduceSum node
    let mut node = Node::new("ReduceSum");
    node.name = "reducesum_node".to_string();
    node.inputs = vec!["input".to_string()];
    node.outputs = vec!["output".to_string()];
    node.attributes
        .insert("axes".to_string(), AttributeValue::Ints(vec![0i64]));
    node.attributes
        .insert("keepdims".to_string(), AttributeValue::Int(1i64));
    graph.add_node(node);

    graph.inputs = vec!["input".to_string()];
    graph.outputs = vec!["output".to_string()];

    graph.metadata.name = "test_reducesum_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

/// End-to-end test: Sum reduction on GPU.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_reducesum_e2e() {
    let graph = make_reducesum_graph();
    graph.validate().expect("Graph validation should succeed");

    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new()
        .await
        .expect("Runtime creation should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Model loading should succeed");

    // Prepare input: [1, 2, 3, 4], sum should be 10
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]);

    let outputs = executor
        .run(&[("input", input)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    assert_eq!(output.len(), 1);
    assert!(
        (output[0] - 10.0).abs() < 1e-5,
        "ReduceSum result incorrect"
    );

    println!("✓ End-to-end ReduceSum test passed!");
    println!("  Input: [1, 2, 3, 4] (shape [4])");
    println!("  Output: [10] (shape [1])");
}
