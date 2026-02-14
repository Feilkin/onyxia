//! End-to-end tests for type conversion operations.
//!
//! Tests: Cast

use onyxia_onnx::{AttributeValue, DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_planner::{KernelRegistry, compile};
use onyxia_runtime::{Runtime, Tensor};
use std::collections::HashMap;

/// End-to-end test: Cast I64 to F32 and verify correct conversion.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_cast_i64_to_f32_e2e() {
    let mut graph = Graph::new();

    // Add input tensor (I64)
    graph.add_tensor(TensorInfo {
        name: "input".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add output tensor (F32)
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Cast node
    let mut node = Node::new("Cast");
    node.name = "cast_node".to_string();
    node.inputs = vec!["input".to_string()];
    node.outputs = vec!["output".to_string()];
    // Add "to" attribute: ONNX FLOAT = 1
    node.attributes
        .insert("to".to_string(), AttributeValue::Int(1));
    graph.add_node(node);

    graph.inputs = vec!["input".to_string()];
    graph.outputs = vec!["output".to_string()];

    // Set metadata
    graph.metadata.name = "test_cast_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    // Validate and compile
    graph.validate().expect("Graph validation should succeed");

    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    // Initialize runtime
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    println!(
        "GPU: {} ({:?})",
        runtime.adapter_info().name,
        runtime.adapter_info().backend
    );

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Test: Cast [1i64, 2i64, 3i64, 4i64] → [1.0f32, 2.0f32, 3.0f32, 4.0f32]
    let input = Tensor::from_vec(vec![1i64, 2, 3, 4], &[4]);

    let outputs = executor
        .run(&[("input", input)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    assert_eq!(
        output,
        vec![1.0, 2.0, 3.0, 4.0],
        "Cast I64→F32 result incorrect"
    );

    println!("✓ End-to-end Cast I64→F32 test passed!");
    println!("  Input: [1i64, 2i64, 3i64, 4i64]");
    println!("  Output: {:?}", output);
}
