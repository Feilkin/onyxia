//! End-to-end tests for normalization operations.
//!
//! Tests: RMSNorm (SimplifiedLayerNormalization)

use onyxia_onnx::{AttributeValue, DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_planner::{KernelRegistry, compile};
use onyxia_runtime::{Runtime, Tensor};
use std::collections::HashMap;

/// End-to-end test: RMS Normalization on GPU and verify correct output.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_rmsnorm_e2e() {
    let mut graph = Graph::new();

    // Add input tensor [2, 4] - 2 sequences, 4 hidden dims
    graph.add_tensor(TensorInfo {
        name: "input".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 4]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add weight tensor [4]
    graph.add_tensor(TensorInfo {
        name: "weight".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add output tensor [2, 4]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 4]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create RMSNorm node
    let mut node = Node::new("SimplifiedLayerNormalization");
    node.name = "rmsnorm_node".to_string();
    node.inputs = vec!["input".to_string(), "weight".to_string()];
    node.outputs = vec!["output".to_string()];
    node.attributes
        .insert("epsilon".to_string(), AttributeValue::Float(1e-5));
    graph.add_node(node);

    graph.inputs = vec!["input".to_string(), "weight".to_string()];
    graph.outputs = vec!["output".to_string()];

    // Compile and execute
    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Test with simple values
    // Input: [[1, 2, 3, 4], [2, 4, 6, 8]]
    // Weight: [1, 1, 1, 1] (no scaling)
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0], &[2, 4]);
    let weight = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0], &[4]);

    let outputs = executor
        .run(&[("input", input), ("weight", weight)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");
    assert_eq!(output.len(), 8);

    // For first sequence [1,2,3,4]:
    // mean_square = (1+4+9+16)/4 = 7.5
    // rms = sqrt(7.5) ≈ 2.7386
    // normalized ≈ [0.365, 0.730, 1.095, 1.460]
    let first_rms = (7.5f32).sqrt();
    let expected_0 = 1.0 / first_rms;
    assert!(
        (output[0] - expected_0).abs() < 0.01,
        "First element should be ~{:.3}, got {:.3}",
        expected_0,
        output[0]
    );

    println!("✓ End-to-end RMSNorm test passed!");
    println!("  Input: [[1, 2, 3, 4], [2, 4, 6, 8]]");
    println!("  Weight: [1, 1, 1, 1]");
    println!("  Output: {:?}", output);
}
