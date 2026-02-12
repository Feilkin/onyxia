//! Core end-to-end tests for plan execution.
//!
//! This file contains general tests for the execution plan infrastructure:
//! - Empty plan handling
//! - Multi-operation graphs
//!
//! Kernel-specific tests have been moved to separate files:
//! - elementwise_ops_test.rs: Add, Sub, Mul
//! - activation_ops_test.rs: Gelu
//! - normalization_ops_test.rs: RMSNorm
//! - matmul_ops_test.rs: MatMul, MatMulNBits
//! - shape_ops_test.rs: Reshape, Transpose, Concat
//! - indexing_ops_test.rs: Gather
//! - reduction_ops_test.rs: ReduceSum
//! - cast_ops_test.rs: Cast

use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_planner::{KernelRegistry, compile};
use onyxia_runtime::{Runtime, Tensor};
use std::collections::HashMap;

/// Test that empty plan can be loaded without crashing.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_empty_plan() {
    use onyxia_planner::{ExecutionPlan, ModelMetadata, TensorRegistry};

    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    // Create an empty execution plan
    let plan = ExecutionPlan {
        operations: Vec::new(),
        shaders: Vec::new(),
        tensors: TensorRegistry::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        metadata: ModelMetadata {
            name: "test_empty".to_string(),
            version: 1,
            ir_version: 9,
            producer: "test".to_string(),
        },
    };

    // Should be able to load an empty plan
    let result = runtime.load_model(plan).await;
    assert!(
        result.is_ok(),
        "Should be able to load empty plan: {:?}",
        result.err()
    );

    println!("✓ Empty plan test passed!");
}

/// Test plan execution with multiple operations.
///
/// Graph: d = (a + b) + c
/// This tests that multiple operations execute in sequence correctly.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_multiple_operations() {
    let mut graph = Graph::new();

    // Add tensors
    graph.add_tensor(TensorInfo {
        name: "a".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Input,
        initializer: None,
    });
    graph.add_tensor(TensorInfo {
        name: "b".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Input,
        initializer: None,
    });
    graph.add_tensor(TensorInfo {
        name: "c".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Input,
        initializer: None,
    });
    graph.add_tensor(TensorInfo {
        name: "temp".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Intermediate,
        initializer: None,
    });
    graph.add_tensor(TensorInfo {
        name: "d".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // First Add: temp = a + b
    let mut node1 = Node::new("Add");
    node1.name = "add1".to_string();
    node1.inputs = vec!["a".to_string(), "b".to_string()];
    node1.outputs = vec!["temp".to_string()];
    graph.add_node(node1);

    // Second Add: d = temp + c
    let mut node2 = Node::new("Add");
    node2.name = "add2".to_string();
    node2.inputs = vec!["temp".to_string(), "c".to_string()];
    node2.outputs = vec!["d".to_string()];
    graph.add_node(node2);

    // Set graph inputs and outputs
    graph.inputs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    graph.outputs = vec!["d".to_string()];

    // Set metadata
    graph.metadata.name = "test_double_add".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    // Compile and execute
    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    assert_eq!(plan.operations.len(), 2, "Should have 2 operations");

    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Execute: d = (1+2) + 3 = 6 for each element
    let a = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0], &[4]);
    let b = Tensor::from_vec(vec![2.0f32, 2.0, 2.0, 2.0], &[4]);
    let c = Tensor::from_vec(vec![3.0f32, 3.0, 3.0, 3.0], &[4]);

    let outputs = executor
        .run(&[("a", a), ("b", b), ("c", c)])
        .expect("Execution should succeed");

    let d = outputs["d"].to_vec::<f32>().expect("Should convert to f32");
    assert_eq!(d, vec![6.0, 6.0, 6.0, 6.0], "Double add result incorrect");

    println!("✓ Multiple operations test passed!");
    println!("  d = (a + b) + c = (1 + 2) + 3 = {:?}", d);
}

/// Test that initializer data is correctly uploaded to GPU.
///
/// This test creates a minimal plan with an initializer tensor and verifies
/// that plan loading succeeds (initializers are uploaded during plan loading).
#[pollster::test]
#[ignore] // Requires GPU
async fn test_initializer_upload() {
    use onyxia_planner::{ExecutionPlan, ModelMetadata, TensorRegistry};

    // Create a tensor registry with a tensor that has initializer data
    let mut tensors = TensorRegistry::new();

    // Add a tensor with known initializer data: [1.0, 2.0, 3.0, 4.0]
    let initializer_data: Vec<u8> = vec![
        0, 0, 128, 63, // 1.0f32 in little-endian
        0, 0, 0, 64, // 2.0f32
        0, 0, 64, 64, // 3.0f32
        0, 0, 128, 64, // 4.0f32
    ];

    tensors.add(TensorInfo {
        name: "weights".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Weight,
        initializer: Some(initializer_data.clone()),
    });

    // Create empty plan (no operations, just the initializer tensor)
    let plan = ExecutionPlan {
        operations: Vec::new(),
        shaders: Vec::new(),
        tensors,
        inputs: Vec::new(),
        outputs: Vec::new(),
        metadata: ModelMetadata {
            name: "test_initializer".to_string(),
            version: 1,
            ir_version: 9,
            producer: "onyxia_test".to_string(),
        },
    };

    // Load plan - this should upload initializer data to GPU
    let runtime = Runtime::new().await.expect("Runtime init should succeed");

    let result = runtime.load_model(plan).await;

    // If plan loading succeeds, initializer upload worked
    assert!(
        result.is_ok(),
        "Plan with initializer should load successfully: {:?}",
        result.err()
    );

    println!("✓ Initializer upload test passed!");
    println!("  Successfully uploaded 4 f32 values as initializer data");
}

/// End-to-end test: Add operation with bias tensor as initializer.
///
/// Graph: output = input + bias
/// Where bias is a constant tensor uploaded as initializer data.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_add_with_bias_e2e() {
    let mut graph = Graph::new();

    // Add input tensor
    graph.add_tensor(TensorInfo {
        name: "input".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add bias tensor with initializer [10.0, 20.0, 30.0, 40.0]
    let bias_data: Vec<u8> = vec![
        0, 0, 32, 65, // 10.0f32 in little-endian
        0, 0, 160, 65, // 20.0f32
        0, 0, 240, 65, // 30.0f32
        0, 0, 32, 66, // 40.0f32
    ];

    graph.add_tensor(TensorInfo {
        name: "bias".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Weight,
        initializer: Some(bias_data),
    });

    // Add output tensor
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Add node: output = input + bias
    let mut node = Node::new("Add");
    node.name = "add_bias".to_string();
    node.inputs = vec!["input".to_string(), "bias".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    // Set graph inputs and outputs
    graph.inputs = vec!["input".to_string()];
    graph.outputs = vec!["output".to_string()];

    // Set metadata
    graph.metadata.name = "test_add_bias".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    // Compile and execute
    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    assert_eq!(plan.operations.len(), 1, "Should have 1 operation");

    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Execute: output = [1,2,3,4] + [10,20,30,40] = [11,22,33,44]
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]);

    let outputs = executor
        .run(&[("input", input)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    assert_eq!(
        output,
        vec![11.0, 22.0, 33.0, 44.0],
        "Add with bias should produce correct result"
    );

    println!("✓ Add with bias test passed!");
    println!("  output = input + bias = {:?}", output);
}
