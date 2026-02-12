//! End-to-end tests for plan execution.
//!
//! These tests verify the complete pipeline from Graph → ExecutionPlan → GPU execution.

use onyxia_codegen::{KernelRegistry, compile_to_plan};
use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_runtime::{Runtime, Tensor};
use std::collections::HashMap;

/// Helper function to create a simple Add graph programmatically.
///
/// Graph structure:
/// - Inputs: a:[f32;4], b:[f32;4]
/// - Operation: Add(a, b) -> c
/// - Output: c:[f32;4]
fn make_add_graph() -> Graph {
    let mut graph = Graph::new();

    // Add input tensor 'a'
    graph.add_tensor(TensorInfo {
        name: "a".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add input tensor 'b'
    graph.add_tensor(TensorInfo {
        name: "b".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add output tensor 'c'
    graph.add_tensor(TensorInfo {
        name: "c".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Add node
    let mut node = Node::new("Add");
    node.name = "add_node".to_string();
    node.inputs = vec!["a".to_string(), "b".to_string()];
    node.outputs = vec!["c".to_string()];
    graph.add_node(node);

    // Set graph inputs and outputs
    graph.inputs = vec!["a".to_string(), "b".to_string()];
    graph.outputs = vec!["c".to_string()];

    // Set metadata
    graph.metadata.name = "test_add_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

/// End-to-end test: Add two vectors on GPU and verify correct output.
///
/// This test verifies:
/// 1. Graph construction (programmatic)
/// 2. Compilation to ExecutionPlan via compile_to_plan()
/// 3. Plan loading into runtime
/// 4. GPU execution with PlanExecutor::run()
/// 5. Correct numerical results
#[pollster::test]
#[ignore] // Requires GPU - ignore in CI or environments without GPU
async fn test_add_e2e() {
    // Step 1: Build graph programmatically
    let graph = make_add_graph();

    // Validate graph structure
    graph.validate().expect("Graph validation should succeed");
    assert_eq!(graph.inputs.len(), 2);
    assert_eq!(graph.outputs.len(), 1);
    assert_eq!(graph.nodes.len(), 1);

    // Step 2: Compile to ExecutionPlan
    // No dynamic dimensions needed - all shapes are static
    let registry = KernelRegistry::with_defaults();
    let dynamic_dimensions = HashMap::new();

    let plan = compile_to_plan(&graph, &registry, &dynamic_dimensions)
        .expect("Compilation should succeed");

    // Verify plan structure
    assert_eq!(
        plan.operations.len(),
        1,
        "Should have exactly 1 operation (Add)"
    );
    assert_eq!(plan.shaders.len(), 1, "Should have exactly 1 shader");
    assert_eq!(plan.inputs.len(), 2, "Should have 2 inputs");
    assert_eq!(plan.outputs.len(), 1, "Should have 1 output");

    // Verify operation details
    let op = &plan.operations[0];
    assert_eq!(op.op_type, "Add");
    assert_eq!(op.inputs.len(), 2);
    assert_eq!(op.outputs.len(), 1);
    assert_eq!(op.steps.len(), 1, "Add should have 1 dispatch step");

    // Step 3: Initialize runtime and load plan
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    println!(
        "GPU: {} ({:?})",
        runtime.adapter_info().name,
        runtime.adapter_info().backend
    );

    let mut executor = runtime
        .load_plan(plan)
        .await
        .expect("Plan loading should succeed");

    // Step 4: Run with concrete inputs
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]);
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], &[4]);

    let outputs = executor
        .run(&[("a", a), ("b", b)])
        .expect("Execution should succeed");

    // Step 5: Verify output
    assert!(outputs.contains_key("c"), "Output 'c' should exist");

    let c = outputs["c"].to_vec::<f32>().expect("Should convert to f32");

    assert_eq!(c.len(), 4, "Output should have 4 elements");
    assert_eq!(c, vec![6.0, 8.0, 10.0, 12.0], "Add result incorrect");

    println!("✓ End-to-end Add test passed!");
    println!("  Input a: [1.0, 2.0, 3.0, 4.0]");
    println!("  Input b: [5.0, 6.0, 7.0, 8.0]");
    println!("  Output c: {:?}", c);
}

/// Test that empty plan can be loaded without crashing.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_empty_plan() {
    use onyxia_codegen::{ExecutionPlan, ModelMetadata, TensorRegistry};

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
    let result = runtime.load_plan(plan).await;
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
    let plan =
        compile_to_plan(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    assert_eq!(plan.operations.len(), 2, "Should have 2 operations");

    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_plan(plan)
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
