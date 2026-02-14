//! End-to-end tests for elementwise operations.
//!
//! Tests: Add, Sub, Mul

mod common;

use common::make_binary_elementwise_graph;
use onyxia_onnx::DataType;
use onyxia_planner::{KernelRegistry, compile};
use onyxia_runtime::{Runtime, Tensor};
use std::collections::HashMap;

/// End-to-end test: Add two vectors on GPU and verify correct output.
#[pollster::test]
#[ignore] // Requires GPU - ignore in CI or environments without GPU
async fn test_add_e2e() {
    // Build graph
    let graph = make_binary_elementwise_graph("Add", "add_node", DataType::F32, &[4]);
    graph.validate().expect("Graph validation should succeed");

    // Compile to ExecutionPlan
    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    // Verify plan structure
    assert_eq!(plan.operations.len(), 1);
    assert_eq!(plan.shaders.len(), 1);
    assert_eq!(plan.inputs.len(), 2);
    assert_eq!(plan.outputs.len(), 1);

    // Initialize runtime and load plan
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

    // Run with concrete inputs
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]);
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], &[4]);

    let outputs = executor
        .run(&[("a", a), ("b", b)])
        .expect("Execution should succeed");

    // Verify output
    let c = outputs["c"].to_vec::<f32>().expect("Should convert to f32");
    assert_eq!(c, vec![6.0, 8.0, 10.0, 12.0], "Add result incorrect");

    println!("✓ End-to-end Add test passed!");
    println!("  Input a: [1.0, 2.0, 3.0, 4.0]");
    println!("  Input b: [5.0, 6.0, 7.0, 8.0]");
    println!("  Output c: {:?}", c);
}

/// End-to-end test: Subtract two vectors on GPU and verify correct output.
#[pollster::test]
#[ignore] // Requires GPU - ignore in CI or environments without GPU
async fn test_sub_e2e() {
    // Build graph
    let graph = make_binary_elementwise_graph("Sub", "sub_node", DataType::F32, &[4]);
    graph.validate().expect("Graph validation should succeed");

    // Compile to ExecutionPlan
    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    // Verify plan structure
    assert_eq!(plan.operations.len(), 1);
    assert_eq!(plan.shaders.len(), 1);

    // Initialize runtime and load plan
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Run with concrete inputs
    let a = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], &[4]);
    let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]);

    let outputs = executor
        .run(&[("a", a), ("b", b)])
        .expect("Execution should succeed");

    // Verify output
    let c = outputs["c"].to_vec::<f32>().expect("Should convert to f32");
    assert_eq!(c, vec![4.0, 4.0, 4.0, 4.0], "Sub result incorrect");

    println!("✓ End-to-end Sub test passed!");
    println!("  Input a: [5.0, 6.0, 7.0, 8.0]");
    println!("  Input b: [1.0, 2.0, 3.0, 4.0]");
    println!("  Output c: {:?}", c);
}

/// End-to-end test: Multiply two vectors on GPU and verify correct output.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_mul_e2e() {
    // Build graph
    let graph = make_binary_elementwise_graph("Mul", "mul_node", DataType::F32, &[4]);
    graph.validate().expect("Graph validation should succeed");

    // Compile to ExecutionPlan
    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    // Initialize runtime and load plan
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Execute: c = a * b
    let a = Tensor::from_vec(vec![2.0f32, 3.0, 4.0, 5.0], &[4]);
    let b = Tensor::from_vec(vec![1.5f32, 2.0, 2.5, 3.0], &[4]);

    let outputs = executor
        .run(&[("a", a), ("b", b)])
        .expect("Execution should succeed");

    let c = outputs["c"].to_vec::<f32>().expect("Should convert to f32");
    assert_eq!(c, vec![3.0, 6.0, 10.0, 15.0], "Mul result incorrect");

    println!("✓ End-to-end Mul test passed!");
    println!("  Input a: [2.0, 3.0, 4.0, 5.0]");
    println!("  Input b: [1.5, 2.0, 2.5, 3.0]");
    println!("  Output c: {:?}", c);
}

/// End-to-end test: Equal comparison on GPU and verify correct output.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_equal_e2e() {
    // Build graph with Bool output type for Equal operator
    let mut graph = onyxia_onnx::Graph::new();

    // Add input tensors (F32)
    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "a".to_string(),
        dtype: DataType::F32,
        shape: onyxia_onnx::TensorShape::Static(vec![4]),
        kind: onyxia_onnx::TensorKind::Input,
        initializer: None,
    });

    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "b".to_string(),
        dtype: DataType::F32,
        shape: onyxia_onnx::TensorShape::Static(vec![4]),
        kind: onyxia_onnx::TensorKind::Input,
        initializer: None,
    });

    // Add output tensor (Bool type per ONNX spec)
    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "c".to_string(),
        dtype: DataType::Bool,
        shape: onyxia_onnx::TensorShape::Static(vec![4]),
        kind: onyxia_onnx::TensorKind::Output,
        initializer: None,
    });

    // Create Equal operation node
    let mut node = onyxia_onnx::Node::new("Equal");
    node.name = "equal_node".to_string();
    node.inputs = vec!["a".to_string(), "b".to_string()];
    node.outputs = vec!["c".to_string()];
    graph.add_node(node);

    // Set graph inputs and outputs
    graph.inputs = vec!["a".to_string(), "b".to_string()];
    graph.outputs = vec!["c".to_string()];

    // Set metadata
    graph.metadata.name = "test_equal_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph.validate().expect("Graph validation should succeed");

    // Compile to ExecutionPlan
    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    // Initialize runtime and load plan
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Execute: c = (a == b)
    // Test with some equal and some not equal values
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]);
    let b = Tensor::from_vec(vec![1.0f32, 2.5, 3.0, 5.0], &[4]);

    let outputs = executor
        .run(&[("a", a), ("b", b)])
        .expect("Execution should succeed");

    // Output is stored as u32 (0 for false, 1 for true)
    let c = outputs["c"].to_vec::<u32>().expect("Should convert to u32");
    assert_eq!(
        c,
        vec![1, 0, 1, 0],
        "Equal result incorrect (1=true, 0=false)"
    );

    println!("✓ End-to-end Equal test passed!");
    println!("  Input a: [1.0, 2.0, 3.0, 4.0]");
    println!("  Input b: [1.0, 2.5, 3.0, 5.0]");
    println!("  Output c: {:?} (1=equal, 0=not equal)", c);
}

/// End-to-end test: Equal with broadcasting (scalar vs tensor).
#[pollster::test]
#[ignore] // Requires GPU
async fn test_equal_broadcast_e2e() {
    // Build graph with broadcasting: scalar == vector
    let mut graph = onyxia_onnx::Graph::new();

    // Add input tensors
    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "a".to_string(),
        dtype: DataType::F32,
        shape: onyxia_onnx::TensorShape::Static(vec![1]),
        kind: onyxia_onnx::TensorKind::Input,
        initializer: None,
    });

    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "b".to_string(),
        dtype: DataType::F32,
        shape: onyxia_onnx::TensorShape::Static(vec![4]),
        kind: onyxia_onnx::TensorKind::Input,
        initializer: None,
    });

    // Add output tensor (Bool type per ONNX spec)
    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "c".to_string(),
        dtype: DataType::Bool,
        shape: onyxia_onnx::TensorShape::Static(vec![4]),
        kind: onyxia_onnx::TensorKind::Output,
        initializer: None,
    });

    // Create Equal operation node
    let mut node = onyxia_onnx::Node::new("Equal");
    node.name = "equal_broadcast_node".to_string();
    node.inputs = vec!["a".to_string(), "b".to_string()];
    node.outputs = vec!["c".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["a".to_string(), "b".to_string()];
    graph.outputs = vec!["c".to_string()];

    graph.metadata.name = "test_equal_broadcast_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph.validate().expect("Graph validation should succeed");

    // Compile and execute
    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Broadcast scalar 2.0 against vector [1.0, 2.0, 3.0, 2.0]
    let a = Tensor::from_vec(vec![2.0f32], &[1]);
    let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 2.0], &[4]);

    let outputs = executor
        .run(&[("a", a), ("b", b)])
        .expect("Execution should succeed");

    let c = outputs["c"].to_vec::<u32>().expect("Should convert to u32");
    assert_eq!(
        c,
        vec![0, 1, 0, 1],
        "Equal broadcast result incorrect (1=true, 0=false)"
    );

    println!("✓ End-to-end Equal broadcast test passed!");
    println!("  Input a: [2.0] (scalar)");
    println!("  Input b: [1.0, 2.0, 3.0, 2.0]");
    println!("  Output c: {:?} (1=equal, 0=not equal)", c);
}
