//! End-to-end tests for elementwise operations.
//!
//! Tests: Add, Sub, Mul, Div, Pow, Max

mod common;

use common::make_binary_elementwise_graph;
use onyxia_onnx::{DataType, Graph, TensorInfo, TensorKind, TensorShape};
use onyxia_compiler::{OperatorRegistry, compile};
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
    let registry = OperatorRegistry::with_defaults();
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
    let registry = OperatorRegistry::with_defaults();
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
    let registry = OperatorRegistry::with_defaults();
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

/// End-to-end test: Divide two vectors on GPU and verify correct output.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_div_e2e() {
    // Build graph
    let graph = make_binary_elementwise_graph("Div", "div_node", DataType::F32, &[4]);
    graph.validate().expect("Graph validation should succeed");

    // Compile to ExecutionPlan
    let registry = OperatorRegistry::with_defaults();
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

    // Execute: c = a / b
    let a = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], &[4]);
    let b = Tensor::from_vec(vec![2.0f32, 4.0, 5.0, 8.0], &[4]);

    let outputs = executor
        .run(&[("a", a), ("b", b)])
        .expect("Execution should succeed");

    let c = outputs["c"].to_vec::<f32>().expect("Should convert to f32");
    assert_eq!(c, vec![5.0, 5.0, 6.0, 5.0], "Div result incorrect");

    println!("✓ End-to-end Div test passed!");
    println!("  Input a: [10.0, 20.0, 30.0, 40.0]");
    println!("  Input b: [2.0, 4.0, 5.0, 8.0]");
    println!("  Output c: {:?}", c);
}

/// End-to-end test: Division by one (identity operation).
#[pollster::test]
#[ignore] // Requires GPU
async fn test_div_by_one_e2e() {
    // Build graph
    let graph = make_binary_elementwise_graph("Div", "div_by_one_node", DataType::F32, &[4]);
    graph.validate().expect("Graph validation should succeed");

    // Compile to ExecutionPlan
    let registry = OperatorRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    // Initialize runtime and load plan
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Execute: c = a / 1
    let a = Tensor::from_vec(vec![1.5f32, 2.75, 3.125, 4.25], &[4]);
    let b = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0], &[4]);

    let outputs = executor
        .run(&[("a", a), ("b", b)])
        .expect("Execution should succeed");

    let c = outputs["c"].to_vec::<f32>().expect("Should convert to f32");
    assert_eq!(
        c,
        vec![1.5, 2.75, 3.125, 4.25],
        "Div by one should preserve values"
    );

    println!("✓ End-to-end Div by one test passed!");
    println!("  Input a: [1.5, 2.75, 3.125, 4.25]");
    println!("  Input b: [1.0, 1.0, 1.0, 1.0]");
    println!("  Output c: {:?}", c);
}

/// End-to-end test: Division with broadcasting (tensor / scalar).
#[pollster::test]
#[ignore] // Requires GPU
async fn test_div_broadcast_scalar_e2e() {
    // Create graph with broadcast: [4] / [1] -> [4]
    let mut graph = onyxia_onnx::Graph::new();

    // Input tensor [4]
    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "a".to_string(),
        dtype: DataType::F32,
        shape: onyxia_onnx::TensorShape::Static(vec![4]),
        kind: onyxia_onnx::TensorKind::Input,
        initializer: None,
    });

    // Scalar (broadcast) [1]
    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "b".to_string(),
        dtype: DataType::F32,
        shape: onyxia_onnx::TensorShape::Static(vec![1]),
        kind: onyxia_onnx::TensorKind::Input,
        initializer: None,
    });

    // Output tensor [4]
    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "c".to_string(),
        dtype: DataType::F32,
        shape: onyxia_onnx::TensorShape::Static(vec![4]),
        kind: onyxia_onnx::TensorKind::Output,
        initializer: None,
    });

    // Create Div operation node
    let mut node = onyxia_onnx::Node::new("Div");
    node.name = "div_broadcast_node".to_string();
    node.inputs = vec!["a".to_string(), "b".to_string()];
    node.outputs = vec!["c".to_string()];
    graph.add_node(node);

    // Set graph inputs and outputs
    graph.inputs = vec!["a".to_string(), "b".to_string()];
    graph.outputs = vec!["c".to_string()];

    // Set metadata
    graph.metadata.name = "test_div_broadcast_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph.validate().expect("Graph validation should succeed");

    // Compile to ExecutionPlan
    let registry = OperatorRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    // Initialize runtime and load plan
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Execute: c = [10, 20, 30, 40] / [2]
    let a = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], &[4]);
    let b = Tensor::from_vec(vec![2.0f32], &[1]);

    let outputs = executor
        .run(&[("a", a), ("b", b)])
        .expect("Execution should succeed");

    let c = outputs["c"].to_vec::<f32>().expect("Should convert to f32");
    assert_eq!(
        c,
        vec![5.0, 10.0, 15.0, 20.0],
        "Div broadcast result incorrect"
    );

    println!("✓ End-to-end Div broadcast test passed!");
    println!("  Input a: [10.0, 20.0, 30.0, 40.0]");
    println!("  Input b (scalar): [2.0]");
    println!("  Output c: {:?}", c);
}

/// End-to-end test: Division by zero produces infinity.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_div_by_zero_e2e() {
    // Build graph
    let graph = make_binary_elementwise_graph("Div", "div_by_zero_node", DataType::F32, &[4]);
    graph.validate().expect("Graph validation should succeed");

    // Compile to ExecutionPlan
    let registry = OperatorRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    // Initialize runtime and load plan
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Execute: c = a / 0
    let a = Tensor::from_vec(vec![1.0f32, -2.0, 0.0, 10.0], &[4]);
    let b = Tensor::from_vec(vec![0.0f32, 0.0, 0.0, 0.0], &[4]);

    let outputs = executor
        .run(&[("a", a), ("b", b)])
        .expect("Execution should succeed");

    let c = outputs["c"].to_vec::<f32>().expect("Should convert to f32");

    // Verify infinity and NaN behavior (IEEE 754 floating point rules)
    assert!(
        c[0].is_infinite() && c[0].is_sign_positive(),
        "1.0/0.0 should be +inf"
    );
    assert!(
        c[1].is_infinite() && c[1].is_sign_negative(),
        "-2.0/0.0 should be -inf"
    );
    assert!(c[2].is_nan(), "0.0/0.0 should be NaN");
    assert!(
        c[3].is_infinite() && c[3].is_sign_positive(),
        "10.0/0.0 should be +inf"
    );

    println!("✓ End-to-end Div by zero test passed!");
    println!("  Input a: [1.0, -2.0, 0.0, 10.0]");
    println!("  Input b: [0.0, 0.0, 0.0, 0.0]");
    println!("  Output c: {:?}", c);
    println!("  Verified: infinity and NaN behavior");
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
    let registry = OperatorRegistry::with_defaults();
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
    let registry = OperatorRegistry::with_defaults();
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

/// End-to-end test: Greater comparison on GPU and verify correct output.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_greater_e2e() {
    // Build graph with Bool output type for Greater operator
    let mut graph = onyxia_onnx::Graph::new();

    // Add input tensors (F32)
    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "a".to_string(),
        dtype: DataType::F32,
        shape: onyxia_onnx::TensorShape::Static(vec![6]),
        kind: onyxia_onnx::TensorKind::Input,
        initializer: None,
    });

    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "b".to_string(),
        dtype: DataType::F32,
        shape: onyxia_onnx::TensorShape::Static(vec![6]),
        kind: onyxia_onnx::TensorKind::Input,
        initializer: None,
    });

    // Add output tensor (Bool type per ONNX spec)
    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "c".to_string(),
        dtype: DataType::Bool,
        shape: onyxia_onnx::TensorShape::Static(vec![6]),
        kind: onyxia_onnx::TensorKind::Output,
        initializer: None,
    });

    // Create Greater operation node
    let mut node = onyxia_onnx::Node::new("Greater");
    node.name = "greater_node".to_string();
    node.inputs = vec!["a".to_string(), "b".to_string()];
    node.outputs = vec!["c".to_string()];
    graph.add_node(node);

    // Set graph inputs and outputs
    graph.inputs = vec!["a".to_string(), "b".to_string()];
    graph.outputs = vec!["c".to_string()];

    // Set metadata
    graph.metadata.name = "test_greater_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph.validate().expect("Graph validation should succeed");

    // Compile to ExecutionPlan
    let registry = OperatorRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    // Initialize runtime and load plan
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Execute: c = (a > b)
    // Test various cases: greater, less, equal, negative numbers, infinity
    let a = Tensor::from_vec(
        vec![5.0f32, 3.0, 5.0, -2.0, f32::INFINITY, f32::NEG_INFINITY],
        &[6],
    );
    let b = Tensor::from_vec(vec![3.0f32, 5.0, 5.0, -5.0, 100.0, -100.0], &[6]);

    let outputs = executor
        .run(&[("a", a), ("b", b)])
        .expect("Execution should succeed");

    // Output is stored as u32 (0 for false, 1 for true)
    let c = outputs["c"].to_vec::<u32>().expect("Should convert to u32");
    assert_eq!(
        c,
        vec![1, 0, 0, 1, 1, 0],
        "Greater result incorrect (1=true, 0=false)"
    );

    println!("✓ End-to-end Greater test passed!");
    println!("  Input a: [5.0, 3.0, 5.0, -2.0, inf, -inf]");
    println!("  Input b: [3.0, 5.0, 5.0, -5.0, 100.0, -100.0]");
    println!("  Output c: {:?} (1=greater, 0=not greater)", c);
}

/// End-to-end test: Greater with broadcasting (scalar vs tensor).
#[pollster::test]
#[ignore] // Requires GPU
async fn test_greater_broadcast_e2e() {
    // Build graph with broadcasting: scalar > vector
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
        shape: onyxia_onnx::TensorShape::Static(vec![5]),
        kind: onyxia_onnx::TensorKind::Input,
        initializer: None,
    });

    // Add output tensor (Bool type per ONNX spec)
    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "c".to_string(),
        dtype: DataType::Bool,
        shape: onyxia_onnx::TensorShape::Static(vec![5]),
        kind: onyxia_onnx::TensorKind::Output,
        initializer: None,
    });

    // Create Greater operation node
    let mut node = onyxia_onnx::Node::new("Greater");
    node.name = "greater_broadcast_node".to_string();
    node.inputs = vec!["a".to_string(), "b".to_string()];
    node.outputs = vec!["c".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["a".to_string(), "b".to_string()];
    graph.outputs = vec!["c".to_string()];

    graph.metadata.name = "test_greater_broadcast_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph.validate().expect("Graph validation should succeed");

    // Compile and execute
    let registry = OperatorRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Broadcast scalar 3.0 against vector [1.0, 2.0, 3.0, 4.0, 5.0]
    // Expected: 3.0 > [1.0, 2.0, 3.0, 4.0, 5.0] = [true, true, false, false, false]
    let a = Tensor::from_vec(vec![3.0f32], &[1]);
    let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]);

    let outputs = executor
        .run(&[("a", a), ("b", b)])
        .expect("Execution should succeed");

    let c = outputs["c"].to_vec::<u32>().expect("Should convert to u32");
    assert_eq!(
        c,
        vec![1, 1, 0, 0, 0],
        "Greater broadcast result incorrect (1=true, 0=false)"
    );

    println!("✓ End-to-end Greater broadcast test passed!");
    println!("  Input a: [3.0] (scalar)");
    println!("  Input b: [1.0, 2.0, 3.0, 4.0, 5.0]");
    println!("  Output c: {:?} (1=greater, 0=not greater)", c);
}

/// End-to-end test: Where operator for conditional element selection.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_where_basic_e2e() {
    // Build graph for Where operator: output = condition ? x : y
    let mut graph = onyxia_onnx::Graph::new();

    // Condition input (Bool represented as I32)
    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "condition".to_string(),
        dtype: DataType::I32,
        shape: onyxia_onnx::TensorShape::Static(vec![4]),
        kind: onyxia_onnx::TensorKind::Input,
        initializer: None,
    });

    // X input (values when condition is true)
    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "x".to_string(),
        dtype: DataType::F32,
        shape: onyxia_onnx::TensorShape::Static(vec![4]),
        kind: onyxia_onnx::TensorKind::Input,
        initializer: None,
    });

    // Y input (values when condition is false)
    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "y".to_string(),
        dtype: DataType::F32,
        shape: onyxia_onnx::TensorShape::Static(vec![4]),
        kind: onyxia_onnx::TensorKind::Input,
        initializer: None,
    });

    // Output tensor
    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: onyxia_onnx::TensorShape::Static(vec![4]),
        kind: onyxia_onnx::TensorKind::Output,
        initializer: None,
    });

    // Create Where operation node
    let mut node = onyxia_onnx::Node::new("Where");
    node.name = "where_node".to_string();
    node.inputs = vec!["condition".to_string(), "x".to_string(), "y".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["condition".to_string(), "x".to_string(), "y".to_string()];
    graph.outputs = vec!["output".to_string()];

    graph.metadata.name = "test_where_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph.validate().expect("Graph validation should succeed");

    // Compile and execute
    let registry = OperatorRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Test: condition = [1, 0, 1, 0] (true, false, true, false)
    //       x = [10.0, 20.0, 30.0, 40.0]
    //       y = [100.0, 200.0, 300.0, 400.0]
    // Expected: [10.0, 200.0, 30.0, 400.0]
    let condition = Tensor::from_vec(vec![1i32, 0, 1, 0], &[4]);
    let x = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], &[4]);
    let y = Tensor::from_vec(vec![100.0f32, 200.0, 300.0, 400.0], &[4]);

    let outputs = executor
        .run(&[("condition", condition), ("x", x), ("y", y)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");
    assert_eq!(
        output,
        vec![10.0, 200.0, 30.0, 400.0],
        "Where result incorrect"
    );

    println!("✓ End-to-end Where basic test passed!");
    println!("  Condition: [1, 0, 1, 0] (1=true, 0=false)");
    println!("  X: [10.0, 20.0, 30.0, 40.0]");
    println!("  Y: [100.0, 200.0, 300.0, 400.0]");
    println!("  Output: {:?}", output);
}

/// End-to-end test: Where operator with all-true condition.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_where_all_true_e2e() {
    let mut graph = onyxia_onnx::Graph::new();

    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "condition".to_string(),
        dtype: DataType::I32,
        shape: onyxia_onnx::TensorShape::Static(vec![3]),
        kind: onyxia_onnx::TensorKind::Input,
        initializer: None,
    });

    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "x".to_string(),
        dtype: DataType::F32,
        shape: onyxia_onnx::TensorShape::Static(vec![3]),
        kind: onyxia_onnx::TensorKind::Input,
        initializer: None,
    });

    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "y".to_string(),
        dtype: DataType::F32,
        shape: onyxia_onnx::TensorShape::Static(vec![3]),
        kind: onyxia_onnx::TensorKind::Input,
        initializer: None,
    });

    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: onyxia_onnx::TensorShape::Static(vec![3]),
        kind: onyxia_onnx::TensorKind::Output,
        initializer: None,
    });

    let mut node = onyxia_onnx::Node::new("Where");
    node.name = "where_node".to_string();
    node.inputs = vec!["condition".to_string(), "x".to_string(), "y".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["condition".to_string(), "x".to_string(), "y".to_string()];
    graph.outputs = vec!["output".to_string()];

    graph.metadata.name = "test_where_all_true_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    let registry = OperatorRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // All true: should select all from x
    let condition = Tensor::from_vec(vec![1i32, 1, 1], &[3]);
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]);
    let y = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], &[3]);

    let outputs = executor
        .run(&[("condition", condition), ("x", x), ("y", y)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");
    assert_eq!(
        output,
        vec![1.0, 2.0, 3.0],
        "Where all-true result incorrect"
    );

    println!("✓ End-to-end Where all-true test passed!");
    println!("  Condition: [1, 1, 1] (all true)");
    println!("  X: [1.0, 2.0, 3.0]");
    println!("  Y: [10.0, 20.0, 30.0]");
    println!("  Output: {:?} (all from X)", output);
}

/// End-to-end test: Where operator with all-false condition.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_where_all_false_e2e() {
    let mut graph = onyxia_onnx::Graph::new();

    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "condition".to_string(),
        dtype: DataType::I32,
        shape: onyxia_onnx::TensorShape::Static(vec![3]),
        kind: onyxia_onnx::TensorKind::Input,
        initializer: None,
    });

    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "x".to_string(),
        dtype: DataType::F32,
        shape: onyxia_onnx::TensorShape::Static(vec![3]),
        kind: onyxia_onnx::TensorKind::Input,
        initializer: None,
    });

    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "y".to_string(),
        dtype: DataType::F32,
        shape: onyxia_onnx::TensorShape::Static(vec![3]),
        kind: onyxia_onnx::TensorKind::Input,
        initializer: None,
    });

    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: onyxia_onnx::TensorShape::Static(vec![3]),
        kind: onyxia_onnx::TensorKind::Output,
        initializer: None,
    });

    let mut node = onyxia_onnx::Node::new("Where");
    node.name = "where_node".to_string();
    node.inputs = vec!["condition".to_string(), "x".to_string(), "y".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["condition".to_string(), "x".to_string(), "y".to_string()];
    graph.outputs = vec!["output".to_string()];

    graph.metadata.name = "test_where_all_false_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    let registry = OperatorRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // All false: should select all from y
    let condition = Tensor::from_vec(vec![0i32, 0, 0], &[3]);
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]);
    let y = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], &[3]);

    let outputs = executor
        .run(&[("condition", condition), ("x", x), ("y", y)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");
    assert_eq!(
        output,
        vec![10.0, 20.0, 30.0],
        "Where all-false result incorrect"
    );

    println!("✓ End-to-end Where all-false test passed!");
    println!("  Condition: [0, 0, 0] (all false)");
    println!("  X: [1.0, 2.0, 3.0]");
    println!("  Y: [10.0, 20.0, 30.0]");
    println!("  Output: {:?} (all from Y)", output);
}

/// End-to-end test: Where operator with scalar condition broadcast.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_where_scalar_condition_e2e() {
    let mut graph = onyxia_onnx::Graph::new();

    // Scalar condition (broadcasts to all elements)
    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "condition".to_string(),
        dtype: DataType::I32,
        shape: onyxia_onnx::TensorShape::Static(vec![1]),
        kind: onyxia_onnx::TensorKind::Input,
        initializer: None,
    });

    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "x".to_string(),
        dtype: DataType::F32,
        shape: onyxia_onnx::TensorShape::Static(vec![4]),
        kind: onyxia_onnx::TensorKind::Input,
        initializer: None,
    });

    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "y".to_string(),
        dtype: DataType::F32,
        shape: onyxia_onnx::TensorShape::Static(vec![4]),
        kind: onyxia_onnx::TensorKind::Input,
        initializer: None,
    });

    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: onyxia_onnx::TensorShape::Static(vec![4]),
        kind: onyxia_onnx::TensorKind::Output,
        initializer: None,
    });

    let mut node = onyxia_onnx::Node::new("Where");
    node.name = "where_node".to_string();
    node.inputs = vec!["condition".to_string(), "x".to_string(), "y".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["condition".to_string(), "x".to_string(), "y".to_string()];
    graph.outputs = vec!["output".to_string()];

    graph.metadata.name = "test_where_scalar_condition_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    let registry = OperatorRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Scalar condition = 1 (true), should select all from x
    let condition = Tensor::from_vec(vec![1i32], &[1]);
    let x = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], &[4]);
    let y = Tensor::from_vec(vec![50.0f32, 60.0, 70.0, 80.0], &[4]);

    let outputs = executor
        .run(&[("condition", condition), ("x", x), ("y", y)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");
    assert_eq!(
        output,
        vec![5.0, 6.0, 7.0, 8.0],
        "Where scalar condition result incorrect"
    );

    println!("✓ End-to-end Where scalar condition test passed!");
    println!("  Condition: [1] (scalar, broadcasts to true for all)");
    println!("  X: [5.0, 6.0, 7.0, 8.0]");
    println!("  Y: [50.0, 60.0, 70.0, 80.0]");
    println!("  Output: {:?} (all from X)", output);
}

/// End-to-end test: Pow operation with basic powers.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_pow_basic_e2e() {
    // Build graph
    let graph = make_binary_elementwise_graph("Pow", "pow_node", DataType::F32, &[4]);
    graph.validate().expect("Graph validation should succeed");

    // Compile to ExecutionPlan
    let registry = OperatorRegistry::with_defaults();
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

    // Execute: z = x ^ y
    // Test: 2^3=8, 3^2=9, 4^2=16, 5^2=25
    let x = Tensor::from_vec(vec![2.0f32, 3.0, 4.0, 5.0], &[4]);
    let y = Tensor::from_vec(vec![3.0f32, 2.0, 2.0, 2.0], &[4]);

    let outputs = executor
        .run(&[("a", x), ("b", y)])
        .expect("Execution should succeed");

    let z = outputs["c"].to_vec::<f32>().expect("Should convert to f32");

    // Allow small floating point error
    assert!((z[0] - 8.0).abs() < 1e-5, "2^3 should be 8, got {}", z[0]);
    assert!((z[1] - 9.0).abs() < 1e-5, "3^2 should be 9, got {}", z[1]);
    assert!((z[2] - 16.0).abs() < 1e-5, "4^2 should be 16, got {}", z[2]);
    assert!((z[3] - 25.0).abs() < 1e-5, "5^2 should be 25, got {}", z[3]);

    println!("✓ End-to-end Pow basic test passed!");
    println!("  Input x: [2.0, 3.0, 4.0, 5.0]");
    println!("  Input y: [3.0, 2.0, 2.0, 2.0]");
    println!("  Output z: {:?}", z);
}

/// End-to-end test: Pow with exponent of zero (x^0 = 1).
#[pollster::test]
#[ignore] // Requires GPU
async fn test_pow_zero_exponent_e2e() {
    let graph = make_binary_elementwise_graph("Pow", "pow_node", DataType::F32, &[4]);
    graph.validate().expect("Graph validation should succeed");

    let registry = OperatorRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Any number to the power of 0 should equal 1
    let x = Tensor::from_vec(vec![2.0f32, 3.0, 100.0, -5.0], &[4]);
    let y = Tensor::from_vec(vec![0.0f32, 0.0, 0.0, 0.0], &[4]);

    let outputs = executor
        .run(&[("a", x), ("b", y)])
        .expect("Execution should succeed");

    let z = outputs["c"].to_vec::<f32>().expect("Should convert to f32");

    for (i, &val) in z.iter().enumerate() {
        assert!(
            (val - 1.0).abs() < 1e-5,
            "x^0 should be 1, got {} at index {}",
            val,
            i
        );
    }

    println!("✓ End-to-end Pow zero exponent test passed!");
    println!("  Input x: [2.0, 3.0, 100.0, -5.0]");
    println!("  Input y: [0.0, 0.0, 0.0, 0.0]");
    println!("  Output z: {:?} (all should be 1.0)", z);
}

/// End-to-end test: Pow with exponent of one (x^1 = x).
#[pollster::test]
#[ignore] // Requires GPU
async fn test_pow_one_exponent_e2e() {
    let graph = make_binary_elementwise_graph("Pow", "pow_node", DataType::F32, &[4]);
    graph.validate().expect("Graph validation should succeed");

    let registry = OperatorRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Any number to the power of 1 should equal itself
    let x = Tensor::from_vec(vec![2.5f32, 3.7, 100.1, 5.9], &[4]);
    let y = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0], &[4]);

    let outputs = executor
        .run(&[("a", x.clone()), ("b", y)])
        .expect("Execution should succeed");

    let z = outputs["c"].to_vec::<f32>().expect("Should convert to f32");
    let x_vals = vec![2.5f32, 3.7, 100.1, 5.9];

    for (i, (&expected, &actual)) in x_vals.iter().zip(z.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "x^1 should equal x, got {} vs {} at index {}",
            actual,
            expected,
            i
        );
    }

    println!("✓ End-to-end Pow one exponent test passed!");
    println!("  Input x: {:?}", x_vals);
    println!("  Input y: [1.0, 1.0, 1.0, 1.0]");
    println!("  Output z: {:?} (should equal x)", z);
}

/// End-to-end test: Pow with fractional exponents (square root).
#[pollster::test]
#[ignore] // Requires GPU
async fn test_pow_fractional_exponent_e2e() {
    let graph = make_binary_elementwise_graph("Pow", "pow_node", DataType::F32, &[4]);
    graph.validate().expect("Graph validation should succeed");

    let registry = OperatorRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Test square roots: x^0.5 = sqrt(x)
    let x = Tensor::from_vec(vec![4.0f32, 9.0, 16.0, 25.0], &[4]);
    let y = Tensor::from_vec(vec![0.5f32, 0.5, 0.5, 0.5], &[4]);

    let outputs = executor
        .run(&[("a", x), ("b", y)])
        .expect("Execution should succeed");

    let z = outputs["c"].to_vec::<f32>().expect("Should convert to f32");
    let expected = vec![2.0f32, 3.0, 4.0, 5.0];

    for (i, (&expected, &actual)) in expected.iter().zip(z.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "sqrt({}) should be {}, got {} at index {}",
            vec![4.0, 9.0, 16.0, 25.0][i],
            expected,
            actual,
            i
        );
    }

    println!("✓ End-to-end Pow fractional exponent test passed!");
    println!("  Input x: [4.0, 9.0, 16.0, 25.0]");
    println!("  Input y: [0.5, 0.5, 0.5, 0.5]");
    println!("  Output z: {:?} (square roots)", z);
}

/// End-to-end test: Pow with scalar broadcasting.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_pow_broadcast_scalar_e2e() {
    // Create graph with broadcasting: [1] ^ [4]
    let mut graph = onyxia_onnx::Graph::new();

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

    graph.add_tensor(onyxia_onnx::TensorInfo {
        name: "c".to_string(),
        dtype: DataType::F32,
        shape: onyxia_onnx::TensorShape::Static(vec![4]),
        kind: onyxia_onnx::TensorKind::Output,
        initializer: None,
    });

    let mut node = onyxia_onnx::Node::new("Pow");
    node.name = "pow_broadcast_node".to_string();
    node.inputs = vec!["a".to_string(), "b".to_string()];
    node.outputs = vec!["c".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["a".to_string(), "b".to_string()];
    graph.outputs = vec!["c".to_string()];

    graph.metadata.name = "test_pow_broadcast_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph.validate().expect("Graph validation should succeed");

    let registry = OperatorRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Broadcast scalar 2.0 to powers [1, 2, 3, 4]
    // Result should be [2^1, 2^2, 2^3, 2^4] = [2, 4, 8, 16]
    let a = Tensor::from_vec(vec![2.0f32], &[1]);
    let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]);

    let outputs = executor
        .run(&[("a", a), ("b", b)])
        .expect("Execution should succeed");

    let c = outputs["c"].to_vec::<f32>().expect("Should convert to f32");
    let expected = vec![2.0f32, 4.0, 8.0, 16.0];

    for (i, (&expected, &actual)) in expected.iter().zip(c.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "2^{} should be {}, got {} at index {}",
            i + 1,
            expected,
            actual,
            i
        );
    }

    println!("✓ End-to-end Pow broadcast scalar test passed!");
    println!("  Input a: [2.0] (scalar, broadcasts)");
    println!("  Input b: [1.0, 2.0, 3.0, 4.0]");
    println!("  Output c: {:?}", c);
}

/// End-to-end test: Max of two vectors on GPU and verify correct output.
#[pollster::test]
#[ignore] // Requires GPU - ignore in CI or environments without GPU
async fn test_max_basic_e2e() {
    // Build graph
    let graph = make_binary_elementwise_graph("Max", "max_node", DataType::F32, &[4]);
    graph.validate().expect("Graph validation should succeed");

    // Compile to ExecutionPlan
    let registry = OperatorRegistry::with_defaults();
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
    let a = Tensor::from_vec(vec![1.0f32, 5.0, 2.0, 8.0], &[4]);
    let b = Tensor::from_vec(vec![3.0f32, 2.0, 6.0, 4.0], &[4]);

    let outputs = executor
        .run(&[("a", a), ("b", b)])
        .expect("Execution should succeed");

    // Verify output: max(a, b) element-wise
    let c = outputs["c"].to_vec::<f32>().expect("Should convert to f32");
    assert_eq!(c, vec![3.0, 5.0, 6.0, 8.0], "Max result incorrect");

    println!("✓ End-to-end Max basic test passed!");
    println!("  Input a: [1.0, 5.0, 2.0, 8.0]");
    println!("  Input b: [3.0, 2.0, 6.0, 4.0]");
    println!("  Output c: {:?}", c);
}

/// End-to-end test: Max with broadcasting (scalar broadcasted to tensor).
#[pollster::test]
#[ignore] // Requires GPU - ignore in CI or environments without GPU
async fn test_max_broadcast_e2e() {
    // Build graph with different shapes
    let mut graph = Graph::new();

    // Input a: scalar [1]
    graph.add_tensor(TensorInfo {
        name: "a".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Input b: vector [4]
    graph.add_tensor(TensorInfo {
        name: "b".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Output c: vector [4] (broadcasted)
    graph.add_tensor(TensorInfo {
        name: "c".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = onyxia_onnx::Node::new("Max");
    node.name = "max_broadcast".to_string();
    node.inputs = vec!["a".to_string(), "b".to_string()];
    node.outputs = vec!["c".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["a".to_string(), "b".to_string()];
    graph.outputs = vec!["c".to_string()];

    graph.validate().expect("Graph validation should succeed");

    // Compile to ExecutionPlan
    let registry = OperatorRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    // Initialize runtime and load plan
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Broadcast scalar 5.0 to vector [1, 8, 3, 4]
    // Result should be [max(5, 1), max(5, 8), max(5, 3), max(5, 4)] = [5, 8, 5, 5]
    let a = Tensor::from_vec(vec![5.0f32], &[1]);
    let b = Tensor::from_vec(vec![1.0f32, 8.0, 3.0, 4.0], &[4]);

    let outputs = executor
        .run(&[("a", a), ("b", b)])
        .expect("Execution should succeed");

    let c = outputs["c"].to_vec::<f32>().expect("Should convert to f32");
    assert_eq!(
        c,
        vec![5.0, 8.0, 5.0, 5.0],
        "Max broadcast result incorrect"
    );

    println!("✓ End-to-end Max broadcast test passed!");
    println!("  Input a: [5.0] (scalar, broadcasts)");
    println!("  Input b: [1.0, 8.0, 3.0, 4.0]");
    println!("  Output c: {:?}", c);
}

/// End-to-end test: Max with negative values and edge cases.
#[pollster::test]
#[ignore] // Requires GPU - ignore in CI or environments without GPU
async fn test_max_negative_e2e() {
    // Build graph
    let graph = make_binary_elementwise_graph("Max", "max_negative", DataType::F32, &[4]);
    graph.validate().expect("Graph validation should succeed");

    // Compile to ExecutionPlan
    let registry = OperatorRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    // Initialize runtime and load plan
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Test with negative values: max(-1, -5), max(-2, 0), max(3, -1), max(-10, -10)
    let a = Tensor::from_vec(vec![-1.0f32, -2.0, 3.0, -10.0], &[4]);
    let b = Tensor::from_vec(vec![-5.0f32, 0.0, -1.0, -10.0], &[4]);

    let outputs = executor
        .run(&[("a", a), ("b", b)])
        .expect("Execution should succeed");

    let c = outputs["c"].to_vec::<f32>().expect("Should convert to f32");
    assert_eq!(
        c,
        vec![-1.0, 0.0, 3.0, -10.0],
        "Max negative result incorrect"
    );

    println!("✓ End-to-end Max negative test passed!");
    println!("  Input a: [-1.0, -2.0, 3.0, -10.0]");
    println!("  Input b: [-5.0, 0.0, -1.0, -10.0]");
    println!("  Output c: {:?}", c);
}

/// End-to-end test: Max with three inputs (tests chaining).
#[pollster::test]
#[ignore] // Requires GPU - ignore in CI or environments without GPU
async fn test_max_three_inputs_e2e() {
    // Build graph with three inputs
    let mut graph = Graph::new();

    // Add three input tensors
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

    // Add output tensor
    graph.add_tensor(TensorInfo {
        name: "d".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Max node with three inputs
    let mut node = onyxia_onnx::Node::new("Max");
    node.name = "max_three".to_string();
    node.inputs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    node.outputs = vec!["d".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    graph.outputs = vec!["d".to_string()];

    graph.validate().expect("Graph validation should succeed");

    // Compile to ExecutionPlan
    let registry = OperatorRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    // Should have 1 operation with 2 steps (chained pairwise max)
    assert_eq!(plan.operations.len(), 1);
    assert_eq!(plan.operations[0].steps.len(), 2);

    // Initialize runtime and load plan
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Compute max(1, 5, 3), max(2, 1, 6), max(8, 4, 2), max(3, 7, 9)
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 8.0, 3.0], &[4]);
    let b = Tensor::from_vec(vec![5.0f32, 1.0, 4.0, 7.0], &[4]);
    let c = Tensor::from_vec(vec![3.0f32, 6.0, 2.0, 9.0], &[4]);

    let outputs = executor
        .run(&[("a", a), ("b", b), ("c", c)])
        .expect("Execution should succeed");

    let d = outputs["d"].to_vec::<f32>().expect("Should convert to f32");
    assert_eq!(d, vec![5.0, 6.0, 8.0, 9.0], "Max three inputs incorrect");

    println!("✓ End-to-end Max three inputs test passed!");
    println!("  Input a: [1.0, 2.0, 8.0, 3.0]");
    println!("  Input b: [5.0, 1.0, 4.0, 7.0]");
    println!("  Input c: [3.0, 6.0, 2.0, 9.0]");
    println!("  Output d: {:?}", d);
}
