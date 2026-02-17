//! End-to-end dispatch tests for the new architecture.
//!
//! These tests require a GPU. Run with:
//! ```sh
//! cargo nextest run -p onyxia-operators --run-ignored=all
//! ```

mod common;

use common::{CompilerPipeline, Runtime, Tensor, make_binary_elementwise_graph};
use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_operators::core_operator_registry;

// ================================================================================
// Add operator tests
// ================================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_add_basic() {
    // Create graph: a[4] + b[4] -> c[4]
    let graph = make_binary_elementwise_graph("Add", "add_op", DataType::F32, &[4]);

    // Compile
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    // Load and execute
    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]);
    let b = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], &[4]);

    let outputs = executor.run(&[("a", a), ("b", b)]).unwrap();
    let result: Vec<f32> = outputs["c"].to_vec().unwrap();

    assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_add_broadcast() {
    // Test [2, 3] + [1] broadcasting
    let mut graph = Graph::new();

    // Input a: [2, 3]
    graph.add_tensor(TensorInfo {
        name: "a".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Input b: [1] (scalar)
    graph.add_tensor(TensorInfo {
        name: "b".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Output c: [2, 3]
    graph.add_tensor(TensorInfo {
        name: "c".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Add");
    node.name = "add_broadcast".to_string();
    node.inputs = vec!["a".to_string(), "b".to_string()];
    node.outputs = vec!["c".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["a".to_string(), "b".to_string()];
    graph.outputs = vec!["c".to_string()];

    // Compile
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    // Execute
    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::from_vec(vec![10.0f32], &[1]);

    let outputs = executor.run(&[("a", a), ("b", b)]).unwrap();
    let result: Vec<f32> = outputs["c"].to_vec().unwrap();

    assert_eq!(result, vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
}

// ================================================================================
// Mul operator tests
// ================================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_mul_basic() {
    // Create graph: a[4] * b[4] -> c[4]
    let graph = make_binary_elementwise_graph("Mul", "mul_op", DataType::F32, &[4]);

    // Compile
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    // Load and execute
    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]);
    let b = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], &[4]);

    let outputs = executor.run(&[("a", a), ("b", b)]).unwrap();
    let result: Vec<f32> = outputs["c"].to_vec().unwrap();

    // 1*10, 2*20, 3*30, 4*40
    assert_eq!(result, vec![10.0, 40.0, 90.0, 160.0]);
}

// ================================================================================
// Multi-op chain tests
// ================================================================================

/// Helper to create a graph with Mul followed by Add
fn make_mul_add_graph() -> Graph {
    let mut graph = Graph::new();

    // Inputs
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

    // Intermediate: mul_result
    graph.add_tensor(TensorInfo {
        name: "mul_result".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Intermediate,
        initializer: None,
    });

    // Output
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Mul node: a * b -> mul_result
    let mut mul_node = Node::new("Mul");
    mul_node.name = "mul_op".to_string();
    mul_node.inputs = vec!["a".to_string(), "b".to_string()];
    mul_node.outputs = vec!["mul_result".to_string()];
    graph.add_node(mul_node);

    // Add node: mul_result + c -> output
    let mut add_node = Node::new("Add");
    add_node.name = "add_op".to_string();
    add_node.inputs = vec!["mul_result".to_string(), "c".to_string()];
    add_node.outputs = vec!["output".to_string()];
    graph.add_node(add_node);

    graph.inputs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    graph.outputs = vec!["output".to_string()];

    graph
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_mul_then_add() {
    // Graph: Mul(a, b) -> Add(result, c) -> output
    // Tests that tensors route correctly between operations via registers.
    let graph = make_mul_add_graph();

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]);
    let b = Tensor::from_vec(vec![2.0f32, 2.0, 2.0, 2.0], &[4]);
    let c = Tensor::from_vec(vec![100.0f32, 100.0, 100.0, 100.0], &[4]);

    let outputs = executor.run(&[("a", a), ("b", b), ("c", c)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // a * b + c = [2, 4, 6, 8] + [100, 100, 100, 100] = [102, 104, 106, 108]
    assert_eq!(result, vec![102.0, 104.0, 106.0, 108.0]);
}

/// Helper to create a graph where one input is used by two operations
fn make_fan_out_graph() -> Graph {
    let mut graph = Graph::new();

    // Input a: shared by both ops
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

    // output1 = a + b
    graph.add_tensor(TensorInfo {
        name: "output1".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // output2 = a * c
    graph.add_tensor(TensorInfo {
        name: "output2".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Add node
    let mut add_node = Node::new("Add");
    add_node.name = "add_op".to_string();
    add_node.inputs = vec!["a".to_string(), "b".to_string()];
    add_node.outputs = vec!["output1".to_string()];
    graph.add_node(add_node);

    // Mul node
    let mut mul_node = Node::new("Mul");
    mul_node.name = "mul_op".to_string();
    mul_node.inputs = vec!["a".to_string(), "c".to_string()];
    mul_node.outputs = vec!["output2".to_string()];
    graph.add_node(mul_node);

    graph.inputs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    graph.outputs = vec!["output1".to_string(), "output2".to_string()];

    graph
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_fan_out_tensor() {
    // Graph: input used by two downstream ops (tests Arc sharing)
    //   a -+-> Add(a, b) -> output1
    //      +-> Mul(a, c) -> output2
    let graph = make_fan_out_graph();

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]);
    let b = Tensor::from_vec(vec![10.0f32, 10.0, 10.0, 10.0], &[4]);
    let c = Tensor::from_vec(vec![5.0f32, 5.0, 5.0, 5.0], &[4]);

    let outputs = executor.run(&[("a", a), ("b", b), ("c", c)]).unwrap();

    let result1: Vec<f32> = outputs["output1"].to_vec().unwrap();
    let result2: Vec<f32> = outputs["output2"].to_vec().unwrap();

    // output1 = a + b = [1, 2, 3, 4] + [10, 10, 10, 10] = [11, 12, 13, 14]
    assert_eq!(result1, vec![11.0, 12.0, 13.0, 14.0]);

    // output2 = a * c = [1, 2, 3, 4] * [5, 5, 5, 5] = [5, 10, 15, 20]
    assert_eq!(result2, vec![5.0, 10.0, 15.0, 20.0]);
}

// ================================================================================
// Reshape operator tests
// ================================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_reshape() {
    // Graph: Reshape(input, shape_tensor) -> output
    // shape_tensor is a weight/initializer with value [2, 6]
    let mut graph = Graph::new();

    // Input tensor: [3, 4] = 12 elements
    graph.add_tensor(TensorInfo {
        name: "input".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![3, 4]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Shape tensor as initializer: [2, 6]
    // ONNX Reshape expects shape as int64
    let shape_data: Vec<i64> = vec![2, 6];
    let shape_bytes: Vec<u8> = shape_data.iter().flat_map(|&x| x.to_le_bytes()).collect();

    graph.add_tensor(TensorInfo {
        name: "shape".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![2]),
        kind: TensorKind::Weight,
        initializer: Some(shape_bytes),
    });

    // Output tensor: [2, 6]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 6]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Reshape node
    let mut node = Node::new("Reshape");
    node.name = "reshape_op".to_string();
    node.inputs = vec!["input".to_string(), "shape".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["input".to_string()];
    graph.outputs = vec!["output".to_string()];

    // Compile
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    // Execute
    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), &[3, 4]);
    let outputs = executor.run(&[("input", input)]).unwrap();

    let result: Vec<f32> = outputs["output"].to_vec().unwrap();
    assert_eq!(result.len(), 12);
    // Data is unchanged, just shape
    assert_eq!(result, (1..=12).map(|x| x as f32).collect::<Vec<_>>());
}
