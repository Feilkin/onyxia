//! End-to-end tests for shape manipulation operations.
//!
//! Tests: Reshape, Transpose, Concat

use onyxia_onnx::{AttributeValue, DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_planner::{KernelRegistry, compile};
use onyxia_runtime::{Runtime, Tensor};
use std::collections::HashMap;

/// End-to-end test: Reshape a tensor and verify data is preserved.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_reshape_e2e() {
    let mut graph = Graph::new();

    // Add input tensor [2, 3]
    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add shape tensor as initializer (constant)
    let shape_data: Vec<u8> = vec![6, 0, 0, 0, 0, 0, 0, 0]; // i64 value: 6
    graph.add_tensor(TensorInfo {
        name: "shape".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Weight,
        initializer: Some(shape_data),
    });

    // Add output tensor [6]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![6]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Reshape node
    let mut node = Node::new("Reshape");
    node.name = "reshape_node".to_string();
    node.inputs = vec!["data".to_string(), "shape".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["data".to_string()];
    graph.outputs = vec!["output".to_string()];

    // Compile and execute
    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Test: reshape [[1,2,3],[4,5,6]] → [1,2,3,4,5,6]
    let data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let outputs = executor
        .run(&[("data", data)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    assert_eq!(
        output,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "Reshape should preserve data"
    );

    println!("✓ End-to-end Reshape test passed!");
    println!("  Input: [[1, 2, 3], [4, 5, 6]] (shape [2, 3])");
    println!("  Output: {:?} (shape [6])", output);
}

/// Helper function to create a Transpose graph.
fn make_transpose_graph() -> Graph {
    let mut graph = Graph::new();

    // Add input tensor [2, 3]
    graph.add_tensor(TensorInfo {
        name: "input".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add output tensor [3, 2]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![3, 2]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Transpose node with perm=[1, 0]
    let mut node = Node::new("Transpose");
    node.name = "transpose_node".to_string();
    node.inputs = vec!["input".to_string()];
    node.outputs = vec!["output".to_string()];
    node.attributes
        .insert("perm".to_string(), AttributeValue::Ints(vec![1i64, 0i64]));
    graph.add_node(node);

    graph.inputs = vec!["input".to_string()];
    graph.outputs = vec!["output".to_string()];

    graph.metadata.name = "test_transpose_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

/// End-to-end test: Transpose a 2D matrix on GPU.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_transpose_2d_e2e() {
    let graph = make_transpose_graph();
    graph.validate().expect("Graph validation should succeed");

    // Compile to ExecutionPlan
    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    // Initialize runtime
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Input: [[1, 2, 3], [4, 5, 6]]
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let outputs = executor
        .run(&[("input", input)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected: [[1, 4], [2, 5], [3, 6]]
    // In row-major order: [1, 4, 2, 5, 3, 6]
    assert_eq!(
        output,
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        "Transpose result incorrect"
    );

    println!("✓ End-to-end Transpose test passed!");
    println!("  Input: [[1, 2, 3], [4, 5, 6]] (shape [2, 3])");
    println!("  Output: [[1, 4], [2, 5], [3, 6]] (shape [3, 2])");
}

/// Helper function to create a Concat graph.
fn make_concat_graph() -> Graph {
    let mut graph = Graph::new();

    // Add input tensor 'a' with 3 elements
    graph.add_tensor(TensorInfo {
        name: "a".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add input tensor 'b' with 4 elements
    graph.add_tensor(TensorInfo {
        name: "b".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add output tensor 'c' with 7 elements
    graph.add_tensor(TensorInfo {
        name: "c".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![7]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Concat node
    let mut node = Node::new("Concat");
    node.name = "concat_node".to_string();
    node.inputs = vec!["a".to_string(), "b".to_string()];
    node.outputs = vec!["c".to_string()];
    node.attributes
        .insert("axis".to_string(), AttributeValue::Int(0i64));
    graph.add_node(node);

    graph.inputs = vec!["a".to_string(), "b".to_string()];
    graph.outputs = vec!["c".to_string()];

    graph.metadata.name = "test_concat_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

/// End-to-end test: Concatenate two vectors on GPU.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_concat_e2e() {
    let graph = make_concat_graph();
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

    // Prepare inputs: a = [1, 2, 3], b = [4, 5, 6, 7]
    let input_a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]);
    let input_b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0, 7.0], &[4]);

    let outputs = executor
        .run(&[("a", input_a), ("b", input_b)])
        .expect("Execution should succeed");

    let output = outputs["c"].to_vec::<f32>().expect("Should convert to f32");

    assert_eq!(
        output,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "Concat result incorrect"
    );

    println!("✓ End-to-end Concat test passed!");
    println!("  Input a: [1, 2, 3] (shape [3])");
    println!("  Input b: [4, 5, 6, 7] (shape [4])");
    println!("  Output: {:?} (shape [7])", output);
}

/// Helper function to create an Expand graph.
fn make_expand_graph(
    input_shape: Vec<usize>,
    target_shape: Vec<i64>,
    output_shape: Vec<usize>,
) -> Graph {
    let mut graph = Graph::new();

    // Add input tensor
    graph.add_tensor(TensorInfo {
        name: "input".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(input_shape),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add shape tensor as initializer (constant)
    let mut shape_data = Vec::new();
    for &dim in &target_shape {
        shape_data.extend_from_slice(&dim.to_le_bytes());
    }
    graph.add_tensor(TensorInfo {
        name: "shape".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![target_shape.len()]),
        kind: TensorKind::Weight,
        initializer: Some(shape_data),
    });

    // Add output tensor
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(output_shape),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Expand node
    let mut node = Node::new("Expand");
    node.name = "expand_node".to_string();
    node.inputs = vec!["input".to_string(), "shape".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["input".to_string()];
    graph.outputs = vec!["output".to_string()];

    graph.metadata.name = "test_expand_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

/// End-to-end test: Expand single dimension [3, 1] to [3, 5].
#[pollster::test]
#[ignore] // Requires GPU
async fn test_expand_single_dimension_e2e() {
    // Test: expand [3, 1] to [3, 5]
    // Input: [[1], [2], [3]]
    // Output: [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]]
    let graph = make_expand_graph(vec![3, 1], vec![3, 5], vec![3, 5]);
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

    // Input: [[1], [2], [3]]
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3, 1]);

    let outputs = executor
        .run(&[("input", input)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected: [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]]
    assert_eq!(
        output,
        vec![
            1.0, 1.0, 1.0, 1.0, 1.0, // row 0
            2.0, 2.0, 2.0, 2.0, 2.0, // row 1
            3.0, 3.0, 3.0, 3.0, 3.0, // row 2
        ],
        "Expand single dimension result incorrect"
    );

    println!("✓ End-to-end Expand (single dimension) test passed!");
    println!("  Input: [[1], [2], [3]] (shape [3, 1])");
    println!("  Output shape: [3, 5]");
}

/// End-to-end test: Expand with new leading dimensions [3, 4] to [2, 3, 4].
#[pollster::test]
#[ignore] // Requires GPU
async fn test_expand_new_dimensions_e2e() {
    // Test: expand [3, 4] to [2, 3, 4]
    // Input: [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    // Output: Two copies of the input along the first dimension
    let graph = make_expand_graph(vec![3, 4], vec![2, 3, 4], vec![2, 3, 4]);
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

    // Input: [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    let input = Tensor::from_vec(
        vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
    );

    let outputs = executor
        .run(&[("input", input)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected: Input repeated twice along the first dimension
    let expected = vec![
        // First copy
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // Second copy
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];

    assert_eq!(
        output, expected,
        "Expand with new dimensions result incorrect"
    );

    println!("✓ End-to-end Expand (new dimensions) test passed!");
    println!("  Input shape: [3, 4]");
    println!("  Output shape: [2, 3, 4]");
}

/// End-to-end test: Expand multiple dimensions [1, 3, 1] to [2, 3, 5].
#[pollster::test]
#[ignore] // Requires GPU
async fn test_expand_multiple_dimensions_e2e() {
    // Test: expand [1, 3, 1] to [2, 3, 5]
    // Input: [[[1], [2], [3]]]
    // Output: Broadcast along first and last dimensions
    let graph = make_expand_graph(vec![1, 3, 1], vec![2, 3, 5], vec![2, 3, 5]);
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

    // Input: [[[1], [2], [3]]]
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[1, 3, 1]);

    let outputs = executor
        .run(&[("input", input)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected: Each element broadcast along first and last dimensions
    // [[[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]],
    //  [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]]]
    let mut expected = Vec::new();
    for _ in 0..2 {
        // Two copies along first dimension
        for val in [1.0f32, 2.0, 3.0] {
            // Three values along second dimension
            for _ in 0..5 {
                // Five copies along last dimension
                expected.push(val);
            }
        }
    }

    assert_eq!(
        output, expected,
        "Expand multiple dimensions result incorrect"
    );

    println!("✓ End-to-end Expand (multiple dimensions) test passed!");
    println!("  Input shape: [1, 3, 1]");
    println!("  Output shape: [2, 3, 5]");
}

/// End-to-end test: Expand identity (no change) [3, 4] to [3, 4].
#[pollster::test]
#[ignore] // Requires GPU
async fn test_expand_identity_e2e() {
    // Test: expand [3, 4] to [3, 4] (identity, no change)
    let graph = make_expand_graph(vec![3, 4], vec![3, 4], vec![3, 4]);
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

    // Input: [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    let input = Tensor::from_vec(
        vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
    );

    let outputs = executor
        .run(&[("input", input.clone())])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected: Same as input
    let expected = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];

    assert_eq!(output, expected, "Expand identity result incorrect");

    println!("✓ End-to-end Expand (identity) test passed!");
    println!("  Input shape: [3, 4]");
    println!("  Output shape: [3, 4] (no change)");
}
