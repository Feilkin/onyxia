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

/// Helper function to create a ConstantOfShape graph.
fn make_constantofshape_graph(
    shape_dims: Vec<i64>,
    output_shape: Vec<usize>,
    value_opt: Option<f32>,
) -> Graph {
    let mut graph = Graph::new();

    // Add shape tensor as initializer (constant)
    let shape_data: Vec<u8> = shape_dims
        .iter()
        .flat_map(|&dim| dim.to_le_bytes())
        .collect();
    graph.add_tensor(TensorInfo {
        name: "shape".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![shape_dims.len()]),
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

    // Create ConstantOfShape node
    let mut node = Node::new("ConstantOfShape");
    node.name = "constantofshape_node".to_string();
    node.inputs = vec!["shape".to_string()];
    node.outputs = vec!["output".to_string()];

    // Add 'value' attribute if specified
    if let Some(value) = value_opt {
        // Create raw bytes for a float32 tensor value
        // Format: directly encode the float as 4 bytes (little-endian)
        let value_bytes = value.to_le_bytes().to_vec();
        node.attributes
            .insert("value".to_string(), AttributeValue::Tensor(value_bytes));
    }

    graph.add_node(node);

    graph.inputs = vec![];
    graph.outputs = vec!["output".to_string()];

    graph
}

/// End-to-end test: ConstantOfShape with default value (0.0).
#[pollster::test]
#[ignore] // Requires GPU
async fn test_constantofshape_default_value_e2e() {
    let graph = make_constantofshape_graph(vec![2, 3], vec![2, 3], None);
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

    let outputs = executor.run(&[]).expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected: All zeros (default value)
    let expected = vec![0.0f32; 6]; // 2 * 3 = 6

    assert_eq!(output, expected, "ConstantOfShape default value incorrect");

    println!("✓ End-to-end ConstantOfShape (default value) test passed!");
    println!("  Output shape: [2, 3]");
    println!("  Fill value: 0.0 (default)");
}

/// End-to-end test: ConstantOfShape with custom value (1.0).
#[pollster::test]
#[ignore] // Requires GPU
async fn test_constantofshape_custom_value_e2e() {
    let graph = make_constantofshape_graph(vec![3, 4], vec![3, 4], Some(1.0));
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

    let outputs = executor.run(&[]).expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected: All ones
    let expected = vec![1.0f32; 12]; // 3 * 4 = 12

    assert_eq!(output, expected, "ConstantOfShape custom value incorrect");

    println!("✓ End-to-end ConstantOfShape (custom value) test passed!");
    println!("  Output shape: [3, 4]");
    println!("  Fill value: 1.0");
}

/// End-to-end test: ConstantOfShape with 1D output.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_constantofshape_1d_e2e() {
    let graph = make_constantofshape_graph(vec![5], vec![5], Some(2.5));
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

    let outputs = executor.run(&[]).expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected: All 2.5
    let expected = vec![2.5f32; 5];

    assert_eq!(output, expected, "ConstantOfShape 1D result incorrect");

    println!("✓ End-to-end ConstantOfShape (1D) test passed!");
    println!("  Output shape: [5]");
    println!("  Fill value: 2.5");
}

/// End-to-end test: ConstantOfShape with 3D output.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_constantofshape_3d_e2e() {
    let graph = make_constantofshape_graph(vec![2, 3, 4], vec![2, 3, 4], Some(-1.0));
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

    let outputs = executor.run(&[]).expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected: All -1.0
    let expected = vec![-1.0f32; 24]; // 2 * 3 * 4 = 24

    assert_eq!(output, expected, "ConstantOfShape 3D result incorrect");

    println!("✓ End-to-end ConstantOfShape (3D) test passed!");
    println!("  Output shape: [2, 3, 4]");
    println!("  Fill value: -1.0");
}

/// End-to-end test: ConstantOfShape with scalar output (empty shape).
#[pollster::test]
#[ignore] // Requires GPU
async fn test_constantofshape_scalar_e2e() {
    let graph = make_constantofshape_graph(vec![], vec![], Some(42.0));
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

    let outputs = executor.run(&[]).expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected: Single value 42.0
    let expected = vec![42.0f32];

    assert_eq!(output, expected, "ConstantOfShape scalar result incorrect");

    println!("✓ End-to-end ConstantOfShape (scalar) test passed!");
    println!("  Output shape: [] (scalar)");
    println!("  Fill value: 42.0");
}

/// Helper function to create a Range graph with integer inputs.
fn make_range_graph_i64(start: i64, limit: i64, delta: i64) -> Graph {
    let mut graph = Graph::new();

    // Add start tensor (scalar I64)
    graph.add_tensor(TensorInfo {
        name: "start".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![]),
        kind: TensorKind::Weight,
        initializer: Some(start.to_le_bytes().to_vec()),
    });

    // Add limit tensor (scalar I64)
    graph.add_tensor(TensorInfo {
        name: "limit".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![]),
        kind: TensorKind::Weight,
        initializer: Some(limit.to_le_bytes().to_vec()),
    });

    // Add delta tensor (scalar I64)
    graph.add_tensor(TensorInfo {
        name: "delta".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![]),
        kind: TensorKind::Weight,
        initializer: Some(delta.to_le_bytes().to_vec()),
    });

    // Calculate output size
    let output_size = ((limit - start) as f64 / delta as f64).ceil().max(0.0) as usize;

    // Add output tensor (1D F32 - Range outputs as F32 on GPU)
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![output_size]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Range node
    let mut node = Node::new("Range");
    node.name = "range_node".to_string();
    node.inputs = vec![
        "start".to_string(),
        "limit".to_string(),
        "delta".to_string(),
    ];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec![];
    graph.outputs = vec!["output".to_string()];

    graph.metadata.name = "test_range_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

/// Helper function to create a Range graph with float inputs.
fn make_range_graph_f32(start: f32, limit: f32, delta: f32) -> Graph {
    let mut graph = Graph::new();

    // Add start tensor (scalar F32)
    graph.add_tensor(TensorInfo {
        name: "start".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![]),
        kind: TensorKind::Weight,
        initializer: Some(start.to_le_bytes().to_vec()),
    });

    // Add limit tensor (scalar F32)
    graph.add_tensor(TensorInfo {
        name: "limit".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![]),
        kind: TensorKind::Weight,
        initializer: Some(limit.to_le_bytes().to_vec()),
    });

    // Add delta tensor (scalar F32)
    graph.add_tensor(TensorInfo {
        name: "delta".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![]),
        kind: TensorKind::Weight,
        initializer: Some(delta.to_le_bytes().to_vec()),
    });

    // Calculate output size
    let output_size = ((limit - start) / delta).ceil().max(0.0) as usize;

    // Add output tensor (1D F32)
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![output_size]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Range node
    let mut node = Node::new("Range");
    node.name = "range_node".to_string();
    node.inputs = vec![
        "start".to_string(),
        "limit".to_string(),
        "delta".to_string(),
    ];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec![];
    graph.outputs = vec!["output".to_string()];

    graph.metadata.name = "test_range_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

/// End-to-end test: Range with integer step=1.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_range_integer_step_1_e2e() {
    let graph = make_range_graph_i64(0, 5, 1);
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

    let outputs = executor.run(&[]).expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected: [0, 1, 2, 3, 4]
    let expected = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];

    assert_eq!(output, expected, "Range(0, 5, 1) result incorrect");

    println!("✓ End-to-end Range(0, 5, 1) test passed!");
    println!("  Expected: [0, 1, 2, 3, 4]");
    println!("  Got: {:?}", output);
}

/// End-to-end test: Range with integer step=2.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_range_integer_step_2_e2e() {
    let graph = make_range_graph_i64(2, 10, 2);
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

    let outputs = executor.run(&[]).expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected: [2, 4, 6, 8]
    let expected = vec![2.0f32, 4.0, 6.0, 8.0];

    assert_eq!(output, expected, "Range(2, 10, 2) result incorrect");

    println!("✓ End-to-end Range(2, 10, 2) test passed!");
    println!("  Expected: [2, 4, 6, 8]");
    println!("  Got: {:?}", output);
}

/// End-to-end test: Range with negative step (descending).
#[pollster::test]
#[ignore] // Requires GPU
async fn test_range_negative_step_e2e() {
    let graph = make_range_graph_i64(10, 0, -2);
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

    let outputs = executor.run(&[]).expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected: [10, 8, 6, 4, 2]
    let expected = vec![10.0f32, 8.0, 6.0, 4.0, 2.0];

    assert_eq!(output, expected, "Range(10, 0, -2) result incorrect");

    println!("✓ End-to-end Range(10, 0, -2) test passed!");
    println!("  Expected: [10, 8, 6, 4, 2]");
    println!("  Got: {:?}", output);
}

/// End-to-end test: Range with floating point values.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_range_float_e2e() {
    let graph = make_range_graph_f32(0.0, 2.5, 0.5);
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

    let outputs = executor.run(&[]).expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected: [0.0, 0.5, 1.0, 1.5, 2.0]
    let expected = vec![0.0f32, 0.5, 1.0, 1.5, 2.0];

    // Use approximate comparison for floating point
    for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "Range float result mismatch at index {}: got {}, expected {}",
            i,
            got,
            exp
        );
    }

    println!("✓ End-to-end Range(0.0, 2.5, 0.5) test passed!");
    println!("  Expected: [0.0, 0.5, 1.0, 1.5, 2.0]");
    println!("  Got: {:?}", output);
}

/// End-to-end test: Range with empty output (start == limit).
#[pollster::test]
#[ignore] // Requires GPU
async fn test_range_empty_e2e() {
    let graph = make_range_graph_i64(5, 5, 1);
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

    let outputs = executor.run(&[]).expect("Execution should succeed");

    let output_tensor = &outputs["output"];

    // Check that the shape is [0]
    assert_eq!(
        output_tensor.shape(),
        &[0],
        "Range(5, 5, 1) shape should be [0]"
    );
    assert!(output_tensor.is_empty(), "Range(5, 5, 1) should be empty");

    println!("✓ End-to-end Range(5, 5, 1) empty test passed!");
    println!("  Expected shape: [0]");
    println!("  Got shape: {:?}", output_tensor.shape());
}
