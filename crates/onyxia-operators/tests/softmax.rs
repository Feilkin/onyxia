//! Tests for the Softmax operator.
//!
//! These tests require a GPU. Run with:
//! ```sh
//! cargo nextest run -p onyxia-operators softmax --run-ignored=all
//! ```

mod common;

use common::{CompilerPipeline, Runtime, Tensor};
use onyxia_onnx::{AttributeValue, DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_operators::core_operator_registry;

/// Helper to assert that two f32 values are approximately equal.
fn assert_approx_eq(a: f32, b: f32, epsilon: f32) {
    let diff = (a - b).abs();
    assert!(
        diff < epsilon,
        "Values not approximately equal: {} vs {} (diff: {})",
        a,
        b,
        diff
    );
}

/// Helper to assert that two f32 vectors are approximately equal element-wise.
fn assert_vec_approx_eq(actual: &[f32], expected: &[f32], epsilon: f32) {
    assert_eq!(actual.len(), expected.len(), "Vector lengths differ");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < epsilon,
            "Element {} differs: {} vs {} (diff: {})",
            i,
            a,
            e,
            diff
        );
    }
}

/// Helper to create a Softmax graph.
fn make_softmax_graph(input_shape: &[usize], axis: i64) -> Graph {
    let mut graph = Graph::new();

    // Add input tensor
    graph.add_tensor(TensorInfo {
        name: "input".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(input_shape.to_vec()),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add output tensor
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(input_shape.to_vec()), // Same shape as input
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Softmax node
    let mut node = Node::new("Softmax");
    node.name = "softmax_op".to_string();
    node.inputs = vec!["input".to_string()];
    node.outputs = vec!["output".to_string()];
    node.attributes
        .insert("axis".to_string(), AttributeValue::Int(axis));
    graph.add_node(node);

    // Set graph inputs and outputs
    graph.inputs = vec!["input".to_string()];
    graph.outputs = vec!["output".to_string()];

    // Set metadata
    graph.metadata.name = "test_softmax_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_softmax_1d() {
    // Simple 1D softmax: [1.0, 2.0, 3.0]
    // Expected output: softmax([1, 2, 3])
    // = exp([1, 2, 3]) / sum(exp([1, 2, 3]))
    // = [exp(1), exp(2), exp(3)] / (exp(1) + exp(2) + exp(3))
    // ≈ [0.09003, 0.24473, 0.66524]

    let graph = make_softmax_graph(&[3], -1);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]);
    let outputs = executor.run(&[("input", input)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Verify output sums to 1.0
    let sum: f32 = result.iter().sum();
    assert_approx_eq(sum, 1.0, 1e-5);

    // Verify expected values
    let expected = vec![0.09003057, 0.24472848, 0.66524095];
    assert_vec_approx_eq(&result, &expected, 1e-4);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_softmax_2d_axis_1() {
    // 2D softmax along axis=1 (last axis)
    // Input: [[1.0, 2.0, 3.0],
    //         [4.0, 5.0, 6.0]]
    // Each row becomes a probability distribution

    let graph = make_softmax_graph(&[2, 3], 1);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let outputs = executor.run(&[("input", input)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Each row should sum to 1.0
    let row1_sum: f32 = result[0..3].iter().sum();
    let row2_sum: f32 = result[3..6].iter().sum();
    assert_approx_eq(row1_sum, 1.0, 1e-5);
    assert_approx_eq(row2_sum, 1.0, 1e-5);

    // Expected values (same for both rows since values are shifted uniformly)
    let expected = vec![
        0.09003057, 0.24472848, 0.66524095, // Row 1: softmax([1, 2, 3])
        0.09003057, 0.24472848, 0.66524095, // Row 2: softmax([4, 5, 6])
    ];
    assert_vec_approx_eq(&result, &expected, 1e-4);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_softmax_2d_axis_0() {
    // 2D softmax along axis=0 (first axis)
    // Input: [[1.0, 2.0],
    //         [3.0, 4.0]]
    // Each column becomes a probability distribution

    let graph = make_softmax_graph(&[2, 2], 0);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
    let outputs = executor.run(&[("input", input)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Each column should sum to 1.0
    // Column 0: result[0] + result[2]
    // Column 1: result[1] + result[3]
    let col1_sum = result[0] + result[2];
    let col2_sum = result[1] + result[3];
    assert_approx_eq(col1_sum, 1.0, 1e-5);
    assert_approx_eq(col2_sum, 1.0, 1e-5);

    // Calculate expected values
    // Column 0: softmax([1, 3]) = [exp(1), exp(3)] / (exp(1) + exp(3))
    // Column 1: softmax([2, 4]) = [exp(2), exp(4)] / (exp(2) + exp(4))
    let expected = vec![
        0.11920292, 0.11920292, // Row 1: [softmax([1,3])[0], softmax([2,4])[0]]
        0.88079708, 0.88079708, // Row 2: [softmax([1,3])[1], softmax([2,4])[1]]
    ];
    assert_vec_approx_eq(&result, &expected, 1e-4);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_softmax_negative_axis() {
    // Test axis=-1 (default, last axis)
    // Input: [[1.0, 2.0], [3.0, 4.0]]
    // Same as axis=1 for 2D tensor

    let graph = make_softmax_graph(&[2, 2], -1);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
    let outputs = executor.run(&[("input", input)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Each row should sum to 1.0
    let row1_sum: f32 = result[0..2].iter().sum();
    let row2_sum: f32 = result[2..4].iter().sum();
    assert_approx_eq(row1_sum, 1.0, 1e-5);
    assert_approx_eq(row2_sum, 1.0, 1e-5);

    // Expected: softmax along last dimension
    let expected = vec![
        0.26894143, 0.7310586, // softmax([1, 2])
        0.26894143, 0.7310586, // softmax([3, 4])
    ];
    assert_vec_approx_eq(&result, &expected, 1e-4);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_softmax_numerical_stability() {
    // Test with large values to verify max-trick prevents overflow
    // Input: [1000.0, 1000.0, 1000.0]
    // Without max-trick: exp(1000) → Inf
    // With max-trick: exp(1000 - 1000) = exp(0) = 1.0
    // Output: [0.33333, 0.33333, 0.33333]

    let graph = make_softmax_graph(&[3], -1);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec(vec![1000.0f32, 1000.0, 1000.0], &[3]);
    let outputs = executor.run(&[("input", input)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Verify no NaN or Inf
    for &val in &result {
        assert!(val.is_finite(), "Result contains non-finite value: {}", val);
    }

    // Verify uniform distribution
    let expected = vec![0.33333334, 0.33333334, 0.33333334];
    assert_vec_approx_eq(&result, &expected, 1e-5);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_softmax_large_values() {
    // Test with large but different values
    // Input: [100.0, 200.0, 300.0]
    // exp(x - 300) keeps values in valid range
    // Output: [≈0, ≈0, ≈1]

    let graph = make_softmax_graph(&[3], -1);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec(vec![100.0f32, 200.0, 300.0], &[3]);
    let outputs = executor.run(&[("input", input)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Verify no NaN or Inf
    for &val in &result {
        assert!(val.is_finite(), "Result contains non-finite value: {}", val);
    }

    // Verify sum is 1
    let sum: f32 = result.iter().sum();
    assert_approx_eq(sum, 1.0, 1e-5);

    // First two values should be very close to 0, last should be ≈1
    assert!(result[0] < 1e-10, "First value should be ≈0: {}", result[0]);
    assert!(
        result[1] < 1e-10,
        "Second value should be ≈0: {}",
        result[1]
    );
    assert_approx_eq(result[2], 1.0, 1e-5);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_softmax_3d_tensor() {
    // Test 3D tensor with axis=2 (last axis)
    // Shape: (2, 2, 3)
    // Each vector along the last dimension becomes a probability distribution

    let graph = make_softmax_graph(&[2, 2, 3], 2);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    // Input data: 2x2x3 tensor
    let input = Tensor::from_vec(
        vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[2, 2, 3],
    );
    let outputs = executor.run(&[("input", input)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Each vector along last dimension should sum to 1
    for i in 0..4 {
        let slice_sum: f32 = result[i * 3..(i + 1) * 3].iter().sum();
        assert_approx_eq(slice_sum, 1.0, 1e-5);
    }

    // Expected: each group of 3 elements is softmax([a, a+1, a+2])
    let expected = vec![
        0.09003057, 0.24472848, 0.66524095, // softmax([1, 2, 3])
        0.09003057, 0.24472848, 0.66524095, // softmax([4, 5, 6])
        0.09003057, 0.24472848, 0.66524095, // softmax([7, 8, 9])
        0.09003057, 0.24472848, 0.66524095, // softmax([10, 11, 12])
    ];
    assert_vec_approx_eq(&result, &expected, 1e-4);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_softmax_3d_middle_axis() {
    // Test 3D tensor with axis=1 (middle axis)
    // Shape: (2, 3, 2)
    // Each vector along the middle dimension becomes a probability distribution

    let graph = make_softmax_graph(&[2, 3, 2], 1);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    // Input data: 2x3x2 tensor
    let input = Tensor::from_vec(
        vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[2, 3, 2],
    );
    let outputs = executor.run(&[("input", input)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // For each (outer, inner) pair, the middle dimension should sum to 1
    // [0, :, 0]: positions 0, 2, 4
    // [0, :, 1]: positions 1, 3, 5
    // [1, :, 0]: positions 6, 8, 10
    // [1, :, 1]: positions 7, 9, 11

    let sum1 = result[0] + result[2] + result[4]; // [0, :, 0]
    let sum2 = result[1] + result[3] + result[5]; // [0, :, 1]
    let sum3 = result[6] + result[8] + result[10]; // [1, :, 0]
    let sum4 = result[7] + result[9] + result[11]; // [1, :, 1]

    assert_approx_eq(sum1, 1.0, 1e-5);
    assert_approx_eq(sum2, 1.0, 1e-5);
    assert_approx_eq(sum3, 1.0, 1e-5);
    assert_approx_eq(sum4, 1.0, 1e-5);
}
