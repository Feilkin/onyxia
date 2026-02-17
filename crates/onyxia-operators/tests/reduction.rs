//! Tests for reduction and element-wise max operators.
//!
//! These tests require a GPU. Run with:
//! ```sh
//! cargo nextest run -p onyxia-operators reduction --run-ignored=all
//! ```

mod common;

use common::{CompilerPipeline, Runtime, Tensor};
use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
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

// ================================================================================
// ReduceMean operator tests
// ================================================================================

/// Helper to create a ReduceMean graph with axes as an initializer.
fn make_reduce_mean_graph(
    input_shape: &[usize],
    output_shape: &[usize],
    axes: &[i64],
    keepdims: bool,
) -> Graph {
    let mut graph = Graph::new();

    // Add input tensor
    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(input_shape.to_vec()),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add axes as initializer tensor (constant)
    let axes_data: Vec<u8> = axes.iter().flat_map(|&x| x.to_le_bytes()).collect();

    graph.add_tensor(TensorInfo {
        name: "axes".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![axes.len()]),
        kind: TensorKind::Weight,
        initializer: Some(axes_data),
    });

    // Add output tensor
    graph.add_tensor(TensorInfo {
        name: "reduced".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(output_shape.to_vec()),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create ReduceMean node
    let mut node = Node::new("ReduceMean");
    node.name = "reduce_mean_op".to_string();
    node.inputs = vec!["data".to_string(), "axes".to_string()];
    node.outputs = vec!["reduced".to_string()];
    node.attributes.insert(
        "keepdims".to_string(),
        onyxia_onnx::AttributeValue::Int(if keepdims { 1 } else { 0 }),
    );
    graph.add_node(node);

    // Set graph inputs and outputs
    graph.inputs = vec!["data".to_string()];
    graph.outputs = vec!["reduced".to_string()];

    // Set metadata
    graph.metadata.name = "test_reduce_mean_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_reduce_mean_single_axis() {
    // Reduce along axis 0: [[1, 2, 3], [4, 5, 6]] -> [2.5, 3.5, 4.5]
    // Input shape: (2, 3), output shape: (3,)
    let graph = make_reduce_mean_graph(&[2, 3], &[3], &[0], false);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let outputs = executor.run(&[("data", input)]).unwrap();
    let result: Vec<f32> = outputs["reduced"].to_vec().unwrap();

    let expected = vec![2.5, 3.5, 4.5];
    assert_vec_approx_eq(&result, &expected, 1e-5);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_reduce_mean_last_axis() {
    // Reduce along last axis: [[1, 2, 3], [4, 5, 6]] -> [2, 5]
    // Input shape: (2, 3), output shape: (2,)
    let graph = make_reduce_mean_graph(&[2, 3], &[2], &[1], false);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let outputs = executor.run(&[("data", input)]).unwrap();
    let result: Vec<f32> = outputs["reduced"].to_vec().unwrap();

    let expected = vec![2.0, 5.0];
    assert_vec_approx_eq(&result, &expected, 1e-5);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_reduce_mean_all_axes() {
    // Reduce all axes: [1, 2, 3, 4] -> [2.5]
    // Input shape: (4,), output shape: (1,) (scalar)
    let graph = make_reduce_mean_graph(&[4], &[1], &[0], false);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]);
    let outputs = executor.run(&[("data", input)]).unwrap();
    let result: Vec<f32> = outputs["reduced"].to_vec().unwrap();

    assert_approx_eq(result[0], 2.5, 1e-5);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_reduce_mean_keepdims() {
    // Reduce with keepdims: [[1, 2, 3]] -> [[2]]
    // Input shape: (1, 3), output shape: (1, 1)
    let graph = make_reduce_mean_graph(&[1, 3], &[1, 1], &[1], true);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[1, 3]);
    let outputs = executor.run(&[("data", input)]).unwrap();
    let result: Vec<f32> = outputs["reduced"].to_vec().unwrap();

    assert_approx_eq(result[0], 2.0, 1e-5);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_reduce_mean_3d_tensor() {
    // Reduce middle axis: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] with axes=[1]
    // Input shape: (2, 2, 2), output shape: (2, 2)
    // Expected: [[(1+3)/2, (2+4)/2], [(5+7)/2, (6+8)/2]] = [[2, 3], [6, 7]]
    let graph = make_reduce_mean_graph(&[2, 2, 2], &[2, 2], &[1], false);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let outputs = executor.run(&[("data", input)]).unwrap();
    let result: Vec<f32> = outputs["reduced"].to_vec().unwrap();

    let expected = vec![2.0, 3.0, 6.0, 7.0];
    assert_vec_approx_eq(&result, &expected, 1e-5);
}

// ================================================================================
// Max operator tests (element-wise maximum)
// ================================================================================

/// Helper to create a Max graph with N inputs.
fn make_max_graph(input_shapes: &[&[usize]], output_shape: &[usize]) -> Graph {
    let mut graph = Graph::new();

    // Add input tensors
    let input_names: Vec<String> = (0..input_shapes.len())
        .map(|i| format!("input_{}", i))
        .collect();

    for (i, shape) in input_shapes.iter().enumerate() {
        graph.add_tensor(TensorInfo {
            name: input_names[i].clone(),
            dtype: DataType::F32,
            shape: TensorShape::Static(shape.to_vec()),
            kind: TensorKind::Input,
            initializer: None,
        });
    }

    // Add output tensor
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(output_shape.to_vec()),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Max node
    let mut node = Node::new("Max");
    node.name = "max_op".to_string();
    node.inputs = input_names.clone();
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    // Set graph inputs and outputs
    graph.inputs = input_names;
    graph.outputs = vec!["output".to_string()];

    // Set metadata
    graph.metadata.name = "test_max_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_max_two_inputs() {
    // Max of two vectors: [1, 5, 3] and [4, 2, 6] -> [4, 5, 6]
    let graph = make_max_graph(&[&[3], &[3]], &[3]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input_0 = Tensor::from_vec(vec![1.0f32, 5.0, 3.0], &[3]);
    let input_1 = Tensor::from_vec(vec![4.0f32, 2.0, 6.0], &[3]);
    let outputs = executor
        .run(&[("input_0", input_0), ("input_1", input_1)])
        .unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_eq!(result, vec![4.0, 5.0, 6.0]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_max_three_inputs() {
    // Max of three vectors: [1, 5, 3], [4, 2, 6], [3, 7, 1] -> [4, 7, 6]
    let graph = make_max_graph(&[&[3], &[3], &[3]], &[3]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input_0 = Tensor::from_vec(vec![1.0f32, 5.0, 3.0], &[3]);
    let input_1 = Tensor::from_vec(vec![4.0f32, 2.0, 6.0], &[3]);
    let input_2 = Tensor::from_vec(vec![3.0f32, 7.0, 1.0], &[3]);
    let outputs = executor
        .run(&[
            ("input_0", input_0),
            ("input_1", input_1),
            ("input_2", input_2),
        ])
        .unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_eq!(result, vec![4.0, 7.0, 6.0]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_max_broadcasting_scalar() {
    // Max with scalar: [1, 2, 3] and [5] -> [5, 5, 5]
    let graph = make_max_graph(&[&[3], &[1]], &[3]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input_0 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]);
    let input_1 = Tensor::from_vec(vec![5.0f32], &[1]);
    let outputs = executor
        .run(&[("input_0", input_0), ("input_1", input_1)])
        .unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_eq!(result, vec![5.0, 5.0, 5.0]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_max_broadcasting_matrix_vector() {
    // Max with broadcasting: [[1, 2], [3, 4]] and [5, 1] -> [[5, 2], [5, 4]]
    // Matrix shape: (2, 2), vector shape: (2,)
    // Vector broadcasts to each row
    let graph = make_max_graph(&[&[2, 2], &[2]], &[2, 2]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input_0 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
    let input_1 = Tensor::from_vec(vec![5.0f32, 1.0], &[2]);
    let outputs = executor
        .run(&[("input_0", input_0), ("input_1", input_1)])
        .unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_eq!(result, vec![5.0, 2.0, 5.0, 4.0]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_max_negative_values() {
    // Max with negative values: [-5, -2, -8] and [-3, -4, -1] -> [-3, -2, -1]
    let graph = make_max_graph(&[&[3], &[3]], &[3]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input_0 = Tensor::from_vec(vec![-5.0f32, -2.0, -8.0], &[3]);
    let input_1 = Tensor::from_vec(vec![-3.0f32, -4.0, -1.0], &[3]);
    let outputs = executor
        .run(&[("input_0", input_0), ("input_1", input_1)])
        .unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_eq!(result, vec![-3.0, -2.0, -1.0]);
}
