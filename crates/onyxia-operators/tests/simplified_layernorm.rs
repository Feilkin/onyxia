//! Tests for SimplifiedLayerNormalization operator.
//!
//! These tests require a GPU. Run with:
//! ```sh
//! cargo nextest run -p onyxia-operators simplified_layernorm --run-ignored=all
//! ```

mod common;

use common::{CompilerPipeline, Runtime, Tensor};
use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_operators::core_operator_registry;

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

/// Helper to create a SimplifiedLayerNormalization graph.
fn make_simplified_layernorm_graph(
    input_shape: &[usize],
    scale_shape: &[usize],
    scale_data: &[f32],
    axis: i64,
    epsilon: f32,
) -> Graph {
    let mut graph = Graph::new();

    // Add input tensor
    graph.add_tensor(TensorInfo {
        name: "X".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(input_shape.to_vec()),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add scale as initializer tensor (constant)
    let scale_bytes: Vec<u8> = scale_data.iter().flat_map(|&x| x.to_le_bytes()).collect();

    graph.add_tensor(TensorInfo {
        name: "scale".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(scale_shape.to_vec()),
        kind: TensorKind::Weight,
        initializer: Some(scale_bytes),
    });

    // Add output tensor
    graph.add_tensor(TensorInfo {
        name: "Y".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(input_shape.to_vec()),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create SimplifiedLayerNormalization node
    let mut node = Node::new("SimplifiedLayerNormalization");
    node.name = "simplified_layernorm_op".to_string();
    node.domain = "com.microsoft".to_string();
    node.inputs = vec!["X".to_string(), "scale".to_string()];
    node.outputs = vec!["Y".to_string()];
    node.attributes
        .insert("axis".to_string(), onyxia_onnx::AttributeValue::Int(axis));
    node.attributes.insert(
        "epsilon".to_string(),
        onyxia_onnx::AttributeValue::Float(epsilon),
    );
    graph.add_node(node);

    // Set graph inputs and outputs
    graph.inputs = vec!["X".to_string()];
    graph.outputs = vec!["Y".to_string()];

    // Set metadata
    graph.metadata.name = "test_simplified_layernorm_graph".to_string();

    graph
}

// =====================================================================
// SimplifiedLayerNormalization operator tests
// ====================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_simplified_layernorm_basic_2d() {
    let input_shape = vec![2, 4];
    let scale_shape = vec![4];
    let scale_data = vec![1.0, 1.0, 1.0, 1.0];
    let epsilon = 1e-5;

    let graph =
        make_simplified_layernorm_graph(&input_shape, &scale_shape, &scale_data, -1, epsilon);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let compiled = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(compiled).await.unwrap();

    let input_data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_tensor = Tensor::from_vec(input_data, &input_shape);

    let outputs = executor.run(&[("X", input_tensor)]).unwrap();
    let result: Vec<f32> = outputs["Y"].to_vec().unwrap();

    let rms_0 = (7.5_f32 + epsilon).sqrt();
    let rms_1 = (43.5_f32 + epsilon).sqrt();

    let expected = vec![
        1.0 / rms_0,
        2.0 / rms_0,
        3.0 / rms_0,
        4.0 / rms_0,
        5.0 / rms_1,
        6.0 / rms_1,
        7.0 / rms_1,
        8.0 / rms_1,
    ];

    assert_vec_approx_eq(&result, &expected, 1e-4);

    println!("✓ SimplifiedLayerNormalization 2D basic test passed!");
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_simplified_layernorm_with_scale() {
    let input_shape = vec![2, 4];
    let scale_shape = vec![4];
    let scale_data = vec![0.5, 2.0, 1.5, 1.0];
    let epsilon = 1e-5;

    let graph =
        make_simplified_layernorm_graph(&input_shape, &scale_shape, &scale_data, -1, epsilon);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let compiled = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(compiled).await.unwrap();

    let input_data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_tensor = Tensor::from_vec(input_data, &input_shape);

    let outputs = executor.run(&[("X", input_tensor)]).unwrap();
    let result: Vec<f32> = outputs["Y"].to_vec().unwrap();

    let rms_0 = (7.5_f32 + epsilon).sqrt();
    let rms_1 = (43.5_f32 + epsilon).sqrt();

    let expected = vec![
        (1.0 / rms_0) * 0.5,
        (2.0 / rms_0) * 2.0,
        (3.0 / rms_0) * 1.5,
        (4.0 / rms_0) * 1.0,
        (5.0 / rms_1) * 0.5,
        (6.0 / rms_1) * 2.0,
        (7.0 / rms_1) * 1.5,
        (8.0 / rms_1) * 1.0,
    ];

    assert_vec_approx_eq(&result, &expected, 1e-4);

    println!("✓ SimplifiedLayerNormalization with scale test passed!");
}
