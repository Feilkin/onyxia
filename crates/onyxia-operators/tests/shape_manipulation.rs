//! Tests for shape manipulation operators (Concat, Expand, Transpose, Unsqueeze).
//!
//! These tests require a GPU. Run with:
//! ```sh
//! cargo nextest run -p onyxia-operators shape_manipulation --run-ignored=all
//! ```

mod common;

use common::{CompilerPipeline, Runtime, Tensor};
use onyxia_onnx::{AttributeValue, DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_operators::core_operator_registry;

// ================================================================================
// Concat operator tests
// ================================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_concat_axis_0() {
    // Concat two tensors along axis 0: [2,3] + [3,3] -> [5,3]
    let mut graph = Graph::new();

    // First input: [2, 3]
    graph.add_tensor(TensorInfo {
        name: "a".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Second input: [3, 3]
    graph.add_tensor(TensorInfo {
        name: "b".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![3, 3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Output: [5, 3]
    graph.add_tensor(TensorInfo {
        name: "c".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![5, 3]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Concat");
    node.name = "concat_axis0".to_string();
    node.inputs = vec!["a".to_string(), "b".to_string()];
    node.outputs = vec!["c".to_string()];
    node.attributes
        .insert("axis".to_string(), AttributeValue::Int(0));
    graph.add_node(node);

    graph.inputs = vec!["a".to_string(), "b".to_string()];
    graph.outputs = vec!["c".to_string()];

    // Compile and execute
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::from_vec(
        vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        &[3, 3],
    );

    let outputs = executor.run(&[("a", a), ("b", b)]).unwrap();
    let result: Vec<f32> = outputs["c"].to_vec().unwrap();

    assert_eq!(
        result,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0
        ]
    );
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_concat_axis_1() {
    // Concat two tensors along axis 1: [2,3] + [2,5] -> [2,8]
    let mut graph = Graph::new();

    // First input: [2, 3]
    graph.add_tensor(TensorInfo {
        name: "a".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Second input: [2, 5]
    graph.add_tensor(TensorInfo {
        name: "b".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 5]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Output: [2, 8]
    graph.add_tensor(TensorInfo {
        name: "c".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 8]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Concat");
    node.name = "concat_axis1".to_string();
    node.inputs = vec!["a".to_string(), "b".to_string()];
    node.outputs = vec!["c".to_string()];
    node.attributes
        .insert("axis".to_string(), AttributeValue::Int(1));
    graph.add_node(node);

    graph.inputs = vec!["a".to_string(), "b".to_string()];
    graph.outputs = vec!["c".to_string()];

    // Compile and execute
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::from_vec(
        vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        &[2, 5],
    );

    let outputs = executor.run(&[("a", a), ("b", b)]).unwrap();
    let result: Vec<f32> = outputs["c"].to_vec().unwrap();

    // Expected: first row = [1,2,3,7,8,9,10,11], second row = [4,5,6,12,13,14,15,16]
    assert_eq!(
        result,
        vec![
            1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 10.0, 11.0, 4.0, 5.0, 6.0, 12.0, 13.0, 14.0, 15.0, 16.0
        ]
    );
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_concat_negative_axis() {
    // Test -1 axis (last axis) on [2,3] tensors
    let mut graph = Graph::new();

    graph.add_tensor(TensorInfo {
        name: "a".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    graph.add_tensor(TensorInfo {
        name: "b".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 2]),
        kind: TensorKind::Input,
        initializer: None,
    });

    graph.add_tensor(TensorInfo {
        name: "c".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 5]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Concat");
    node.name = "concat_neg_axis".to_string();
    node.inputs = vec!["a".to_string(), "b".to_string()];
    node.outputs = vec!["c".to_string()];
    node.attributes
        .insert("axis".to_string(), AttributeValue::Int(-1));
    graph.add_node(node);

    graph.inputs = vec!["a".to_string(), "b".to_string()];
    graph.outputs = vec!["c".to_string()];

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::from_vec(vec![7.0f32, 8.0, 9.0, 10.0], &[2, 2]);

    let outputs = executor.run(&[("a", a), ("b", b)]).unwrap();
    let result: Vec<f32> = outputs["c"].to_vec().unwrap();

    // Expected: [1,2,3,7,8] [4,5,6,9,10]
    assert_eq!(
        result,
        vec![1.0, 2.0, 3.0, 7.0, 8.0, 4.0, 5.0, 6.0, 9.0, 10.0]
    );
}

// ================================================================================
// Expand operator tests
// ================================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_expand_broadcast_column() {
    // Expand [3, 1] -> [3, 4] (broadcast column to 4 columns)
    let mut graph = Graph::new();

    // Input data: [3, 1]
    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![3, 1]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Shape input: [2] containing [3, 4]
    let shape_bytes = bytemuck::cast_slice(&[3i64, 4i64]).to_vec();
    graph.add_tensor(TensorInfo {
        name: "shape".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![2]),
        kind: TensorKind::Weight,
        initializer: Some(shape_bytes),
    });

    // Output: [3, 4]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![3, 4]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Expand");
    node.name = "expand_col".to_string();
    node.inputs = vec!["data".to_string(), "shape".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["data".to_string()];
    graph.outputs = vec!["output".to_string()];

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3, 1]);

    let outputs = executor.run(&[("data", data)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Expected: each row repeated 4 times
    assert_eq!(
        result,
        vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
    );
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_expand_broadcast_row() {
    // Expand [1, 4] -> [3, 4] (broadcast row to 3 rows)
    let mut graph = Graph::new();

    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![1, 4]),
        kind: TensorKind::Input,
        initializer: None,
    });

    let shape_bytes = bytemuck::cast_slice(&[3i64, 4i64]).to_vec();
    graph.add_tensor(TensorInfo {
        name: "shape".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![2]),
        kind: TensorKind::Weight,
        initializer: Some(shape_bytes),
    });

    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![3, 4]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Expand");
    node.name = "expand_row".to_string();
    node.inputs = vec!["data".to_string(), "shape".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["data".to_string()];
    graph.outputs = vec!["output".to_string()];

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 4]);

    let outputs = executor.run(&[("data", data)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Expected: [1,2,3,4] repeated 3 times
    assert_eq!(
        result,
        vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
    );
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_expand_scalar() {
    // Expand [1] -> [2, 3] (broadcast scalar)
    let mut graph = Graph::new();

    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Input,
        initializer: None,
    });

    let shape_bytes = bytemuck::cast_slice(&[2i64, 3i64]).to_vec();
    graph.add_tensor(TensorInfo {
        name: "shape".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![2]),
        kind: TensorKind::Weight,
        initializer: Some(shape_bytes),
    });

    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Expand");
    node.name = "expand_scalar".to_string();
    node.inputs = vec!["data".to_string(), "shape".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["data".to_string()];
    graph.outputs = vec!["output".to_string()];

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(vec![42.0f32], &[1]);

    let outputs = executor.run(&[("data", data)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Expected: 42.0 repeated 6 times
    assert_eq!(result, vec![42.0; 6]);
}

// ================================================================================
// Transpose operator tests
// ================================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_transpose_2d() {
    // Transpose [2, 3] -> [3, 2] (simple 2D transpose)
    let mut graph = Graph::new();

    graph.add_tensor(TensorInfo {
        name: "input".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![3, 2]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Transpose");
    node.name = "transpose_2d".to_string();
    node.inputs = vec!["input".to_string()];
    node.outputs = vec!["output".to_string()];
    // No perm attribute means default reverse: [1, 0]
    graph.add_node(node);

    graph.inputs = vec!["input".to_string()];
    graph.outputs = vec!["output".to_string()];

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    // Input: [[1, 2, 3], [4, 5, 6]]
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let outputs = executor.run(&[("input", input)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Expected: [[1, 4], [2, 5], [3, 6]]
    assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_transpose_3d() {
    // Transpose [2, 3, 4] with perm=[2, 0, 1] -> [4, 2, 3]
    let mut graph = Graph::new();

    graph.add_tensor(TensorInfo {
        name: "input".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3, 4]),
        kind: TensorKind::Input,
        initializer: None,
    });

    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4, 2, 3]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Transpose");
    node.name = "transpose_3d".to_string();
    node.inputs = vec!["input".to_string()];
    node.outputs = vec!["output".to_string()];
    node.attributes
        .insert("perm".to_string(), AttributeValue::Ints(vec![2, 0, 1]));
    graph.add_node(node);

    graph.inputs = vec!["input".to_string()];
    graph.outputs = vec!["output".to_string()];

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    // Input: 2x3x4 tensor filled with sequential values
    let input_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let input = Tensor::from_vec(input_data, &[2, 3, 4]);

    let outputs = executor.run(&[("input", input)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Verify a few key elements
    // Original [0,0,0] = 0 -> new [0,0,0] = 0
    assert_eq!(result[0], 0.0);
    // Original [0,0,1] = 1 -> new [1,0,0] = 1
    assert_eq!(result[6], 1.0); // index [1,0,0] in [4,2,3] = 1*6 + 0*3 + 0 = 6
    // Original [1,2,3] = 23 -> new [3,1,2] = 23
    assert_eq!(result[23], 23.0); // index [3,1,2] in [4,2,3] = 3*6 + 1*3 + 2 = 23
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_transpose_identity() {
    // Transpose with identity permutation [0, 1, 2] should be no-op
    let mut graph = Graph::new();

    graph.add_tensor(TensorInfo {
        name: "input".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3, 4]),
        kind: TensorKind::Input,
        initializer: None,
    });

    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3, 4]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Transpose");
    node.name = "transpose_identity".to_string();
    node.inputs = vec!["input".to_string()];
    node.outputs = vec!["output".to_string()];
    node.attributes
        .insert("perm".to_string(), AttributeValue::Ints(vec![0, 1, 2]));
    graph.add_node(node);

    graph.inputs = vec!["input".to_string()];
    graph.outputs = vec!["output".to_string()];

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let input = Tensor::from_vec(input_data.clone(), &[2, 3, 4]);

    let outputs = executor.run(&[("input", input)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Should be identical to input
    assert_eq!(result, input_data);
}

// ================================================================================
// Unsqueeze operator tests
// ================================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_unsqueeze_single_axis() {
    // Unsqueeze [2, 3] at axis 0 -> [1, 2, 3]
    let mut graph = Graph::new();

    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    let axes_bytes = bytemuck::cast_slice(&[0i64]).to_vec();
    graph.add_tensor(TensorInfo {
        name: "axes".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Weight,
        initializer: Some(axes_bytes),
    });

    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![1, 2, 3]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Unsqueeze");
    node.name = "unsqueeze_axis0".to_string();
    node.inputs = vec!["data".to_string(), "axes".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["data".to_string()];
    graph.outputs = vec!["output".to_string()];

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let outputs = executor.run(&[("data", data)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Data should be unchanged, only shape changes
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_unsqueeze_multiple_axes() {
    // Unsqueeze [2, 3] at axes [0, 3] -> [1, 2, 3, 1]
    let mut graph = Graph::new();

    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    let axes_bytes = bytemuck::cast_slice(&[0i64, 3i64]).to_vec();
    graph.add_tensor(TensorInfo {
        name: "axes".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![2]),
        kind: TensorKind::Weight,
        initializer: Some(axes_bytes),
    });

    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![1, 2, 3, 1]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Unsqueeze");
    node.name = "unsqueeze_multi".to_string();
    node.inputs = vec!["data".to_string(), "axes".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["data".to_string()];
    graph.outputs = vec!["output".to_string()];

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let outputs = executor.run(&[("data", data)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Data should be unchanged
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_unsqueeze_trailing() {
    // Unsqueeze [2] at axes [1, 2] -> [2, 1, 1]
    let mut graph = Graph::new();

    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2]),
        kind: TensorKind::Input,
        initializer: None,
    });

    let axes_bytes = bytemuck::cast_slice(&[1i64, 2i64]).to_vec();
    graph.add_tensor(TensorInfo {
        name: "axes".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![2]),
        kind: TensorKind::Weight,
        initializer: Some(axes_bytes),
    });

    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 1, 1]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Unsqueeze");
    node.name = "unsqueeze_trailing".to_string();
    node.inputs = vec!["data".to_string(), "axes".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["data".to_string()];
    graph.outputs = vec!["output".to_string()];

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(vec![42.0f32, 99.0], &[2]);

    let outputs = executor.run(&[("data", data)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_eq!(result, vec![42.0, 99.0]);
}

// ================================================================================
// Shape operator tests
// ================================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_shape_1d() {
    // Shape of [5] -> [1]
    let mut graph = Graph::new();

    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![5]),
        kind: TensorKind::Input,
        initializer: None,
    });

    graph.add_tensor(TensorInfo {
        name: "shape".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Shape");
    node.name = "shape_1d".to_string();
    node.inputs = vec!["data".to_string()];
    node.outputs = vec!["shape".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["data".to_string()];
    graph.outputs = vec!["shape".to_string()];

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]);

    let outputs = executor.run(&[("data", data)]).unwrap();
    let result: Vec<i64> = outputs["shape"].to_vec().unwrap();

    assert_eq!(result, vec![5]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_shape_2d() {
    // Shape of [2, 3] -> [2]
    let mut graph = Graph::new();

    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    graph.add_tensor(TensorInfo {
        name: "shape".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![2]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Shape");
    node.name = "shape_2d".to_string();
    node.inputs = vec!["data".to_string()];
    node.outputs = vec!["shape".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["data".to_string()];
    graph.outputs = vec!["shape".to_string()];

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let outputs = executor.run(&[("data", data)]).unwrap();
    let result: Vec<i64> = outputs["shape"].to_vec().unwrap();

    assert_eq!(result, vec![2, 3]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_shape_4d() {
    // Shape of [1, 3, 224, 224] -> [4]
    let mut graph = Graph::new();

    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![1, 3, 224, 224]),
        kind: TensorKind::Input,
        initializer: None,
    });

    graph.add_tensor(TensorInfo {
        name: "shape".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Shape");
    node.name = "shape_4d".to_string();
    node.inputs = vec!["data".to_string()];
    node.outputs = vec!["shape".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["data".to_string()];
    graph.outputs = vec!["shape".to_string()];

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    // Create a small dummy tensor with the right shape (not full 1×3×224×224)
    let data = Tensor::from_vec(vec![1.0f32; 1 * 3 * 224 * 224], &[1, 3, 224, 224]);

    let outputs = executor.run(&[("data", data)]).unwrap();
    let result: Vec<i64> = outputs["shape"].to_vec().unwrap();

    assert_eq!(result, vec![1, 3, 224, 224]);
}

// ================================================================================
// ConstantOfShape operator tests
// ================================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_constant_of_shape_zeros() {
    // Create tensor of shape [2, 3] filled with 0.0
    let mut graph = Graph::new();

    let shape_bytes = bytemuck::cast_slice(&[2i64, 3i64]).to_vec();
    graph.add_tensor(TensorInfo {
        name: "shape".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![2]),
        kind: TensorKind::Weight,
        initializer: Some(shape_bytes),
    });

    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("ConstantOfShape");
    node.name = "constant_of_shape_zeros".to_string();
    node.inputs = vec!["shape".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec![];
    graph.outputs = vec!["output".to_string()];

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let outputs = executor.run(&[]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_eq!(result, vec![0.0; 6]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_constant_of_shape_1d() {
    // Create tensor of shape [5] filled with 0.0
    let mut graph = Graph::new();

    let shape_bytes = bytemuck::cast_slice(&[5i64]).to_vec();
    graph.add_tensor(TensorInfo {
        name: "shape".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Weight,
        initializer: Some(shape_bytes),
    });

    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![5]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("ConstantOfShape");
    node.name = "constant_of_shape_1d".to_string();
    node.inputs = vec!["shape".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec![];
    graph.outputs = vec!["output".to_string()];

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let outputs = executor.run(&[]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_eq!(result, vec![0.0; 5]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_constant_of_shape_large() {
    // Create tensor of shape [100, 100] filled with 0.0
    let mut graph = Graph::new();

    let shape_bytes = bytemuck::cast_slice(&[100i64, 100i64]).to_vec();
    graph.add_tensor(TensorInfo {
        name: "shape".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![2]),
        kind: TensorKind::Weight,
        initializer: Some(shape_bytes),
    });

    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![100, 100]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("ConstantOfShape");
    node.name = "constant_of_shape_large".to_string();
    node.inputs = vec!["shape".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec![];
    graph.outputs = vec!["output".to_string()];

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let outputs = executor.run(&[]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_eq!(result.len(), 10000);
    assert!(result.iter().all(|&x| x == 0.0));
}
