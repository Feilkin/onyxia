//! Integration tests for special operators (Range, Trilu, Where).

mod common;

use common::{CompilerPipeline, Runtime};
use onyxia_onnx::{AttributeValue, DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_operators::core_operator_registry;

/// Helper to assert that two f32 vectors are approximately equal element-wise.
fn assert_vec_eq(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "Vector lengths differ");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-4,
            "Element {} differs: {} vs {}",
            i,
            a,
            e
        );
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Range tests
// ══════════════════════════════════════════════════════════════════════════════

/// Helper to create a Range graph.
fn make_range_graph(start: f32, limit: f32, delta: f32, output_len: usize) -> Graph {
    let mut graph = Graph::new();

    // Add start initializer
    graph.add_tensor(TensorInfo {
        name: "start".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![]),
        kind: TensorKind::Weight,
        initializer: Some(bytemuck::cast_slice(&[start]).to_vec()),
    });

    // Add limit initializer
    graph.add_tensor(TensorInfo {
        name: "limit".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![]),
        kind: TensorKind::Weight,
        initializer: Some(bytemuck::cast_slice(&[limit]).to_vec()),
    });

    // Add delta initializer
    graph.add_tensor(TensorInfo {
        name: "delta".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![]),
        kind: TensorKind::Weight,
        initializer: Some(bytemuck::cast_slice(&[delta]).to_vec()),
    });

    // Add output tensor
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![output_len]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Range node
    let mut node = Node::new("Range");
    node.name = "range_op".to_string();
    node.inputs = vec![
        "start".to_string(),
        "limit".to_string(),
        "delta".to_string(),
    ];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    // Set graph inputs and outputs
    graph.inputs = vec![];
    graph.outputs = vec!["output".to_string()];

    // Set metadata
    graph.metadata.name = "test_range_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_range_basic() {
    let graph = make_range_graph(0.0, 5.0, 1.0, 5);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let outputs = executor.run(&[]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_vec_eq(&result, &[0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_range_float_step() {
    let graph = make_range_graph(0.0, 1.0, 0.25, 4);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let outputs = executor.run(&[]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_vec_eq(&result, &[0.0, 0.25, 0.5, 0.75]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_range_negative_step() {
    let graph = make_range_graph(5.0, 0.0, -1.0, 5);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let outputs = executor.run(&[]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_vec_eq(&result, &[5.0, 4.0, 3.0, 2.0, 1.0]);
}

// ══════════════════════════════════════════════════════════════════════════════
// Trilu tests
// ══════════════════════════════════════════════════════════════════════════════

/// Helper to create a Trilu graph.
fn make_trilu_graph(input_data: &[f32], shape: &[usize], upper: bool, k: Option<i64>) -> Graph {
    let mut graph = Graph::new();

    // Add input initializer
    graph.add_tensor(TensorInfo {
        name: "input".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(shape.to_vec()),
        kind: TensorKind::Weight,
        initializer: Some(bytemuck::cast_slice(input_data).to_vec()),
    });

    // Add optional k parameter
    if let Some(k_val) = k {
        graph.add_tensor(TensorInfo {
            name: "k".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![]),
            kind: TensorKind::Weight,
            initializer: Some(bytemuck::cast_slice(&[k_val]).to_vec()),
        });
    }

    // Add output tensor
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(shape.to_vec()),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Trilu node
    let mut node = Node::new("Trilu");
    node.name = "trilu_op".to_string();
    node.inputs = if k.is_some() {
        vec!["input".to_string(), "k".to_string()]
    } else {
        vec!["input".to_string()]
    };
    node.outputs = vec!["output".to_string()];
    node.attributes.insert(
        "upper".to_string(),
        AttributeValue::Int(if upper { 1 } else { 0 }),
    );
    graph.add_node(node);

    // Set graph inputs and outputs
    graph.inputs = vec![];
    graph.outputs = vec!["output".to_string()];

    // Set metadata
    graph.metadata.name = "test_trilu_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_trilu_upper() {
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let graph = make_trilu_graph(&input_data, &[3, 3], true, None);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let outputs = executor.run(&[]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Expected: [[1,2,3], [0,5,6], [0,0,9]]
    assert_vec_eq(&result, &[1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_trilu_lower() {
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let graph = make_trilu_graph(&input_data, &[3, 3], false, None);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let outputs = executor.run(&[]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Expected: [[1,0,0], [4,5,0], [7,8,9]]
    assert_vec_eq(&result, &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_trilu_upper_offset() {
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let graph = make_trilu_graph(&input_data, &[3, 3], true, Some(1));

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let outputs = executor.run(&[]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Expected: [[0,2,3], [0,0,6], [0,0,0]] (above main diagonal)
    assert_vec_eq(&result, &[0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]);
}

// ══════════════════════════════════════════════════════════════════════════════
// Where tests
// ══════════════════════════════════════════════════════════════════════════════

/// Helper to create a Where graph.
fn make_where_graph(
    cond_data: &[u8],
    cond_shape: &[usize],
    x_data: &[f32],
    x_shape: &[usize],
    y_data: &[f32],
    y_shape: &[usize],
    output_shape: &[usize],
) -> Graph {
    let mut graph = Graph::new();

    // Add condition initializer (bool data)
    graph.add_tensor(TensorInfo {
        name: "condition".to_string(),
        dtype: DataType::Bool,
        shape: TensorShape::Static(cond_shape.to_vec()),
        kind: TensorKind::Weight,
        initializer: Some(cond_data.to_vec()),
    });

    // Add X initializer
    graph.add_tensor(TensorInfo {
        name: "x".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(x_shape.to_vec()),
        kind: TensorKind::Weight,
        initializer: Some(bytemuck::cast_slice(x_data).to_vec()),
    });

    // Add Y initializer
    graph.add_tensor(TensorInfo {
        name: "y".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(y_shape.to_vec()),
        kind: TensorKind::Weight,
        initializer: Some(bytemuck::cast_slice(y_data).to_vec()),
    });

    // Add output tensor
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(output_shape.to_vec()),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Where node
    let mut node = Node::new("Where");
    node.name = "where_op".to_string();
    node.inputs = vec!["condition".to_string(), "x".to_string(), "y".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    // Set graph inputs and outputs
    graph.inputs = vec![];
    graph.outputs = vec!["output".to_string()];

    // Set metadata
    graph.metadata.name = "test_where_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_where_basic() {
    // Condition: [true, false, true]
    let cond_data = bytemuck::cast_slice(&[1u32, 0u32, 1u32]).to_vec();
    let x_data = vec![1.0f32, 2.0, 3.0];
    let y_data = vec![10.0f32, 20.0, 30.0];

    let graph = make_where_graph(&cond_data, &[3], &x_data, &[3], &y_data, &[3], &[3]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let outputs = executor.run(&[]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Expected: [1, 20, 3] (select from X where condition is true)
    assert_vec_eq(&result, &[1.0, 20.0, 3.0]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_where_broadcast() {
    // Test with same-shape inputs (no actual broadcasting needed)
    // TODO: Implement proper NumPy-style broadcasting instead of simple modulo
    let cond_data = bytemuck::cast_slice(&[1u32, 0u32, 1u32, 0u32]).to_vec();
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let y_data = vec![5.0f32, 6.0, 7.0, 8.0];

    let graph = make_where_graph(&cond_data, &[4], &x_data, &[4], &y_data, &[4], &[4]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let outputs = executor.run(&[]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Expected: [1, 6, 3, 8] (condition selects alternating elements)
    assert_vec_eq(&result, &[1.0, 6.0, 3.0, 8.0]);
}
