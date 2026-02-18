//! Tests for indexing operators (Gather, ScatterND).
//!
//! These tests require a GPU. Run with:
//! ```sh
//! cargo nextest run -p onyxia-operators indexing --run-ignored=all
//! ```

mod common;

use common::{CompilerPipeline, Runtime, Tensor};
use onyxia_onnx::{AttributeValue, DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_operators::core_operator_registry;

// ================================================================================
// Gather operator tests
// ================================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_gather_1d_axis_0() {
    // Gather from 1D tensor along axis 0
    // data: [1, 2, 3, 4, 5]
    // indices: [0, 2, 4]
    // output: [1, 3, 5]
    let mut graph = Graph::new();

    // Data input: [5]
    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![5]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Indices input: [3]
    graph.add_tensor(TensorInfo {
        name: "indices".to_string(),
        dtype: DataType::I32,
        shape: TensorShape::Static(vec![3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Output: [3]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![3]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Gather");
    node.name = "gather_op".to_string();
    node.inputs = vec!["data".to_string(), "indices".to_string()];
    node.outputs = vec!["output".to_string()];
    node.attributes
        .insert("axis".to_string(), AttributeValue::Int(0));
    graph.add_node(node);

    graph.inputs = vec!["data".to_string(), "indices".to_string()];
    graph.outputs = vec!["output".to_string()];

    // Compile and execute
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]);
    let indices = Tensor::from_vec(vec![0i32, 2, 4], &[3]);

    let outputs = executor
        .run(&[("data", data), ("indices", indices)])
        .unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_eq!(result, vec![1.0, 3.0, 5.0]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_gather_2d_axis_0() {
    // Gather from 2D tensor along axis 0
    // data: [[1, 2], [3, 4], [5, 6]]  shape: (3, 2)
    // indices: [2, 0]  shape: (2,)
    // output: [[5, 6], [1, 2]]  shape: (2, 2)
    let mut graph = Graph::new();

    // Data input: [3, 2]
    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![3, 2]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Indices input: [2]
    graph.add_tensor(TensorInfo {
        name: "indices".to_string(),
        dtype: DataType::I32,
        shape: TensorShape::Static(vec![2]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Output: [2, 2]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 2]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Gather");
    node.name = "gather_op".to_string();
    node.inputs = vec!["data".to_string(), "indices".to_string()];
    node.outputs = vec!["output".to_string()];
    node.attributes
        .insert("axis".to_string(), AttributeValue::Int(0));
    graph.add_node(node);

    graph.inputs = vec!["data".to_string(), "indices".to_string()];
    graph.outputs = vec!["output".to_string()];

    // Compile and execute
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let indices = Tensor::from_vec(vec![2i32, 0], &[2]);

    let outputs = executor
        .run(&[("data", data), ("indices", indices)])
        .unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Expected: [[5, 6], [1, 2]] = [5, 6, 1, 2]
    assert_eq!(result, vec![5.0, 6.0, 1.0, 2.0]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_gather_2d_axis_1() {
    // Gather from 2D tensor along axis 1
    // data: [[1, 2, 3], [4, 5, 6]]  shape: (2, 3)
    // indices: [[0, 2], [1, 0]]  shape: (2, 2)
    // output: [[[1, 3], [2, 1]], [[4, 6], [5, 4]]]  shape: (2, 2, 2)
    let mut graph = Graph::new();

    // Data input: [2, 3]
    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Indices input: [2, 2]
    graph.add_tensor(TensorInfo {
        name: "indices".to_string(),
        dtype: DataType::I32,
        shape: TensorShape::Static(vec![2, 2]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Output: [2, 2, 2]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 2, 2]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Gather");
    node.name = "gather_op".to_string();
    node.inputs = vec!["data".to_string(), "indices".to_string()];
    node.outputs = vec!["output".to_string()];
    node.attributes
        .insert("axis".to_string(), AttributeValue::Int(1));
    graph.add_node(node);

    graph.inputs = vec!["data".to_string(), "indices".to_string()];
    graph.outputs = vec!["output".to_string()];

    // Compile and execute
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let indices = Tensor::from_vec(vec![0i32, 2, 1, 0], &[2, 2]);

    let outputs = executor
        .run(&[("data", data), ("indices", indices)])
        .unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Expected: [[[1, 3], [2, 1]], [[4, 6], [5, 4]]]
    // Row-major layout: [1, 3, 2, 1, 4, 6, 5, 4]
    assert_eq!(result, vec![1.0, 3.0, 2.0, 1.0, 4.0, 6.0, 5.0, 4.0]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_gather_negative_indices() {
    // Gather with negative indices
    // data: [1, 2, 3, 4, 5]
    // indices: [0, -1, -2]
    // output: [1, 5, 4]
    let mut graph = Graph::new();

    // Data input: [5]
    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![5]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Indices input: [3]
    graph.add_tensor(TensorInfo {
        name: "indices".to_string(),
        dtype: DataType::I32,
        shape: TensorShape::Static(vec![3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Output: [3]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![3]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Gather");
    node.name = "gather_op".to_string();
    node.inputs = vec!["data".to_string(), "indices".to_string()];
    node.outputs = vec!["output".to_string()];
    node.attributes
        .insert("axis".to_string(), AttributeValue::Int(0));
    graph.add_node(node);

    graph.inputs = vec!["data".to_string(), "indices".to_string()];
    graph.outputs = vec!["output".to_string()];

    // Compile and execute
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]);
    let indices = Tensor::from_vec(vec![0i32, -1, -2], &[3]);

    let outputs = executor
        .run(&[("data", data), ("indices", indices)])
        .unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_eq!(result, vec![1.0, 5.0, 4.0]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_gather_f32_with_i64_indices() {
    let mut graph = Graph::new();

    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![6]),
        kind: TensorKind::Input,
        initializer: None,
    });
    graph.add_tensor(TensorInfo {
        name: "indices".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![3]),
        kind: TensorKind::Input,
        initializer: None,
    });
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![3]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Gather");
    node.name = "gather_i64_indices".to_string();
    node.inputs = vec!["data".to_string(), "indices".to_string()];
    node.outputs = vec!["output".to_string()];
    node.attributes
        .insert("axis".to_string(), AttributeValue::Int(0));
    graph.add_node(node);

    graph.inputs = vec!["data".to_string(), "indices".to_string()];
    graph.outputs = vec!["output".to_string()];

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(vec![10.0f32, 11.0, 12.0, 13.0, 14.0, 15.0], &[6]);
    let indices = Tensor::from_vec(vec![5i64, 0, 3], &[3]);

    let outputs = executor
        .run(&[("data", data), ("indices", indices)])
        .unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_eq!(result, vec![15.0, 10.0, 13.0]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_gather_i64_data_cpu_fallback() {
    let mut graph = Graph::new();

    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![2, 3]),
        kind: TensorKind::Input,
        initializer: None,
    });
    graph.add_tensor(TensorInfo {
        name: "indices".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![2]),
        kind: TensorKind::Input,
        initializer: None,
    });
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![2, 2]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Gather");
    node.name = "gather_i64_data".to_string();
    node.inputs = vec!["data".to_string(), "indices".to_string()];
    node.outputs = vec!["output".to_string()];
    node.attributes
        .insert("axis".to_string(), AttributeValue::Int(1));
    graph.add_node(node);

    graph.inputs = vec!["data".to_string(), "indices".to_string()];
    graph.outputs = vec!["output".to_string()];

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(vec![1i64, 2, 3, 4, 5, 6], &[2, 3]);
    let indices = Tensor::from_vec(vec![2i64, 0], &[2]);

    let outputs = executor
        .run(&[("data", data), ("indices", indices)])
        .unwrap();
    let result: Vec<i64> = outputs["output"].to_vec().unwrap();

    assert_eq!(result, vec![3, 1, 6, 4]);
}

// ================================================================================
// ScatterND operator tests
// ================================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_scatter_nd_1d() {
    // ScatterND on 1D tensor
    // data: [1, 2, 3, 4, 5]
    // indices: [[1], [3]]  shape: (2, 1)
    // updates: [10, 20]  shape: (2,)
    // output: [1, 10, 3, 20, 5]
    let mut graph = Graph::new();

    // Data input: [5]
    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![5]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Indices input: [2, 1]
    graph.add_tensor(TensorInfo {
        name: "indices".to_string(),
        dtype: DataType::I32,
        shape: TensorShape::Static(vec![2, 1]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Updates input: [2]
    graph.add_tensor(TensorInfo {
        name: "updates".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Output: [5]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![5]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("ScatterND");
    node.name = "scatter_nd_op".to_string();
    node.inputs = vec![
        "data".to_string(),
        "indices".to_string(),
        "updates".to_string(),
    ];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec![
        "data".to_string(),
        "indices".to_string(),
        "updates".to_string(),
    ];
    graph.outputs = vec!["output".to_string()];

    // Compile and execute
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]);
    let indices = Tensor::from_vec(vec![1i32, 3], &[2, 1]);
    let updates = Tensor::from_vec(vec![10.0f32, 20.0], &[2]);

    let outputs = executor
        .run(&[("data", data), ("indices", indices), ("updates", updates)])
        .unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_eq!(result, vec![1.0, 10.0, 3.0, 20.0, 5.0]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_scatter_nd_2d_point_updates() {
    // ScatterND on 2D tensor with point updates
    // data: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  shape: (3, 3)
    // indices: [[0, 0], [1, 1]]  shape: (2, 2)
    // updates: [100, 200]  shape: (2,)
    // output: [[100, 2, 3], [4, 200, 6], [7, 8, 9]]
    let mut graph = Graph::new();

    // Data input: [3, 3]
    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![3, 3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Indices input: [2, 2]
    graph.add_tensor(TensorInfo {
        name: "indices".to_string(),
        dtype: DataType::I32,
        shape: TensorShape::Static(vec![2, 2]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Updates input: [2]
    graph.add_tensor(TensorInfo {
        name: "updates".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Output: [3, 3]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![3, 3]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("ScatterND");
    node.name = "scatter_nd_op".to_string();
    node.inputs = vec![
        "data".to_string(),
        "indices".to_string(),
        "updates".to_string(),
    ];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec![
        "data".to_string(),
        "indices".to_string(),
        "updates".to_string(),
    ];
    graph.outputs = vec!["output".to_string()];

    // Compile and execute
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
    );
    let indices = Tensor::from_vec(vec![0i32, 0, 1, 1], &[2, 2]);
    let updates = Tensor::from_vec(vec![100.0f32, 200.0], &[2]);

    let outputs = executor
        .run(&[("data", data), ("indices", indices), ("updates", updates)])
        .unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Expected: [[100, 2, 3], [4, 200, 6], [7, 8, 9]]
    assert_eq!(
        result,
        vec![100.0, 2.0, 3.0, 4.0, 200.0, 6.0, 7.0, 8.0, 9.0]
    );
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_scatter_nd_2d_slice_updates() {
    // ScatterND on 2D tensor with slice updates
    // data: [[1, 2, 3], [4, 5, 6]]  shape: (2, 3)
    // indices: [[1]]  shape: (1, 1) - update row 1
    // updates: [[10, 20, 30]]  shape: (1, 3)
    // output: [[1, 2, 3], [10, 20, 30]]
    let mut graph = Graph::new();

    // Data input: [2, 3]
    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Indices input: [1, 1]
    graph.add_tensor(TensorInfo {
        name: "indices".to_string(),
        dtype: DataType::I32,
        shape: TensorShape::Static(vec![1, 1]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Updates input: [1, 3]
    graph.add_tensor(TensorInfo {
        name: "updates".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![1, 3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Output: [2, 3]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("ScatterND");
    node.name = "scatter_nd_op".to_string();
    node.inputs = vec![
        "data".to_string(),
        "indices".to_string(),
        "updates".to_string(),
    ];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec![
        "data".to_string(),
        "indices".to_string(),
        "updates".to_string(),
    ];
    graph.outputs = vec!["output".to_string()];

    // Compile and execute
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let indices = Tensor::from_vec(vec![1i32], &[1, 1]);
    let updates = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], &[1, 3]);

    let outputs = executor
        .run(&[("data", data), ("indices", indices), ("updates", updates)])
        .unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Expected: [[1, 2, 3], [10, 20, 30]]
    assert_eq!(result, vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
}
