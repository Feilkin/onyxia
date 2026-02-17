//! Tests for Slice operator.
//!
//! These tests require a GPU. Run with:
//! ```sh
//! cargo nextest run -p onyxia-operators slice --run-ignored=all
//! ```

mod common;

use common::{CompilerPipeline, Runtime, Tensor};
use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_operators::core_operator_registry;

// ================================================================================
// Helper function to create a slice graph
// ================================================================================

fn create_slice_graph(
    data_shape: Vec<usize>,
    starts: Vec<i64>,
    ends: Vec<i64>,
    axes: Option<Vec<i64>>,
    steps: Option<Vec<i64>>,
    output_shape: Vec<usize>,
) -> Graph {
    let mut graph = Graph::new();

    // Data input
    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(data_shape),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Starts input (i64 tensor)
    graph.add_tensor(TensorInfo {
        name: "starts".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![starts.len()]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Ends input (i64 tensor)
    graph.add_tensor(TensorInfo {
        name: "ends".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![ends.len()]),
        kind: TensorKind::Input,
        initializer: None,
    });

    let mut node_inputs = vec!["data".to_string(), "starts".to_string(), "ends".to_string()];

    // Axes input (optional)
    if let Some(ref axes_vec) = axes {
        graph.add_tensor(TensorInfo {
            name: "axes".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![axes_vec.len()]),
            kind: TensorKind::Input,
            initializer: None,
        });
        node_inputs.push("axes".to_string());
    }

    // Steps input (optional)
    if let Some(ref steps_vec) = steps {
        graph.add_tensor(TensorInfo {
            name: "steps".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![steps_vec.len()]),
            kind: TensorKind::Input,
            initializer: None,
        });
        node_inputs.push("steps".to_string());
    }

    // Output
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(output_shape),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut node = Node::new("Slice");
    node.name = "slice_node".to_string();
    node.inputs = node_inputs;
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["data".to_string(), "starts".to_string(), "ends".to_string()];
    if axes.is_some() {
        graph.inputs.push("axes".to_string());
    }
    if steps.is_some() {
        graph.inputs.push("steps".to_string());
    }
    graph.outputs = vec!["output".to_string()];

    graph
}

// ================================================================================
// Basic slicing tests
// ================================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_slice_1d_basic() {
    // Extract middle portion: [0,1,2,3,4,5,6,7,8,9] -> [2,3,4,5,6]
    let graph = create_slice_graph(
        vec![10],
        vec![2],
        vec![7],
        Some(vec![0]),
        Some(vec![1]),
        vec![5],
    );

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(
        vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[10],
    );
    let starts = Tensor::from_vec(vec![2i64], &[1]);
    let ends = Tensor::from_vec(vec![7i64], &[1]);
    let axes = Tensor::from_vec(vec![0i64], &[1]);
    let steps = Tensor::from_vec(vec![1i64], &[1]);

    let outputs = executor
        .run(&[
            ("data", data),
            ("starts", starts),
            ("ends", ends),
            ("axes", axes),
            ("steps", steps),
        ])
        .unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_slice_2d_basic() {
    // 2D slice: [[0,1,2,3], [4,5,6,7], [8,9,10,11]] -> [[5,6], [9,10]]
    let graph = create_slice_graph(
        vec![3, 4],
        vec![1, 1],
        vec![3, 3],
        Some(vec![0, 1]),
        Some(vec![1, 1]),
        vec![2, 2],
    );

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(
        vec![
            0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
        ],
        &[3, 4],
    );
    let starts = Tensor::from_vec(vec![1i64, 1], &[2]);
    let ends = Tensor::from_vec(vec![3i64, 3], &[2]);
    let axes = Tensor::from_vec(vec![0i64, 1], &[2]);
    let steps = Tensor::from_vec(vec![1i64, 1], &[2]);

    let outputs = executor
        .run(&[
            ("data", data),
            ("starts", starts),
            ("ends", ends),
            ("axes", axes),
            ("steps", steps),
        ])
        .unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Expected: [[5,6], [9,10]]
    assert_eq!(result, vec![5.0, 6.0, 9.0, 10.0]);
}

// ================================================================================
// Negative indices tests
// ================================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_slice_negative_indices() {
    // Negative start/end: [0,1,2,3,4] with starts=[-3], ends=[-1] -> [2,3]
    let graph = create_slice_graph(
        vec![5],
        vec![-3],
        vec![-1],
        Some(vec![0]),
        Some(vec![1]),
        vec![2],
    );

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0, 4.0], &[5]);
    let starts = Tensor::from_vec(vec![-3i64], &[1]);
    let ends = Tensor::from_vec(vec![-1i64], &[1]);
    let axes = Tensor::from_vec(vec![0i64], &[1]);
    let steps = Tensor::from_vec(vec![1i64], &[1]);

    let outputs = executor
        .run(&[
            ("data", data),
            ("starts", starts),
            ("ends", ends),
            ("axes", axes),
            ("steps", steps),
        ])
        .unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_eq!(result, vec![2.0, 3.0]);
}

// ================================================================================
// Step size tests
// ================================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_slice_step_2() {
    // Step > 1 (skip elements): [0,1,2,3,4,5,6,7,8,9] with step=2 -> [0,2,4,6,8]
    let graph = create_slice_graph(
        vec![10],
        vec![0],
        vec![10],
        Some(vec![0]),
        Some(vec![2]),
        vec![5],
    );

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(
        vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[10],
    );
    let starts = Tensor::from_vec(vec![0i64], &[1]);
    let ends = Tensor::from_vec(vec![10i64], &[1]);
    let axes = Tensor::from_vec(vec![0i64], &[1]);
    let steps = Tensor::from_vec(vec![2i64], &[1]);

    let outputs = executor
        .run(&[
            ("data", data),
            ("starts", starts),
            ("ends", ends),
            ("axes", axes),
            ("steps", steps),
        ])
        .unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_eq!(result, vec![0.0, 2.0, 4.0, 6.0, 8.0]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_slice_negative_step() {
    // Reverse slice (step < 0): [0,1,2,3,4] with start=4, end=0, step=-1 -> [4,3,2,1]
    let graph = create_slice_graph(
        vec![5],
        vec![4],
        vec![0],
        Some(vec![0]),
        Some(vec![-1]),
        vec![4],
    );

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0, 4.0], &[5]);
    let starts = Tensor::from_vec(vec![4i64], &[1]);
    let ends = Tensor::from_vec(vec![0i64], &[1]);
    let axes = Tensor::from_vec(vec![0i64], &[1]);
    let steps = Tensor::from_vec(vec![-1i64], &[1]);

    let outputs = executor
        .run(&[
            ("data", data),
            ("starts", starts),
            ("ends", ends),
            ("axes", axes),
            ("steps", steps),
        ])
        .unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_eq!(result, vec![4.0, 3.0, 2.0, 1.0]);
}

// ================================================================================
// Multi-axis slicing tests
// ================================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_slice_multi_axis() {
    // 3D tensor: [2,3,4] -> slice on multiple axes
    let graph = create_slice_graph(
        vec![2, 3, 4],
        vec![0, 1, 1],
        vec![2, 3, 3],
        Some(vec![0, 1, 2]),
        Some(vec![1, 1, 1]),
        vec![2, 2, 2],
    );

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec((0..24).map(|x| x as f32).collect(), &[2, 3, 4]);
    let starts = Tensor::from_vec(vec![0i64, 1, 1], &[3]);
    let ends = Tensor::from_vec(vec![2i64, 3, 3], &[3]);
    let axes = Tensor::from_vec(vec![0i64, 1, 2], &[3]);
    let steps = Tensor::from_vec(vec![1i64, 1, 1], &[3]);

    let outputs = executor
        .run(&[
            ("data", data),
            ("starts", starts),
            ("ends", ends),
            ("axes", axes),
            ("steps", steps),
        ])
        .unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Extract [0:2, 1:3, 1:3] from [2,3,4]
    // First slice: rows 0-1 (all), columns 1-2 (of 0-2), depth 1-2 (of 0-3)
    // Shape: [2, 2, 2]
    // First block (slice 0): rows 1-2 of [[0,1,2,3],[4,5,6,7],[8,9,10,11]], cols 1-2
    //   [[5,6],[9,10]]
    // Second block (slice 1): rows 1-2 of [[12,13,14,15],[16,17,18,19],[20,21,22,23]], cols 1-2
    //   [[17,18],[21,22]]
    assert_eq!(result, vec![5.0, 6.0, 9.0, 10.0, 17.0, 18.0, 21.0, 22.0]);
}

// ================================================================================
// Default parameters tests
// ================================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_slice_default_axes_and_steps() {
    // Omit axes and steps (should default to [0] and [1])
    let graph = create_slice_graph(
        vec![10],
        vec![2],
        vec![7],
        None, // No axes
        None, // No steps
        vec![5],
    );

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(
        vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[10],
    );
    let starts = Tensor::from_vec(vec![2i64], &[1]);
    let ends = Tensor::from_vec(vec![7i64], &[1]);

    let outputs = executor
        .run(&[("data", data), ("starts", starts), ("ends", ends)])
        .unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0, 6.0]);
}

// ================================================================================
// Edge cases
// ================================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_slice_full_range() {
    // Slice entire tensor: [0,1,2,3,4] with start=0, end=5 -> [0,1,2,3,4]
    let graph = create_slice_graph(
        vec![5],
        vec![0],
        vec![5],
        Some(vec![0]),
        Some(vec![1]),
        vec![5],
    );

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0, 4.0], &[5]);
    let starts = Tensor::from_vec(vec![0i64], &[1]);
    let ends = Tensor::from_vec(vec![5i64], &[1]);
    let axes = Tensor::from_vec(vec![0i64], &[1]);
    let steps = Tensor::from_vec(vec![1i64], &[1]);

    let outputs = executor
        .run(&[
            ("data", data),
            ("starts", starts),
            ("ends", ends),
            ("axes", axes),
            ("steps", steps),
        ])
        .unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_eq!(result, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_slice_single_element() {
    // Slice single element: [0,1,2,3,4] with start=2, end=3 -> [2]
    let graph = create_slice_graph(
        vec![5],
        vec![2],
        vec![3],
        Some(vec![0]),
        Some(vec![1]),
        vec![1],
    );

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let data = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0, 4.0], &[5]);
    let starts = Tensor::from_vec(vec![2i64], &[1]);
    let ends = Tensor::from_vec(vec![3i64], &[1]);
    let axes = Tensor::from_vec(vec![0i64], &[1]);
    let steps = Tensor::from_vec(vec![1i64], &[1]);

    let outputs = executor
        .run(&[
            ("data", data),
            ("starts", starts),
            ("ends", ends),
            ("axes", axes),
            ("steps", steps),
        ])
        .unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_eq!(result, vec![2.0]);
}
