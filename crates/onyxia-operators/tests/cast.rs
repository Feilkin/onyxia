//! Tests for the Cast operator.

mod common;

use common::{CompilerPipeline, Runtime, Tensor};
use onyxia_onnx::{AttributeValue, DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_operators::core_operator_registry;

/// Helper to create a Cast graph with specified input/output dtypes.
fn make_cast_graph(input_dtype: DataType, output_dtype: DataType, shape: &[usize]) -> Graph {
    let mut graph = Graph::new();

    // Add input tensor
    graph.add_tensor(TensorInfo {
        name: "input".to_string(),
        dtype: input_dtype,
        shape: TensorShape::Static(shape.to_vec()),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add output tensor
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: output_dtype,
        shape: TensorShape::Static(shape.to_vec()),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Cast node with "to" attribute
    let onnx_dtype_code = match output_dtype {
        DataType::F32 => 1,
        DataType::I32 => 6,
        DataType::I64 => 7,
        DataType::Bool => 9,
        DataType::U32 => 12,
        _ => panic!("Unsupported dtype for test"),
    };

    let mut node = Node::new("Cast");
    node.name = "cast_node".to_string();
    node.inputs = vec!["input".to_string()];
    node.outputs = vec!["output".to_string()];
    node.attributes
        .insert("to".to_string(), AttributeValue::Int(onnx_dtype_code));
    graph.add_node(node);

    // Set graph inputs and outputs
    graph.inputs = vec!["input".to_string()];
    graph.outputs = vec!["output".to_string()];

    // Set metadata
    graph.metadata.name = "test_cast_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

/// Helper to compile and run a Cast operation from f32 input.
async fn run_cast_f32(
    input_data: Vec<f32>,
    input_shape: &[usize],
    target_dtype: DataType,
) -> Vec<u8> {
    let graph = make_cast_graph(DataType::F32, target_dtype, input_shape);

    // Compile model
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    // Create runtime and execute
    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();
    let input_tensor = Tensor::from_vec(input_data, input_shape);
    let outputs = executor.run(&[("input", input_tensor)]).unwrap();

    // Read output
    outputs["output"].as_bytes().unwrap().to_vec()
}

/// Helper to compile and run a Cast operation from i32 input.
async fn run_cast_i32(
    input_data: Vec<i32>,
    input_shape: &[usize],
    target_dtype: DataType,
) -> Vec<u8> {
    let graph = make_cast_graph(DataType::I32, target_dtype, input_shape);

    // Compile model
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    // Create runtime and execute
    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();
    let input_tensor = Tensor::from_vec(input_data, input_shape);
    let outputs = executor.run(&[("input", input_tensor)]).unwrap();

    outputs["output"].as_bytes().unwrap().to_vec()
}

#[pollster::test]
#[ignore = "requires GPU"]
async fn test_cast_f32_to_i32() {
    let input = vec![1.7, 2.3, -3.9, 0.0];
    let output_bytes = run_cast_f32(input, &[4], DataType::I32).await;

    // Parse i32 values
    let output: Vec<i32> = output_bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    // Truncate towards zero
    assert_eq!(output, vec![1, 2, -3, 0]);
}

#[pollster::test]
#[ignore = "requires GPU"]
async fn test_cast_f32_to_bool() {
    let input = vec![0.0, 1.0, -1.0, 0.5];
    let output_bytes = run_cast_f32(input, &[4], DataType::Bool).await;

    // Parse bool values (stored as u32)
    let output: Vec<u32> = output_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    // 0.0 → false (0), others → true (1)
    assert_eq!(output, vec![0, 1, 1, 1]);
}

#[pollster::test]
#[ignore = "requires GPU"]
async fn test_cast_f32_to_u32() {
    let input = vec![1.5, 2.9, 0.0, 100.0];
    let output_bytes = run_cast_f32(input, &[4], DataType::U32).await;

    // Parse u32 values
    let output: Vec<u32> = output_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    // Truncate towards zero
    assert_eq!(output, vec![1, 2, 0, 100]);
}

#[pollster::test]
#[ignore = "requires GPU"]
async fn test_cast_i32_to_f32() {
    let input = vec![1, 2, -3, 0];
    let output_bytes = run_cast_i32(input, &[4], DataType::F32).await;

    // Parse f32 values
    let output: Vec<f32> = output_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    assert_eq!(output, vec![1.0, 2.0, -3.0, 0.0]);
}

#[pollster::test]
#[ignore = "requires GPU"]
async fn test_cast_i32_to_bool() {
    let input = vec![0, 1, -1, 100];
    let output_bytes = run_cast_i32(input, &[4], DataType::Bool).await;

    // Parse bool values (stored as u32)
    let output: Vec<u32> = output_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    // 0 → false (0), others → true (1)
    assert_eq!(output, vec![0, 1, 1, 1]);
}

#[pollster::test]
#[ignore = "requires GPU"]
async fn test_cast_2d_tensor() {
    // Test that Cast works with multi-dimensional tensors
    let input = vec![1.5, 2.3, -3.7, 0.0];
    let output_bytes = run_cast_f32(input, &[2, 2], DataType::I32).await;

    let output: Vec<i32> = output_bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    // Truncate towards zero
    assert_eq!(output, vec![1, 2, -3, 0]);
}

#[pollster::test]
#[ignore = "requires GPU"]
async fn test_cast_negative_float_to_unsigned() {
    // Negative values should be clamped to 0 when casting to unsigned
    let input = vec![-1.0, -2.5, 3.0, 0.0];
    let output_bytes = run_cast_f32(input, &[4], DataType::U32).await;

    let output: Vec<u32> = output_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    // Negative values clamped to 0
    assert_eq!(output, vec![0, 0, 3, 0]);
}
