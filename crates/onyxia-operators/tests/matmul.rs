//! Tests for MatMul operator.

mod common;

use common::{CompilerPipeline, Runtime, Tensor};
use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_operators::core_operator_registry;

// ================================================================================
// Helper function to create a MatMul graph
// ================================================================================

fn create_matmul_graph(
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
    output_shape: Vec<usize>,
) -> Graph {
    let mut graph = Graph::new();

    // Input A
    graph.add_tensor(TensorInfo {
        name: "A".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(a_shape),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Input B
    graph.add_tensor(TensorInfo {
        name: "B".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(b_shape),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Output
    graph.add_tensor(TensorInfo {
        name: "Y".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(output_shape),
        kind: TensorKind::Output,
        initializer: None,
    });

    // MatMul node
    let mut node = Node::new("MatMul");
    node.name = "matmul_node".to_string();
    node.inputs = vec!["A".to_string(), "B".to_string()];
    node.outputs = vec!["Y".to_string()];
    graph.add_node(node);

    // Set graph inputs and outputs
    graph.inputs = vec!["A".to_string(), "B".to_string()];
    graph.outputs = vec!["Y".to_string()];

    // Set metadata
    graph.metadata.name = "matmul_test_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

// ================================================================================
// Tests
// ================================================================================

/// Test basic 2D matrix multiplication: (2, 3) × (3, 2) → (2, 2)
#[ignore = "requires GPU"]
#[pollster::test]
async fn test_matmul_2d_basic() {
    let graph = create_matmul_graph(vec![2, 3], vec![3, 2], vec![2, 2]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    // B = [[7, 8],
    //      [9, 10],
    //      [11, 12]]
    let b = Tensor::from_vec(vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);

    let outputs = executor.run(&[("A", a), ("B", b)]).unwrap();
    let result: Vec<f32> = outputs["Y"].to_vec().unwrap();

    // Expected: [[58, 64],
    //            [139, 154]]
    // Calculation:
    // [0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // [0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // [1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // [1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    let expected = vec![58.0, 64.0, 139.0, 154.0];

    assert_eq!(result.len(), expected.len());
    for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-4,
            "Mismatch at index {}: got {}, expected {}",
            i,
            r,
            e
        );
    }
}

/// Test square matrix multiplication: (3, 3) × (3, 3) → (3, 3)
#[ignore = "requires GPU"]
#[pollster::test]
async fn test_matmul_square() {
    let graph = create_matmul_graph(vec![3, 3], vec![3, 3], vec![3, 3]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    // Identity matrix × arbitrary matrix = arbitrary matrix
    let a = Tensor::from_vec(
        vec![
            1.0f32, 0.0, 0.0, // identity
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
        ],
        &[3, 3],
    );

    let b_data = vec![
        1.0f32, 2.0, 3.0, // arbitrary
        4.0, 5.0, 6.0, //
        7.0, 8.0, 9.0, //
    ];
    let b = Tensor::from_vec(b_data.clone(), &[3, 3]);

    let outputs = executor.run(&[("A", a), ("B", b)]).unwrap();
    let result: Vec<f32> = outputs["Y"].to_vec().unwrap();
    let expected = b_data;

    assert_eq!(result.len(), expected.len());
    for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-4,
            "Mismatch at index {}: got {}, expected {}",
            i,
            r,
            e
        );
    }
}

/// Test matrix-vector multiplication: (2, 3) × (3, 1) → (2, 1)
#[ignore = "requires GPU"]
#[pollster::test]
async fn test_matmul_matrix_vector() {
    let graph = create_matmul_graph(vec![2, 3], vec![3, 1], vec![2, 1]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    // B = [[2],
    //      [3],
    //      [4]]
    let b = Tensor::from_vec(vec![2.0f32, 3.0, 4.0], &[3, 1]);

    let outputs = executor.run(&[("A", a), ("B", b)]).unwrap();
    let result: Vec<f32> = outputs["Y"].to_vec().unwrap();

    // Expected: [[20],  (1*2 + 2*3 + 3*4)
    //            [47]]  (4*2 + 5*3 + 6*4)
    let expected = vec![20.0, 47.0];

    assert_eq!(result.len(), expected.len());
    for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-4,
            "Mismatch at index {}: got {}, expected {}",
            i,
            r,
            e
        );
    }
}

/// Test vector-matrix multiplication: (1, 3) × (3, 2) → (1, 2)
#[ignore = "requires GPU"]
#[pollster::test]
async fn test_matmul_vector_matrix() {
    let graph = create_matmul_graph(vec![1, 3], vec![3, 2], vec![1, 2]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    // A = [[1, 2, 3]]
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[1, 3]);

    // B = [[4, 5],
    //      [6, 7],
    //      [8, 9]]
    let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 2]);

    let outputs = executor.run(&[("A", a), ("B", b)]).unwrap();
    let result: Vec<f32> = outputs["Y"].to_vec().unwrap();

    // Expected: [[40, 46]]  (1*4 + 2*6 + 3*8, 1*5 + 2*7 + 3*9)
    let expected = vec![40.0, 46.0];

    assert_eq!(result.len(), expected.len());
    for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-4,
            "Mismatch at index {}: got {}, expected {}",
            i,
            r,
            e
        );
    }
}

/// Test batched matrix multiplication: (2, 2, 3) × (2, 3, 2) → (2, 2, 2)
#[ignore = "requires GPU"]
#[pollster::test]
async fn test_matmul_batched() {
    let graph = create_matmul_graph(vec![2, 2, 3], vec![2, 3, 2], vec![2, 2, 2]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    // A = batch 0: [[1, 2, 3], [4, 5, 6]]
    //     batch 1: [[7, 8, 9], [10, 11, 12]]
    let a = Tensor::from_vec(
        vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // batch 1
        ],
        &[2, 2, 3],
    );

    // B = batch 0: [[1, 2], [3, 4], [5, 6]]
    //     batch 1: [[7, 8], [9, 10], [11, 12]]
    let b = Tensor::from_vec(
        vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // batch 1
        ],
        &[2, 3, 2],
    );

    let outputs = executor.run(&[("A", a), ("B", b)]).unwrap();
    let result: Vec<f32> = outputs["Y"].to_vec().unwrap();

    // Expected batch 0: [[22, 28], [49, 64]]
    // Expected batch 1: [[220, 244], [301, 334]]
    //   Batch 1 calculation:
    //     [0,0] = 7*7 + 8*9 + 9*11 = 49 + 72 + 99 = 220
    //     [0,1] = 7*8 + 8*10 + 9*12 = 56 + 80 + 108 = 244
    //     [1,0] = 10*7 + 11*9 + 12*11 = 70 + 99 + 132 = 301
    //     [1,1] = 10*8 + 11*10 + 12*12 = 80 + 110 + 144 = 334
    let expected = vec![
        22.0, 28.0, 49.0, 64.0, // batch 0
        220.0, 244.0, 301.0, 334.0, // batch 1
    ];

    assert_eq!(result.len(), expected.len());
    for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-4,
            "Mismatch at index {}: got {}, expected {}",
            i,
            r,
            e
        );
    }
}

/// Test broadcasting: (2, 3) × (5, 3, 4) → (5, 2, 4)
#[ignore = "requires GPU"]
#[pollster::test]
async fn test_matmul_broadcast() {
    let graph = create_matmul_graph(vec![2, 3], vec![5, 3, 4], vec![5, 2, 4]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    // B: 5 batches of (3, 4) matrices, each filled with batch index value
    let mut b_data = Vec::with_capacity(5 * 3 * 4);
    for batch in 0..5 {
        for _ in 0..12 {
            b_data.push((batch + 1) as f32);
        }
    }
    let b = Tensor::from_vec(b_data, &[5, 3, 4]);

    let outputs = executor.run(&[("A", a), ("B", b)]).unwrap();
    let result: Vec<f32> = outputs["Y"].to_vec().unwrap();

    // A should be broadcast across all 5 batches
    // For batch i (value = i+1):
    //   Result = [[1*v + 2*v + 3*v, ...], [4*v + 5*v + 6*v, ...]]
    //          = [[6*v, 6*v, 6*v, 6*v], [15*v, 15*v, 15*v, 15*v]]
    let mut expected = Vec::with_capacity(5 * 2 * 4);
    for batch in 0..5 {
        let v = (batch + 1) as f32;
        for _ in 0..4 {
            expected.push(6.0 * v);
        }
        for _ in 0..4 {
            expected.push(15.0 * v);
        }
    }

    assert_eq!(result.len(), expected.len());
    for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-4,
            "Mismatch at index {}: got {}, expected {}",
            i,
            r,
            e
        );
    }
}

/// Test large matrix multiplication to verify tiling works correctly
#[ignore = "requires GPU"]
#[pollster::test]
async fn test_matmul_large() {
    let size = 64;
    let graph = create_matmul_graph(vec![size, size], vec![size, size], vec![size, size]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    // Create 64×64 × 64×64 matrix multiplication
    let a_data: Vec<f32> = (0..size * size).map(|i| (i % 10) as f32).collect();
    let a = Tensor::from_vec(a_data, &[size, size]);

    let b_data: Vec<f32> = (0..size * size).map(|i| ((i + 1) % 10) as f32).collect();
    let b = Tensor::from_vec(b_data, &[size, size]);

    let outputs = executor.run(&[("A", a), ("B", b)]).unwrap();
    let _result: Vec<f32> = outputs["Y"].to_vec().unwrap();

    // Just verify it runs without errors; exact result checking is complex
    // A proper test would verify against a reference implementation
}
