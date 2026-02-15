//! End-to-end tests for matrix multiplication operations.
//!
//! Tests: MatMul, MatMulNBits (Q4 quantization)

use onyxia_compiler::{OperatorRegistry, compile};
use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_runtime::{Runtime, Tensor};
use std::collections::HashMap;

/// End-to-end test: Matrix multiplication on GPU and verify correct output.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_matmul_e2e() {
    let mut graph = Graph::new();

    // Matrix A: [2, 3]
    graph.add_tensor(TensorInfo {
        name: "A".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Matrix B: [3, 2]
    graph.add_tensor(TensorInfo {
        name: "B".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![3, 2]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Matrix C: [2, 2]
    graph.add_tensor(TensorInfo {
        name: "C".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 2]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create MatMul node
    let mut node = Node::new("MatMul");
    node.name = "matmul_node".to_string();
    node.inputs = vec!["A".to_string(), "B".to_string()];
    node.outputs = vec!["C".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["A".to_string(), "B".to_string()];
    graph.outputs = vec!["C".to_string()];

    // Compile and execute
    let registry = OperatorRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Test with simple matrices
    // A = [[1, 2, 3],     B = [[1, 2],
    //      [4, 5, 6]]          [3, 4],
    //                          [5, 6]]
    //
    // C = A × B = [[22, 28],
    //              [49, 64]]
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

    let outputs = executor
        .run(&[("A", a), ("B", b)])
        .expect("Execution should succeed");

    let c = outputs["C"].to_vec::<f32>().expect("Should convert to f32");
    assert_eq!(c, vec![22.0, 28.0, 49.0, 64.0], "MatMul result incorrect");

    println!("✓ End-to-end MatMul test passed!");
    println!("  A: [[1, 2, 3], [4, 5, 6]]");
    println!("  B: [[1, 2], [3, 4], [5, 6]]");
    println!("  C: {:?}", c);
}

/// Helper function to create a MatMulNBits graph with Q4 quantization.
fn make_matmul_nbits_graph() -> Graph {
    let mut graph = Graph::new();

    // Add input tensor 'A' [2, 8]
    graph.add_tensor(TensorInfo {
        name: "A".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 8]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add packed weights 'B' [N=2, n_blocks=1, blob_size=4]
    graph.add_tensor(TensorInfo {
        name: "B".to_string(),
        dtype: DataType::U8,
        shape: TensorShape::Static(vec![2, 1, 4]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add scales [N=2, n_blocks=1]
    graph.add_tensor(TensorInfo {
        name: "scales".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 1]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add output tensor 'C' [2, 2]
    graph.add_tensor(TensorInfo {
        name: "C".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 2]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create MatMulNBits node
    let mut node = Node::new("MatMulNBits");
    node.name = "matmul_nbits_node".to_string();
    node.inputs = vec!["A".to_string(), "B".to_string(), "scales".to_string()];
    node.outputs = vec!["C".to_string()];

    // Add attributes
    node.attributes
        .insert("K".to_string(), onyxia_onnx::AttributeValue::Int(8));
    node.attributes
        .insert("N".to_string(), onyxia_onnx::AttributeValue::Int(2));
    node.attributes
        .insert("bits".to_string(), onyxia_onnx::AttributeValue::Int(4));
    node.attributes.insert(
        "block_size".to_string(),
        onyxia_onnx::AttributeValue::Int(8),
    );

    graph.add_node(node);

    graph.inputs = vec!["A".to_string(), "B".to_string(), "scales".to_string()];
    graph.outputs = vec!["C".to_string()];

    graph.metadata.name = "test_matmul_nbits_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

/// Pack Q4 values into u32 format (8 values per u32).
fn pack_q4_values(values: &[u8]) -> Vec<u32> {
    assert!(
        values.iter().all(|&v| v < 16),
        "All values must be 4-bit (0-15)"
    );

    let mut packed = Vec::new();
    for chunk in values.chunks(8) {
        let mut val: u32 = 0;
        for (i, &v) in chunk.iter().enumerate() {
            val |= (v as u32) << (i * 4);
        }
        packed.push(val);
    }
    packed
}

/// End-to-end test: MatMulNBits with Q4 quantization on GPU.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_matmul_q4_e2e() {
    let graph = make_matmul_nbits_graph();
    graph.validate().expect("Graph validation should succeed");

    // Compile to ExecutionPlan
    let registry = OperatorRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    // Initialize runtime
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    println!(
        "GPU: {} ({:?})",
        runtime.adapter_info().name,
        runtime.adapter_info().backend
    );

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Prepare test data
    let a_data = vec![
        1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    ];
    let a = Tensor::from_vec(a_data, &[2, 8]);

    // Pack Q4 weights
    let col0_q4 = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let col1_q4 = vec![2u8, 3, 4, 5, 6, 7, 8, 9];

    let col0_packed = pack_q4_values(&col0_q4);
    let col1_packed = pack_q4_values(&col1_q4);

    let mut b_data: Vec<u8> = Vec::new();
    b_data.extend_from_slice(&col0_packed[0].to_le_bytes());
    b_data.extend_from_slice(&col1_packed[0].to_le_bytes());

    let b = Tensor::from_vec(b_data, &[2, 1, 4]);

    let scales_data = vec![1.0f32, 1.0];
    let scales = Tensor::from_vec(scales_data, &[2, 1]);

    // Run inference
    let outputs = executor
        .run(&[("A", a), ("B", b), ("scales", scales)])
        .expect("Execution should succeed");

    let c = outputs["C"].to_vec::<f32>().expect("Should convert to f32");
    assert_eq!(c.len(), 4);

    // Expected with zero_point=8: [-28, -20, -56, -40]
    let expected = vec![-28.0f32, -20.0, -56.0, -40.0];

    for (i, (&actual, &expected)) in c.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 0.1,
            "Mismatch at index {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }

    println!("✓ End-to-end MatMulNBits Q4 test passed!");
    println!("  Output C: {:?}", c);
}
