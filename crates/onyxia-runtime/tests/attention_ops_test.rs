//! End-to-end tests for attention operations.
//!
//! Tests: RotaryEmbedding

use onyxia_onnx::{AttributeValue, DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_planner::{KernelRegistry, compile};
use onyxia_runtime::{Runtime, Tensor};
use std::collections::HashMap;

/// End-to-end test: RotaryEmbedding on GPU with known cos/sin values.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_rotary_embedding_e2e() {
    // Build graph for RotaryEmbedding test
    // Input shape: [batch=1, seq_len=2, num_heads=1, head_dim=4]
    let mut graph = Graph::new();

    // Input tensor: [1, 2, 1, 4]
    graph.add_tensor(TensorInfo {
        name: "input".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![1, 2, 1, 4]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Position IDs: [1, 2] (I64)
    graph.add_tensor(TensorInfo {
        name: "position_ids".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1, 2]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Cos cache: [max_seq=4, head_dim/2=2]
    graph.add_tensor(TensorInfo {
        name: "cos_cache".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4, 2]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Sin cache: [max_seq=4, head_dim/2=2]
    graph.add_tensor(TensorInfo {
        name: "sin_cache".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4, 2]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Output tensor: [1, 2, 1, 4]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![1, 2, 1, 4]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create RotaryEmbedding node
    let mut node = Node::new("RotaryEmbedding");
    node.name = "rope_node".to_string();
    node.inputs = vec![
        "input".to_string(),
        "position_ids".to_string(),
        "cos_cache".to_string(),
        "sin_cache".to_string(),
    ];
    node.outputs = vec!["output".to_string()];
    node.attributes
        .insert("num_heads".to_string(), AttributeValue::Int(1));
    node.attributes
        .insert("interleaved".to_string(), AttributeValue::Int(0)); // split-half

    graph.add_node(node);
    graph.inputs = vec![
        "input".to_string(),
        "position_ids".to_string(),
        "cos_cache".to_string(),
        "sin_cache".to_string(),
    ];
    graph.outputs = vec!["output".to_string()];

    graph.metadata.name = "test_rotary_embedding".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph.validate().expect("Graph validation should succeed");

    // Compile and execute
    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Test data:
    // Input: [1, 2, 1, 4] -> two sequence positions, each with 4 values
    // Split-half layout: [x0, x1, y0, y1] where (x0,y0) and (x1,y1) are pairs
    let input_data = vec![
        1.0f32, 2.0, 3.0, 4.0, // seq=0: [x0=1, x1=2 | y0=3, y1=4]
        5.0, 6.0, 7.0, 8.0, // seq=1: [x0=5, x1=6 | y0=7, y1=8]
    ];

    // Position IDs: [0, 1] (as I64, stored as pairs of u32 in little-endian)
    let position_ids_data: Vec<i64> = vec![0, 1];

    // Cos cache: [4, 2] - we'll use simple values for positions 0 and 1
    // For position 0: cos = [1.0, 1.0] (no rotation)
    // For position 1: cos = [0.0, 1.0]
    let cos_cache_data = vec![
        1.0f32, 1.0, // pos=0
        0.0, 1.0, // pos=1
        0.0, 0.0, // pos=2 (unused)
        0.0, 0.0, // pos=3 (unused)
    ];

    // Sin cache: [4, 2]
    // For position 0: sin = [0.0, 0.0] (no rotation)
    // For position 1: sin = [1.0, 0.0]
    let sin_cache_data = vec![
        0.0f32, 0.0, // pos=0
        1.0, 0.0, // pos=1
        0.0, 0.0, // pos=2 (unused)
        0.0, 0.0, // pos=3 (unused)
    ];

    let input = Tensor::from_vec(input_data, &[1, 2, 1, 4]);
    let position_ids = Tensor::from_vec(position_ids_data, &[1, 2]);
    let cos_cache = Tensor::from_vec(cos_cache_data, &[4, 2]);
    let sin_cache = Tensor::from_vec(sin_cache_data, &[4, 2]);

    let outputs = executor
        .run(&[
            ("input", input),
            ("position_ids", position_ids),
            ("cos_cache", cos_cache),
            ("sin_cache", sin_cache),
        ])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");
    assert_eq!(output.len(), 8);

    // Expected output calculation:
    // Seq 0, pos=0: cos=[1,1], sin=[0,0]
    //   pair 0: x0'=1*1-3*0=1, y0'=3*1+1*0=3
    //   pair 1: x1'=2*1-4*0=2, y1'=4*1+2*0=4
    //   Result: [1, 2, 3, 4]
    //
    // Seq 1, pos=1: cos=[0,1], sin=[1,0]
    //   pair 0: x0'=5*0-7*1=-7, y0'=7*0+5*1=5
    //   pair 1: x1'=6*1-8*0=6, y1'=8*1+6*0=8
    //   Result: [-7, 6, 5, 8]

    let expected = vec![
        1.0f32, 2.0, 3.0, 4.0, // seq=0 (no rotation)
        -7.0, 6.0, 5.0, 8.0, // seq=1 (rotated)
    ];

    for (i, (out, exp)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (out - exp).abs() < 1e-5,
            "Output mismatch at index {}: expected {}, got {}",
            i,
            exp,
            out
        );
    }

    println!("✓ End-to-end RotaryEmbedding test passed!");
    println!("  Input shape: [1, 2, 1, 4]");
    println!("  Position IDs: [0, 1]");
    println!("  Output: {:?}", output);
}
