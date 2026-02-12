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

/// End-to-end test: GroupQueryAttention with no cache (past_seq_len=0).
#[pollster::test]
#[ignore] // Requires GPU
async fn test_gqa_e2e_no_cache() {
    // Build graph for GQA with no cache
    // Query: [batch=1, seq_len=2, num_heads=2, head_dim=4] → [1, 2, 8]
    let mut graph = Graph::new();

    let batch = 1;
    let seq_len = 2;
    let past_seq_len = 0; // No cache
    let num_heads = 2;
    let kv_num_heads = 1;
    let head_dim = 4;

    // Input 0: query [1, 2, 8]
    graph.add_tensor(TensorInfo {
        name: "query".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, seq_len, num_heads * head_dim]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Input 1: key [1, 2, 4]
    graph.add_tensor(TensorInfo {
        name: "key".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, seq_len, kv_num_heads * head_dim]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Input 2: value [1, 2, 4]
    graph.add_tensor(TensorInfo {
        name: "value".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, seq_len, kv_num_heads * head_dim]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Input 3: past_key [1, 1, 0, 4] (empty)
    graph.add_tensor(TensorInfo {
        name: "past_key".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, kv_num_heads, past_seq_len, head_dim]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Input 4: past_value [1, 1, 0, 4] (empty)
    graph.add_tensor(TensorInfo {
        name: "past_value".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, kv_num_heads, past_seq_len, head_dim]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Input 5: seqlens_k [1]
    graph.add_tensor(TensorInfo {
        name: "seqlens_k".to_string(),
        dtype: DataType::I32,
        shape: TensorShape::Static(vec![batch]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Input 6: total_sequence_length (scalar)
    graph.add_tensor(TensorInfo {
        name: "total_sequence_length".to_string(),
        dtype: DataType::I32,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Output 0: output [1, 2, 8]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, seq_len, num_heads * head_dim]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Output 1: present_key [1, 1, 2, 4]
    graph.add_tensor(TensorInfo {
        name: "present_key".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, kv_num_heads, seq_len, head_dim]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Output 2: present_value [1, 1, 2, 4]
    graph.add_tensor(TensorInfo {
        name: "present_value".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, kv_num_heads, seq_len, head_dim]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create GroupQueryAttention node
    let mut node = Node::new("GroupQueryAttention");
    node.name = "gqa_node".to_string();
    node.inputs = vec![
        "query".to_string(),
        "key".to_string(),
        "value".to_string(),
        "past_key".to_string(),
        "past_value".to_string(),
        "seqlens_k".to_string(),
        "total_sequence_length".to_string(),
    ];
    node.outputs = vec![
        "output".to_string(),
        "present_key".to_string(),
        "present_value".to_string(),
    ];
    node.attributes.insert(
        "num_heads".to_string(),
        AttributeValue::Int(num_heads as i64),
    );
    node.attributes.insert(
        "kv_num_heads".to_string(),
        AttributeValue::Int(kv_num_heads as i64),
    );

    graph.add_node(node);
    graph.inputs = vec![
        "query".to_string(),
        "key".to_string(),
        "value".to_string(),
        "past_key".to_string(),
        "past_value".to_string(),
        "seqlens_k".to_string(),
        "total_sequence_length".to_string(),
    ];
    graph.outputs = vec![
        "output".to_string(),
        "present_key".to_string(),
        "present_value".to_string(),
    ];

    graph.metadata.name = "test_gqa_no_cache".to_string();
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

    // Test data: Simple values for verification
    // Query: [1, 2, 8] - 2 positions, 2 heads, 4 dims each
    let query_data = vec![
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // pos 0: head0=[1,0,0,0], head1=[0,1,0,0]
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, // pos 1: head0=[0,0,1,0], head1=[0,0,0,1]
    ];

    // Key: [1, 2, 4] - 2 positions, 1 kv_head, 4 dims
    let key_data = vec![
        1.0f32, 0.0, 0.0, 0.0, // pos 0: [1,0,0,0]
        0.0, 1.0, 0.0, 0.0, // pos 1: [0,1,0,0]
    ];

    // Value: [1, 2, 4] - same structure
    let value_data = vec![
        2.0f32, 0.0, 0.0, 0.0, // pos 0: [2,0,0,0]
        0.0, 3.0, 0.0, 0.0, // pos 1: [0,3,0,0]
    ];

    // Empty past cache (0 elements)
    let past_key_data: Vec<f32> = vec![];
    let past_value_data: Vec<f32> = vec![];

    // Sequence lengths and total
    let seqlens_k_data = vec![2i32];
    let total_seq_data = vec![2i32];

    let query = Tensor::from_vec(query_data.clone(), &[1, 2, 8]);
    let key = Tensor::from_vec(key_data.clone(), &[1, 2, 4]);
    let value = Tensor::from_vec(value_data.clone(), &[1, 2, 4]);
    let past_key = Tensor::from_vec(past_key_data, &[1, 1, 0, 4]);
    let past_value = Tensor::from_vec(past_value_data, &[1, 1, 0, 4]);
    let seqlens_k = Tensor::from_vec(seqlens_k_data, &[1]);
    let total_seq = Tensor::from_vec(total_seq_data, &[1]);

    let outputs = executor
        .run(&[
            ("query", query),
            ("key", key),
            ("value", value),
            ("past_key", past_key),
            ("past_value", past_value),
            ("seqlens_k", seqlens_k),
            ("total_sequence_length", total_seq),
        ])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");
    let present_key = outputs["present_key"]
        .to_vec::<f32>()
        .expect("Should convert to f32");
    let present_value = outputs["present_value"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Verify output shape
    assert_eq!(output.len(), 16); // 1*2*8
    assert_eq!(present_key.len(), 8); // 1*1*2*4
    assert_eq!(present_value.len(), 8); // 1*1*2*4

    // Verify present_key contains the key data (no past cache)
    for (i, (&pk, &k)) in present_key.iter().zip(key_data.iter()).enumerate() {
        assert!(
            (pk - k).abs() < 1e-5,
            "present_key mismatch at index {}: expected {}, got {}",
            i,
            k,
            pk
        );
    }

    println!("✓ End-to-end GroupQueryAttention (no cache) test passed!");
    println!("  Query shape: [1, 2, 8]");
    println!("  Output shape: [1, 2, 8]");
    println!("  Present key shape: [1, 1, 2, 4]");
}

/// End-to-end test: GroupQueryAttention with KV cache.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_gqa_e2e_with_cache() {
    // Build graph for GQA with cache
    let mut graph = Graph::new();

    let batch = 1;
    let seq_len = 1; // Process 1 new token
    let past_seq_len = 2; // 2 tokens in cache
    let num_heads = 2;
    let kv_num_heads = 1;
    let head_dim = 4;
    let total_seq_len = past_seq_len + seq_len;

    // Input 0: query [1, 1, 8]
    graph.add_tensor(TensorInfo {
        name: "query".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, seq_len, num_heads * head_dim]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Input 1: key [1, 1, 4]
    graph.add_tensor(TensorInfo {
        name: "key".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, seq_len, kv_num_heads * head_dim]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Input 2: value [1, 1, 4]
    graph.add_tensor(TensorInfo {
        name: "value".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, seq_len, kv_num_heads * head_dim]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Input 3: past_key [1, 1, 2, 4]
    graph.add_tensor(TensorInfo {
        name: "past_key".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, kv_num_heads, past_seq_len, head_dim]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Input 4: past_value [1, 1, 2, 4]
    graph.add_tensor(TensorInfo {
        name: "past_value".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, kv_num_heads, past_seq_len, head_dim]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Input 5: seqlens_k [1]
    graph.add_tensor(TensorInfo {
        name: "seqlens_k".to_string(),
        dtype: DataType::I32,
        shape: TensorShape::Static(vec![batch]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Input 6: total_sequence_length (scalar)
    graph.add_tensor(TensorInfo {
        name: "total_sequence_length".to_string(),
        dtype: DataType::I32,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Output 0: output [1, 1, 8]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, seq_len, num_heads * head_dim]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Output 1: present_key [1, 1, 3, 4]
    graph.add_tensor(TensorInfo {
        name: "present_key".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, kv_num_heads, total_seq_len, head_dim]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Output 2: present_value [1, 1, 3, 4]
    graph.add_tensor(TensorInfo {
        name: "present_value".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, kv_num_heads, total_seq_len, head_dim]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create GroupQueryAttention node
    let mut node = Node::new("GroupQueryAttention");
    node.name = "gqa_node".to_string();
    node.inputs = vec![
        "query".to_string(),
        "key".to_string(),
        "value".to_string(),
        "past_key".to_string(),
        "past_value".to_string(),
        "seqlens_k".to_string(),
        "total_sequence_length".to_string(),
    ];
    node.outputs = vec![
        "output".to_string(),
        "present_key".to_string(),
        "present_value".to_string(),
    ];
    node.attributes.insert(
        "num_heads".to_string(),
        AttributeValue::Int(num_heads as i64),
    );
    node.attributes.insert(
        "kv_num_heads".to_string(),
        AttributeValue::Int(kv_num_heads as i64),
    );

    graph.add_node(node);
    graph.inputs = vec![
        "query".to_string(),
        "key".to_string(),
        "value".to_string(),
        "past_key".to_string(),
        "past_value".to_string(),
        "seqlens_k".to_string(),
        "total_sequence_length".to_string(),
    ];
    graph.outputs = vec![
        "output".to_string(),
        "present_key".to_string(),
        "present_value".to_string(),
    ];

    graph.metadata.name = "test_gqa_with_cache".to_string();
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

    // Test data
    // Query: [1, 1, 8] - 1 new position
    let query_data = vec![
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // pos 2: head0=[1,0,0,0], head1=[0,1,0,0]
    ];

    // Key: [1, 1, 4] - 1 new position
    let key_data = vec![
        0.0f32, 0.0, 1.0, 0.0, // pos 2: [0,0,1,0]
    ];

    // Value: [1, 1, 4]
    let value_data = vec![
        0.0f32, 0.0, 4.0, 0.0, // pos 2: [0,0,4,0]
    ];

    // Past cache: [1, 1, 2, 4]
    let past_key_data = vec![
        1.0f32, 0.0, 0.0, 0.0, // pos 0: [1,0,0,0]
        0.0, 1.0, 0.0, 0.0, // pos 1: [0,1,0,0]
    ];

    let past_value_data = vec![
        2.0f32, 0.0, 0.0, 0.0, // pos 0: [2,0,0,0]
        0.0, 3.0, 0.0, 0.0, // pos 1: [0,3,0,0]
    ];

    // Sequence lengths
    let seqlens_k_data = vec![3i32]; // Total 3 tokens
    let total_seq_data = vec![3i32];

    let query = Tensor::from_vec(query_data.clone(), &[1, 1, 8]);
    let key = Tensor::from_vec(key_data.clone(), &[1, 1, 4]);
    let value = Tensor::from_vec(value_data.clone(), &[1, 1, 4]);
    let past_key = Tensor::from_vec(past_key_data.clone(), &[1, 1, 2, 4]);
    let past_value = Tensor::from_vec(past_value_data.clone(), &[1, 1, 2, 4]);
    let seqlens_k = Tensor::from_vec(seqlens_k_data, &[1]);
    let total_seq = Tensor::from_vec(total_seq_data, &[1]);

    let outputs = executor
        .run(&[
            ("query", query),
            ("key", key),
            ("value", value),
            ("past_key", past_key),
            ("past_value", past_value),
            ("seqlens_k", seqlens_k),
            ("total_sequence_length", total_seq),
        ])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");
    let present_key = outputs["present_key"]
        .to_vec::<f32>()
        .expect("Should convert to f32");
    let present_value = outputs["present_value"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Verify output shape
    assert_eq!(output.len(), 8); // 1*1*8
    assert_eq!(present_key.len(), 12); // 1*1*3*4
    assert_eq!(present_value.len(), 12);

    // Verify present_key contains past_key + new key
    let expected_present_key: Vec<f32> = past_key_data
        .iter()
        .chain(key_data.iter())
        .copied()
        .collect();
    for (i, (&pk, &exp)) in present_key
        .iter()
        .zip(expected_present_key.iter())
        .enumerate()
    {
        assert!(
            (pk - exp).abs() < 1e-5,
            "present_key mismatch at index {}: expected {}, got {}",
            i,
            exp,
            pk
        );
    }

    // Verify present_value contains past_value + new value
    let expected_present_value: Vec<f32> = past_value_data
        .iter()
        .chain(value_data.iter())
        .copied()
        .collect();
    for (i, (&pv, &exp)) in present_value
        .iter()
        .zip(expected_present_value.iter())
        .enumerate()
    {
        assert!(
            (pv - exp).abs() < 1e-5,
            "present_value mismatch at index {}: expected {}, got {}",
            i,
            exp,
            pv
        );
    }

    println!("✓ End-to-end GroupQueryAttention (with cache) test passed!");
    println!("  Query shape: [1, 1, 8]");
    println!("  Past cache: 2 tokens");
    println!("  Output shape: [1, 1, 8]");
    println!("  Present key shape: [1, 1, 3, 4]");
    println!("  Cache concatenation verified!");
}
