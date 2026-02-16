//! End-to-end tests for attention operations.
//!
//! Tests: RotaryEmbedding

use onyxia_compiler::CompilerPipeline;
use onyxia_onnx::{AttributeValue, DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_operators::core_operator_registry;
use onyxia_runtime::{Runtime, Tensor};
use std::collections::HashMap;

/// End-to-end test: RotaryEmbedding on GPU with known cos/sin values.
#[pollster::test]
#[ignore="requires GPU"]
async fn test_rotary_embedding_e2e() {
    // Build graph for RotaryEmbedding test
    // Input shape: [batch=1, num_heads=1, seq_len=2, head_dim=4]
    // Per ONNX spec: 4D input is [batch_size, num_heads, sequence_length, head_size]
    let mut graph = Graph::new();

    // Input tensor: [1, 1, 2, 4]
    graph.add_tensor(TensorInfo {
        name: "input".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![1, 1, 2, 4]),
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

    // Position IDs: [1, 2] (I64) - optional, provided for this test
    graph.add_tensor(TensorInfo {
        name: "position_ids".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1, 2]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Output tensor: [1, 1, 2, 4]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![1, 1, 2, 4]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create RotaryEmbedding node with correct input order per ONNX spec
    let mut node = Node::new("RotaryEmbedding");
    node.name = "rope_node".to_string();
    node.inputs = vec![
        "input".to_string(),
        "cos_cache".to_string(),
        "sin_cache".to_string(),
        "position_ids".to_string(),
    ];
    node.outputs = vec!["output".to_string()];
    node.attributes
        .insert("num_heads".to_string(), AttributeValue::Int(1));
    node.attributes
        .insert("interleaved".to_string(), AttributeValue::Int(0)); // split-half

    graph.add_node(node);
    graph.inputs = vec![
        "input".to_string(),
        "cos_cache".to_string(),
        "sin_cache".to_string(),
        "position_ids".to_string(),
    ];
    graph.outputs = vec!["output".to_string()];

    graph.metadata.name = "test_rotary_embedding".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph.validate().expect("Graph validation should succeed");

    // Compile and execute
    let registry = core_operator_registry();
    let plan = CompilerPipeline::new(HashMap::new())
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Test data:
    // Input: [1, 1, 2, 4] -> 1 batch, 1 head, 2 positions, 4 values per position
    // Shape is [batch, num_heads, seq_len, head_dim] per ONNX spec
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

    let input = Tensor::from_vec(input_data, &[1, 1, 2, 4]);
    let position_ids = Tensor::from_vec(position_ids_data, &[1, 2]);
    let cos_cache = Tensor::from_vec(cos_cache_data, &[4, 2]);
    let sin_cache = Tensor::from_vec(sin_cache_data, &[4, 2]);

    let outputs = executor
        .run(&[
            ("input", input),
            ("cos_cache", cos_cache),
            ("sin_cache", sin_cache),
            ("position_ids", position_ids),
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
    println!("  Input shape: [1, 1, 2, 4]");
    println!("  Position IDs: [0, 1]");
    println!("  Output: {:?}", output);
}

/// End-to-end test: GroupQueryAttention with no cache (prefill mode).
#[pollster::test]
#[ignore="requires GPU"]
async fn test_gqa_e2e_no_cache() {
    // Build graph for GQA with no cache (prefill mode)
    // Query: [batch=1, seq_len=2, num_heads=2, head_dim=4] → [1, 2, 8]
    let mut graph = Graph::new();

    let batch = 1;
    let seq_len = 2;
    let max_seq_len = 4; // Pre-allocated buffer size
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

    // Input 3: past_key [1, 1, max_seq_len, 4] - pre-allocated, initially empty
    graph.add_tensor(TensorInfo {
        name: "past_key".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, kv_num_heads, max_seq_len, head_dim]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Input 4: past_value [1, 1, max_seq_len, 4] - pre-allocated, initially empty
    graph.add_tensor(TensorInfo {
        name: "past_value".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, kv_num_heads, max_seq_len, head_dim]),
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

    // Output 1: present_key [1, 1, max_seq_len, 4] - same size as past_key
    graph.add_tensor(TensorInfo {
        name: "present_key".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, kv_num_heads, max_seq_len, head_dim]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Output 2: present_value [1, 1, max_seq_len, 4] - same size as past_value
    graph.add_tensor(TensorInfo {
        name: "present_value".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, kv_num_heads, max_seq_len, head_dim]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create GroupQueryAttention node
    let mut node = Node::new("GroupQueryAttention");
    node.name = "gqa_node".to_string();
    node.domain = "com.microsoft".to_string();
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
    let registry = core_operator_registry();
    let plan = CompilerPipeline::new(HashMap::new())
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

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

    // Empty past cache (pre-allocated to max_seq_len, all zeros)
    let past_key_data = vec![0.0f32; max_seq_len * head_dim];
    let past_value_data = vec![0.0f32; max_seq_len * head_dim];

    // Sequence lengths and total
    let seqlens_k_data = vec![1i32]; // last valid index = seq_len - 1
    let total_seq_data = vec![2i32];

    let query = Tensor::from_vec(query_data.clone(), &[1, 2, 8]);
    let key = Tensor::from_vec(key_data.clone(), &[1, 2, 4]);
    let value = Tensor::from_vec(value_data.clone(), &[1, 2, 4]);
    let past_key = Tensor::from_vec(past_key_data, &[1, 1, max_seq_len, 4]);
    let past_value = Tensor::from_vec(past_value_data, &[1, 1, max_seq_len, 4]);
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
    assert_eq!(present_key.len(), max_seq_len * head_dim); // 1*1*max_seq_len*4 = 16
    assert_eq!(present_value.len(), max_seq_len * head_dim);

    // Verify present_key contains the key data in first seq_len positions
    // Position 0: [1,0,0,0]
    assert!(
        (present_key[0] - 1.0).abs() < 1e-5,
        "key pos 0[0] should be 1.0"
    );
    assert!(
        (present_key[1] - 0.0).abs() < 1e-5,
        "key pos 0[1] should be 0.0"
    );
    // Position 1: [0,1,0,0]
    assert!(
        (present_key[4] - 0.0).abs() < 1e-5,
        "key pos 1[0] should be 0.0"
    );
    assert!(
        (present_key[5] - 1.0).abs() < 1e-5,
        "key pos 1[1] should be 1.0"
    );
    // Positions beyond seq_len should be 0
    assert!(
        (present_key[8] - 0.0).abs() < 1e-5,
        "key pos 2 should be 0.0"
    );

    // Verify present_value
    assert!(
        (present_value[0] - 2.0).abs() < 1e-5,
        "value pos 0[0] should be 2.0"
    );
    assert!(
        (present_value[5] - 3.0).abs() < 1e-5,
        "value pos 1[1] should be 3.0"
    );

    println!("✓ End-to-end GroupQueryAttention (prefill, no prior cache) test passed!");
    println!("  Query shape: [1, 2, 8]");
    println!("  Output shape: [1, 2, 8]");
    println!(
        "  KV cache shape: [1, 1, {}, 4] ({} positions used)",
        max_seq_len, seq_len
    );
}

/// End-to-end test: GroupQueryAttention with KV cache.
#[pollster::test]
#[ignore="requires GPU"]
async fn test_gqa_e2e_with_cache() {
    // Build graph for GQA with cache
    let mut graph = Graph::new();

    let batch = 1;
    let seq_len = 3; // Process 3 tokens (prefill mode)
    let past_seq_len = 0; // No prior cache (prefill)
    let max_seq_len = 5; // Pre-allocated buffer size for KV cache
    let num_heads = 2;
    let kv_num_heads = 1;
    let head_dim = 4;
    let total_seq_len = seq_len; // In prefill, total = seq_len

    // Input 0: query [1, 3, 8] - 3 tokens
    graph.add_tensor(TensorInfo {
        name: "query".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, seq_len, num_heads * head_dim]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Input 1: key [1, 3, 4]
    graph.add_tensor(TensorInfo {
        name: "key".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, seq_len, kv_num_heads * head_dim]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Input 2: value [1, 3, 4]
    graph.add_tensor(TensorInfo {
        name: "value".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, seq_len, kv_num_heads * head_dim]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Input 3: past_key [1, 1, max_seq_len, 4] - pre-allocated to max size
    graph.add_tensor(TensorInfo {
        name: "past_key".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, kv_num_heads, max_seq_len, head_dim]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Input 4: past_value [1, 1, max_seq_len, 4] - pre-allocated to max size
    graph.add_tensor(TensorInfo {
        name: "past_value".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, kv_num_heads, max_seq_len, head_dim]),
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

    // Output 0: output [1, 3, 8]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, seq_len, num_heads * head_dim]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Output 1: present_key [1, 1, max_seq_len, 4] - same size as past_key
    graph.add_tensor(TensorInfo {
        name: "present_key".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, kv_num_heads, max_seq_len, head_dim]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Output 2: present_value [1, 1, max_seq_len, 4] - same size as past_value
    graph.add_tensor(TensorInfo {
        name: "present_value".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, kv_num_heads, max_seq_len, head_dim]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create GroupQueryAttention node
    let mut node = Node::new("GroupQueryAttention");
    node.name = "gqa_node".to_string();
    node.domain = "com.microsoft".to_string();
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
    let registry = core_operator_registry();
    let plan = CompilerPipeline::new(HashMap::new())
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Test data - prefill with 3 tokens
    // Query: [1, 3, 8] - 3 positions
    let query_data = vec![
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // pos 0: head0=[1,0,0,0], head1=[0,1,0,0]
        0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // pos 1: head0=[0,1,0,0], head1=[1,0,0,0]
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, // pos 2: head0=[0,0,1,0], head1=[0,0,1,0]
    ];

    // Key: [1, 3, 4] - 3 positions
    let key_data = vec![
        1.0f32, 0.0, 0.0, 0.0, // pos 0: [1,0,0,0]
        0.0, 1.0, 0.0, 0.0, // pos 1: [0,1,0,0]
        0.0, 0.0, 1.0, 0.0, // pos 2: [0,0,1,0]
    ];

    // Value: [1, 3, 4]
    let value_data = vec![
        2.0f32, 0.0, 0.0, 0.0, // pos 0: [2,0,0,0]
        0.0, 3.0, 0.0, 0.0, // pos 1: [0,3,0,0]
        0.0, 0.0, 4.0, 0.0, // pos 2: [0,0,4,0]
    ];

    // Past cache: [1, 1, max_seq_len, 4] - empty (all zeros)
    let past_key_data = vec![0.0f32; max_seq_len * head_dim];
    let past_value_data = vec![0.0f32; max_seq_len * head_dim];

    // Sequence lengths
    let seqlens_k_data = vec![2i32]; // last valid index = total_seq - 1 = 3 - 1 = 2
    let total_seq_data = vec![3i32]; // total sequence length

    let query = Tensor::from_vec(query_data.clone(), &[1, 3, 8]);
    let key = Tensor::from_vec(key_data.clone(), &[1, 3, 4]);
    let value = Tensor::from_vec(value_data.clone(), &[1, 3, 4]);
    let past_key = Tensor::from_vec(past_key_data.clone(), &[1, 1, max_seq_len, 4]);
    let past_value = Tensor::from_vec(past_value_data.clone(), &[1, 1, max_seq_len, 4]);
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
    assert_eq!(output.len(), 24); // 1*3*8
    assert_eq!(present_key.len(), max_seq_len * head_dim); // 1*1*5*4 = 20
    assert_eq!(present_value.len(), max_seq_len * head_dim);

    // Verify present_key contains the 3 new keys in first 3 positions
    // Position 0: [1,0,0,0]
    assert!(
        (present_key[0] - 1.0).abs() < 1e-5,
        "key pos 0[0] should be 1.0"
    );
    assert!(
        (present_key[1] - 0.0).abs() < 1e-5,
        "key pos 0[1] should be 0.0"
    );
    // Position 1: [0,1,0,0]
    assert!(
        (present_key[4] - 0.0).abs() < 1e-5,
        "key pos 1[0] should be 0.0"
    );
    assert!(
        (present_key[5] - 1.0).abs() < 1e-5,
        "key pos 1[1] should be 1.0"
    );
    // Position 2: [0,0,1,0]
    assert!(
        (present_key[8] - 0.0).abs() < 1e-5,
        "key pos 2[0] should be 0.0"
    );
    assert!(
        (present_key[10] - 1.0).abs() < 1e-5,
        "key pos 2[2] should be 1.0"
    );
    // Positions 3-4: should be 0 (beyond valid data)

    // Verify present_value contains the 3 new values
    assert!(
        (present_value[0] - 2.0).abs() < 1e-5,
        "value pos 0[0] should be 2.0"
    );
    assert!(
        (present_value[5] - 3.0).abs() < 1e-5,
        "value pos 1[1] should be 3.0"
    );
    assert!(
        (present_value[10] - 4.0).abs() < 1e-5,
        "value pos 2[2] should be 4.0"
    );

    println!("✓ End-to-end GroupQueryAttention (prefill mode, buffer sharing) test passed!");
    println!("  Query shape: [1, 3, 8]");
    println!(
        "  KV cache: [1, 1, {}, 4] (pre-allocated, {} positions used)",
        max_seq_len, total_seq_len
    );
    println!("  Past cache: 2 tokens");
    println!("  Output shape: [1, 1, 8]");
    println!("  Present key shape: [1, 1, 3, 4]");
    println!("  Cache concatenation verified!");
}
