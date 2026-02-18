//! End-to-end test for GroupQueryAttention operator.

mod common;

use common::{CompilerPipeline, Runtime, Tensor};
use onyxia_core::DataType;
use onyxia_onnx::{AttributeValue, Dimension, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_operators::core_operator_registry;

#[pollster::test]
#[ignore = "requires GPU"]
async fn test_group_query_attention_basic() {
    // Create a minimal GQA graph
    let mut graph = Graph::new();

    let batch = 1;
    let seq_len = 4;
    let num_heads = 8;
    let kv_num_heads = 2;
    let head_size = 16;

    // Query: [batch, seq, num_heads * head_size]
    graph.add_tensor(TensorInfo {
        name: "query".into(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, seq_len, num_heads * head_size]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Key: [batch, seq, kv_num_heads * head_size]
    graph.add_tensor(TensorInfo {
        name: "key".into(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, seq_len, kv_num_heads * head_size]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Value: [batch, seq, kv_num_heads * head_size]
    graph.add_tensor(TensorInfo {
        name: "value".into(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, seq_len, kv_num_heads * head_size]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Output
    graph.add_tensor(TensorInfo {
        name: "output".into(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, seq_len, num_heads * head_size]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // present_key output: [batch, kv_num_heads, seq_len, head_size]
    graph.add_tensor(TensorInfo {
        name: "present_key".into(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, kv_num_heads, seq_len, head_size]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // present_value output: [batch, kv_num_heads, seq_len, head_size]
    graph.add_tensor(TensorInfo {
        name: "present_value".into(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![batch, kv_num_heads, seq_len, head_size]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut gqa_node = Node::new("GroupQueryAttention");
    gqa_node.domain = "com.microsoft".into();
    gqa_node.inputs = vec!["query".into(), "key".into(), "value".into()];
    gqa_node.outputs = vec![
        "output".into(),
        "present_key".into(),
        "present_value".into(),
    ];
    gqa_node
        .attributes
        .insert("num_heads".into(), AttributeValue::Int(num_heads as i64));
    gqa_node.attributes.insert(
        "kv_num_heads".into(),
        AttributeValue::Int(kv_num_heads as i64),
    );
    graph.nodes.push(gqa_node);

    graph.inputs = vec!["query".into(), "key".into(), "value".into()];
    graph.outputs = vec![
        "output".into(),
        "present_key".into(),
        "present_value".into(),
    ];

    // Compile
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let compiled = pipeline
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    // Execute
    let runtime = Runtime::new().await.expect("Runtime init");
    let mut executor = runtime.load_model(compiled).await.expect("Load model");

    // Create test inputs (simple uniform values)
    let query = Tensor::from_vec(
        vec![1.0f32; batch * seq_len * num_heads * head_size],
        &[batch, seq_len, num_heads * head_size],
    );
    let key = Tensor::from_vec(
        vec![1.0f32; batch * seq_len * kv_num_heads * head_size],
        &[batch, seq_len, kv_num_heads * head_size],
    );
    let value = Tensor::from_vec(
        vec![1.0f32; batch * seq_len * kv_num_heads * head_size],
        &[batch, seq_len, kv_num_heads * head_size],
    );

    let outputs = executor
        .run(&[("query", query), ("key", key), ("value", value)])
        .expect("Execution");

    let result: Vec<f32> = outputs["output"].to_vec().expect("Convert to f32");
    let present_key: Vec<f32> = outputs["present_key"].to_vec().expect("Convert to f32");
    let present_value: Vec<f32> = outputs["present_value"].to_vec().expect("Convert to f32");

    assert_eq!(result.len(), batch * seq_len * num_heads * head_size);
    assert_eq!(
        present_key.len(),
        batch * kv_num_heads * seq_len * head_size
    );
    assert_eq!(
        present_value.len(),
        batch * kv_num_heads * seq_len * head_size
    );

    println!("✓ GroupQueryAttention test passed!");
}

#[pollster::test]
#[ignore = "requires GPU"]
async fn test_group_query_attention_kv_cache() {
    // Test KV cache concatenation: prefill followed by decode steps
    let mut graph = Graph::new();

    let batch = 1;
    let prefill_seq_len = 4;
    let decode_seq_len = 1; // decode one token at a time
    let num_heads = 8;
    let kv_num_heads = 2;
    let head_size = 16;

    // Query: [batch, seq, num_heads * head_size]
    graph.add_tensor(TensorInfo {
        name: "query".into(),
        dtype: DataType::F32,
        shape: TensorShape::Dynamic(vec![
            Dimension::Static(batch),
            Dimension::Named("seq".into()),
            Dimension::Static(num_heads * head_size),
        ]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Key: [batch, seq, kv_num_heads * head_size]
    graph.add_tensor(TensorInfo {
        name: "key".into(),
        dtype: DataType::F32,
        shape: TensorShape::Dynamic(vec![
            Dimension::Static(batch),
            Dimension::Named("seq".into()),
            Dimension::Static(kv_num_heads * head_size),
        ]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Value: [batch, seq, kv_num_heads * head_size]
    graph.add_tensor(TensorInfo {
        name: "value".into(),
        dtype: DataType::F32,
        shape: TensorShape::Dynamic(vec![
            Dimension::Static(batch),
            Dimension::Named("seq".into()),
            Dimension::Static(kv_num_heads * head_size),
        ]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // past_key: [batch, kv_num_heads, past_seq, head_size]
    graph.add_tensor(TensorInfo {
        name: "past_key".into(),
        dtype: DataType::F32,
        shape: TensorShape::Dynamic(vec![
            Dimension::Static(batch),
            Dimension::Static(kv_num_heads),
            Dimension::Named("past_seq".into()),
            Dimension::Static(head_size),
        ]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // past_value: [batch, kv_num_heads, past_seq, head_size]
    graph.add_tensor(TensorInfo {
        name: "past_value".into(),
        dtype: DataType::F32,
        shape: TensorShape::Dynamic(vec![
            Dimension::Static(batch),
            Dimension::Static(kv_num_heads),
            Dimension::Named("past_seq".into()),
            Dimension::Static(head_size),
        ]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Output
    graph.add_tensor(TensorInfo {
        name: "output".into(),
        dtype: DataType::F32,
        shape: TensorShape::Dynamic(vec![
            Dimension::Static(batch),
            Dimension::Named("seq".into()),
            Dimension::Static(num_heads * head_size),
        ]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // present_key: [batch, kv_num_heads, total_seq, head_size]
    graph.add_tensor(TensorInfo {
        name: "present_key".into(),
        dtype: DataType::F32,
        shape: TensorShape::Dynamic(vec![
            Dimension::Static(batch),
            Dimension::Static(kv_num_heads),
            Dimension::Named("total_seq".into()),
            Dimension::Static(head_size),
        ]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // present_value: [batch, kv_num_heads, total_seq, head_size]
    graph.add_tensor(TensorInfo {
        name: "present_value".into(),
        dtype: DataType::F32,
        shape: TensorShape::Dynamic(vec![
            Dimension::Static(batch),
            Dimension::Static(kv_num_heads),
            Dimension::Named("total_seq".into()),
            Dimension::Static(head_size),
        ]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut gqa_node = Node::new("GroupQueryAttention");
    gqa_node.domain = "com.microsoft".into();
    gqa_node.inputs = vec![
        "query".into(),
        "key".into(),
        "value".into(),
        "past_key".into(),
        "past_value".into(),
    ];
    gqa_node.outputs = vec![
        "output".into(),
        "present_key".into(),
        "present_value".into(),
    ];
    gqa_node
        .attributes
        .insert("num_heads".into(), AttributeValue::Int(num_heads as i64));
    gqa_node.attributes.insert(
        "kv_num_heads".into(),
        AttributeValue::Int(kv_num_heads as i64),
    );
    graph.nodes.push(gqa_node);

    graph.inputs = vec![
        "query".into(),
        "key".into(),
        "value".into(),
        "past_key".into(),
        "past_value".into(),
    ];
    graph.outputs = vec![
        "output".into(),
        "present_key".into(),
        "present_value".into(),
    ];

    // Compile
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let compiled = pipeline
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    // Execute
    let runtime = Runtime::new().await.expect("Runtime init");
    let mut executor = runtime.load_model(compiled).await.expect("Load model");

    // Step 1: Prefill with empty past cache
    let query_prefill = Tensor::from_vec(
        vec![1.0f32; batch * prefill_seq_len * num_heads * head_size],
        &[batch, prefill_seq_len, num_heads * head_size],
    );
    let key_prefill = Tensor::from_vec(
        vec![2.0f32; batch * prefill_seq_len * kv_num_heads * head_size],
        &[batch, prefill_seq_len, kv_num_heads * head_size],
    );
    let value_prefill = Tensor::from_vec(
        vec![3.0f32; batch * prefill_seq_len * kv_num_heads * head_size],
        &[batch, prefill_seq_len, kv_num_heads * head_size],
    );
    // Empty past cache (0 sequence length)
    let past_key_empty = Tensor::from_vec(vec![] as Vec<f32>, &[batch, kv_num_heads, 0, head_size]);
    let past_value_empty =
        Tensor::from_vec(vec![] as Vec<f32>, &[batch, kv_num_heads, 0, head_size]);

    let outputs_prefill = executor
        .run(&[
            ("query", query_prefill),
            ("key", key_prefill),
            ("value", value_prefill),
            ("past_key", past_key_empty),
            ("past_value", past_value_empty),
        ])
        .expect("Prefill execution");

    let present_key_after_prefill: Vec<f32> = outputs_prefill["present_key"]
        .to_vec()
        .expect("Convert to f32");
    let present_value_after_prefill: Vec<f32> = outputs_prefill["present_value"]
        .to_vec()
        .expect("Convert to f32");

    // Verify present cache has correct length after prefill
    assert_eq!(
        present_key_after_prefill.len(),
        batch * kv_num_heads * prefill_seq_len * head_size
    );
    assert_eq!(
        present_value_after_prefill.len(),
        batch * kv_num_heads * prefill_seq_len * head_size
    );

    println!(
        "✓ Prefill: present cache has correct size (seq_len={})",
        prefill_seq_len
    );

    // Step 2: Decode step 1 - use present from prefill as past
    let query_decode = Tensor::from_vec(
        vec![1.5f32; batch * decode_seq_len * num_heads * head_size],
        &[batch, decode_seq_len, num_heads * head_size],
    );
    let key_decode = Tensor::from_vec(
        vec![2.5f32; batch * decode_seq_len * kv_num_heads * head_size],
        &[batch, decode_seq_len, kv_num_heads * head_size],
    );
    let value_decode = Tensor::from_vec(
        vec![3.5f32; batch * decode_seq_len * kv_num_heads * head_size],
        &[batch, decode_seq_len, kv_num_heads * head_size],
    );
    let past_key_decode = Tensor::from_vec(
        present_key_after_prefill.clone(),
        &[batch, kv_num_heads, prefill_seq_len, head_size],
    );
    let past_value_decode = Tensor::from_vec(
        present_value_after_prefill.clone(),
        &[batch, kv_num_heads, prefill_seq_len, head_size],
    );

    let outputs_decode = executor
        .run(&[
            ("query", query_decode),
            ("key", key_decode),
            ("value", value_decode),
            ("past_key", past_key_decode),
            ("past_value", past_value_decode),
        ])
        .expect("Decode execution");

    let present_key_after_decode: Vec<f32> = outputs_decode["present_key"]
        .to_vec()
        .expect("Convert to f32");
    let present_value_after_decode: Vec<f32> = outputs_decode["present_value"]
        .to_vec()
        .expect("Convert to f32");

    // Verify present cache has grown by 1 token
    let expected_total_seq = prefill_seq_len + decode_seq_len;
    assert_eq!(
        present_key_after_decode.len(),
        batch * kv_num_heads * expected_total_seq * head_size
    );
    assert_eq!(
        present_value_after_decode.len(),
        batch * kv_num_heads * expected_total_seq * head_size
    );

    println!(
        "✓ Decode step 1: present cache grew to seq_len={}",
        expected_total_seq
    );

    // Step 3: Another decode step to verify continued concatenation
    let query_decode2 = Tensor::from_vec(
        vec![1.8f32; batch * decode_seq_len * num_heads * head_size],
        &[batch, decode_seq_len, num_heads * head_size],
    );
    let key_decode2 = Tensor::from_vec(
        vec![2.8f32; batch * decode_seq_len * kv_num_heads * head_size],
        &[batch, decode_seq_len, kv_num_heads * head_size],
    );
    let value_decode2 = Tensor::from_vec(
        vec![3.8f32; batch * decode_seq_len * kv_num_heads * head_size],
        &[batch, decode_seq_len, kv_num_heads * head_size],
    );
    let past_key_decode2 = Tensor::from_vec(
        present_key_after_decode.clone(),
        &[batch, kv_num_heads, expected_total_seq, head_size],
    );
    let past_value_decode2 = Tensor::from_vec(
        present_value_after_decode.clone(),
        &[batch, kv_num_heads, expected_total_seq, head_size],
    );

    let outputs_decode2 = executor
        .run(&[
            ("query", query_decode2),
            ("key", key_decode2),
            ("value", value_decode2),
            ("past_key", past_key_decode2),
            ("past_value", past_value_decode2),
        ])
        .expect("Decode step 2 execution");

    let present_key_after_decode2: Vec<f32> = outputs_decode2["present_key"]
        .to_vec()
        .expect("Convert to f32");
    let present_value_after_decode2: Vec<f32> = outputs_decode2["present_value"]
        .to_vec()
        .expect("Convert to f32");

    // Verify present cache has grown by another token
    let expected_total_seq2 = expected_total_seq + decode_seq_len;
    assert_eq!(
        present_key_after_decode2.len(),
        batch * kv_num_heads * expected_total_seq2 * head_size
    );
    assert_eq!(
        present_value_after_decode2.len(),
        batch * kv_num_heads * expected_total_seq2 * head_size
    );

    println!(
        "✓ Decode step 2: present cache grew to seq_len={}",
        expected_total_seq2
    );
    println!("✓ KV cache concatenation test passed!");
}
