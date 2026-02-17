//! End-to-end test for GroupQueryAttention operator.

mod common;

use common::{CompilerPipeline, Runtime, Tensor};
use onyxia_core::DataType;
use onyxia_onnx::{AttributeValue, Graph, Node, TensorInfo, TensorKind, TensorShape};
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

    let mut gqa_node = Node::new("GroupQueryAttention");
    gqa_node.domain = "com.microsoft".into();
    gqa_node.inputs = vec!["query".into(), "key".into(), "value".into()];
    gqa_node.outputs = vec!["output".into()];
    gqa_node
        .attributes
        .insert("num_heads".into(), AttributeValue::Int(num_heads as i64));
    gqa_node.attributes.insert(
        "kv_num_heads".into(),
        AttributeValue::Int(kv_num_heads as i64),
    );
    graph.nodes.push(gqa_node);

    graph.inputs = vec!["query".into(), "key".into(), "value".into()];
    graph.outputs = vec!["output".into()];

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

    assert_eq!(result.len(), batch * seq_len * num_heads * head_size);

    println!("✓ GroupQueryAttention test passed!");
}
