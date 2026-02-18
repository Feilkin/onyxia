//! End-to-end test for RotaryEmbedding operator.

mod common;

use common::{CompilerPipeline, Runtime, Tensor};
use onyxia_core::DataType;
use onyxia_onnx::{AttributeValue, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_operators::core_operator_registry;

#[pollster::test]
#[ignore = "requires GPU"]
async fn test_rotary_embedding_basic() {
    // Create a minimal graph with RotaryEmbedding
    let mut graph = Graph::new();

    // Input: [1, 4, 64] (batch=1, seq=4, hidden=64)
    graph.add_tensor(TensorInfo {
        name: "input".into(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![1, 4, 64]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Position IDs: [1, 4]
    graph.add_tensor(TensorInfo {
        name: "position_ids".into(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1, 4]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Cos cache: [1024, 32] (max_seq=1024, rotary_dim/2=32)
    graph.add_tensor(TensorInfo {
        name: "cos_cache".into(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![1024, 32]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Sin cache: [1024, 32]
    graph.add_tensor(TensorInfo {
        name: "sin_cache".into(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![1024, 32]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Output
    graph.add_tensor(TensorInfo {
        name: "output".into(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![1, 4, 64]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut rope_node = Node::new("RotaryEmbedding");
    rope_node.domain = "com.microsoft".into();
    rope_node.inputs = vec![
        "input".into(),
        "position_ids".into(),
        "cos_cache".into(),
        "sin_cache".into(),
    ];
    rope_node.outputs = vec!["output".into()];
    rope_node
        .attributes
        .insert("interleaved".into(), AttributeValue::Int(0));
    graph.nodes.push(rope_node);

    graph.inputs = vec![
        "input".into(),
        "position_ids".into(),
        "cos_cache".into(),
        "sin_cache".into(),
    ];
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

    // Create test inputs
    let input = Tensor::from_vec(vec![1.0f32; 1 * 4 * 64], &[1, 4, 64]);
    let position_ids = Tensor::from_vec(vec![0i64, 1, 2, 3], &[1, 4]);
    let cos_cache = Tensor::from_vec(vec![1.0f32; 1024 * 32], &[1024, 32]);
    let sin_cache = Tensor::from_vec(vec![0.0f32; 1024 * 32], &[1024, 32]);

    let outputs = executor
        .run(&[
            ("input", input),
            ("position_ids", position_ids),
            ("cos_cache", cos_cache),
            ("sin_cache", sin_cache),
        ])
        .expect("Execution");

    let result: Vec<f32> = outputs["output"].to_vec().expect("Convert to f32");

    assert_eq!(result.len(), 1 * 4 * 64);

    // With cos=1 and sin=0, the rotation should preserve the input
    // for the first half and negate for second half in non-interleaved mode
    println!("✓ RotaryEmbedding test passed!");
}

#[pollster::test]
#[ignore = "requires GPU"]
async fn test_rotary_embedding_applies_per_head() {
    let mut graph = Graph::new();

    // hidden=8 with 2 heads => head_size=4, rotary_dim=4 per head
    graph.add_tensor(TensorInfo {
        name: "input".into(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![1, 1, 8]),
        kind: TensorKind::Input,
        initializer: None,
    });
    graph.add_tensor(TensorInfo {
        name: "position_ids".into(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1, 1]),
        kind: TensorKind::Input,
        initializer: None,
    });
    graph.add_tensor(TensorInfo {
        name: "cos_cache".into(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![1, 2]),
        kind: TensorKind::Input,
        initializer: None,
    });
    graph.add_tensor(TensorInfo {
        name: "sin_cache".into(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![1, 2]),
        kind: TensorKind::Input,
        initializer: None,
    });
    graph.add_tensor(TensorInfo {
        name: "output".into(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![1, 1, 8]),
        kind: TensorKind::Output,
        initializer: None,
    });

    let mut rope_node = Node::new("RotaryEmbedding");
    rope_node.domain = "com.microsoft".into();
    rope_node.inputs = vec![
        "input".into(),
        "position_ids".into(),
        "cos_cache".into(),
        "sin_cache".into(),
    ];
    rope_node.outputs = vec!["output".into()];
    rope_node
        .attributes
        .insert("interleaved".into(), AttributeValue::Int(0));
    rope_node
        .attributes
        .insert("num_heads".into(), AttributeValue::Int(2));
    rope_node
        .attributes
        .insert("rotary_embedding_dim".into(), AttributeValue::Int(4));
    graph.nodes.push(rope_node);

    graph.inputs = vec![
        "input".into(),
        "position_ids".into(),
        "cos_cache".into(),
        "sin_cache".into(),
    ];
    graph.outputs = vec!["output".into()];

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let compiled = pipeline
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    let runtime = Runtime::new().await.expect("Runtime init");
    let mut executor = runtime.load_model(compiled).await.expect("Load model");

    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 1, 8]);
    let position_ids = Tensor::from_vec(vec![0i64], &[1, 1]);

    // cos=0, sin=1 performs quarter-turn rotation in each head
    let cos_cache = Tensor::from_vec(vec![0.0f32, 0.0], &[1, 2]);
    let sin_cache = Tensor::from_vec(vec![1.0f32, 1.0], &[1, 2]);

    let outputs = executor
        .run(&[
            ("input", input),
            ("position_ids", position_ids),
            ("cos_cache", cos_cache),
            ("sin_cache", sin_cache),
        ])
        .expect("Execution");

    let result: Vec<f32> = outputs["output"].to_vec().expect("Convert to f32");
    let expected = [-3.0, -4.0, 1.0, 2.0, -7.0, -8.0, 5.0, 6.0];

    assert_eq!(result.len(), expected.len());
    for (actual, expected) in result.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "{} != {}",
            actual,
            expected
        );
    }
}
