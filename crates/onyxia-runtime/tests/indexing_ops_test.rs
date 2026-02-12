//! End-to-end tests for indexing operations.
//!
//! Tests: Gather (simple and embedding lookup)

use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_planner::{compile, KernelRegistry};
use onyxia_runtime::{Runtime, Tensor};
use std::collections::HashMap;

/// Helper function to create a simple Gather graph.
fn make_gather_graph() -> Graph {
    let mut graph = Graph::new();

    // Add data input tensor (embedding table) [4, 3]
    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4, 3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add indices input tensor (I64) [2]
    graph.add_tensor(TensorInfo {
        name: "indices".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![2]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add output tensor [2, 3]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Gather node
    let mut node = Node::new("Gather");
    node.name = "gather_node".to_string();
    node.inputs = vec!["data".to_string(), "indices".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["data".to_string(), "indices".to_string()];
    graph.outputs = vec!["output".to_string()];

    graph.metadata.name = "test_gather_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

/// End-to-end test: Gather operation with small embedding table.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_gather_e2e() {
    let graph = make_gather_graph();
    graph
        .validate()
        .expect("Graph validation should succeed");

    // Compile to ExecutionPlan
    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    // Initialize runtime
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Data: 4×3 embedding table
    let data = Tensor::from_vec(
        vec![
            1.0f32, 2.0, 3.0, // Row 0
            4.0, 5.0, 6.0, // Row 1
            7.0, 8.0, 9.0, // Row 2
            10.0, 11.0, 12.0, // Row 3
        ],
        &[4, 3],
    );

    // Indices: [1, 3] → gather rows 1 and 3
    let indices = Tensor::from_vec(vec![1i64, 3], &[2]);

    let outputs = executor
        .run(&[("data", data), ("indices", indices)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected: [4.0, 5.0, 6.0, 10.0, 11.0, 12.0] (rows 1 and 3)
    assert_eq!(
        output,
        vec![4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
        "Gather result incorrect"
    );

    println!("✓ End-to-end Gather test passed!");
    println!("  Data shape: [4, 3]");
    println!("  Indices: [1, 3]");
    println!("  Output: {:?}", output);
}

/// End-to-end test: Gather operation simulating token embedding lookup.
///
/// Simulates a realistic use case like Gemma's embedding layer:
/// - Small embedding table [8, 4] (vocab_size × hidden_dim)
/// - Token indices [2, 3] (batch × seq)
/// - Output embeddings [2, 3, 4] (batch × seq × hidden_dim)
#[pollster::test]
#[ignore] // Requires GPU
async fn test_gather_embedding_e2e() {
    let mut graph = Graph::new();

    // Add embedding table [8, 4]
    graph.add_tensor(TensorInfo {
        name: "embedding_table".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![8, 4]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add token indices [2, 3] (batch=2, seq=3)
    graph.add_tensor(TensorInfo {
        name: "token_ids".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![2, 3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add output tensor [2, 3, 4]
    graph.add_tensor(TensorInfo {
        name: "embeddings".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3, 4]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Gather node
    let mut node = Node::new("Gather");
    node.name = "embedding_lookup".to_string();
    node.inputs = vec!["embedding_table".to_string(), "token_ids".to_string()];
    node.outputs = vec!["embeddings".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["embedding_table".to_string(), "token_ids".to_string()];
    graph.outputs = vec!["embeddings".to_string()];

    graph.metadata.name = "test_embedding_gather".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
        .validate()
        .expect("Graph validation should succeed");

    // Compile and load
    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Create embedding table [8, 4]
    let embedding_table = Tensor::from_vec(
        vec![
            0.1f32, 0.2, 0.3, 0.4, // Token 0
            1.1, 1.2, 1.3, 1.4, // Token 1
            2.1, 2.2, 2.3, 2.4, // Token 2
            3.1, 3.2, 3.3, 3.4, // Token 3
            4.1, 4.2, 4.3, 4.4, // Token 4
            5.1, 5.2, 5.3, 5.4, // Token 5
            6.1, 6.2, 6.3, 6.4, // Token 6
            7.1, 7.2, 7.3, 7.4, // Token 7
        ],
        &[8, 4],
    );

    // Token IDs: [[0, 2, 4], [1, 3, 5]]
    let token_ids = Tensor::from_vec(
        vec![
            0i64, 2, 4, // Batch 0
            1, 3, 5, // Batch 1
        ],
        &[2, 3],
    );

    let outputs = executor
        .run(&[
            ("embedding_table", embedding_table),
            ("token_ids", token_ids),
        ])
        .expect("Execution should succeed");

    let embeddings = outputs["embeddings"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    let expected = vec![
        0.1, 0.2, 0.3, 0.4, // Token 0
        2.1, 2.2, 2.3, 2.4, // Token 2
        4.1, 4.2, 4.3, 4.4, // Token 4
        1.1, 1.2, 1.3, 1.4, // Token 1
        3.1, 3.2, 3.3, 3.4, // Token 3
        5.1, 5.2, 5.3, 5.4, // Token 5
    ];

    assert_eq!(embeddings, expected, "Embedding lookup result incorrect");

    println!("✓ End-to-end embedding Gather test passed!");
    println!("  Embedding table: [8, 4]");
    println!("  Token IDs: [2, 3] = [[0, 2, 4], [1, 3, 5]]");
    println!("  Output embeddings shape: [2, 3, 4]");
}
