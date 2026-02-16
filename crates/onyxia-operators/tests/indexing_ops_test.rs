//! End-to-end tests for indexing operations.
//!
//! Tests: Gather (simple and embedding lookup)

use onyxia_compiler::CompilerPipeline;
use onyxia_onnx::{AttributeValue, DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_operators::core_operator_registry;
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
#[ignore="requires GPU"]
async fn test_gather_e2e() {
    let graph = make_gather_graph();
    graph.validate().expect("Graph validation should succeed");

    // Compile to CompiledModel
    let registry = core_operator_registry();
    let plan = CompilerPipeline::new(HashMap::new())
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

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
#[ignore="requires GPU"]
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

    graph.validate().expect("Graph validation should succeed");

    // Compile and load
    let registry = core_operator_registry();
    let plan = CompilerPipeline::new(HashMap::new())
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

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

/// Helper function to create i64 tensor bytes from a slice.
fn i64_bytes(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|&v| v.to_le_bytes()).collect()
}

/// End-to-end test: Slice operation with basic single-axis slicing.
#[pollster::test]
#[ignore="requires GPU"]
async fn test_slice_basic_e2e() {
    let mut graph = Graph::new();

    // Add data input tensor [10]
    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![10]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add starts tensor (constant)
    graph.add_tensor(TensorInfo {
        name: "starts".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Weight,
        initializer: Some(i64_bytes(&[2])),
    });

    // Add ends tensor (constant)
    graph.add_tensor(TensorInfo {
        name: "ends".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Weight,
        initializer: Some(i64_bytes(&[7])),
    });

    // Add output tensor [5]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![5]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Slice node
    let mut node = Node::new("Slice");
    node.name = "slice_node".to_string();
    node.inputs = vec!["data".to_string(), "starts".to_string(), "ends".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["data".to_string()];
    graph.outputs = vec!["output".to_string()];

    graph.metadata.name = "test_slice_basic".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph.validate().expect("Graph validation should succeed");

    // Compile and load
    let registry = core_operator_registry();
    let plan = CompilerPipeline::new(HashMap::new())
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    let data = Tensor::from_vec((0..10).map(|x| x as f32).collect(), &[10]);

    let outputs = executor
        .run(&[("data", data)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected: [2, 3, 4, 5, 6]
    assert_eq!(
        output,
        vec![2.0, 3.0, 4.0, 5.0, 6.0],
        "Slice result incorrect"
    );

    println!("✓ End-to-end Slice basic test passed!");
    println!("  Input: [0..10]");
    println!("  Slice [2:7]: {:?}", output);
}

/// End-to-end test: Slice operation with multi-axis slicing.
#[pollster::test]
#[ignore="requires GPU"]
async fn test_slice_multiaxis_e2e() {
    let mut graph = Graph::new();

    // Add data input tensor [4, 5, 6]
    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4, 5, 6]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add starts tensor: [1, 0, 2]
    graph.add_tensor(TensorInfo {
        name: "starts".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![3]),
        kind: TensorKind::Weight,
        initializer: Some(i64_bytes(&[1, 0, 2])),
    });

    // Add ends tensor: [3, 4, 5]
    graph.add_tensor(TensorInfo {
        name: "ends".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![3]),
        kind: TensorKind::Weight,
        initializer: Some(i64_bytes(&[3, 4, 5])),
    });

    // Add output tensor [2, 4, 3]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 4, 3]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Slice node
    let mut node = Node::new("Slice");
    node.name = "slice_multiaxis".to_string();
    node.inputs = vec!["data".to_string(), "starts".to_string(), "ends".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["data".to_string()];
    graph.outputs = vec!["output".to_string()];

    graph.metadata.name = "test_slice_multiaxis".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph.validate().expect("Graph validation should succeed");

    // Compile and load
    let registry = core_operator_registry();
    let plan = CompilerPipeline::new(HashMap::new())
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Create data [4, 5, 6] with sequential values
    let data: Vec<f32> = (0..120).map(|x| x as f32).collect();
    let data_tensor = Tensor::from_vec(data, &[4, 5, 6]);

    let outputs = executor
        .run(&[("data", data_tensor)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Calculate expected output manually
    // Slicing [1:3, 0:4, 2:5] from [4, 5, 6]
    let mut expected = Vec::new();
    for i in 1..3 {
        // dim 0: 1 to 3
        for j in 0..4 {
            // dim 1: 0 to 4
            for k in 2..5 {
                // dim 2: 2 to 5
                let idx = i * 30 + j * 6 + k;
                expected.push(idx as f32);
            }
        }
    }

    assert_eq!(output, expected, "Multi-axis slice result incorrect");

    println!("✓ End-to-end Slice multi-axis test passed!");
    println!("  Input shape: [4, 5, 6]");
    println!("  Slice [1:3, 0:4, 2:5]");
    println!("  Output shape: [2, 4, 3]");
}

/// End-to-end test: Slice with negative indices.
#[pollster::test]
#[ignore="requires GPU"]
async fn test_slice_negative_indices_e2e() {
    let mut graph = Graph::new();

    // Add data input tensor [8]
    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![8]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add starts tensor: [-5] (should be 3)
    graph.add_tensor(TensorInfo {
        name: "starts".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Weight,
        initializer: Some(i64_bytes(&[-5])),
    });

    // Add ends tensor: [-1] (should be 7)
    graph.add_tensor(TensorInfo {
        name: "ends".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Weight,
        initializer: Some(i64_bytes(&[-1])),
    });

    // Add output tensor [4]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Slice node
    let mut node = Node::new("Slice");
    node.name = "slice_neg".to_string();
    node.inputs = vec!["data".to_string(), "starts".to_string(), "ends".to_string()];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["data".to_string()];
    graph.outputs = vec!["output".to_string()];

    graph.metadata.name = "test_slice_negative".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph.validate().expect("Graph validation should succeed");

    // Compile and load
    let registry = core_operator_registry();
    let plan = CompilerPipeline::new(HashMap::new())
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Data: [0, 1, 2, 3, 4, 5, 6, 7]
    let data = Tensor::from_vec((0..8).map(|x| x as f32).collect(), &[8]);

    let outputs = executor
        .run(&[("data", data)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected: [3, 4, 5, 6]
    assert_eq!(
        output,
        vec![3.0, 4.0, 5.0, 6.0],
        "Slice with negative indices incorrect"
    );

    println!("✓ End-to-end Slice negative indices test passed!");
    println!("  Input: [0..8]");
    println!("  Slice [-5:-1]: {:?}", output);
}

/// End-to-end test: Slice with steps > 1 (strided slicing).
#[pollster::test]
#[ignore="requires GPU"]
async fn test_slice_strided_e2e() {
    let mut graph = Graph::new();

    // Add data input tensor [10]
    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![10]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add starts tensor: [0]
    graph.add_tensor(TensorInfo {
        name: "starts".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Weight,
        initializer: Some(i64_bytes(&[0])),
    });

    // Add ends tensor: [10]
    graph.add_tensor(TensorInfo {
        name: "ends".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Weight,
        initializer: Some(i64_bytes(&[10])),
    });

    // Add axes tensor: [0]
    graph.add_tensor(TensorInfo {
        name: "axes".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Weight,
        initializer: Some(i64_bytes(&[0])),
    });

    // Add steps tensor: [2]
    graph.add_tensor(TensorInfo {
        name: "steps".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Weight,
        initializer: Some(i64_bytes(&[2])),
    });

    // Add output tensor [5]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![5]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Slice node with all 5 inputs
    let mut node = Node::new("Slice");
    node.name = "slice_strided".to_string();
    node.inputs = vec![
        "data".to_string(),
        "starts".to_string(),
        "ends".to_string(),
        "axes".to_string(),
        "steps".to_string(),
    ];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["data".to_string()];
    graph.outputs = vec!["output".to_string()];

    graph.metadata.name = "test_slice_strided".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph.validate().expect("Graph validation should succeed");

    // Compile and load
    let registry = core_operator_registry();
    let plan = CompilerPipeline::new(HashMap::new())
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    let data = Tensor::from_vec((0..10).map(|x| x as f32).collect(), &[10]);

    let outputs = executor
        .run(&[("data", data)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected: [0, 2, 4, 6, 8] (every 2nd element)
    assert_eq!(
        output,
        vec![0.0, 2.0, 4.0, 6.0, 8.0],
        "Strided slice result incorrect"
    );

    println!("✓ End-to-end Slice strided test passed!");
    println!("  Input: [0..10]");
    println!("  Slice [0:10:2]: {:?}", output);
}

/// End-to-end test: Slice with negative steps (reverse slicing).
#[pollster::test]
#[ignore="requires GPU"]
async fn test_slice_reverse_e2e() {
    let mut graph = Graph::new();

    // Add data input tensor [6]
    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![6]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add starts tensor: [5]
    graph.add_tensor(TensorInfo {
        name: "starts".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Weight,
        initializer: Some(i64_bytes(&[5])),
    });

    // Add ends tensor: [0]
    graph.add_tensor(TensorInfo {
        name: "ends".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Weight,
        initializer: Some(i64_bytes(&[0])),
    });

    // Add axes tensor: [0]
    graph.add_tensor(TensorInfo {
        name: "axes".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Weight,
        initializer: Some(i64_bytes(&[0])),
    });

    // Add steps tensor: [-1]
    graph.add_tensor(TensorInfo {
        name: "steps".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Weight,
        initializer: Some(i64_bytes(&[-1])),
    });

    // Add output tensor [5]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![5]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Slice node
    let mut node = Node::new("Slice");
    node.name = "slice_reverse".to_string();
    node.inputs = vec![
        "data".to_string(),
        "starts".to_string(),
        "ends".to_string(),
        "axes".to_string(),
        "steps".to_string(),
    ];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec!["data".to_string()];
    graph.outputs = vec!["output".to_string()];

    graph.metadata.name = "test_slice_reverse".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph.validate().expect("Graph validation should succeed");

    // Compile and load
    let registry = core_operator_registry();
    let plan = CompilerPipeline::new(HashMap::new())
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Data: [0, 1, 2, 3, 4, 5]
    let data = Tensor::from_vec((0..6).map(|x| x as f32).collect(), &[6]);

    let outputs = executor
        .run(&[("data", data)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected: [5, 4, 3, 2, 1] (reversed from 5 to 1)
    assert_eq!(
        output,
        vec![5.0, 4.0, 3.0, 2.0, 1.0],
        "Reverse slice result incorrect"
    );

    println!("✓ End-to-end Slice reverse test passed!");
    println!("  Input: [0..6]");
    println!("  Slice [5:0:-1]: {:?}", output);
}

/// Helper function to create a Trilu graph.
fn make_trilu_graph(upper: i64) -> Graph {
    let mut graph = Graph::new();

    // Add input tensor [3, 3]
    graph.add_tensor(TensorInfo {
        name: "input".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![3, 3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add output tensor [3, 3]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![3, 3]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Trilu node
    let mut node = Node::new("Trilu");
    node.name = "trilu_node".to_string();
    node.inputs = vec!["input".to_string()];
    node.outputs = vec!["output".to_string()];
    node.attributes
        .insert("upper".to_string(), AttributeValue::Int(upper));
    graph.add_node(node);

    graph.inputs = vec!["input".to_string()];
    graph.outputs = vec!["output".to_string()];

    graph.metadata.name = "test_trilu_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

/// End-to-end test: Trilu operation - upper triangle.
#[pollster::test]
#[ignore="requires GPU"]
async fn test_trilu_upper_triangle_e2e() {
    let graph = make_trilu_graph(1);
    graph.validate().expect("Graph validation should succeed");

    // Compile to CompiledModel
    let registry = core_operator_registry();
    let plan = CompilerPipeline::new(HashMap::new())
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    // Initialize runtime
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Input: 3×3 matrix
    #[rustfmt::skip]
    let input = Tensor::from_vec(
        vec![
            1.0f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ],
        &[3, 3],
    );

    let outputs = executor
        .run(&[("input", input)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected upper triangle (upper=1, k=0):
    // [[1, 2, 3],
    //  [0, 5, 6],
    //  [0, 0, 9]]
    #[rustfmt::skip]
    let expected = vec![
        1.0, 2.0, 3.0,
        0.0, 5.0, 6.0,
        0.0, 0.0, 9.0,
    ];

    assert_eq!(output.len(), expected.len());
    for (i, (&actual, &expected)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-6,
            "Mismatch at index {}: got {}, expected {}",
            i,
            actual,
            expected
        );
    }

    println!("✓ End-to-end Trilu upper triangle test passed!");
    println!("  Output: {:?}", output);
}

/// End-to-end test: Trilu operation - lower triangle.
#[pollster::test]
#[ignore="requires GPU"]
async fn test_trilu_lower_triangle_e2e() {
    let graph = make_trilu_graph(0);
    graph.validate().expect("Graph validation should succeed");

    // Compile to CompiledModel
    let registry = core_operator_registry();
    let plan = CompilerPipeline::new(HashMap::new())
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    // Initialize runtime
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Input: 3×3 matrix
    #[rustfmt::skip]
    let input = Tensor::from_vec(
        vec![
            1.0f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ],
        &[3, 3],
    );

    let outputs = executor
        .run(&[("input", input)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected lower triangle (upper=0, k=0):
    // [[1, 0, 0],
    //  [4, 5, 0],
    //  [7, 8, 9]]
    #[rustfmt::skip]
    let expected = vec![
        1.0, 0.0, 0.0,
        4.0, 5.0, 0.0,
        7.0, 8.0, 9.0,
    ];

    assert_eq!(output.len(), expected.len());
    for (i, (&actual, &expected)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-6,
            "Mismatch at index {}: got {}, expected {}",
            i,
            actual,
            expected
        );
    }

    println!("✓ End-to-end Trilu lower triangle test passed!");
    println!("  Output: {:?}", output);
}

/// Helper function to create a Trilu graph with batched input.
fn make_trilu_batched_graph(upper: i64) -> Graph {
    let mut graph = Graph::new();

    // Add input tensor [2, 3, 3] (batch of 2 matrices)
    graph.add_tensor(TensorInfo {
        name: "input".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3, 3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add output tensor [2, 3, 3]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 3, 3]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Trilu node
    let mut node = Node::new("Trilu");
    node.name = "trilu_node".to_string();
    node.inputs = vec!["input".to_string()];
    node.outputs = vec!["output".to_string()];
    node.attributes
        .insert("upper".to_string(), AttributeValue::Int(upper));
    graph.add_node(node);

    graph.inputs = vec!["input".to_string()];
    graph.outputs = vec!["output".to_string()];

    graph.metadata.name = "test_trilu_batched_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

/// End-to-end test: Trilu operation with batched input (3D tensor).
#[pollster::test]
#[ignore="requires GPU"]
async fn test_trilu_batched_e2e() {
    let graph = make_trilu_batched_graph(1);
    graph.validate().expect("Graph validation should succeed");

    // Compile to CompiledModel
    let registry = core_operator_registry();
    let plan = CompilerPipeline::new(HashMap::new())
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    // Initialize runtime
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Input: 2×3×3 batched matrices
    #[rustfmt::skip]
    let input = Tensor::from_vec(
        vec![
            // First matrix
            1.0f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            // Second matrix
            10.0, 11.0, 12.0,
            13.0, 14.0, 15.0,
            16.0, 17.0, 18.0,
        ],
        &[2, 3, 3],
    );

    let outputs = executor
        .run(&[("input", input)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected upper triangle for both matrices:
    // First: [[1, 2, 3], [0, 5, 6], [0, 0, 9]]
    // Second: [[10, 11, 12], [0, 14, 15], [0, 0, 18]]
    #[rustfmt::skip]
    let expected = vec![
        1.0, 2.0, 3.0,
        0.0, 5.0, 6.0,
        0.0, 0.0, 9.0,
        10.0, 11.0, 12.0,
        0.0, 14.0, 15.0,
        0.0, 0.0, 18.0,
    ];

    assert_eq!(output.len(), expected.len());
    for (i, (&actual, &expected)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-6,
            "Mismatch at index {}: got {}, expected {}",
            i,
            actual,
            expected
        );
    }

    println!("✓ End-to-end Trilu batched test passed!");
    println!("  Output: {:?}", output);
}

/// Helper function to create a Trilu graph with non-square matrices.
fn make_trilu_nonsquare_graph(upper: i64) -> Graph {
    let mut graph = Graph::new();

    // Add input tensor [2, 4] (non-square matrix)
    graph.add_tensor(TensorInfo {
        name: "input".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 4]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add output tensor [2, 4]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2, 4]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create Trilu node
    let mut node = Node::new("Trilu");
    node.name = "trilu_node".to_string();
    node.inputs = vec!["input".to_string()];
    node.outputs = vec!["output".to_string()];
    node.attributes
        .insert("upper".to_string(), AttributeValue::Int(upper));
    graph.add_node(node);

    graph.inputs = vec!["input".to_string()];
    graph.outputs = vec!["output".to_string()];

    graph.metadata.name = "test_trilu_nonsquare_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

/// End-to-end test: Trilu operation with non-square matrix.
#[pollster::test]
#[ignore="requires GPU"]
async fn test_trilu_nonsquare_e2e() {
    let graph = make_trilu_nonsquare_graph(1);
    graph.validate().expect("Graph validation should succeed");

    // Compile to CompiledModel
    let registry = core_operator_registry();
    let plan = CompilerPipeline::new(HashMap::new())
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    // Initialize runtime
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Input: 2×4 non-square matrix
    #[rustfmt::skip]
    let input = Tensor::from_vec(
        vec![
            1.0f32, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ],
        &[2, 4],
    );

    let outputs = executor
        .run(&[("input", input)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected upper triangle (upper=1, k=0):
    // [[1, 2, 3, 4],
    //  [0, 6, 7, 8]]
    #[rustfmt::skip]
    let expected = vec![
        1.0, 2.0, 3.0, 4.0,
        0.0, 6.0, 7.0, 8.0,
    ];

    assert_eq!(output.len(), expected.len());
    for (i, (&actual, &expected)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-6,
            "Mismatch at index {}: got {}, expected {}",
            i,
            actual,
            expected
        );
    }

    println!("✓ End-to-end Trilu non-square test passed!");
    println!("  Output: {:?}", output);
}

// ============================================================================
// ScatterND Tests
// ============================================================================

/// Helper function to create a simple ScatterND graph with reduction="none".
fn make_scatternd_graph(reduction: &str) -> Graph {
    let mut graph = Graph::new();

    // Add data input tensor [3, 3]
    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![3, 3]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add indices input tensor (I64) [2, 2]
    graph.add_tensor(TensorInfo {
        name: "indices".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![2, 2]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add updates input tensor [2]
    graph.add_tensor(TensorInfo {
        name: "updates".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![2]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add output tensor [3, 3]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![3, 3]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create ScatterND node
    let mut node = Node::new("ScatterND");
    node.name = "scatternd_node".to_string();
    node.inputs = vec![
        "data".to_string(),
        "indices".to_string(),
        "updates".to_string(),
    ];
    node.outputs = vec!["output".to_string()];
    node.attributes.insert(
        "reduction".to_string(),
        AttributeValue::String(reduction.to_string()),
    );
    graph.add_node(node);

    graph.inputs = vec![
        "data".to_string(),
        "indices".to_string(),
        "updates".to_string(),
    ];
    graph.outputs = vec!["output".to_string()];

    graph.metadata.name = "test_scatternd_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

/// End-to-end test: ScatterND operation with reduction="none".
#[pollster::test]
#[ignore="requires GPU"]
async fn test_scatternd_none_e2e() {
    let graph = make_scatternd_graph("none");
    graph.validate().expect("Graph validation should succeed");

    // Compile to CompiledModel
    let registry = core_operator_registry();
    let plan = CompilerPipeline::new(HashMap::new())
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    // Initialize runtime
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Data: 3×3 matrix
    #[rustfmt::skip]
    let data = Tensor::from_vec(
        vec![
            1.0f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ],
        &[3, 3],
    );

    // Indices: [[0, 0], [2, 1]] - scatter to (0,0) and (2,1)
    let indices = Tensor::from_vec(vec![0i64, 0, 2, 1], &[2, 2]);

    // Updates: [10.0, 20.0]
    let updates = Tensor::from_vec(vec![10.0f32, 20.0], &[2]);

    let outputs = executor
        .run(&[("data", data), ("indices", indices), ("updates", updates)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected: data with 10.0 at [0,0] and 20.0 at [2,1]
    // [[10, 2, 3],
    //  [4,  5, 6],
    //  [7, 20, 9]]
    #[rustfmt::skip]
    let expected = vec![
        10.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 20.0, 9.0,
    ];

    assert_eq!(output.len(), expected.len());
    for (i, (&actual, &expected)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Mismatch at index {}: got {}, expected {}",
            i,
            actual,
            expected
        );
    }

    println!("✓ End-to-end ScatterND (none) test passed!");
    println!("  Output: {:?}", output);
}

/// End-to-end test: ScatterND operation with reduction="add".
#[pollster::test]
#[ignore="requires GPU"]
async fn test_scatternd_add_e2e() {
    let graph = make_scatternd_graph("add");
    graph.validate().expect("Graph validation should succeed");

    // Compile to CompiledModel
    let registry = core_operator_registry();
    let plan = CompilerPipeline::new(HashMap::new())
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    // Initialize runtime
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Data: 3×3 matrix
    #[rustfmt::skip]
    let data = Tensor::from_vec(
        vec![
            1.0f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ],
        &[3, 3],
    );

    // Indices: [[0, 0], [2, 1]] - scatter to (0,0) and (2,1)
    let indices = Tensor::from_vec(vec![0i64, 0, 2, 1], &[2, 2]);

    // Updates: [10.0, 20.0]
    let updates = Tensor::from_vec(vec![10.0f32, 20.0], &[2]);

    let outputs = executor
        .run(&[("data", data), ("indices", indices), ("updates", updates)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected: data with 10.0 added to [0,0] and 20.0 added to [2,1]
    // [[11, 2, 3],
    //  [4,  5, 6],
    //  [7, 28, 9]]
    #[rustfmt::skip]
    let expected = vec![
        11.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 28.0, 9.0,
    ];

    assert_eq!(output.len(), expected.len());
    for (i, (&actual, &expected)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Mismatch at index {}: got {}, expected {}",
            i,
            actual,
            expected
        );
    }

    println!("✓ End-to-end ScatterND (add) test passed!");
    println!("  Output: {:?}", output);
}

/// Helper function to create a 1D ScatterND test.
fn make_scatternd_1d_graph() -> Graph {
    let mut graph = Graph::new();

    // Add data input tensor [8]
    graph.add_tensor(TensorInfo {
        name: "data".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![8]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add indices input tensor (I64) [4, 1] - 4 indices, each of rank 1
    graph.add_tensor(TensorInfo {
        name: "indices".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![4, 1]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add updates input tensor [4]
    graph.add_tensor(TensorInfo {
        name: "updates".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![4]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Add output tensor [8]
    graph.add_tensor(TensorInfo {
        name: "output".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![8]),
        kind: TensorKind::Output,
        initializer: None,
    });

    // Create ScatterND node
    let mut node = Node::new("ScatterND");
    node.name = "scatternd_node".to_string();
    node.inputs = vec![
        "data".to_string(),
        "indices".to_string(),
        "updates".to_string(),
    ];
    node.outputs = vec!["output".to_string()];
    graph.add_node(node);

    graph.inputs = vec![
        "data".to_string(),
        "indices".to_string(),
        "updates".to_string(),
    ];
    graph.outputs = vec!["output".to_string()];

    graph.metadata.name = "test_scatternd_1d_graph".to_string();
    graph.metadata.ir_version = 9;
    graph.metadata.producer_name = "onyxia_test".to_string();
    graph.metadata.model_version = 1;

    graph
}

/// End-to-end test: ScatterND operation with 1D data.
#[pollster::test]
#[ignore="requires GPU"]
async fn test_scatternd_1d_e2e() {
    let graph = make_scatternd_1d_graph();
    graph.validate().expect("Graph validation should succeed");

    // Compile to CompiledModel
    let registry = core_operator_registry();
    let plan = CompilerPipeline::new(HashMap::new())
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    // Initialize runtime
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Data: [0, 0, 0, 0, 0, 0, 0, 0]
    let data = Tensor::from_vec(vec![0.0f32; 8], &[8]);

    // Indices: [[1], [3], [5], [7]] - scatter to positions 1, 3, 5, 7
    let indices = Tensor::from_vec(vec![1i64, 3, 5, 7], &[4, 1]);

    // Updates: [10.0, 20.0, 30.0, 40.0]
    let updates = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], &[4]);

    let outputs = executor
        .run(&[("data", data), ("indices", indices), ("updates", updates)])
        .expect("Execution should succeed");

    let output = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");

    // Expected: [0, 10, 0, 20, 0, 30, 0, 40]
    let expected = vec![0.0, 10.0, 0.0, 20.0, 0.0, 30.0, 0.0, 40.0];

    assert_eq!(output.len(), expected.len());
    for (i, (&actual, &expected)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Mismatch at index {}: got {}, expected {}",
            i,
            actual,
            expected
        );
    }

    println!("✓ End-to-end ScatterND 1D test passed!");
    println!("  Output: {:?}", output);
}
