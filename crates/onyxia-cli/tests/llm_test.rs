//! Integration tests for LLM session functionality.

use onyxia_cli::llm::{LlmConfig, LlmSession};
use onyxia_compiler::plan::ExecutionPlan;
use onyxia_compiler::{ModelMetadata, TensorRegistry};
use onyxia_onnx::{DataType, TensorInfo, TensorKind, TensorShape};
use onyxia_runtime::Runtime;

/// Create a minimal execution plan that mimics an LLM model structure.
fn create_minimal_llm_plan(num_layers: usize, vocab_size: usize) -> ExecutionPlan {
    let mut tensors = TensorRegistry::new();

    // Register input tensors (what the model expects)
    let _input_ids = tensors.add(TensorInfo {
        name: "input_ids".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1, 1]), // batch=1, seq=1 for decode
        kind: TensorKind::Input,
        initializer: None,
    });

    let _attention_mask = tensors.add(TensorInfo {
        name: "attention_mask".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1, 1]),
        kind: TensorKind::Input,
        initializer: None,
    });

    let _position_ids = tensors.add(TensorInfo {
        name: "position_ids".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1, 1]),
        kind: TensorKind::Input,
        initializer: None,
    });

    let _past_seq_len = tensors.add(TensorInfo {
        name: "past_sequence_length".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Input,
        initializer: None,
    });

    let _total_seq_len = tensors.add(TensorInfo {
        name: "total_sequence_length".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Input,
        initializer: None,
    });

    let _seqlens_k = tensors.add(TensorInfo {
        name: "seqlens_k".to_string(),
        dtype: DataType::I64,
        shape: TensorShape::Static(vec![1]),
        kind: TensorKind::Input,
        initializer: None,
    });

    // Register KV cache input/output pairs for each layer
    let mut inputs = vec![];
    let mut outputs = vec![];

    for layer in 0..num_layers {
        let past_key_id = tensors.add(TensorInfo {
            name: format!("past_key_values.{}.key", layer),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 1, 64, 128]), // [batch, heads, seq, dim]
            kind: TensorKind::Input,
            initializer: None,
        });

        let past_value_id = tensors.add(TensorInfo {
            name: format!("past_key_values.{}.value", layer),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 1, 64, 128]),
            kind: TensorKind::Input,
            initializer: None,
        });

        let present_key_id = tensors.add(TensorInfo {
            name: format!("present.{}.key", layer),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 1, 64, 128]),
            kind: TensorKind::Output,
            initializer: None,
        });

        let present_value_id = tensors.add(TensorInfo {
            name: format!("present.{}.value", layer),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 1, 64, 128]),
            kind: TensorKind::Output,
            initializer: None,
        });

        inputs.push(past_key_id);
        inputs.push(past_value_id);
        outputs.push(present_key_id);
        outputs.push(present_value_id);
    }

    // Register logits output
    let logits_id = tensors.add(TensorInfo {
        name: "logits".to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Static(vec![1, vocab_size]),
        kind: TensorKind::Output,
        initializer: None,
    });

    outputs.push(logits_id);

    ExecutionPlan {
        operations: Vec::new(), // No actual operations for this minimal plan
        shaders: Vec::new(),
        tensors,
        inputs,
        outputs,
        metadata: ModelMetadata {
            name: "minimal_llm_test".to_string(),
            version: 1,
            ir_version: 9,
            producer: "test".to_string(),
        },
    }
}

#[pollster::test]
#[ignore] // Requires GPU
async fn test_llm_session_creation() {
    let runtime = Runtime::new().await.expect("Failed to create runtime");

    let num_layers = 2;
    let vocab_size = 1000;
    let plan = create_minimal_llm_plan(num_layers, vocab_size);

    let executor = runtime
        .load_model(plan)
        .await
        .expect("Failed to load model");

    let config = LlmConfig {
        max_seq_len: 64,
        num_layers,
    };

    let session = LlmSession::new(executor, &config);

    // Verify initial state
    assert_eq!(session.sequence_length(), 0);
    assert!(session.kv_pairs().len() > 0, "Should have KV cache pairs");
}

#[pollster::test]
#[ignore] // Requires GPU
async fn test_llm_session_sequence_length_tracking() {
    let runtime = Runtime::new().await.expect("Failed to create runtime");

    let num_layers = 2;
    let vocab_size = 1000;
    let plan = create_minimal_llm_plan(num_layers, vocab_size);

    let executor = runtime
        .load_model(plan)
        .await
        .expect("Failed to load model");

    let config = LlmConfig {
        max_seq_len: 64,
        num_layers,
    };

    let mut session = LlmSession::new(executor, &config);

    // Initial sequence length should be 0
    assert_eq!(session.sequence_length(), 0);

    // Note: We can't actually run prefill because we have no compute operations.
    // This test verifies the session can be created and tracks state correctly.
    // In a real implementation with actual operations, we would test:
    // session.prefill(&[100, 200, 300]);
    // assert_eq!(session.sequence_length(), 3);
}

#[pollster::test]
#[ignore] // Requires GPU
async fn test_llm_session_reset() {
    let runtime = Runtime::new().await.expect("Failed to create runtime");

    let num_layers = 2;
    let vocab_size = 1000;
    let plan = create_minimal_llm_plan(num_layers, vocab_size);

    let executor = runtime
        .load_model(plan)
        .await
        .expect("Failed to load model");

    let config = LlmConfig {
        max_seq_len: 64,
        num_layers,
    };

    let mut session = LlmSession::new(executor, &config);

    // Initial state
    assert_eq!(session.sequence_length(), 0);

    // Reset on initial state should be a no-op
    session.reset();

    // Sequence length should still be 0
    assert_eq!(
        session.sequence_length(),
        0,
        "Reset should work on initial state"
    );

    // Note: In a real implementation with actual operations, we would test:
    // session.prefill(&[100, 200, 300]);
    // assert_eq!(session.sequence_length(), 3);
    // session.reset();
    // assert_eq!(session.sequence_length(), 0);
}

#[pollster::test]
#[ignore] // Requires GPU
async fn test_llm_session_kv_pairs_discovery() {
    let runtime = Runtime::new().await.expect("Failed to create runtime");

    let num_layers = 3;
    let vocab_size = 1000;
    let plan = create_minimal_llm_plan(num_layers, vocab_size);

    let executor = runtime
        .load_model(plan)
        .await
        .expect("Failed to load model");

    let config = LlmConfig {
        max_seq_len: 64,
        num_layers,
    };

    let session = LlmSession::new(executor, &config);

    // Verify KV pairs were discovered
    let kv_pairs = session.kv_pairs();
    assert!(
        kv_pairs.len() >= num_layers * 2,
        "Should have at least 2 pairs (key+value) per layer"
    );

    // Check that pairs follow expected naming pattern
    let has_expected_pattern = kv_pairs.iter().any(|(present, past): &(String, String)| {
        present.starts_with("present.") && past.starts_with("past_key_values.")
    });

    assert!(
        has_expected_pattern,
        "KV pairs should follow present.N.key/value → past_key_values.N.key/value pattern"
    );
}

#[test]
fn test_llm_config_creation() {
    let config = LlmConfig {
        max_seq_len: 2048,
        num_layers: 18,
    };

    assert_eq!(config.max_seq_len, 2048);
    assert_eq!(config.num_layers, 18);
}

#[test]
fn test_llm_config_clone() {
    let config = LlmConfig {
        max_seq_len: 1024,
        num_layers: 12,
    };

    let cloned = config.clone();

    assert_eq!(config.max_seq_len, cloned.max_seq_len);
    assert_eq!(config.num_layers, cloned.num_layers);
}

#[test]
fn test_minimal_plan_structure() {
    let plan = create_minimal_llm_plan(2, 1000);

    // Verify plan has expected inputs
    assert!(!plan.inputs.is_empty(), "Plan should have inputs");

    // Verify plan has expected outputs (logits + KV cache)
    assert!(!plan.outputs.is_empty(), "Plan should have outputs");

    // Verify we can find key tensors by name
    assert!(
        plan.tensors.find_by_name("input_ids").is_some(),
        "Should have input_ids tensor"
    );
    assert!(
        plan.tensors.find_by_name("logits").is_some(),
        "Should have logits tensor"
    );
    assert!(
        plan.tensors.find_by_name("present.0.key").is_some(),
        "Should have present.0.key tensor"
    );
    assert!(
        plan.tensors.find_by_name("past_key_values.0.key").is_some(),
        "Should have past_key_values.0.key tensor"
    );
}
