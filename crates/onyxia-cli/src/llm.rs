//! LLM-specific session management with KV cache support.

use anyhow::{Context, Result};
use onyxia_runtime::{PlanExecutor, Tensor};

/// Configuration for LLM session.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// Maximum sequence length (buffers pre-allocated to this size).
    pub max_seq_len: usize,
    /// Number of transformer layers (for KV cache discovery).
    pub num_layers: usize,
}

/// High-level LLM inference session with KV cache management.
///
/// Wraps a `PlanExecutor` and manages KV cache aliasing, sequence length
/// tracking, and the prefill/decode workflow. All LLM-specific logic lives
/// here, keeping the runtime generic.
pub struct LlmSession {
    executor: PlanExecutor,
    /// Current past sequence length (grows each decode step).
    past_seq_len: usize,
    /// Maximum sequence length (buffers pre-allocated to this).
    max_seq_len: usize,
    /// KV cache tensor name pairs: (present_output_name, past_input_name).
    kv_pairs: Vec<(String, String)>,
}

impl LlmSession {
    /// Create from a loaded PlanExecutor + model config.
    ///
    /// Discovers KV cache pairs by scanning tensor names in the execution plan.
    pub fn new(executor: PlanExecutor, config: &LlmConfig) -> Self {
        let kv_pairs = discover_kv_pairs(&executor);

        Self {
            executor,
            past_seq_len: 0,
            max_seq_len: config.max_seq_len,
            kv_pairs,
        }
    }

    /// Prefill: process the full prompt.
    ///
    /// - Passes past_sequence_length = 0 via model inputs
    /// - Runs with seq_len = prompt_len
    /// - After run, aliases present.* → past_key_values.* for next step
    /// - Returns logits [vocab_size] for last position
    pub fn prefill(&mut self, input_ids: &[i64]) -> Result<Vec<f32>> {
        let prompt_len = input_ids.len();

        if prompt_len == 0 {
            anyhow::bail!("Cannot prefill with empty input_ids");
        }

        if prompt_len > self.max_seq_len {
            anyhow::bail!(
                "Input length {} exceeds max_seq_len {}",
                prompt_len,
                self.max_seq_len
            );
        }

        // Create input tensors (non-borrowing)
        let inputs = create_prefill_inputs(input_ids, 0);

        // Run only downloading the logits output
        let outputs = self
            .executor
            .run_with_outputs(&inputs, &["logits"])
            .with_context(|| {
                format!(
                    "Prefill execution failed (prompt_len={}, inputs={:?})",
                    prompt_len,
                    inputs
                        .iter()
                        .map(|(name, tensor)| { format!("{}:{:?}", name, tensor.shape()) })
                        .collect::<Vec<_>>()
                )
            })?;

        // Copy present.* → past_key_values.* for next decode step
        self.executor
            .copy_tensors(&self.kv_pairs)
            .context("Failed to copy KV cache after prefill")?;

        // Update sequence length
        self.past_seq_len = prompt_len;

        // Extract logits
        let logits = outputs
            .get("logits")
            .context("Missing logits output")?
            .to_vec::<f32>()
            .context("Failed to convert logits to f32")?;

        Ok(logits)
    }

    /// Decode: generate one next-token logit vector.
    ///
    /// - Increments past_sequence_length
    /// - Runs with seq_len = 1
    /// - After run, aliases present.* → past_key_values.*
    /// - Returns logits [vocab_size]
    pub fn decode(&mut self, token_id: i64) -> Result<Vec<f32>> {
        if self.past_seq_len >= self.max_seq_len {
            anyhow::bail!(
                "Sequence length {} would exceed max_seq_len {}",
                self.past_seq_len + 1,
                self.max_seq_len
            );
        }

        // Create input tensors (non-borrowing)
        let inputs = create_decode_inputs(token_id, self.past_seq_len);

        // Run only downloading the logits output
        let outputs = self
            .executor
            .run_with_outputs(&inputs, &["logits"])
            .with_context(|| {
                format!(
                    "Decode execution failed (token={}, past_seq_len={}, inputs={:?})",
                    token_id,
                    self.past_seq_len,
                    inputs
                        .iter()
                        .map(|(name, tensor)| { format!("{}:{:?}", name, tensor.shape()) })
                        .collect::<Vec<_>>()
                )
            })?;

        // Copy present.* → past_key_values.* for next decode step
        self.executor
            .copy_tensors(&self.kv_pairs)
            .context("Failed to copy KV cache after decode")?;

        // Update sequence length
        self.past_seq_len += 1;

        // Extract logits
        let logits = outputs
            .get("logits")
            .context("Missing logits output")?
            .to_vec::<f32>()
            .context("Failed to convert logits to f32")?;

        Ok(logits)
    }

    /// Reset state for a new conversation.
    pub fn reset(&mut self) {
        self.past_seq_len = 0;
    }

    /// Get current sequence length.
    pub fn sequence_length(&self) -> usize {
        self.past_seq_len
    }

    /// Get KV cache pairs (for testing).
    pub fn kv_pairs(&self) -> &[(String, String)] {
        &self.kv_pairs
    }
}

/// Create input tensors for prefill phase.
fn create_prefill_inputs(input_ids: &[i64], _past_seq_len: usize) -> Vec<(&'static str, Tensor)> {
    let prompt_len = input_ids.len();

    vec![
        (
            "input_ids",
            Tensor::from_vec(input_ids.to_vec(), &[1, prompt_len]),
        ),
        (
            "attention_mask",
            Tensor::from_vec(vec![1i64; prompt_len], &[1, prompt_len]),
        ),
        (
            "position_ids",
            Tensor::from_vec((0..prompt_len as i64).collect::<Vec<_>>(), &[1, prompt_len]),
        ),
    ]
}

/// Create input tensors for decode phase.
fn create_decode_inputs(token_id: i64, past_seq_len: usize) -> Vec<(&'static str, Tensor)> {
    vec![
        ("input_ids", Tensor::from_vec(vec![token_id], &[1, 1])),
        ("attention_mask", Tensor::from_vec(vec![1i64], &[1, 1])),
        (
            "position_ids",
            Tensor::from_vec(vec![past_seq_len as i64], &[1, 1]),
        ),
    ]
}

/// Discover KV cache tensor pairs by scanning plan tensor names.
///
/// Matches patterns like:
/// - Output: `present.{N}.key`, `present.{N}.value`
/// - Input: `past_key_values.{N}.key`, `past_key_values.{N}.value`
fn discover_kv_pairs(executor: &PlanExecutor) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    let plan = executor.plan();

    // Scan all tensors to find present.* outputs
    for tensor in plan.tensors.all() {
        let name = &tensor.name;

        // Look for tensors with pattern "present.{layer}.{type}"
        if name.starts_with("present.") {
            // Extract the layer number and type (key or value)
            // Example: "present.0.key" -> layer=0, type="key"
            let parts: Vec<&str> = name.split('.').collect();
            if parts.len() == 3 && parts[0] == "present" {
                let layer = parts[1];
                let kv_type = parts[2]; // "key" or "value"

                // Construct the corresponding past_key_values name
                let past_name = format!("past_key_values.{}.{}", layer, kv_type);

                // Verify that the past tensor exists in the plan
                if plan.tensors.find_by_name(&past_name).is_some() {
                    pairs.push((name.clone(), past_name));
                }
            }
        }
    }

    pairs
}
