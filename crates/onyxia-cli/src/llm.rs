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
/// Wraps a `PlanExecutor` and manages KV cache via explicit CPU-side storage.
/// After each forward pass, present.* outputs are downloaded and stored,
/// then uploaded as past_key_values.* inputs on the next call.
pub struct LlmSession {
    executor: PlanExecutor,
    /// Current past sequence length (grows each decode step).
    past_seq_len: usize,
    /// Maximum sequence length (buffers pre-allocated to this).
    max_seq_len: usize,
    /// KV cache tensor name pairs: (present_output_name, past_input_name).
    kv_pairs: Vec<(String, String)>,
    /// Stored KV cache tensors (downloaded from present.* after each forward pass).
    /// Keyed by past_key_values.*.* name.
    kv_cache: std::collections::HashMap<String, Tensor>,
}

impl LlmSession {
    /// Create from a loaded PlanExecutor + model config.
    ///
    /// Discovers KV cache pairs by scanning tensor names in the execution plan
    /// and initializes them with zeros.
    pub fn new(executor: PlanExecutor, config: &LlmConfig) -> Self {
        let kv_pairs = discover_kv_pairs(&executor);

        // Initialize KV cache with zeros
        let mut kv_cache = std::collections::HashMap::new();
        let plan = executor.plan();

        for (_present_name, past_name) in &kv_pairs {
            if let Some((_, tensor_info)) = plan.tensors.find_by_name(past_name) {
                // Extract static shape
                if let onyxia_core::TensorShape::Static(dims) = &tensor_info.shape {
                    let size: usize = dims.iter().product();

                    // Create zero-filled tensor matching the dtype
                    let tensor = match tensor_info.dtype {
                        onyxia_core::DataType::F32 => Tensor::from_vec(vec![0.0f32; size], dims),
                        onyxia_core::DataType::F16 => {
                            // F16 zeros as u16 values (0x0000 = 0.0 in half precision)
                            Tensor::from_vec(vec![0u16; size], dims)
                        }
                        _ => {
                            panic!("Unsupported KV cache dtype: {:?}", tensor_info.dtype);
                        }
                    };

                    kv_cache.insert(past_name.clone(), tensor);
                }
            }
        }

        Self {
            executor,
            past_seq_len: 0,
            max_seq_len: config.max_seq_len,
            kv_pairs,
            kv_cache,
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

        // Create input tensors including KV caches
        let base_inputs = create_prefill_inputs(input_ids, 0);

        // Collect KV cache tensors (need owned strings for lifetime)
        let kv_input_data: Vec<(String, Tensor)> = self
            .kv_pairs
            .iter()
            .filter_map(|(_present_name, past_name)| {
                self.kv_cache
                    .get(past_name)
                    .map(|t| (past_name.clone(), t.clone()))
            })
            .collect();

        // Combine into one input slice
        let mut inputs: Vec<(&str, Tensor)> = base_inputs;
        for (name, tensor) in &kv_input_data {
            inputs.push((name.as_str(), tensor.clone()));
        }

        // Run and download both logits and present.* outputs
        let output_names: Vec<&str> = std::iter::once("logits")
            .chain(
                self.kv_pairs
                    .iter()
                    .map(|(present_name, _)| present_name.as_str()),
            )
            .collect();

        let outputs = self
            .executor
            .run_with_outputs(&inputs, &output_names)
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

        // Download and store present.* outputs as KV cache for next step
        for (present_name, past_name) in &self.kv_pairs {
            if let Some(tensor) = outputs.get(present_name) {
                self.kv_cache.insert(past_name.clone(), tensor.clone());
            }
        }

        // Update sequence length
        self.past_seq_len = prompt_len;

        // Extract logits for the last position only
        // Logits shape is [batch=1, seq_len=max_seq_len, vocab_size]
        // We need logits[0, prompt_len-1, :] (last token of the prompt)
        let logits_tensor = outputs.get("logits").context("Missing logits output")?;
        let logits_full = logits_tensor
            .to_vec::<f32>()
            .context("Failed to convert logits to f32")?;

        // Extract the vocab_size from the shape
        let shape = logits_tensor.shape();
        if shape.len() != 3 || shape[0] != 1 {
            anyhow::bail!(
                "Expected logits shape [1, seq_len, vocab_size], got {:?}",
                shape
            );
        }

        let vocab_size = shape[2];

        // Calculate offset to last valid position: [0, prompt_len-1, :]
        let last_pos = prompt_len - 1;
        let offset = last_pos * vocab_size;
        let logits = logits_full[offset..offset + vocab_size].to_vec();

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

        // Create input tensors including KV caches
        let base_inputs = create_decode_inputs(token_id, self.past_seq_len);

        // Collect KV cache tensors (need owned strings for lifetime)
        let kv_input_data: Vec<(String, Tensor)> = self
            .kv_pairs
            .iter()
            .filter_map(|(_present_name, past_name)| {
                self.kv_cache
                    .get(past_name)
                    .map(|t| (past_name.clone(), t.clone()))
            })
            .collect();

        // Combine into one input slice
        let mut inputs: Vec<(&str, Tensor)> = base_inputs;
        for (name, tensor) in &kv_input_data {
            inputs.push((name.as_str(), tensor.clone()));
        }

        // Run and download both logits and present.* outputs
        let output_names: Vec<&str> = std::iter::once("logits")
            .chain(
                self.kv_pairs
                    .iter()
                    .map(|(present_name, _)| present_name.as_str()),
            )
            .collect();

        let outputs = self
            .executor
            .run_with_outputs(&inputs, &output_names)
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

        // Download and store present.* outputs as KV cache for next step
        for (present_name, past_name) in &self.kv_pairs {
            if let Some(tensor) = outputs.get(present_name) {
                self.kv_cache.insert(past_name.clone(), tensor.clone());
            }
        }

        // Update sequence length
        self.past_seq_len += 1;

        // Extract logits for the current position (position 0 in decode mode)
        // Logits shape is [batch=1, seq_len=1, vocab_size] in decode mode
        let logits_tensor = outputs.get("logits").context("Missing logits output")?;
        let logits_full = logits_tensor
            .to_vec::<f32>()
            .context("Failed to convert logits to f32")?;

        // Extract the vocab_size from the shape
        let shape = logits_tensor.shape();
        if shape.len() != 3 || shape[0] != 1 {
            anyhow::bail!(
                "Expected logits shape [1, seq_len, vocab_size], got {:?}",
                shape
            );
        }

        let vocab_size = shape[2];

        // For decode with seq_len=1, we want position 0: [0, 0, :]
        let offset = 0;
        let logits = logits_full[offset..offset + vocab_size].to_vec();

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
    // total_sequence_length = past_seq_len (already generated) + 1 (current token)
    let total_seq_len = past_seq_len + 1;

    vec![
        ("input_ids", Tensor::from_vec(vec![token_id], &[1, 1])),
        // attention_mask covers all tokens: past + current
        (
            "attention_mask",
            Tensor::from_vec(vec![1i64; total_seq_len], &[1, total_seq_len]),
        ),
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
