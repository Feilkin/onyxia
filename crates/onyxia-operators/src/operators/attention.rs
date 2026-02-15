//! Attention mechanism operators.

use onyxia_core::{BindingDesc, InferenceCtx, Operator, PlanCtx, Result, Step, TensorShape};
use std::collections::HashMap;

/// Rotary positional embedding operator.
///
/// Used in transformer attention to inject positional information.
pub struct RotaryEmbeddingOp;

impl Operator for RotaryEmbeddingOp {
    fn name(&self) -> &str {
        "RotaryEmbedding"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        if ctx.input_count() == 0 {
            return Err(onyxia_core::Error::ShapeInference(
                "RotaryEmbedding requires at least one input".to_string(),
            ));
        }
        // Output shape = input shape (same tensor with rotated values)
        Ok(vec![ctx.input_shape(0)?.clone()])
    }

    fn plan(&self, _ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // TODO: Implement RotaryEmbedding GPU planning
        Err(onyxia_core::Error::Planning(
            "RotaryEmbedding GPU planning not yet implemented".to_string(),
        ))
    }
}

/// Grouped Query Attention operator.
///
/// Implements multi-head attention with grouped query attention (GQA) where multiple
/// query heads share the same key/value head, reducing KV cache memory.
///
/// Inputs:
/// 0. query: [batch, seq_len, num_heads * head_dim]
/// 1. key: [batch, seq_len, kv_num_heads * head_dim]
/// 2. value: [batch, seq_len, kv_num_heads * head_dim]
/// 3. past_key: [batch, kv_num_heads, past_seq_len, head_dim] (KV cache)
/// 4. past_value: [batch, kv_num_heads, past_seq_len, head_dim]
/// 5. seqlens_k: [batch] I32 (total sequence lengths)
/// 6. total_sequence_length: scalar I32
/// 7. cos_cache: RoPE cache (may be empty)
/// 8. sin_cache: RoPE cache (may be empty)
///
/// Outputs:
/// 0. output: [batch, seq_len, num_heads * head_dim]
/// 1. present_key: [batch, kv_num_heads, total_seq_len, head_dim]
/// 2. present_value: [batch, kv_num_heads, total_seq_len, head_dim]
pub struct GroupQueryAttentionOp;

impl Operator for GroupQueryAttentionOp {
    fn name(&self) -> &str {
        "GroupQueryAttention"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        if ctx.input_count() < 7 {
            return Err(onyxia_core::Error::ShapeInference(
                "GroupQueryAttention requires at least 7 inputs".to_string(),
            ));
        }

        // Output 0: attention output, same shape as query input
        let output_shape = ctx.input_shape(0)?.clone();

        // Outputs 1 and 2: present_key and present_value
        // In buffer-sharing mode, these have the SAME shape as past_key/past_value
        let present_kv_shape = ctx.input_shape(3)?.clone(); // Same as past_key shape

        Ok(vec![
            output_shape,
            present_kv_shape.clone(),
            present_kv_shape,
        ])
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // Read attributes
        let num_heads: i64 = ctx.attr_i64("num_heads")?;
        let kv_num_heads: i64 = ctx.attr_i64("kv_num_heads")?;

        // Read optional attributes with defaults per ONNX spec
        let local_window_size: i64 = ctx.attr_i64("local_window_size").unwrap_or(-1);
        let softcap: f32 = ctx.attr_f32("softcap").unwrap_or(0.0);

        // Get query shape: [batch, seq_len, num_heads * head_dim]
        let query_tensor = ctx.input_tensor(0)?;
        let query_shape = ctx.static_dims(&query_tensor.shape)?;

        if query_shape.len() != 3 {
            return Err(onyxia_core::Error::Planning(format!(
                "GroupQueryAttention expects query shape [batch, seq_len, num_heads*head_dim], got {:?}",
                query_shape
            )));
        }

        let batch_size = query_shape[0];
        let seq_len = query_shape[1];
        let query_hidden = query_shape[2];

        // Calculate head_dim
        let head_dim = query_hidden / num_heads as usize;

        // Validate dimensions
        if query_hidden != (num_heads as usize) * head_dim {
            return Err(onyxia_core::Error::Planning(format!(
                "Query hidden dimension {} is not divisible by num_heads {}",
                query_hidden, num_heads
            )));
        }

        // Get past_key shape: [batch, kv_num_heads, max_seq_len, head_dim]
        let past_key_tensor = ctx.input_tensor(3)?;
        let past_key_shape = ctx.static_dims(&past_key_tensor.shape)?;

        if past_key_shape.len() != 4 {
            return Err(onyxia_core::Error::Planning(format!(
                "GroupQueryAttention expects past_key shape [batch, kv_heads, max_seq, head_dim], got {:?}",
                past_key_shape
            )));
        }

        let max_seq_len = past_key_shape[2];

        // Calculate scale (default: 1 / sqrt(head_dim))
        let scale: f32 = ctx
            .attr_f32("scale")
            .unwrap_or(1.0 / (head_dim as f32).sqrt());

        let workgroup_size: u32 = 256;

        // Step 1: Update past_key with current key → present_key
        let update_key_elements = batch_size * (kv_num_heads as usize) * max_seq_len * head_dim;
        let update_key_shader = ctx.compile_shader(
            "gqa_update_kv",
            include_str!("../../shaders/attention/gqa_update_kv.wgsl"),
            &HashMap::from([("WORKGROUP_SIZE".to_string(), workgroup_size.to_string())]),
        )?;

        let past_seq_len = 0; // TODO: Read from seqlens_k at runtime

        let mut update_key_immediates = Vec::new();
        update_key_immediates.extend_from_slice(&(batch_size as u32).to_le_bytes());
        update_key_immediates.extend_from_slice(&(seq_len as u32).to_le_bytes());
        update_key_immediates.extend_from_slice(&(past_seq_len as u32).to_le_bytes());
        update_key_immediates.extend_from_slice(&(max_seq_len as u32).to_le_bytes());
        update_key_immediates.extend_from_slice(&(kv_num_heads as u32).to_le_bytes());
        update_key_immediates.extend_from_slice(&(head_dim as u32).to_le_bytes());

        let update_key_step = Step::Dispatch {
            shader_index: update_key_shader,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(3)?, // past_key
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(1)?, // current key
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(1)?, // present_key
                    read_only: false,
                },
            ],
            workgroups: [
                (update_key_elements as u32 + workgroup_size - 1) / workgroup_size,
                1,
                1,
            ],
            immediates: Some(update_key_immediates),
        };

        // Step 2: Update past_value with current value → present_value
        let update_value_elements = batch_size * (kv_num_heads as usize) * max_seq_len * head_dim;
        let update_value_shader = update_key_shader; // Reuse same shader

        let mut update_value_immediates = Vec::new();
        update_value_immediates.extend_from_slice(&(batch_size as u32).to_le_bytes());
        update_value_immediates.extend_from_slice(&(seq_len as u32).to_le_bytes());
        update_value_immediates.extend_from_slice(&(past_seq_len as u32).to_le_bytes());
        update_value_immediates.extend_from_slice(&(max_seq_len as u32).to_le_bytes());
        update_value_immediates.extend_from_slice(&(kv_num_heads as u32).to_le_bytes());
        update_value_immediates.extend_from_slice(&(head_dim as u32).to_le_bytes());

        let update_value_step = Step::Dispatch {
            shader_index: update_value_shader,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(4)?, // past_value
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(2)?, // current value
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(2)?, // present_value
                    read_only: false,
                },
            ],
            workgroups: [
                (update_value_elements as u32 + workgroup_size - 1) / workgroup_size,
                1,
                1,
            ],
            immediates: Some(update_value_immediates),
        };

        let total_seq_len = max_seq_len;

        // Step 3: Compute attention scores (Q @ K^T * scale)
        let scores_elements = batch_size * (num_heads as usize) * seq_len * total_seq_len;
        let scores_size = (scores_elements * std::mem::size_of::<f32>()) as u64;

        // Allocate scratch buffer for attention scores
        let scores_buffer = ctx.alloc_scratch(scores_size, "gqa_scores".to_string());

        let scores_shader = ctx.compile_shader(
            "gqa_scores",
            include_str!("../../shaders/attention/gqa_scores.wgsl"),
            &HashMap::from([("WORKGROUP_SIZE".to_string(), workgroup_size.to_string())]),
        )?;

        let mut scores_immediates = Vec::new();
        scores_immediates.extend_from_slice(&(batch_size as u32).to_le_bytes());
        scores_immediates.extend_from_slice(&(seq_len as u32).to_le_bytes());
        scores_immediates.extend_from_slice(&(total_seq_len as u32).to_le_bytes());
        scores_immediates.extend_from_slice(&(num_heads as u32).to_le_bytes());
        scores_immediates.extend_from_slice(&(kv_num_heads as u32).to_le_bytes());
        scores_immediates.extend_from_slice(&(head_dim as u32).to_le_bytes());
        scores_immediates.extend_from_slice(&scale.to_le_bytes());
        scores_immediates.extend_from_slice(&softcap.to_le_bytes());

        let scores_step = Step::Dispatch {
            shader_index: scores_shader,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0)?, // query
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(1)?, // present_key
                    read_only: true,
                },
                BindingDesc {
                    buffer: scores_buffer,
                    read_only: false,
                },
            ],
            workgroups: [
                (scores_elements as u32 + workgroup_size - 1) / workgroup_size,
                1,
                1,
            ],
            immediates: Some(scores_immediates),
        };

        // Step 4: Apply causal mask, sliding window, and softmax
        let softmax_rows = batch_size * (num_heads as usize) * seq_len;
        let softmax_shader = ctx.compile_shader(
            "gqa_softmax",
            include_str!("../../shaders/attention/gqa_softmax.wgsl"),
            &HashMap::from([("WORKGROUP_SIZE".to_string(), workgroup_size.to_string())]),
        )?;

        let mut softmax_immediates = Vec::new();
        softmax_immediates.extend_from_slice(&(batch_size as u32).to_le_bytes());
        softmax_immediates.extend_from_slice(&(seq_len as u32).to_le_bytes());
        softmax_immediates.extend_from_slice(&(past_seq_len as u32).to_le_bytes());
        softmax_immediates.extend_from_slice(&(total_seq_len as u32).to_le_bytes());
        softmax_immediates.extend_from_slice(&(num_heads as u32).to_le_bytes());
        softmax_immediates.extend_from_slice(&(local_window_size as i32).to_le_bytes());

        let softmax_step = Step::Dispatch {
            shader_index: softmax_shader,
            bindings: vec![BindingDesc {
                buffer: scores_buffer,
                read_only: false,
            }],
            workgroups: [
                (softmax_rows as u32 + workgroup_size - 1) / workgroup_size,
                1,
                1,
            ],
            immediates: Some(softmax_immediates),
        };

        // Step 5: Compute output (attn_weights @ V)
        let output_elements = batch_size * (num_heads as usize) * seq_len * head_dim;
        let output_shader = ctx.compile_shader(
            "gqa_output",
            include_str!("../../shaders/attention/gqa_output.wgsl"),
            &HashMap::from([("WORKGROUP_SIZE".to_string(), workgroup_size.to_string())]),
        )?;

        let mut output_immediates = Vec::new();
        output_immediates.extend_from_slice(&(batch_size as u32).to_le_bytes());
        output_immediates.extend_from_slice(&(seq_len as u32).to_le_bytes());
        output_immediates.extend_from_slice(&(total_seq_len as u32).to_le_bytes());
        output_immediates.extend_from_slice(&(num_heads as u32).to_le_bytes());
        output_immediates.extend_from_slice(&(kv_num_heads as u32).to_le_bytes());
        output_immediates.extend_from_slice(&(head_dim as u32).to_le_bytes());

        let output_step = Step::Dispatch {
            shader_index: output_shader,
            bindings: vec![
                BindingDesc {
                    buffer: scores_buffer,
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(2)?, // present_value
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(0)?, // output
                    read_only: false,
                },
            ],
            workgroups: [
                (output_elements as u32 + workgroup_size - 1) / workgroup_size,
                1,
                1,
            ],
            immediates: Some(output_immediates),
        };

        // Return all steps
        Ok(vec![
            update_key_step,
            update_value_step,
            scores_step,
            softmax_step,
            output_step,
        ])
    }
}
