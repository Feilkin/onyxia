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

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // RotaryEmbedding inputs per ONNX spec:
        // 0: X - input tensor (required)
        //    - 4D: [batch_size, num_heads, sequence_length, head_size]
        //    - 3D: [batch_size, sequence_length, hidden_size]
        // 1: cos_cache (required)
        //    - 2D: [max_position_id_plus_1, head_size/2] when position_ids provided
        //    - 3D: [batch_size, sequence_length, head_size/2] when position_ids not provided
        // 2: sin_cache (required) - same shape as cos_cache
        // 3: position_ids (optional) - 2D: [batch_size, sequence_length] (I64)

        if ctx.input_count() < 3 {
            return Err(onyxia_core::Error::Planning(format!(
                "RotaryEmbedding requires at least 3 inputs (X, cos_cache, sin_cache), got {}",
                ctx.input_count()
            )));
        }

        // Get input shape and determine format (3D or 4D)
        let input_tensor = ctx.input_tensor(0)?;
        let input_shape = ctx.static_dims(&input_tensor.shape)?;

        // Read num_heads attribute (required for 3D input)
        let num_heads_attr = ctx.attr_i64("num_heads").unwrap_or(0);

        let (batch_size, seq_len, num_heads, head_dim) = match input_shape.len() {
            // 4D format: [batch_size, num_heads, sequence_length, head_size]
            4 => {
                let batch = input_shape[0];
                let heads = input_shape[1];
                let seq = input_shape[2];
                let head_size = input_shape[3];

                // Validate num_heads attribute if provided
                if num_heads_attr > 0 && heads != num_heads_attr as usize {
                    return Err(onyxia_core::Error::Planning(format!(
                        "RotaryEmbedding: num_heads attribute ({}) doesn't match input shape num_heads ({})",
                        num_heads_attr, heads
                    )));
                }

                (batch, seq, heads, head_size)
            }
            // 3D format: [batch_size, sequence_length, hidden_size]
            3 => {
                let batch = input_shape[0];
                let seq = input_shape[1];
                let hidden = input_shape[2];

                let heads = if num_heads_attr > 0 {
                    // num_heads provided as attribute
                    num_heads_attr as usize
                } else {
                    // Infer num_heads from cos_cache shape
                    // cos_cache is input 1 with shape [max_seq, head_dim/2]
                    let cos_cache_tensor = ctx.input_tensor(1)?;
                    let cos_cache_shape = ctx.static_dims(&cos_cache_tensor.shape)?;

                    if cos_cache_shape.len() != 2 {
                        return Err(onyxia_core::Error::Planning(format!(
                            "RotaryEmbedding: cos_cache should be 2D, got {:?}",
                            cos_cache_shape
                        )));
                    }

                    // Infer head_dim from cos_cache: head_dim = cos_cache.shape[1] * 2
                    let head_dim_inferred = cos_cache_shape[1] * 2;

                    if hidden % head_dim_inferred != 0 {
                        return Err(onyxia_core::Error::Planning(format!(
                            "RotaryEmbedding: hidden_size ({}) must be divisible by inferred head_dim ({})",
                            hidden, head_dim_inferred
                        )));
                    }

                    hidden / head_dim_inferred
                };

                if hidden % heads != 0 {
                    return Err(onyxia_core::Error::Planning(format!(
                        "RotaryEmbedding: hidden_size ({}) must be divisible by num_heads ({})",
                        hidden, heads
                    )));
                }

                let head_size = hidden / heads;
                (batch, seq, heads, head_size)
            }
            _ => {
                return Err(onyxia_core::Error::Planning(format!(
                    "RotaryEmbedding expects 3D [batch, seq, hidden] or 4D [batch, heads, seq, head_size] input, got {:?}",
                    input_shape
                )));
            }
        };

        if head_dim % 2 != 0 {
            return Err(onyxia_core::Error::Planning(format!(
                "RotaryEmbedding requires even head dimension, got {}",
                head_dim
            )));
        }

        // Check if position_ids is provided (input 3)
        let has_position_ids = ctx.input_count() >= 4;

        // Read interleaved attribute (default 0 for split-half)
        let interleaved: u32 = ctx.attr_i64("interleaved").unwrap_or(0) as u32;

        // Determine if input is 3D or 4D
        let is_3d_input = input_shape.len() == 3;

        // Calculate total number of pairs to rotate
        let total_pairs = batch_size * seq_len * num_heads * (head_dim / 2);

        let workgroup_size: u32 = 256;
        let num_workgroups = (total_pairs as u32).div_ceil(workgroup_size);

        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), workgroup_size.to_string());
        if has_position_ids {
            shader_defs.insert("HAS_POSITION_IDS".to_string(), "1".to_string());
        }
        if is_3d_input {
            shader_defs.insert("INPUT_3D".to_string(), "1".to_string());
        }

        let shader_index = ctx.compile_shader(
            "rotary_embedding",
            include_str!("../../shaders/attention/rotary_embedding.wgsl"),
            &shader_defs,
        )?;

        // Encode immediate constants
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(batch_size as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(seq_len as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(num_heads as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(head_dim as u32).to_le_bytes());
        immediates_data.extend_from_slice(&interleaved.to_le_bytes());
        immediates_data.extend_from_slice(&(head_dim as u32).to_le_bytes()); // rotation_dim = head_dim (full rotation)
        immediates_data.extend_from_slice(&1.0f32.to_le_bytes()); // scale = 1.0

        // Build binding list: input, cos_cache, sin_cache, [position_ids], output
        let mut bindings = vec![
            BindingDesc {
                buffer: ctx.input(0)?, // X (input)
                read_only: true,
            },
            BindingDesc {
                buffer: ctx.input(1)?, // cos_cache
                read_only: true,
            },
            BindingDesc {
                buffer: ctx.input(2)?, // sin_cache
                read_only: true,
            },
        ];

        if has_position_ids {
            bindings.push(BindingDesc {
                buffer: ctx.input(3)?, // position_ids (I64)
                read_only: true,
            });
        }

        bindings.push(BindingDesc {
            buffer: ctx.output(0)?, // output
            read_only: false,
        });

        Ok(vec![Step::Dispatch {
            shader_index,
            bindings,
            workgroups: [num_workgroups, 1, 1],
            immediates: Some(immediates_data),
        }])
    }
}

/// Microsoft Rotary Embedding operator (com.microsoft domain).
///
/// Similar to ai.onnx.RotaryEmbedding but with additional attributes and
/// required position_ids input.
pub struct MicrosoftRotaryEmbeddingOp;

impl Operator for MicrosoftRotaryEmbeddingOp {
    fn name(&self) -> &str {
        "com.microsoft.RotaryEmbedding"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        if ctx.input_count() < 4 {
            return Err(onyxia_core::Error::ShapeInference(
                "com.microsoft.RotaryEmbedding requires 4 inputs".to_string(),
            ));
        }
        // Output shape = input shape
        Ok(vec![ctx.input_shape(0)?.clone()])
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // com.microsoft.RotaryEmbedding inputs:
        // 0: input - 3D [batch, seq, hidden] or 4D [batch, num_heads, seq, head_size]
        // 1: position_ids - 1D [1] or 2D [batch, seq] (required, I64)
        // 2: cos_cache - 2D [max_seq, head_size/2] or [max_seq, rotary_embedding_dim/2]
        // 3: sin_cache - 2D [max_seq, head_size/2] or [max_seq, rotary_embedding_dim/2]

        if ctx.input_count() < 4 {
            return Err(onyxia_core::Error::Planning(format!(
                "com.microsoft.RotaryEmbedding requires 4 inputs, got {}",
                ctx.input_count()
            )));
        }

        // Get input shape and determine format (3D or 4D)
        let input_tensor = ctx.input_tensor(0)?;
        let input_shape = ctx.static_dims(&input_tensor.shape)?;
        let input_dtype = input_tensor.dtype; // Copy dtype before mutable borrows

        // Read attributes
        let num_heads_attr = ctx.attr_i64("num_heads").unwrap_or(0);
        let interleaved: u32 = ctx.attr_i64("interleaved").unwrap_or(0) as u32;
        let rotary_embedding_dim = ctx.attr_i64("rotary_embedding_dim").unwrap_or(0);
        let scale: f32 = ctx.attr_f32("scale").unwrap_or(1.0);
        let _is_packed_batching = ctx.attr_i64("is_packed_batching").unwrap_or(0);

        let (batch_size, seq_len, num_heads, head_dim) = match input_shape.len() {
            // 4D format: [batch_size, num_heads, sequence_length, head_size]
            4 => {
                let batch = input_shape[0];
                let heads = input_shape[1];
                let seq = input_shape[2];
                let head_size = input_shape[3];

                if num_heads_attr > 0 && heads != num_heads_attr as usize {
                    return Err(onyxia_core::Error::Planning(format!(
                        "com.microsoft.RotaryEmbedding: num_heads attribute ({}) doesn't match input shape ({})",
                        num_heads_attr, heads
                    )));
                }

                (batch, seq, heads, head_size)
            }
            // 3D format: [batch_size, sequence_length, hidden_size]
            3 => {
                let batch = input_shape[0];
                let seq = input_shape[1];
                let hidden = input_shape[2];

                let heads = if num_heads_attr > 0 {
                    // num_heads provided as attribute
                    num_heads_attr as usize
                } else {
                    // Infer num_heads from cos_cache shape
                    // cos_cache is input 2 with shape [max_seq, head_dim/2] or [max_seq, rotary_embedding_dim/2]
                    let cos_cache_tensor = ctx.input_tensor(2)?;
                    let cos_cache_shape = ctx.static_dims(&cos_cache_tensor.shape)?;

                    if cos_cache_shape.len() != 2 {
                        return Err(onyxia_core::Error::Planning(format!(
                            "com.microsoft.RotaryEmbedding: cos_cache should be 2D, got {:?}",
                            cos_cache_shape
                        )));
                    }

                    // If rotary_embedding_dim is specified, use it; otherwise infer from cos_cache
                    let effective_dim = if rotary_embedding_dim > 0 {
                        rotary_embedding_dim as usize
                    } else {
                        // Infer from cos_cache: dim = cos_cache.shape[1] * 2
                        cos_cache_shape[1] * 2
                    };

                    if hidden % effective_dim != 0 {
                        return Err(onyxia_core::Error::Planning(format!(
                            "com.microsoft.RotaryEmbedding: hidden_size ({}) must be divisible by effective_dim ({})",
                            hidden, effective_dim
                        )));
                    }

                    hidden / effective_dim
                };

                if hidden % heads != 0 {
                    return Err(onyxia_core::Error::Planning(format!(
                        "com.microsoft.RotaryEmbedding: hidden_size ({}) must be divisible by num_heads ({})",
                        hidden, heads
                    )));
                }

                let head_size = hidden / heads;
                (batch, seq, heads, head_size)
            }
            _ => {
                return Err(onyxia_core::Error::Planning(format!(
                    "com.microsoft.RotaryEmbedding expects 3D or 4D input, got {:?}",
                    input_shape
                )));
            }
        };

        // Determine effective rotation dimension
        let rotation_dim = if rotary_embedding_dim > 0 {
            rotary_embedding_dim as usize
        } else {
            head_dim
        };

        if rotation_dim % 2 != 0 {
            return Err(onyxia_core::Error::Planning(format!(
                "com.microsoft.RotaryEmbedding requires even rotation dimension, got {}",
                rotation_dim
            )));
        }

        let is_3d_input = input_shape.len() == 3;

        // Calculate total number of pairs to rotate
        let total_pairs = batch_size * seq_len * num_heads * (rotation_dim / 2);

        let workgroup_size: u32 = 256;
        let num_workgroups = (total_pairs as u32).div_ceil(workgroup_size);

        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), workgroup_size.to_string());
        shader_defs.insert("HAS_POSITION_IDS".to_string(), "1".to_string());
        if is_3d_input {
            shader_defs.insert("INPUT_3D".to_string(), "1".to_string());
        }
        if rotation_dim < head_dim {
            shader_defs.insert("PARTIAL_ROTATION".to_string(), "1".to_string());
        }

        let shader_index = ctx.compile_shader(
            "microsoft_rotary_embedding",
            include_str!("../../shaders/attention/rotary_embedding.wgsl"),
            &shader_defs,
        )?;

        // Encode immediate constants
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(batch_size as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(seq_len as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(num_heads as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(head_dim as u32).to_le_bytes());
        immediates_data.extend_from_slice(&interleaved.to_le_bytes());
        immediates_data.extend_from_slice(&(rotation_dim as u32).to_le_bytes());
        immediates_data.extend_from_slice(&scale.to_le_bytes());

        let mut steps = Vec::new();

        // If partial rotation, first copy input to output
        if rotation_dim < head_dim {
            let element_count: usize = input_shape.iter().product();
            let dtype_bytes = match input_dtype {
                onyxia_core::DataType::F32
                | onyxia_core::DataType::I32
                | onyxia_core::DataType::U32
                | onyxia_core::DataType::Bool => 4,
                onyxia_core::DataType::F16 => 2,
                onyxia_core::DataType::I64 => 8,
                onyxia_core::DataType::U8
                | onyxia_core::DataType::Q4
                | onyxia_core::DataType::Q8 => 1,
            };
            let bytes = element_count * dtype_bytes;
            steps.push(Step::CopyBuffer {
                src: ctx.input(0)?,
                src_offset: 0,
                dst: ctx.output(0)?,
                dst_offset: 0,
                size: bytes as u64,
            });
        }

        steps.push(Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0)?, // input
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(1)?, // position_ids (I64, required)
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(2)?, // cos_cache
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(3)?, // sin_cache
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(0)?, // output
                    read_only: false,
                },
            ],
            workgroups: [num_workgroups, 1, 1],
            immediates: Some(immediates_data),
        });

        Ok(steps)
    }
}

/// Gemma Rotary Embedding operator (com.microsoft domain).
///
/// This is a specialized RoPE implementation used in Google Gemma models.
/// It computes sin/cos from the embedding input and applies rotation to q and k tensors.
///
/// Formula:
/// - sin_val = Sin(emb), cos_val = Cos(emb)
/// - q_embed = (q * cos_val) + (q_rot * sin_val)
/// - k_embed = (k * cos_val) + (k_rot * sin_val)
pub struct GemmaRotaryEmbeddingOp;

impl Operator for GemmaRotaryEmbeddingOp {
    fn name(&self) -> &str {
        "com.microsoft.GemmaRotaryEmbedding"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        if ctx.input_count() < 5 {
            return Err(onyxia_core::Error::ShapeInference(
                "GemmaRotaryEmbedding requires 5 inputs".to_string(),
            ));
        }
        // Outputs have same shape as q and k inputs (inputs 1 and 3)
        Ok(vec![
            ctx.input_shape(1)?.clone(), // q_embed shape = q shape
            ctx.input_shape(3)?.clone(), // k_embed shape = k shape
        ])
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // GemmaRotaryEmbedding inputs per ONNX spec:
        // 0: emb - [batch_size, seq_len, dim] (F32)
        // 1: q - [batch_size, num_heads, seq_len, dim] (F16)
        // 2: q_rot - [batch_size, num_heads, seq_len, dim] (F16) - half rotated q
        // 3: k - [batch_size, num_heads, seq_len, dim] (F16)
        // 4: k_rot - [batch_size, num_heads, seq_len, dim] (F16) - half rotated k

        if ctx.input_count() < 5 {
            return Err(onyxia_core::Error::Planning(format!(
                "GemmaRotaryEmbedding requires 5 inputs, got {}",
                ctx.input_count()
            )));
        }

        // Get shapes
        let emb_tensor = ctx.input_tensor(0)?;
        let emb_shape = ctx.static_dims(&emb_tensor.shape)?;

        let q_tensor = ctx.input_tensor(1)?;
        let q_shape = ctx.static_dims(&q_tensor.shape)?;

        // Validate shapes
        if emb_shape.len() != 3 {
            return Err(onyxia_core::Error::Planning(format!(
                "GemmaRotaryEmbedding emb expects 3D [batch, seq, dim], got {:?}",
                emb_shape
            )));
        }

        if q_shape.len() != 4 {
            return Err(onyxia_core::Error::Planning(format!(
                "GemmaRotaryEmbedding q expects 4D [batch, num_heads, seq, dim], got {:?}",
                q_shape
            )));
        }

        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let seq_len = q_shape[2];
        let dim = q_shape[3];

        // Total elements to process (one thread per element in q/k)
        let total_elements = batch_size * num_heads * seq_len * dim;

        let workgroup_size: u32 = 256;
        let num_workgroups = (total_elements as u32).div_ceil(workgroup_size);

        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), workgroup_size.to_string());

        let shader_index = ctx.compile_shader(
            "gemma_rotary_embedding",
            include_str!("../../shaders/attention/gemma_rotary_embedding.wgsl"),
            &shader_defs,
        )?;

        // Encode immediate constants
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(batch_size as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(num_heads as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(seq_len as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(dim as u32).to_le_bytes());

        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0)?, // emb
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(1)?, // q
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(2)?, // q_rot
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(3)?, // k
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(4)?, // k_rot
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(0)?, // q_embed
                    read_only: false,
                },
                BindingDesc {
                    buffer: ctx.output(1)?, // k_embed
                    read_only: false,
                },
            ],
            workgroups: [num_workgroups, 1, 1],
            immediates: Some(immediates_data),
        }])
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
            workgroups: [(update_key_elements as u32).div_ceil(workgroup_size), 1, 1],
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
                (update_value_elements as u32).div_ceil(workgroup_size),
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
            workgroups: [(scores_elements as u32).div_ceil(workgroup_size), 1, 1],
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
            workgroups: [(softmax_rows as u32).div_ceil(workgroup_size), 1, 1],
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
            workgroups: [(output_elements as u32).div_ceil(workgroup_size), 1, 1],
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
