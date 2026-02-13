//! GroupQueryAttentionKernel implementation for fused multi-head attention with KV cache.

use crate::error::{CodegenError, Result};
use crate::inference::InferenceContext;
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::{BindingDesc, ScratchBufferDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::TensorShape;
use std::collections::HashMap;

/// Kernel for GroupQueryAttention (Microsoft contrib op).
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
///
/// Attributes:
/// - num_heads: number of query attention heads
/// - kv_num_heads: number of key/value heads (GQA)
/// - scale: optional attention scale (default: 1/sqrt(head_dim))
pub struct GroupQueryAttentionKernel;

impl OpKernel for GroupQueryAttentionKernel {
    fn name(&self) -> &str {
        "GroupQueryAttention"
    }

    fn infer_output_shapes(&self, ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>> {
        if ctx.input_shapes.len() < 7 {
            return Err(CodegenError::InvalidShape(
                "GroupQueryAttention requires at least 7 inputs".to_string(),
            ));
        }

        // Output 0: attention output, same shape as query input
        let output_shape = ctx.input_shapes[0].clone();

        // Outputs 1 and 2 are present_key and present_value
        // We need to infer these from past_key/past_value + current key/value shapes
        // For now, return Unknown and let runtime handle it
        // TODO: Properly infer KV cache output shapes
        let present_key_shape = TensorShape::Unknown;
        let present_value_shape = TensorShape::Unknown;

        Ok(vec![output_shape, present_key_shape, present_value_shape])
    }

    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        // Read attributes
        let num_heads: i64 = ctx.node.attr("num_heads")?;
        let kv_num_heads: i64 = ctx.node.attr("kv_num_heads")?;

        // Get query shape: [batch, seq_len, num_heads * head_dim]
        let query_info = ctx.input_info(0)?;
        let query_shape = ctx.static_shape(&query_info.shape)?;

        if query_shape.len() != 3 {
            return Err(CodegenError::InvalidShape(format!(
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
            return Err(CodegenError::InvalidShape(format!(
                "Query hidden dimension {} is not divisible by num_heads {}",
                query_hidden, num_heads
            )));
        }

        // Get past_key shape: [batch, kv_num_heads, past_seq_len, head_dim]
        let past_key_info = ctx.input_info(3)?;
        let past_key_shape = ctx.static_shape(&past_key_info.shape)?;

        if past_key_shape.len() != 4 {
            return Err(CodegenError::InvalidShape(format!(
                "GroupQueryAttention expects past_key shape [batch, kv_heads, past_seq, head_dim], got {:?}",
                past_key_shape
            )));
        }

        let past_seq_len = past_key_shape[2];
        let total_seq_len = past_seq_len + seq_len;

        // Calculate scale (default: 1 / sqrt(head_dim))
        let scale: f32 = ctx
            .node
            .attr::<f32>("scale")
            .unwrap_or(1.0 / (head_dim as f32).sqrt());

        let workgroup_size: u32 = 256;

        // Step 1: Concatenate past_key with current key → present_key
        let present_key_elements = batch_size * (kv_num_heads as usize) * total_seq_len * head_dim;
        let concat_key_shader = ctx.compile_shader(
            "gqa_concat_kv",
            include_str!("../../shaders/attention/gqa_concat_kv.wgsl"),
            HashMap::from([(
                "WORKGROUP_SIZE".to_string(),
                ShaderDefValue::UInt(workgroup_size),
            )]),
        )?;

        let mut concat_key_immediates = Vec::new();
        concat_key_immediates.extend_from_slice(&(batch_size as u32).to_le_bytes());
        concat_key_immediates.extend_from_slice(&(seq_len as u32).to_le_bytes());
        concat_key_immediates.extend_from_slice(&(past_seq_len as u32).to_le_bytes());
        concat_key_immediates.extend_from_slice(&(total_seq_len as u32).to_le_bytes());
        concat_key_immediates.extend_from_slice(&(kv_num_heads as u32).to_le_bytes());
        concat_key_immediates.extend_from_slice(&(head_dim as u32).to_le_bytes());

        let concat_key_step = Step::Dispatch {
            shader_index: concat_key_shader,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(3), // past_key
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(1), // current key
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(1), // present_key
                    read_only: false,
                },
            ],
            workgroups: [(present_key_elements as u32).div_ceil(workgroup_size), 1, 1],
            immediates: Some(concat_key_immediates),
        };

        // Step 2: Concatenate past_value with current value → present_value
        let present_value_elements =
            batch_size * (kv_num_heads as usize) * total_seq_len * head_dim;
        // Reuse the same shader
        let concat_value_shader = concat_key_shader;

        let mut concat_value_immediates = Vec::new();
        concat_value_immediates.extend_from_slice(&(batch_size as u32).to_le_bytes());
        concat_value_immediates.extend_from_slice(&(seq_len as u32).to_le_bytes());
        concat_value_immediates.extend_from_slice(&(past_seq_len as u32).to_le_bytes());
        concat_value_immediates.extend_from_slice(&(total_seq_len as u32).to_le_bytes());
        concat_value_immediates.extend_from_slice(&(kv_num_heads as u32).to_le_bytes());
        concat_value_immediates.extend_from_slice(&(head_dim as u32).to_le_bytes());

        let concat_value_step = Step::Dispatch {
            shader_index: concat_value_shader,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(4), // past_value
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(2), // current value
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(2), // present_value
                    read_only: false,
                },
            ],
            workgroups: [
                (present_value_elements as u32).div_ceil(workgroup_size),
                1,
                1,
            ],
            immediates: Some(concat_value_immediates),
        };

        // Step 3: Compute attention scores (Q @ K^T / scale)
        let scores_elements = batch_size * (num_heads as usize) * seq_len * total_seq_len;
        let scores_size = (scores_elements * std::mem::size_of::<f32>()) as u64;

        // Allocate scratch buffer for attention scores
        let scores_buffer = ctx.alloc_scratch(ScratchBufferDesc {
            size: scores_size,
            label: "gqa_scores".to_string(),
        });

        let scores_shader = ctx.compile_shader(
            "gqa_scores",
            include_str!("../../shaders/attention/gqa_scores.wgsl"),
            HashMap::from([(
                "WORKGROUP_SIZE".to_string(),
                ShaderDefValue::UInt(workgroup_size),
            )]),
        )?;

        let mut scores_immediates = Vec::new();
        scores_immediates.extend_from_slice(&(batch_size as u32).to_le_bytes());
        scores_immediates.extend_from_slice(&(seq_len as u32).to_le_bytes());
        scores_immediates.extend_from_slice(&(total_seq_len as u32).to_le_bytes());
        scores_immediates.extend_from_slice(&(num_heads as u32).to_le_bytes());
        scores_immediates.extend_from_slice(&(kv_num_heads as u32).to_le_bytes());
        scores_immediates.extend_from_slice(&(head_dim as u32).to_le_bytes());
        scores_immediates.extend_from_slice(&scale.to_le_bytes());

        let scores_step = Step::Dispatch {
            shader_index: scores_shader,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0), // query
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(1), // present_key
                    read_only: true,
                },
                BindingDesc {
                    buffer: scores_buffer.clone(),
                    read_only: false,
                },
            ],
            workgroups: [(scores_elements as u32).div_ceil(workgroup_size), 1, 1],
            immediates: Some(scores_immediates),
        };

        // Step 4: Apply causal mask and softmax
        let softmax_rows = batch_size * (num_heads as usize) * seq_len;
        let softmax_shader = ctx.compile_shader(
            "gqa_softmax",
            include_str!("../../shaders/attention/gqa_softmax.wgsl"),
            HashMap::from([(
                "WORKGROUP_SIZE".to_string(),
                ShaderDefValue::UInt(workgroup_size),
            )]),
        )?;

        let mut softmax_immediates = Vec::new();
        softmax_immediates.extend_from_slice(&(batch_size as u32).to_le_bytes());
        softmax_immediates.extend_from_slice(&(seq_len as u32).to_le_bytes());
        softmax_immediates.extend_from_slice(&(past_seq_len as u32).to_le_bytes());
        softmax_immediates.extend_from_slice(&(total_seq_len as u32).to_le_bytes());
        softmax_immediates.extend_from_slice(&(num_heads as u32).to_le_bytes());

        let softmax_step = Step::Dispatch {
            shader_index: softmax_shader,
            bindings: vec![BindingDesc {
                buffer: scores_buffer.clone(),
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
            HashMap::from([(
                "WORKGROUP_SIZE".to_string(),
                ShaderDefValue::UInt(workgroup_size),
            )]),
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
                    buffer: scores_buffer, // After softmax, these are attention weights
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(2), // present_value
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(0), // output
                    read_only: false,
                },
            ],
            workgroups: [(output_elements as u32).div_ceil(workgroup_size), 1, 1],
            immediates: Some(output_immediates),
        };

        // Return all steps
        Ok(vec![
            concat_key_step,
            concat_value_step,
            scores_step,
            softmax_step,
            output_step,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::InferenceContext;
    use onyxia_onnx::{AttributeValue, DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};

    fn create_gqa_test_graph() -> (Graph, Node) {
        let mut graph = Graph::new();

        let batch = 1;
        let seq_len = 2;
        let past_seq_len = 4;
        let num_heads = 4;
        let kv_num_heads = 1;
        let head_dim = 8;

        // Input 0: query [batch, seq_len, num_heads * head_dim]
        graph.add_tensor(TensorInfo {
            name: "query".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![batch, seq_len, num_heads * head_dim]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Input 1: key [batch, seq_len, kv_num_heads * head_dim]
        graph.add_tensor(TensorInfo {
            name: "key".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![batch, seq_len, kv_num_heads * head_dim]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Input 2: value [batch, seq_len, kv_num_heads * head_dim]
        graph.add_tensor(TensorInfo {
            name: "value".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![batch, seq_len, kv_num_heads * head_dim]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Input 3: past_key [batch, kv_num_heads, past_seq_len, head_dim]
        graph.add_tensor(TensorInfo {
            name: "past_key".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![batch, kv_num_heads, past_seq_len, head_dim]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Input 4: past_value [batch, kv_num_heads, past_seq_len, head_dim]
        graph.add_tensor(TensorInfo {
            name: "past_value".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![batch, kv_num_heads, past_seq_len, head_dim]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Input 5: seqlens_k [batch]
        graph.add_tensor(TensorInfo {
            name: "seqlens_k".to_string(),
            dtype: DataType::I32,
            shape: TensorShape::Static(vec![batch]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Input 6: total_sequence_length (scalar)
        graph.add_tensor(TensorInfo {
            name: "total_sequence_length".to_string(),
            dtype: DataType::I32,
            shape: TensorShape::Static(vec![1]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Output 0: output [batch, seq_len, num_heads * head_dim]
        graph.add_tensor(TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![batch, seq_len, num_heads * head_dim]),
            kind: TensorKind::Output,
            initializer: None,
        });

        // Output 1: present_key [batch, kv_num_heads, total_seq_len, head_dim]
        graph.add_tensor(TensorInfo {
            name: "present_key".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![batch, kv_num_heads, past_seq_len + seq_len, head_dim]),
            kind: TensorKind::Output,
            initializer: None,
        });

        // Output 2: present_value [batch, kv_num_heads, total_seq_len, head_dim]
        graph.add_tensor(TensorInfo {
            name: "present_value".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![batch, kv_num_heads, past_seq_len + seq_len, head_dim]),
            kind: TensorKind::Output,
            initializer: None,
        });

        let mut node = Node::new("GroupQueryAttention");
        node.inputs = vec![
            "query".to_string(),
            "key".to_string(),
            "value".to_string(),
            "past_key".to_string(),
            "past_value".to_string(),
            "seqlens_k".to_string(),
            "total_sequence_length".to_string(),
        ];
        node.outputs = vec![
            "output".to_string(),
            "present_key".to_string(),
            "present_value".to_string(),
        ];
        node.attributes.insert(
            "num_heads".to_string(),
            AttributeValue::Int(num_heads as i64),
        );
        node.attributes.insert(
            "kv_num_heads".to_string(),
            AttributeValue::Int(kv_num_heads as i64),
        );

        (graph, node)
    }

    #[test]
    fn test_gqa_kernel_shape_inference() {
        let kernel = GroupQueryAttentionKernel;
        let (_graph, node) = create_gqa_test_graph();

        let query_shape = TensorShape::Static(vec![1, 2, 32]); // [batch, seq, num_heads*head_dim]
        let key_shape = TensorShape::Static(vec![1, 2, 8]); // [batch, seq, kv_heads*head_dim]
        let value_shape = TensorShape::Static(vec![1, 2, 8]);
        let past_key_shape = TensorShape::Static(vec![1, 1, 4, 8]); // [batch, kv_heads, past_seq, head_dim]
        let past_value_shape = TensorShape::Static(vec![1, 1, 4, 8]);
        let seqlens_shape = TensorShape::Static(vec![1]);
        let total_seq_shape = TensorShape::Static(vec![1]);

        let input_shapes = vec![
            query_shape,
            key_shape,
            value_shape,
            past_key_shape,
            past_value_shape,
            seqlens_shape,
            total_seq_shape,
        ];

        let graph = onyxia_onnx::Graph::new();
        let output_shapes = kernel
            .infer_output_shapes(&{
            let input_values = vec![None; input_shapes.len()];
            InferenceContext::new(&node, &graph, input_shapes.clone(), input_values)
        })
            .expect("Shape inference should succeed");

        assert_eq!(output_shapes.len(), 3);
        // Output 0 should match query shape
        assert_eq!(output_shapes[0], TensorShape::Static(vec![1, 2, 32]));
    }

    #[test]
    fn test_gqa_kernel_plan_structure() {
        let (graph, node) = create_gqa_test_graph();
        let input_ids = vec![0, 1, 2, 3, 4, 5, 6];
        let output_ids = vec![7, 8, 9];
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let steps = GroupQueryAttentionKernel
            .plan(&mut ctx)
            .expect("Planning should succeed");

        // Verify we got 5 dispatch steps
        assert_eq!(steps.len(), 5);

        // Verify all steps are Dispatch steps
        for step in &steps {
            assert!(matches!(step, Step::Dispatch { .. }));
        }

        // Verify scratch buffer was allocated for scores
        assert_eq!(ctx.scratch_buffers.len(), 1);
        assert_eq!(ctx.scratch_buffers[0].label, "gqa_scores");

        // Verify shaders were compiled (concat_kv, scores, softmax, output = 4 unique shaders)
        assert_eq!(shaders.len(), 4);
    }

    #[test]
    fn test_gqa_kernel_attributes() {
        let (graph, mut node) = create_gqa_test_graph();

        // Test with explicit scale attribute
        node.attributes
            .insert("scale".to_string(), AttributeValue::Float(0.125));

        let input_ids = vec![0, 1, 2, 3, 4, 5, 6];
        let output_ids = vec![7, 8, 9];
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let steps = GroupQueryAttentionKernel
            .plan(&mut ctx)
            .expect("Planning should succeed with explicit scale");

        assert_eq!(steps.len(), 5);
    }

    #[test]
    fn test_gqa_kernel_no_cache() {
        // Test with zero past_seq_len (no cache)
        let mut graph = Graph::new();
        let batch = 1;
        let seq_len = 2;
        let past_seq_len = 0; // No cache
        let num_heads = 2;
        let kv_num_heads = 1;
        let head_dim = 4;

        // Add all required tensors
        graph.add_tensor(TensorInfo {
            name: "query".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![batch, seq_len, num_heads * head_dim]),
            kind: TensorKind::Input,
            initializer: None,
        });
        graph.add_tensor(TensorInfo {
            name: "key".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![batch, seq_len, kv_num_heads * head_dim]),
            kind: TensorKind::Input,
            initializer: None,
        });
        graph.add_tensor(TensorInfo {
            name: "value".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![batch, seq_len, kv_num_heads * head_dim]),
            kind: TensorKind::Input,
            initializer: None,
        });
        graph.add_tensor(TensorInfo {
            name: "past_key".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![batch, kv_num_heads, past_seq_len, head_dim]),
            kind: TensorKind::Input,
            initializer: None,
        });
        graph.add_tensor(TensorInfo {
            name: "past_value".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![batch, kv_num_heads, past_seq_len, head_dim]),
            kind: TensorKind::Input,
            initializer: None,
        });
        graph.add_tensor(TensorInfo {
            name: "seqlens_k".to_string(),
            dtype: DataType::I32,
            shape: TensorShape::Static(vec![batch]),
            kind: TensorKind::Input,
            initializer: None,
        });
        graph.add_tensor(TensorInfo {
            name: "total_sequence_length".to_string(),
            dtype: DataType::I32,
            shape: TensorShape::Static(vec![1]),
            kind: TensorKind::Input,
            initializer: None,
        });
        graph.add_tensor(TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![batch, seq_len, num_heads * head_dim]),
            kind: TensorKind::Output,
            initializer: None,
        });
        graph.add_tensor(TensorInfo {
            name: "present_key".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![batch, kv_num_heads, seq_len, head_dim]),
            kind: TensorKind::Output,
            initializer: None,
        });
        graph.add_tensor(TensorInfo {
            name: "present_value".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![batch, kv_num_heads, seq_len, head_dim]),
            kind: TensorKind::Output,
            initializer: None,
        });

        let mut node = Node::new("GroupQueryAttention");
        node.inputs = vec![
            "query".to_string(),
            "key".to_string(),
            "value".to_string(),
            "past_key".to_string(),
            "past_value".to_string(),
            "seqlens_k".to_string(),
            "total_sequence_length".to_string(),
        ];
        node.outputs = vec![
            "output".to_string(),
            "present_key".to_string(),
            "present_value".to_string(),
        ];
        node.attributes.insert(
            "num_heads".to_string(),
            AttributeValue::Int(num_heads as i64),
        );
        node.attributes.insert(
            "kv_num_heads".to_string(),
            AttributeValue::Int(kv_num_heads as i64),
        );

        let input_ids = vec![0, 1, 2, 3, 4, 5, 6];
        let output_ids = vec![7, 8, 9];
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let steps = GroupQueryAttentionKernel
            .plan(&mut ctx)
            .expect("Planning should succeed with no cache");

        // Should still generate 5 steps
        assert_eq!(steps.len(), 5);
    }
}
