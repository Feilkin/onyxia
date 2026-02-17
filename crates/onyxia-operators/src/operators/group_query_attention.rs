//! GroupQueryAttention operator implementation (Microsoft contrib).
//!
//! Implements multi-head grouped-query attention with decomposed GPU operations.

use onyxia_core::{CompileCtx, DispatchCtx, Error, OpDispatch, Operator, Result, RuntimeTensor};
use std::collections::HashMap;
use std::sync::Arc;

/// Shader sources
const RESHAPE_QKV_SHADER: &str = include_str!("../../shaders/gqa_reshape_qkv.wgsl");
const REPEAT_KV_SHADER: &str = include_str!("../../shaders/gqa_repeat_kv.wgsl");
const MATMUL_QK_SHADER: &str = include_str!("../../shaders/gqa_matmul_qk.wgsl");
const MASK_SCALE_SHADER: &str = include_str!("../../shaders/gqa_mask_scale.wgsl");
const SOFTMAX_MAX_SHADER: &str = include_str!("../../shaders/gqa_softmax_max.wgsl");
const SOFTMAX_NORMALIZE_SHADER: &str = include_str!("../../shaders/gqa_softmax_normalize.wgsl");
const MATMUL_AV_SHADER: &str = include_str!("../../shaders/gqa_matmul_av.wgsl");
const RESHAPE_FROM_HEADS_SHADER: &str = include_str!("../../shaders/gqa_reshape_from_heads.wgsl");

/// GroupQueryAttention operator (Microsoft contrib).
///
/// Implements multi-head attention with grouped queries for efficient KV caching.
///
/// **ONNX Specification:**
/// - Domain: com.microsoft
/// - Inputs:
///   - query (T) - [batch, seq_len, num_heads * head_size]
///   - key (T) - [batch, kv_seq_len, kv_num_heads * head_size]
///   - value (T) - [batch, kv_seq_len, kv_num_heads * head_size]
///   - past_key (optional) - Not implemented yet
///   - past_value (optional) - Not implemented yet
/// - Outputs:
///   - output (T) - [batch, seq_len, num_heads * head_size]
///   - present_key (optional) - Not implemented yet
///   - present_value (optional) - Not implemented yet
/// - Attributes:
///   - num_heads (int) - Number of query heads
///   - kv_num_heads (int) - Number of key/value heads (≤ num_heads)
///   - scale (float, optional) - Attention scale factor (default: 1/sqrt(head_size))
///
/// **Implementation:**
/// Uses a decomposed multi-stage pipeline:
/// 1. Reshape Q, K, V to separate heads
/// 2. Repeat KV heads to match Q heads (for grouped attention)
/// 3. Compute attention scores: Q @ K^T
/// 4. Apply causal mask and scale
/// 5. Compute softmax
/// 6. Compute attention output: softmax @ V
/// 7. Reshape back to original format
pub struct GroupQueryAttentionOp;

impl Operator for GroupQueryAttentionOp {
    fn name(&self) -> &str {
        "com.microsoft::GroupQueryAttention"
    }

    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        let num_heads = match ctx.attr("num_heads") {
            Some(onyxia_onnx::AttributeValue::Int(v)) => *v as usize,
            _ => {
                return Err(Error::Compilation(
                    "GroupQueryAttention requires 'num_heads' attribute".into(),
                ));
            }
        };

        let kv_num_heads = match ctx.attr("kv_num_heads") {
            Some(onyxia_onnx::AttributeValue::Int(v)) => *v as usize,
            _ => {
                return Err(Error::Compilation(
                    "GroupQueryAttention requires 'kv_num_heads' attribute".into(),
                ));
            }
        };

        if num_heads % kv_num_heads != 0 {
            return Err(Error::Compilation(
                "num_heads must be divisible by kv_num_heads for grouped attention".into(),
            ));
        }

        // Scale defaults to 1/sqrt(head_size), but we'll compute it at runtime
        let scale_override = ctx.attr("scale").and_then(|v| match v {
            onyxia_onnx::AttributeValue::Float(f) => Some(*f),
            _ => None,
        });

        // Compile all shaders
        let shader_defs = HashMap::new();

        let reshape_module = ctx.compile_shader("GQA_Reshape", RESHAPE_QKV_SHADER, &shader_defs)?;
        let repeat_kv_module =
            ctx.compile_shader("GQA_RepeatKV", REPEAT_KV_SHADER, &shader_defs)?;
        let matmul_qk_module =
            ctx.compile_shader("GQA_MatMulQK", MATMUL_QK_SHADER, &shader_defs)?;
        let mask_scale_module =
            ctx.compile_shader("GQA_MaskScale", MASK_SCALE_SHADER, &shader_defs)?;
        let softmax_max_module =
            ctx.compile_shader("GQA_SoftmaxMax", SOFTMAX_MAX_SHADER, &shader_defs)?;
        let softmax_normalize_module = ctx.compile_shader(
            "GQA_SoftmaxNormalize",
            SOFTMAX_NORMALIZE_SHADER,
            &shader_defs,
        )?;
        let matmul_av_module =
            ctx.compile_shader("GQA_MatMulAV", MATMUL_AV_SHADER, &shader_defs)?;
        let reshape_from_heads_module = ctx.compile_shader(
            "GQA_ReshapeFromHeads",
            RESHAPE_FROM_HEADS_SHADER,
            &shader_defs,
        )?;

        Ok(Box::new(GroupQueryAttentionDispatch {
            reshape_module,
            repeat_kv_module,
            matmul_qk_module,
            mask_scale_module,
            softmax_max_module,
            softmax_normalize_module,
            matmul_av_module,
            reshape_from_heads_module,
            num_heads,
            kv_num_heads,
            scale_override,
        }))
    }
}

/// Runtime dispatch for GroupQueryAttention.
struct GroupQueryAttentionDispatch {
    reshape_module: naga::Module,
    repeat_kv_module: naga::Module,
    matmul_qk_module: naga::Module,
    mask_scale_module: naga::Module,
    softmax_max_module: naga::Module,
    softmax_normalize_module: naga::Module,
    matmul_av_module: naga::Module,
    reshape_from_heads_module: naga::Module,
    num_heads: usize,
    kv_num_heads: usize,
    scale_override: Option<f32>,
}

impl OpDispatch for GroupQueryAttentionDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> Result<Vec<RuntimeTensor>> {
        // Extract inputs
        let query = &inputs[0];
        let key = &inputs[1];
        let value = &inputs[2];

        // Extract dimensions
        if query.shape.len() != 3 || key.shape.len() != 3 || value.shape.len() != 3 {
            return Err(Error::Shape(
                "GQA inputs must be 3D [batch, seq, hidden]".into(),
            ));
        }

        let batch_size = query.shape[0];
        let seq_len = query.shape[1];
        let kv_seq_len = key.shape[1];
        let head_size = query.shape[2] / self.num_heads;

        // Compute scale factor
        let scale = self
            .scale_override
            .unwrap_or(1.0 / (head_size as f32).sqrt());

        // Step 1: Reshape Q, K, V to [batch, heads, seq, head_size]
        let q_shaped =
            self.reshape_to_heads(ctx, query, batch_size, seq_len, self.num_heads, head_size)?;
        let k_shaped = self.reshape_to_heads(
            ctx,
            key,
            batch_size,
            kv_seq_len,
            self.kv_num_heads,
            head_size,
        )?;
        let v_shaped = self.reshape_to_heads(
            ctx,
            value,
            batch_size,
            kv_seq_len,
            self.kv_num_heads,
            head_size,
        )?;

        // Step 2: Repeat KV heads if grouped (kv_num_heads < num_heads)
        let k_repeated = if self.num_heads != self.kv_num_heads {
            self.repeat_kv_heads(ctx, &k_shaped, batch_size, kv_seq_len, head_size)?
        } else {
            k_shaped
        };

        let v_repeated = if self.num_heads != self.kv_num_heads {
            self.repeat_kv_heads(ctx, &v_shaped, batch_size, kv_seq_len, head_size)?
        } else {
            v_shaped
        };

        // Step 3: Compute Q @ K^T -> scores [batch, heads, seq_q, seq_k]
        let mut scores = self.matmul_qk(
            ctx,
            &q_shaped,
            &k_repeated,
            batch_size,
            seq_len,
            kv_seq_len,
            head_size,
        )?;

        // Step 4: Apply mask and scale
        self.apply_mask_scale(ctx, &mut scores, batch_size, seq_len, kv_seq_len, scale)?;

        // Step 5: Softmax along last dimension
        let attn_weights = self.apply_softmax(ctx, &scores, batch_size, seq_len, kv_seq_len)?;

        // Step 6: Compute attn @ V -> [batch, heads, seq_q, head_size]
        let attn_output = self.matmul_av(
            ctx,
            &attn_weights,
            &v_repeated,
            batch_size,
            seq_len,
            kv_seq_len,
            head_size,
        )?;

        // Step 7: Reshape back to [batch, seq, heads * head_size]
        let output = self.reshape_from_heads(
            ctx,
            &attn_output,
            batch_size,
            seq_len,
            self.num_heads,
            head_size,
        )?;

        // TODO: Handle present_key and present_value outputs for KV caching
        Ok(vec![output])
    }
}

impl GroupQueryAttentionDispatch {
    /// Reshape [batch, seq, heads*size] -> [batch, heads, seq, size]
    fn reshape_to_heads(
        &self,
        ctx: &mut DispatchCtx,
        input: &RuntimeTensor,
        batch: usize,
        seq: usize,
        heads: usize,
        size: usize,
    ) -> Result<RuntimeTensor> {
        let output_shape = vec![batch, heads, seq, size];
        let output = ctx.create_output_tensor(&output_shape, input.dtype)?;

        let mut params = Vec::new();
        params.extend_from_slice(&(batch as u32).to_le_bytes());
        params.extend_from_slice(&(seq as u32).to_le_bytes());
        params.extend_from_slice(&(heads as u32).to_le_bytes());
        params.extend_from_slice(&(size as u32).to_le_bytes());

        let (pipeline, bind_group_layout) =
            ctx.get_or_create_pipeline("GQA_Reshape", &self.reshape_module, "main")?;

        let params_buffer = create_params_buffer(ctx, &params)?;
        let entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.buffer.as_entire_binding(),
            },
        ];

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gqa_reshape_bind_group"),
            layout: &bind_group_layout,
            entries: &entries,
        });

        let total = (batch * heads * seq * size) as u32;
        ctx.dispatch_compute(&pipeline, &bind_group, [total.div_ceil(256), 1, 1], None)?;

        Ok(output)
    }

    /// Repeat KV heads: [batch, kv_heads, seq, size] -> [batch, num_heads, seq, size]
    fn repeat_kv_heads(
        &self,
        ctx: &mut DispatchCtx,
        kv: &RuntimeTensor,
        batch: usize,
        seq: usize,
        size: usize,
    ) -> Result<RuntimeTensor> {
        let output_shape = vec![batch, self.num_heads, seq, size];
        let output = ctx.create_output_tensor(&output_shape, kv.dtype)?;

        let mut params = Vec::new();
        params.extend_from_slice(&(batch as u32).to_le_bytes());
        params.extend_from_slice(&(self.kv_num_heads as u32).to_le_bytes());
        params.extend_from_slice(&(self.num_heads as u32).to_le_bytes());
        params.extend_from_slice(&(seq as u32).to_le_bytes());
        params.extend_from_slice(&(size as u32).to_le_bytes());

        let (pipeline, bind_group_layout) =
            ctx.get_or_create_pipeline("GQA_RepeatKV", &self.repeat_kv_module, "main")?;

        let params_buffer = create_params_buffer(ctx, &params)?;
        let entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: kv.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.buffer.as_entire_binding(),
            },
        ];

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gqa_repeat_kv_bind_group"),
            layout: &bind_group_layout,
            entries: &entries,
        });

        let total = (batch * self.num_heads * seq * size) as u32;
        ctx.dispatch_compute(&pipeline, &bind_group, [total.div_ceil(256), 1, 1], None)?;

        Ok(output)
    }

    /// Compute Q @ K^T -> [batch, heads, seq_q, seq_k]
    fn matmul_qk(
        &self,
        ctx: &mut DispatchCtx,
        q: &RuntimeTensor,
        k: &RuntimeTensor,
        batch: usize,
        seq_q: usize,
        seq_k: usize,
        head_size: usize,
    ) -> Result<RuntimeTensor> {
        let output_shape = vec![batch, self.num_heads, seq_q, seq_k];
        let output = ctx.create_output_tensor(&output_shape, q.dtype)?;

        let mut params = Vec::new();
        params.extend_from_slice(&(batch as u32).to_le_bytes());
        params.extend_from_slice(&(self.num_heads as u32).to_le_bytes());
        params.extend_from_slice(&(seq_q as u32).to_le_bytes());
        params.extend_from_slice(&(seq_k as u32).to_le_bytes());
        params.extend_from_slice(&(head_size as u32).to_le_bytes());

        let (pipeline, bind_group_layout) =
            ctx.get_or_create_pipeline("GQA_MatMulQK", &self.matmul_qk_module, "main")?;

        let params_buffer = create_params_buffer(ctx, &params)?;
        let entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: q.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: k.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output.buffer.as_entire_binding(),
            },
        ];

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gqa_matmul_qk_bind_group"),
            layout: &bind_group_layout,
            entries: &entries,
        });

        // Workgroups: (seq_k/16, seq_q/16, batch*heads)
        let workgroups_x = (seq_k as u32).div_ceil(16);
        let workgroups_y = (seq_q as u32).div_ceil(16);
        let workgroups_z = (batch * self.num_heads) as u32;

        ctx.dispatch_compute(
            &pipeline,
            &bind_group,
            [workgroups_x, workgroups_y, workgroups_z],
            None,
        )?;

        Ok(output)
    }

    /// Apply causal mask and scale to scores (in-place)
    fn apply_mask_scale(
        &self,
        ctx: &mut DispatchCtx,
        scores: &mut RuntimeTensor,
        batch: usize,
        seq_q: usize,
        seq_k: usize,
        scale: f32,
    ) -> Result<()> {
        let mut params = Vec::new();
        params.extend_from_slice(&(batch as u32).to_le_bytes());
        params.extend_from_slice(&(self.num_heads as u32).to_le_bytes());
        params.extend_from_slice(&(seq_q as u32).to_le_bytes());
        params.extend_from_slice(&(seq_k as u32).to_le_bytes());
        params.extend_from_slice(&scale.to_le_bytes());

        let (pipeline, bind_group_layout) =
            ctx.get_or_create_pipeline("GQA_MaskScale", &self.mask_scale_module, "main")?;

        let params_buffer = create_params_buffer(ctx, &params)?;
        let entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: scores.buffer.as_entire_binding(),
            },
        ];

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gqa_mask_scale_bind_group"),
            layout: &bind_group_layout,
            entries: &entries,
        });

        let total = (batch * self.num_heads * seq_q * seq_k) as u32;
        ctx.dispatch_compute(&pipeline, &bind_group, [total.div_ceil(256), 1, 1], None)?;

        Ok(())
    }

    /// Apply softmax along last dimension
    fn apply_softmax(
        &self,
        ctx: &mut DispatchCtx,
        scores: &RuntimeTensor,
        batch: usize,
        seq_q: usize,
        seq_k: usize,
    ) -> Result<RuntimeTensor> {
        // Create output buffer (will be modified in-place)
        let output = ctx.create_output_tensor(&scores.shape, scores.dtype)?;

        // Copy scores to output (we'll modify in-place)
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("softmax_copy"),
            });
        encoder.copy_buffer_to_buffer(
            &scores.buffer,
            0,
            &output.buffer,
            0,
            scores.size_bytes as u64,
        );
        ctx.queue.submit(std::iter::once(encoder.finish()));

        let total_rows = batch * self.num_heads * seq_q;

        // Create intermediate buffer for max values
        let max_buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gqa_softmax_max"),
            size: (total_rows * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let mut params = Vec::new();
        params.extend_from_slice(&(batch as u32).to_le_bytes());
        params.extend_from_slice(&(self.num_heads as u32).to_le_bytes());
        params.extend_from_slice(&(seq_q as u32).to_le_bytes());
        params.extend_from_slice(&(seq_k as u32).to_le_bytes());

        // Pass 1: Find max values
        {
            let (pipeline, bind_group_layout) =
                ctx.get_or_create_pipeline("GQA_SoftmaxMax", &self.softmax_max_module, "main")?;

            let params_buffer = create_params_buffer(ctx, &params)?;
            let entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: max_buffer.as_entire_binding(),
                },
            ];

            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gqa_softmax_max_bind_group"),
                layout: &bind_group_layout,
                entries: &entries,
            });

            ctx.dispatch_compute(
                &pipeline,
                &bind_group,
                [(total_rows as u32).div_ceil(256), 1, 1],
                None,
            )?;
        }

        // Pass 2: Compute exp and normalize
        {
            let (pipeline, bind_group_layout) = ctx.get_or_create_pipeline(
                "GQA_SoftmaxNormalize",
                &self.softmax_normalize_module,
                "main",
            )?;

            let params_buffer = create_params_buffer(ctx, &params)?;
            let entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: max_buffer.as_entire_binding(),
                },
            ];

            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gqa_softmax_normalize_bind_group"),
                layout: &bind_group_layout,
                entries: &entries,
            });

            ctx.dispatch_compute(
                &pipeline,
                &bind_group,
                [(total_rows as u32).div_ceil(256), 1, 1],
                None,
            )?;
        }

        Ok(output)
    }

    /// Compute attn @ V -> [batch, heads, seq_q, head_size]
    fn matmul_av(
        &self,
        ctx: &mut DispatchCtx,
        attn: &RuntimeTensor,
        v: &RuntimeTensor,
        batch: usize,
        seq_q: usize,
        seq_k: usize,
        head_size: usize,
    ) -> Result<RuntimeTensor> {
        let output_shape = vec![batch, self.num_heads, seq_q, head_size];
        let output = ctx.create_output_tensor(&output_shape, attn.dtype)?;

        let mut params = Vec::new();
        params.extend_from_slice(&(batch as u32).to_le_bytes());
        params.extend_from_slice(&(self.num_heads as u32).to_le_bytes());
        params.extend_from_slice(&(seq_q as u32).to_le_bytes());
        params.extend_from_slice(&(seq_k as u32).to_le_bytes());
        params.extend_from_slice(&(head_size as u32).to_le_bytes());

        let (pipeline, bind_group_layout) =
            ctx.get_or_create_pipeline("GQA_MatMulAV", &self.matmul_av_module, "main")?;

        let params_buffer = create_params_buffer(ctx, &params)?;
        let entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: attn.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: v.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output.buffer.as_entire_binding(),
            },
        ];

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gqa_matmul_av_bind_group"),
            layout: &bind_group_layout,
            entries: &entries,
        });

        // Workgroups: (head_size/16, seq_q/16, batch*heads)
        let workgroups_x = (head_size as u32).div_ceil(16);
        let workgroups_y = (seq_q as u32).div_ceil(16);
        let workgroups_z = (batch * self.num_heads) as u32;

        ctx.dispatch_compute(
            &pipeline,
            &bind_group,
            [workgroups_x, workgroups_y, workgroups_z],
            None,
        )?;

        Ok(output)
    }

    /// Reshape [batch, heads, seq, size] -> [batch, seq, heads*size]
    fn reshape_from_heads(
        &self,
        ctx: &mut DispatchCtx,
        input: &RuntimeTensor,
        batch: usize,
        seq: usize,
        heads: usize,
        size: usize,
    ) -> Result<RuntimeTensor> {
        let output_shape = vec![batch, seq, heads * size];
        let output = ctx.create_output_tensor(&output_shape, input.dtype)?;

        let mut params = Vec::new();
        params.extend_from_slice(&(batch as u32).to_le_bytes());
        params.extend_from_slice(&(seq as u32).to_le_bytes());
        params.extend_from_slice(&(heads as u32).to_le_bytes());
        params.extend_from_slice(&(size as u32).to_le_bytes());

        let (pipeline, bind_group_layout) = ctx.get_or_create_pipeline(
            "GQA_ReshapeFromHeads",
            &self.reshape_from_heads_module,
            "main",
        )?;

        let params_buffer = create_params_buffer(ctx, &params)?;
        let entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.buffer.as_entire_binding(),
            },
        ];

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gqa_reshape_from_heads_bind_group"),
            layout: &bind_group_layout,
            entries: &entries,
        });

        let total = (batch * seq * heads * size) as u32;
        ctx.dispatch_compute(&pipeline, &bind_group, [total.div_ceil(256), 1, 1], None)?;

        Ok(output)
    }
}

/// Helper to create a parameters buffer.
fn create_params_buffer(ctx: &mut DispatchCtx, params: &[u8]) -> Result<Arc<wgpu::Buffer>> {
    let buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gqa_params"),
        size: params.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));

    ctx.queue.write_buffer(&buffer, 0, params);
    Ok(buffer)
}
