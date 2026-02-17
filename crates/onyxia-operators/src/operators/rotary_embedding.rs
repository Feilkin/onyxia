//! RotaryEmbedding operator implementation (Microsoft contrib).
//!
//! Applies rotary position embeddings (RoPE) using precomputed cos/sin caches.

use onyxia_core::{
    CompileCtx, DataType, DispatchCtx, Error, OpDispatch, Operator, Result, RuntimeTensor,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Shader source for rotary embeddings.
const ROTARY_EMBEDDING_SHADER: &str = include_str!("../../shaders/rotary_embedding.wgsl");

/// RotaryEmbedding operator (Microsoft contrib).
///
/// Applies rotary position embeddings using precomputed cosine and sine caches.
///
/// **ONNX Specification:**
/// - Domain: com.microsoft
/// - Inputs:
///   - input (T) - Input tensor [batch, seq_len, hidden_size]
///   - position_ids (int64) - Position indices [batch, seq_len]
///   - cos_cache (T) - Precomputed cosine cache [max_seq_len, rotary_dim // 2]
///   - sin_cache (T) - Precomputed sine cache [max_seq_len, rotary_dim // 2]
/// - Outputs:
///   - output (T) - Output with rotary embeddings applied [batch, seq_len, hidden_size]
/// - Attributes:
///   - interleaved (int, default: 0) - Rotation pattern (0=non-interleaved, 1=interleaved)
///
/// **Algorithm:**
/// - Non-interleaved mode: First half of rotary_dim rotates with second half
/// - Interleaved mode: Adjacent pairs (0,1), (2,3), ... rotate together
/// - Dimensions beyond rotary_dim are copied unchanged
pub struct RotaryEmbeddingOp;

impl Operator for RotaryEmbeddingOp {
    fn name(&self) -> &str {
        "com.microsoft::RotaryEmbedding"
    }

    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        let interleaved = match ctx.attr("interleaved") {
            Some(onyxia_onnx::AttributeValue::Int(v)) => *v != 0,
            _ => false,
        };

        let shader_defs = HashMap::new();
        let module =
            ctx.compile_shader("RotaryEmbedding", ROTARY_EMBEDDING_SHADER, &shader_defs)?;

        Ok(Box::new(RotaryEmbeddingDispatch {
            module,
            interleaved,
        }))
    }
}

/// Runtime dispatch for RotaryEmbedding operation.
struct RotaryEmbeddingDispatch {
    /// Pre-compiled naga module.
    module: naga::Module,

    /// Whether to use interleaved rotation mode.
    interleaved: bool,
}

impl OpDispatch for RotaryEmbeddingDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> Result<Vec<RuntimeTensor>> {
        let input = &inputs[0];
        let position_ids = &inputs[1];
        let cos_cache = &inputs[2];
        let sin_cache = &inputs[3];

        // Validate position_ids dtype
        if position_ids.dtype != DataType::I64 {
            return Err(Error::Compilation(
                "RotaryEmbedding position_ids must be int64".into(),
            ));
        }

        // Convert position_ids from i64 to u32 for GPU (WGSL doesn't support i64)
        let position_ids_data = ctx.download_tensor(position_ids)?;
        let position_ids_i64: &[i64] = bytemuck::cast_slice(&position_ids_data);
        let position_ids_u32: Vec<u32> = position_ids_i64
            .iter()
            .map(|&x| x as u32)
            .collect();
        let position_ids_u32_bytes: &[u8] = bytemuck::cast_slice(&position_ids_u32);
        let position_ids_gpu = ctx.upload_tensor(
            position_ids_u32_bytes,
            &position_ids.shape,
            DataType::U32,
        )?;

        // Extract dimensions
        if input.shape.len() != 3 {
            return Err(Error::Shape(format!(
                "RotaryEmbedding input must be 3D [batch, seq, hidden], got {:?}",
                input.shape
            )));
        }

        let batch_size = input.shape[0];
        let seq_len = input.shape[1];
        let hidden_size = input.shape[2];

        // Rotary dimension is 2 * last dimension of cos_cache
        let rotary_dim = cos_cache.shape[cos_cache.shape.len() - 1] * 2;

        if rotary_dim > hidden_size {
            return Err(Error::Shape(format!(
                "Rotary dimension {} exceeds hidden size {}",
                rotary_dim, hidden_size
            )));
        }

        // Create output tensor
        let output = ctx.create_output_tensor(&input.shape, input.dtype)?;

        // Prepare immediates
        let mut immediates = Vec::new();
        immediates.extend_from_slice(&(batch_size as u32).to_le_bytes());
        immediates.extend_from_slice(&(seq_len as u32).to_le_bytes());
        immediates.extend_from_slice(&(hidden_size as u32).to_le_bytes());
        immediates.extend_from_slice(&(rotary_dim as u32).to_le_bytes());
        immediates.extend_from_slice(&(self.interleaved as u32).to_le_bytes());

        // Compute total elements and workgroups
        let total_elements = (batch_size * seq_len * hidden_size) as u32;
        let workgroups = total_elements.div_ceil(256);

        // Get or create pipeline
        let (pipeline, bind_group_layout) =
            ctx.get_or_create_pipeline("RotaryEmbedding", &self.module, "main")?;

        // Create immediates buffer
        let immediates_buffer = create_immediates_buffer(ctx, &immediates)?;

        // Create bind group
        let entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: immediates_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: position_ids_gpu.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cos_cache.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: sin_cache.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: output.buffer.as_entire_binding(),
            },
        ];

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rotary_embedding_bind_group"),
            layout: &bind_group_layout,
            entries: &entries,
        });

        // Dispatch compute shader
        ctx.dispatch_compute(&pipeline, &bind_group, [workgroups, 1, 1], None)?;

        Ok(vec![output])
    }
}

/// Helper to create an immediates buffer.
fn create_immediates_buffer(
    ctx: &mut DispatchCtx,
    immediates: &[u8],
) -> Result<Arc<wgpu::Buffer>> {
    let buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rotary_embedding_immediates"),
        size: immediates.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));

    ctx.queue.write_buffer(&buffer, 0, immediates);
    Ok(buffer)
}
