//! MatMul operator - Matrix multiplication with GPU optimization.

use onyxia_core::{CompileCtx, DispatchCtx, Error, OpDispatch, Operator, Result, RuntimeTensor};
use std::collections::HashMap;

/// Shader source for the MatMul operator.
const MATMUL_SHADER: &str = include_str!("../../shaders/matmul.wgsl");

/// Tile size for the tiled matrix multiplication algorithm.
const TILE_SIZE: u32 = 16;

/// MatMul operator - Matrix multiplication.
///
/// Computes Y = A × B where:
/// - A: [..., M, K] - First input tensor
/// - B: [..., K, N] - Second input tensor  
/// - Y: [..., M, N] - Output tensor
///
/// Supports:
/// - Non-batched: (M, K) × (K, N) → (M, N)
/// - Batched: (B, M, K) × (B, K, N) → (B, M, N)
/// - Broadcasting: (M, K) × (B, K, N) → (B, M, N)
///
/// Uses a tiled algorithm with workgroup memory for efficient GPU execution.
pub struct MatMulOp;

/// Runtime dispatch for MatMul.
struct MatMulDispatch {
    /// Pre-compiled naga module for the shader.
    module: naga::Module,
}

impl OpDispatch for MatMulDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> Result<Vec<RuntimeTensor>> {
        let a = &inputs[0];
        let b = &inputs[1];

        // Validate inputs have at least 2 dimensions (or will be treated as such)
        if a.shape.is_empty() || b.shape.is_empty() {
            return Err(Error::Shape(
                "MatMul requires at least 1-D inputs".to_string(),
            ));
        }

        // Normalize shapes to at least 2D
        let a_shape = normalize_matmul_shape(&a.shape);
        let b_shape = normalize_matmul_shape(&b.shape);

        // Extract matrix dimensions
        let m = a_shape[a_shape.len() - 2];
        let k_a = a_shape[a_shape.len() - 1];
        let k_b = b_shape[b_shape.len() - 2];
        let n = b_shape[b_shape.len() - 1];

        // Validate inner dimensions match
        if k_a != k_b {
            return Err(Error::Shape(format!(
                "MatMul: incompatible dimensions for multiplication: \
                 A[..., {}, {}] × B[..., {}, {}]",
                m, k_a, k_b, n
            )));
        }
        let k = k_a;

        // Compute batch dimensions with broadcasting
        let a_batch = &a_shape[..a_shape.len() - 2];
        let b_batch = &b_shape[..b_shape.len() - 2];
        let output_batch = broadcast_batch_dims(a_batch, b_batch)?;
        let batch_size: usize = if output_batch.is_empty() {
            1
        } else {
            output_batch.iter().product()
        };

        // Build output shape: [...batch..., M, N]
        let mut output_shape = output_batch.clone();
        output_shape.push(m);
        output_shape.push(n);

        // Handle broadcasting by reshaping inputs if needed
        let a_batched = reshape_for_batch(a, &a_shape, batch_size, m, k, ctx)?;
        let b_batched = reshape_for_batch(b, &b_shape, batch_size, k, n, ctx)?;

        // Allocate output buffer
        let output = ctx.create_output_tensor(&output_shape, a.dtype)?;

        // Encode immediates (must match ImmediateConstants struct in shader)
        let mut immediates = Vec::with_capacity(16);
        immediates.extend_from_slice(&(m as u32).to_le_bytes());
        immediates.extend_from_slice(&(n as u32).to_le_bytes());
        immediates.extend_from_slice(&(k as u32).to_le_bytes());
        immediates.extend_from_slice(&(batch_size as u32).to_le_bytes());

        // Compute workgroups: (N/TILE_SIZE, M/TILE_SIZE, batch_size)
        let workgroups_x = (n as u32).div_ceil(TILE_SIZE);
        let workgroups_y = (m as u32).div_ceil(TILE_SIZE);
        let workgroups_z = batch_size as u32;

        // Get or create pipeline
        let (pipeline, bind_group_layout) =
            ctx.get_or_create_pipeline("MatMul", &self.module, "main")?;

        // Create bind group
        let entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_batched.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_batched.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.buffer.as_entire_binding(),
            },
        ];

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bind_group"),
            layout: &bind_group_layout,
            entries: &entries,
        });

        // Dispatch compute shader
        ctx.dispatch_compute(
            &pipeline,
            &bind_group,
            [workgroups_x, workgroups_y, workgroups_z],
            Some(&immediates),
        )?;

        Ok(vec![output])
    }
}

impl Operator for MatMulOp {
    fn name(&self) -> &str {
        "MatMul"
    }

    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        let shader_defs = HashMap::new();
        let module = ctx.compile_shader("MatMul", MATMUL_SHADER, &shader_defs)?;

        Ok(Box::new(MatMulDispatch { module }))
    }
}

/// Normalize a shape to at least 2D for matrix multiplication.
///
/// - Scalars and 1-D tensors are treated as row or column vectors.
/// - 1-D (N,) → (1, N)
fn normalize_matmul_shape(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1],        // scalar → 1×1 matrix
        1 => vec![1, shape[0]], // vector → row vector
        _ => shape.to_vec(),
    }
}

/// Broadcast batch dimensions of two tensors.
///
/// Returns the output batch shape, or an error if shapes are incompatible.
fn broadcast_batch_dims(a_batch: &[usize], b_batch: &[usize]) -> Result<Vec<usize>> {
    let max_len = a_batch.len().max(b_batch.len());
    let mut result = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let a_idx = a_batch.len().saturating_sub(max_len - i);
        let b_idx = b_batch.len().saturating_sub(max_len - i);

        let a_dim = if a_idx < a_batch.len() {
            a_batch[a_idx]
        } else {
            1
        };
        let b_dim = if b_idx < b_batch.len() {
            b_batch[b_idx]
        } else {
            1
        };

        if a_dim != b_dim && a_dim != 1 && b_dim != 1 {
            return Err(Error::Shape(format!(
                "MatMul: incompatible batch dimensions: {:?} vs {:?}",
                a_batch, b_batch
            )));
        }

        result.push(a_dim.max(b_dim));
    }

    Ok(result)
}

/// Reshape a tensor for batched matmul, handling broadcasting.
///
/// If the input needs broadcasting (its batch dims are smaller than target),
/// we replicate data by copying. Returns a tensor of shape (batch_size, rows, cols).
fn reshape_for_batch(
    tensor: &RuntimeTensor,
    normalized_shape: &[usize],
    target_batch_size: usize,
    rows: usize,
    cols: usize,
    ctx: &mut DispatchCtx,
) -> Result<RuntimeTensor> {
    let tensor_batch = &normalized_shape[..normalized_shape.len() - 2];
    let tensor_batch_size: usize = if tensor_batch.is_empty() {
        1
    } else {
        tensor_batch.iter().product()
    };

    // If batch sizes match, just reshape to 3D
    if tensor_batch_size == target_batch_size {
        let reshaped_shape = vec![target_batch_size, rows, cols];
        return Ok(RuntimeTensor {
            buffer: tensor.buffer.clone(),
            shape: reshaped_shape,
            dtype: tensor.dtype,
            size_bytes: tensor.size_bytes,
        });
    }

    // Need to broadcast: replicate the data
    // For simplicity, handle the common case where tensor_batch_size == 1
    if tensor_batch_size != 1 {
        return Err(Error::Shape(format!(
            "MatMul: complex broadcasting not yet supported: \
             tensor has batch size {}, target is {}",
            tensor_batch_size, target_batch_size
        )));
    }

    // Replicate the single matrix across all batches
    // TODO: This could be optimized with a dedicated broadcast kernel
    let matrix_size = rows * cols;
    let output_shape = vec![target_batch_size, rows, cols];

    // Download original data and replicate
    let data = ctx.download_tensor(tensor)?;
    let f32_data: &[f32] = bytemuck::cast_slice(&data);

    let mut broadcast_data = Vec::with_capacity(target_batch_size * matrix_size);
    for _ in 0..target_batch_size {
        broadcast_data.extend_from_slice(&f32_data[..matrix_size]);
    }

    // Upload broadcast data as a new tensor
    let broadcast_bytes: &[u8] = bytemuck::cast_slice(&broadcast_data);
    ctx.upload_tensor(broadcast_bytes, &output_shape, tensor.dtype)
}
