//! SimplifiedLayerNormalization operator implementation (Microsoft contrib).
//!
//! Implements RMS (Root Mean Square) normalization:
//!   output = input / RMS(input) * scale
//!   where RMS(input) = sqrt(mean(input^2) + epsilon)

use onyxia_core::{CompileCtx, DispatchCtx, Error, OpDispatch, Operator, Result, RuntimeTensor};
use std::collections::HashMap;
use std::sync::Arc;

/// Shader source for RMS computation pass.
const RMS_SHADER: &str = include_str!("../../shaders/simplified_layernorm_rms.wgsl");

/// Shader source for normalization pass.
const NORMALIZE_SHADER: &str = include_str!("../../shaders/simplified_layernorm_normalize.wgsl");

/// SimplifiedLayerNormalization operator (Microsoft contrib).
///
/// Implements RMS normalization, a simplified version of layer normalization
/// that only uses root mean square normalization without mean centering.
///
/// **ONNX Specification:**
/// - Domain: com.microsoft
/// - Inputs:
///   - X (T) - Input tensor
///   - scale (T) - Scale tensor (1D, size = normalized dimension size)
/// - Outputs:
///   - Y (T) - Normalized output tensor (same shape as X)
/// - Attributes:
///   - axis (int, default=-1) - The first normalization dimension
///   - epsilon (float, default=1e-5) - Small value to avoid division by zero
///   - stash_type (int, default=1) - Type hint for intermediate results (not used)
///
/// **Implementation:**
/// - Uses a two-pass algorithm:
///   1. Compute RMS = sqrt(mean(x^2) + epsilon) for each batch element
///   2. Apply normalization: output = input / RMS * scale
/// - Each pass uses a GPU compute shader for parallelism
pub struct SimplifiedLayerNormOp;

impl Operator for SimplifiedLayerNormOp {
    fn name(&self) -> &str {
        "com.microsoft::SimplifiedLayerNormalization"
    }

    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        // Get attributes
        let axis = match ctx.attr("axis") {
            Some(onyxia_onnx::AttributeValue::Int(v)) => *v,
            _ => -1, // default
        };

        let epsilon = match ctx.attr("epsilon") {
            Some(onyxia_onnx::AttributeValue::Float(v)) => *v,
            _ => 1e-5, // default
        };

        // Compile shaders
        let shader_defs = HashMap::new();

        let rms_module = ctx.compile_shader("SimplifiedLayerNorm_RMS", RMS_SHADER, &shader_defs)?;

        let normalize_module = ctx.compile_shader(
            "SimplifiedLayerNorm_Normalize",
            NORMALIZE_SHADER,
            &shader_defs,
        )?;

        Ok(Box::new(SimplifiedLayerNormDispatch {
            rms_module,
            normalize_module,
            axis,
            epsilon,
        }))
    }
}

/// Runtime dispatch for SimplifiedLayerNormalization operation.
struct SimplifiedLayerNormDispatch {
    /// Pre-compiled naga module for RMS computation.
    rms_module: naga::Module,

    /// Pre-compiled naga module for normalization.
    normalize_module: naga::Module,

    /// First normalization dimension.
    axis: i64,

    /// Small value to avoid division by zero.
    epsilon: f32,
}

impl OpDispatch for SimplifiedLayerNormDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> Result<Vec<RuntimeTensor>> {
        let input = &inputs[0];
        let scale = &inputs[1];

        // Normalize axis
        let rank = input.shape.len() as i64;
        let norm_axis = if self.axis < 0 {
            (rank + self.axis) as usize
        } else {
            self.axis as usize
        };

        if norm_axis >= input.shape.len() {
            return Err(Error::Shape(format!(
                "Normalization axis {} out of range for shape {:?}",
                self.axis, input.shape
            )));
        }

        // Dimensions
        let norm_size = input.shape[norm_axis..].iter().product::<usize>();
        let batch_size = input.shape[..norm_axis].iter().product::<usize>();

        // Verify scale shape
        let scale_size = scale.shape.iter().product::<usize>();
        if scale_size != norm_size {
            return Err(Error::Shape(format!(
                "Scale size {} does not match normalization size {}",
                scale_size, norm_size
            )));
        }

        // Create output tensor
        let output = ctx.create_output_tensor(&input.shape, input.dtype)?;

        // Two-pass algorithm:
        // Pass 1: Compute RMS = sqrt(mean(x^2) + epsilon) for each batch element
        // Pass 2: Normalize output = input / RMS * scale

        // Create intermediate RMS buffer [batch_size] (f32)
        let rms_buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simplified_layernorm_rms"),
            size: (batch_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Encode immediates for Pass 1
        let mut immediates = Vec::new();
        immediates.extend_from_slice(&(batch_size as u32).to_le_bytes());
        immediates.extend_from_slice(&(norm_size as u32).to_le_bytes());
        immediates.extend_from_slice(&self.epsilon.to_le_bytes());

        // Pass 1: Compute RMS
        let (rms_pipeline, rms_bind_group_layout) =
            ctx.get_or_create_pipeline("SimplifiedLayerNorm_RMS", &self.rms_module, "compute_rms")?;

        let rms_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("simplified_layernorm_rms_bind_group"),
            layout: &rms_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rms_buffer.as_entire_binding(),
                },
            ],
        });

        ctx.dispatch_compute(
            &rms_pipeline,
            &rms_bind_group,
            [(batch_size as u32).div_ceil(256), 1, 1],
            Some(&immediates),
        )?;

        // Encode immediates for Pass 2
        let mut immediates2 = Vec::new();
        immediates2.extend_from_slice(&(batch_size as u32).to_le_bytes());
        immediates2.extend_from_slice(&(norm_size as u32).to_le_bytes());

        // Pass 2: Normalize with scale
        let (normalize_pipeline, normalize_bind_group_layout) = ctx.get_or_create_pipeline(
            "SimplifiedLayerNorm_Normalize",
            &self.normalize_module,
            "apply_normalization",
        )?;

        let normalize_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("simplified_layernorm_normalize_bind_group"),
            layout: &normalize_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scale.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.buffer.as_entire_binding(),
                },
            ],
        });

        ctx.dispatch_compute(
            &normalize_pipeline,
            &normalize_bind_group,
            [((batch_size * norm_size) as u32).div_ceil(256), 1, 1],
            Some(&immediates2),
        )?;

        Ok(vec![output])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplified_layer_norm_operator_name() {
        let op = SimplifiedLayerNormOp;
        assert_eq!(op.name(), "com.microsoft::SimplifiedLayerNormalization");
    }
}
