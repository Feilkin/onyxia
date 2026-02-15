//! Normalization operators.

use onyxia_core::{BindingDesc, InferenceCtx, Operator, PlanCtx, Result, Step, TensorShape};
use std::collections::HashMap;

/// RMS Normalization operator (SimplifiedLayerNormalization in ONNX).
///
/// RMSNorm(x) = x / RMS(x) * weight where RMS(x) = sqrt(mean(x²) + epsilon).
/// Used in modern LLMs like Llama and Gemma.
pub struct RmsNormOp;

impl Operator for RmsNormOp {
    fn name(&self) -> &str {
        "SimplifiedLayerNormalization"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        // RMSNorm normalizes over the last dimension: output shape equals input shape
        if ctx.input_count() == 0 {
            return Err(onyxia_core::Error::ShapeInference(
                "RMSNorm requires at least one input".to_string(),
            ));
        }

        let input_shape = ctx.input_shape(0)?;
        Ok(vec![input_shape.clone()])
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // Extract epsilon from node attributes (default to 1e-5 if not present)
        let epsilon: f32 = ctx.attr_f32("epsilon").unwrap_or(1e-5);

        // Get input shape
        let input_tensor = ctx.input_tensor(0)?;
        let input_shape = ctx.static_dims(&input_tensor.shape)?;

        // Validate input shape
        if input_shape.len() < 2 {
            return Err(onyxia_core::Error::Planning(format!(
                "RMSNorm expects input with at least 2 dimensions, got {:?}",
                input_shape
            )));
        }

        // Calculate dimensions
        let hidden_dim = *input_shape.last().unwrap();
        let batch_seq: usize = input_shape[..input_shape.len() - 1].iter().product();

        // Split batch_seq for the shader
        let batch_size = 1u32;
        let seq_len = batch_seq as u32;

        // Configure workgroup size
        let workgroup_size: u32 = 256;
        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), workgroup_size.to_string());

        // Compile shader
        let shader_index = ctx.compile_shader(
            "rmsnorm",
            include_str!("../../shaders/normalization/rmsnorm.wgsl"),
            &shader_defs,
        )?;

        // Each workgroup handles one [batch, seq] position
        let num_workgroups = batch_seq as u32;

        // Encode immediate data
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&batch_size.to_le_bytes());
        immediates_data.extend_from_slice(&seq_len.to_le_bytes());
        immediates_data.extend_from_slice(&(hidden_dim as u32).to_le_bytes());
        immediates_data.extend_from_slice(&epsilon.to_le_bytes());

        // Create dispatch step
        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0)?,
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(1)?,
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(0)?,
                    read_only: false,
                },
            ],
            workgroups: [num_workgroups, 1, 1],
            immediates: Some(immediates_data),
        }])
    }
}
