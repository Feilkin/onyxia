//! Activation operators.

use onyxia_core::{BindingDesc, InferenceCtx, Operator, PlanCtx, Result, Step, TensorShape};
use std::collections::HashMap;

/// GELU (Gaussian Error Linear Unit) activation operator.
///
/// GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function
/// of the standard normal distribution.
/// Uses tanh approximation for efficiency.
pub struct GeluOp;

impl Operator for GeluOp {
    fn name(&self) -> &str {
        "Gelu"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        // Gelu is a unary operation: output shape equals input shape
        if ctx.input_count() == 0 {
            return Err(onyxia_core::Error::ShapeInference(
                "Gelu requires one input".to_string(),
            ));
        }
        let shape = ctx.input_shape(0)?;
        Ok(vec![shape.clone()])
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // Get output tensor and shape
        let output_tensor = ctx.output_tensor(0)?;
        let output_shape = ctx.static_dims(&output_tensor.shape)?;
        let num_elements: usize = output_shape.iter().product();

        // Configure workgroup size
        let workgroup_size: u32 = 256;
        let num_workgroups = (num_elements as u32).div_ceil(workgroup_size);

        // Prepare shader definitions
        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), workgroup_size.to_string());

        // Compile shader
        let shader_index = ctx.compile_shader(
            "gelu",
            include_str!("../../shaders/activation/gelu.wgsl"),
            &shader_defs,
        )?;

        // Encode immediate data
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(num_elements as u32).to_le_bytes());

        // Create dispatch step
        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0)?,
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

/// Softmax activation operator.
///
/// Computes softmax activation along a specified axis:
///   output[i] = exp(input[i]) / sum(exp(input[j]) for all j in slice)
///
/// Uses numerically stable implementation:
///   max_val = max(input along axis)
///   output[i] = exp(input[i] - max_val) / sum(exp(input[j] - max_val))
pub struct SoftmaxOp;

impl Operator for SoftmaxOp {
    fn name(&self) -> &str {
        "Softmax"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        // Softmax: output shape equals input shape
        if ctx.input_count() == 0 {
            return Err(onyxia_core::Error::ShapeInference(
                "Softmax requires one input".to_string(),
            ));
        }
        let shape = ctx.input_shape(0)?;
        Ok(vec![shape.clone()])
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // Get input shape
        let input_tensor = ctx.input_tensor(0)?;
        let input_shape = ctx.static_dims(&input_tensor.shape)?;
        let rank = input_shape.len();

        // Get axis attribute (default to -1, which means last axis)
        let axis: i64 = ctx.attr_i64("axis").unwrap_or(-1);
        let normalized_axis = if axis < 0 {
            (rank as i64 + axis) as usize
        } else {
            axis as usize
        };

        if normalized_axis >= rank {
            return Err(onyxia_core::Error::Planning(format!(
                "Softmax axis {} is out of bounds for rank {}",
                axis, rank
            )));
        }

        // Calculate dimensions
        let axis_dim = input_shape[normalized_axis];
        let outer_size: usize = input_shape[..normalized_axis].iter().product();
        let outer_size = if outer_size == 0 { 1 } else { outer_size };
        let num_elements: usize = input_shape.iter().product();

        // Configure workgroup size
        let workgroup_size: u32 = 256;
        let num_workgroups = (outer_size as u32).div_ceil(workgroup_size);

        // Prepare shader definitions
        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), workgroup_size.to_string());

        // Compile shader
        let shader_index = ctx.compile_shader(
            "softmax",
            include_str!("../../shaders/activation/softmax.wgsl"),
            &shader_defs,
        )?;

        // Encode immediate data
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(num_elements as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(outer_size as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(axis_dim as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(normalized_axis as u32).to_le_bytes());

        // Create dispatch step
        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0)?, // input
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(0)?, // output
                    read_only: false,
                },
            ],
            workgroups: [num_workgroups, 1, 1],
            immediates: Some(immediates_data),
        }])
    }
}
