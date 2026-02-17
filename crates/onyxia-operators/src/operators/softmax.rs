//! Softmax operator implementation.
//!
//! Converts a vector of real values into a probability distribution using
//! the softmax function with numerical stability via the max-trick.

use onyxia_core::{CompileCtx, DispatchCtx, Error, OpDispatch, Operator, Result, RuntimeTensor};
use std::collections::HashMap;
use std::sync::Arc;

/// Shader sources for the Softmax operator (three-pass algorithm).
const SOFTMAX_MAX_SHADER: &str = include_str!("../../shaders/softmax_max.wgsl");
const SOFTMAX_EXP_SUM_SHADER: &str = include_str!("../../shaders/softmax_exp_sum.wgsl");
const SOFTMAX_NORMALIZE_SHADER: &str = include_str!("../../shaders/softmax_normalize.wgsl");

/// Softmax operator.
///
/// Applies the softmax activation function along a specified axis, converting
/// input values into a probability distribution.
///
/// **ONNX Specification:**
/// - Opset: 13
/// - Inputs:
///   - input (T) - Input tensor
/// - Outputs:
///   - output (T) - Output tensor (same shape as input)
/// - Attributes:
///   - axis (int, default=-1) - Axis along which to compute softmax
///
/// **Implementation:**
/// - Uses a three-pass algorithm for numerical stability:
///   1. Find maximum value along axis
///   2. Compute exp(x - max) and sum
///   3. Normalize: output = exp(x - max) / sum
/// - The max-trick prevents overflow in exp() for large values
pub struct SoftmaxOp;

impl Operator for SoftmaxOp {
    fn name(&self) -> &str {
        "Softmax"
    }

    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        // Read axis attribute (default=-1, meaning last axis)
        let axis = match ctx.attr("axis") {
            Some(onyxia_onnx::AttributeValue::Int(v)) => *v,
            _ => -1, // default is -1 (last axis)
        };

        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), "256".to_string());

        // Compile all three shader passes
        let max_module = ctx.compile_shader("Softmax_Max", SOFTMAX_MAX_SHADER, &shader_defs)?;
        let exp_sum_module =
            ctx.compile_shader("Softmax_ExpSum", SOFTMAX_EXP_SUM_SHADER, &shader_defs)?;
        let normalize_module =
            ctx.compile_shader("Softmax_Normalize", SOFTMAX_NORMALIZE_SHADER, &shader_defs)?;

        Ok(Box::new(SoftmaxDispatch {
            max_module,
            exp_sum_module,
            normalize_module,
            label: "Softmax".to_string(),
            axis,
        }))
    }
}

/// Runtime dispatch for Softmax operation.
struct SoftmaxDispatch {
    /// Pre-compiled naga module for finding max.
    max_module: naga::Module,

    /// Pre-compiled naga module for computing exp sum.
    exp_sum_module: naga::Module,

    /// Pre-compiled naga module for normalization.
    normalize_module: naga::Module,

    /// Label for pipeline caching.
    label: String,

    /// Axis along which to compute softmax (negative values wrap around).
    axis: i64,
}

impl OpDispatch for SoftmaxDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> Result<Vec<RuntimeTensor>> {
        let input = &inputs[0];

        // Normalize negative axis
        let ndim = input.shape.len() as i64;
        let normalized_axis = if self.axis < 0 {
            ndim + self.axis
        } else {
            self.axis
        };

        if normalized_axis < 0 || normalized_axis >= ndim {
            return Err(Error::Shape(format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                self.axis, ndim
            )));
        }

        let axis = normalized_axis as usize;

        // Compute outer_size, axis_size, inner_size
        let outer_size: usize = input.shape[..axis].iter().product::<usize>().max(1);
        let axis_size: usize = input.shape[axis];
        let inner_size: usize = input.shape[axis + 1..].iter().product::<usize>().max(1);

        // Output has the same shape as input
        let output = ctx.create_output_tensor(&input.shape, input.dtype)?;

        // Allocate intermediate buffers for max_values and exp_sums
        // Shape: (outer_size, inner_size)
        let intermediate_size = outer_size * inner_size;

        let max_values = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("softmax_max_values"),
            size: (intermediate_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let exp_sums = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("softmax_exp_sums"),
            size: (intermediate_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Encode immediates (must match ImmediateConstants in shaders)
        let mut immediates = Vec::with_capacity(16);
        immediates.extend_from_slice(&(outer_size as u32).to_le_bytes());
        immediates.extend_from_slice(&(axis_size as u32).to_le_bytes());
        immediates.extend_from_slice(&(inner_size as u32).to_le_bytes());
        immediates.extend_from_slice(&0u32.to_le_bytes()); // padding

        // Pass 1: Find max along axis
        let (max_pipeline, max_bind_group_layout) =
            ctx.get_or_create_pipeline(&format!("{}_Max", self.label), &self.max_module, "main")?;

        let max_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax_max_bind_group"),
            layout: &max_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: max_values.as_entire_binding(),
                },
            ],
        });

        ctx.dispatch_compute(
            &max_pipeline,
            &max_bind_group,
            [intermediate_size as u32, 1, 1],
            Some(&immediates),
        )?;

        // Pass 2: Compute exp sum
        let (exp_sum_pipeline, exp_sum_bind_group_layout) = ctx.get_or_create_pipeline(
            &format!("{}_ExpSum", self.label),
            &self.exp_sum_module,
            "main",
        )?;

        let exp_sum_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax_exp_sum_bind_group"),
            layout: &exp_sum_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: max_values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: exp_sums.as_entire_binding(),
                },
            ],
        });

        ctx.dispatch_compute(
            &exp_sum_pipeline,
            &exp_sum_bind_group,
            [intermediate_size as u32, 1, 1],
            Some(&immediates),
        )?;

        // Pass 3: Normalize
        let (normalize_pipeline, normalize_bind_group_layout) = ctx.get_or_create_pipeline(
            &format!("{}_Normalize", self.label),
            &self.normalize_module,
            "main",
        )?;

        let normalize_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax_normalize_bind_group"),
            layout: &normalize_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: max_values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: exp_sums.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.buffer.as_entire_binding(),
                },
            ],
        });

        let total_size = input.shape.iter().product::<usize>();
        let num_workgroups = total_size.div_ceil(256);

        ctx.dispatch_compute(
            &normalize_pipeline,
            &normalize_bind_group,
            [num_workgroups as u32, 1, 1],
            Some(&immediates),
        )?;

        Ok(vec![output])
    }
}
