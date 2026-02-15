//! Reduction operator family.
//!
//! Covers: ReduceSum, ReduceMean

use onyxia_core::{BindingDesc, Error, InferenceCtx, Operator, PlanCtx, Result, Step, TensorShape};
use std::collections::HashMap;

/// Reduction operator family.
///
/// All reduction operations share the same structure:
/// - Axis-based shape inference (reduce dimensions specified by axes attribute)
/// - Reduction computation for constant folding
/// - WGSL shader dispatch for GPU execution
///
/// The only differences are:
/// - Shader source code (sum vs mean computation)
/// - Fold function (sum vs mean)
pub struct ReductionOp {
    name: &'static str,
    shader_source: &'static str,
    #[allow(dead_code)]
    fold_fn: fn(&[f32]) -> f32,
}

impl ReductionOp {
    /// Create a ReduceSum operator.
    pub fn reduce_sum() -> Self {
        Self {
            name: "ReduceSum",
            shader_source: include_str!("../../shaders/reduction/reducesum.wgsl"),
            fold_fn: |values| values.iter().sum(),
        }
    }

    /// Create a ReduceMean operator.
    pub fn reduce_mean() -> Self {
        Self {
            name: "ReduceMean",
            shader_source: include_str!("../../shaders/reduction/reducemean.wgsl"),
            fold_fn: |values| {
                let sum: f32 = values.iter().sum();
                sum / values.len() as f32
            },
        }
    }
}

impl Operator for ReductionOp {
    fn name(&self) -> &str {
        self.name
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        if ctx.input_count() == 0 {
            return Err(Error::ShapeInference(format!(
                "{} requires at least one input",
                self.name
            )));
        }

        // Get input shape
        let input_dims = ctx.require_static(0)?;

        // Get axes to reduce over
        // For ONNX opset 13+, axes can be:
        // - From second input (optional, for opset 18+)
        // - From "axes" attribute (for older opsets)
        // - Empty means reduce all axes
        let axes: Vec<i64> = if ctx.input_count() > 1 {
            // Axes from second input - try to read from constant-folded value
            if let Some(val) = ctx.input_value(1) {
                if let Some(axes_i64) = val.as_i64() {
                    axes_i64.to_vec()
                } else if let Some(axes_i32) = val.as_i32() {
                    axes_i32.iter().map(|&v| v as i64).collect()
                } else {
                    // Axes value has wrong type
                    return Err(Error::ShapeInference(format!(
                        "{}: axes must be i64 or i32, got {:?}",
                        self.name, val
                    )));
                }
            } else {
                // Axes value not available at compile time - error
                return Err(Error::ShapeInference(format!(
                    "{}: axes input must be a compile-time constant",
                    self.name
                )));
            }
        } else {
            // Try to get axes from attribute
            ctx.attr_ints("axes")
                .ok()
                .map(|v| v.to_vec())
                .unwrap_or_else(|| {
                    // Default: reduce all axes
                    (0..input_dims.len() as i64).collect()
                })
        };

        // Get keepdims (defaults to 1 in ONNX)
        let keepdims = ctx.attr_i64_or("keepdims", 1);

        // Compute output shape
        let mut output_dims = Vec::new();
        for (i, &dim) in input_dims.iter().enumerate() {
            if axes.contains(&(i as i64)) || axes.contains(&(i as i64 - input_dims.len() as i64)) {
                // This axis is reduced
                if keepdims == 1 {
                    output_dims.push(1);
                }
                // If keepdims == 0, don't include this dimension
            } else {
                // This axis is preserved
                output_dims.push(dim);
            }
        }

        // If all axes reduced and keepdims=0, output is a scalar (shape [])
        if output_dims.is_empty() {
            output_dims.push(1); // Represent scalar as [1]
        }

        Ok(vec![TensorShape::Static(output_dims)])
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // For MVP, only support single-axis reduction
        let axes: Vec<i64> = ctx
            .attr_ints("axes")
            .ok()
            .map(|v| v.to_vec())
            .unwrap_or_else(|| vec![1]);

        if axes.len() != 1 {
            return Err(Error::Planning(format!(
                "{} currently only supports single-axis reduction, got {} axes",
                self.name,
                axes.len()
            )));
        }

        let axis = axes[0];

        // Get input and output shapes
        let input_tensor = ctx.input_tensor(0)?;
        let input_shape = ctx.static_dims(&input_tensor.shape)?;

        let output_tensor = ctx.output_tensor(0)?;
        let output_shape = ctx.static_dims(&output_tensor.shape)?;

        // Normalize negative axis
        let rank = input_shape.len() as i64;
        let normalized_axis = if axis < 0 { rank + axis } else { axis } as usize;

        if normalized_axis >= input_shape.len() {
            return Err(Error::Planning(format!(
                "Reduction axis {} out of bounds for rank {}",
                axis, rank
            )));
        }

        // Calculate dimension sizes
        let outer_size: usize = input_shape[..normalized_axis].iter().product();
        let reduce_size: usize = input_shape[normalized_axis];
        let inner_size: usize = input_shape[normalized_axis + 1..].iter().product();

        let input_size: usize = input_shape.iter().product();
        let output_size: usize = output_shape.iter().product();

        // Configure workgroup size
        let workgroup_size: u32 = 256;

        // Number of workgroups = number of output elements
        // Each workgroup computes one output element by summing along the reduction axis
        let num_workgroups = output_size as u32;

        // Prepare shader definitions
        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), workgroup_size.to_string());

        // Compile shader
        let shader_index = ctx.compile_shader(self.name, self.shader_source, &shader_defs)?;

        // Encode immediate data (must match ImmediateConstants struct in shader)
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(input_size as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(output_size as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(reduce_size as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(outer_size as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(inner_size as u32).to_le_bytes());

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
