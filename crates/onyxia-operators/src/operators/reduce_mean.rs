//! ReduceMean operator implementation.
//!
//! Computes the arithmetic mean of input tensor elements along specified axes.

use onyxia_core::{CompileCtx, DispatchCtx, Error, OpDispatch, Operator, Result, RuntimeTensor};
use std::collections::HashMap;

/// Shader source for the ReduceMean operator.
const REDUCE_MEAN_SHADER: &str = include_str!("../../shaders/reduce_mean.wgsl");

/// ReduceMean operator.
///
/// Computes the arithmetic mean of input tensor elements across specified dimensions.
///
/// **ONNX Specification:**
/// - Opset: 18
/// - Inputs:
///   - data (T) - Input tensor
///   - axes (optional, tensor(int64)) - Axes to reduce over
/// - Outputs:
///   - reduced (T) - Reduced tensor
/// - Attributes:
///   - keepdims (int, default=1) - Keep reduced dimensions as size 1
///   - noop_with_empty_axes (int, default=0) - Identity operation if axes is empty
///
/// **Implementation:**
/// - Uses a two-pass reduction algorithm for numerical stability
/// - Pass 1: Sum all elements along reduction axes
/// - Pass 2: Divide by the number of elements
pub struct ReduceMeanOp;

impl Operator for ReduceMeanOp {
    fn name(&self) -> &str {
        "ReduceMean"
    }

    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        // Read axes from:
        // 1. Second input if provided (as constant tensor)
        // 2. Attribute "axes" (older ONNX versions)
        // 3. Default to empty (reduce all axes)
        let axes: Vec<i64> = if ctx.input_count() > 1 {
            // Try to read from constant input
            if let Some(value) = ctx.input_value(1)? {
                value
                    .as_i64()
                    .ok_or_else(|| Error::Compilation("axes input must be i64".into()))?
                    .to_vec()
            } else {
                // Axes is a runtime input, not supported yet
                return Err(Error::Unsupported(
                    "ReduceMean with runtime axes is not yet supported".into(),
                ));
            }
        } else if let Some(onyxia_onnx::AttributeValue::Ints(axes_attr)) = ctx.attr("axes") {
            axes_attr.clone()
        } else {
            // Empty axes (noop_with_empty_axes will determine behavior)
            vec![]
        };

        // Read keepdims attribute (default=1)
        let keepdims = match ctx.attr("keepdims") {
            Some(onyxia_onnx::AttributeValue::Int(v)) => *v != 0,
            _ => true, // default is 1 (true)
        };

        // Read noop_with_empty_axes attribute (default=0)
        let noop_with_empty_axes = match ctx.attr("noop_with_empty_axes") {
            Some(onyxia_onnx::AttributeValue::Int(v)) => *v != 0,
            _ => false, // default is 0 (false)
        };

        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), "256".to_string());

        let module = ctx.compile_shader("ReduceMean", REDUCE_MEAN_SHADER, &shader_defs)?;

        Ok(Box::new(ReduceMeanDispatch {
            module,
            label: "ReduceMean".to_string(),
            axes,
            keepdims,
            noop_with_empty_axes,
        }))
    }
}

/// Runtime dispatch for ReduceMean operation.
struct ReduceMeanDispatch {
    /// Pre-compiled naga module for the shader.
    module: naga::Module,

    /// Label for pipeline caching.
    label: String,

    /// Axes to reduce over (empty means reduce all axes if noop_with_empty_axes=false).
    axes: Vec<i64>,

    /// Keep reduced dimensions as size 1.
    keepdims: bool,

    /// If true and axes is empty, perform identity operation.
    noop_with_empty_axes: bool,
}

impl OpDispatch for ReduceMeanDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> Result<Vec<RuntimeTensor>> {
        let data = &inputs[0];

        // Handle noop_with_empty_axes
        if self.axes.is_empty() && self.noop_with_empty_axes {
            // Identity operation
            return Ok(vec![data.clone()]);
        }

        // Determine axes to reduce (all axes if empty)
        let axes_to_reduce: Vec<i64> = if self.axes.is_empty() {
            (0..data.shape.len() as i64).collect()
        } else {
            self.axes.clone()
        };

        // Normalize negative axes
        let ndim = data.shape.len() as i64;
        let normalized_axes: Vec<usize> = axes_to_reduce
            .iter()
            .map(|&axis| {
                let normalized = if axis < 0 { ndim + axis } else { axis };
                if normalized < 0 || normalized >= ndim {
                    return Err(Error::Shape(format!(
                        "Axis {} out of bounds for tensor with {} dimensions",
                        axis, ndim
                    )));
                }
                Ok(normalized as usize)
            })
            .collect::<Result<Vec<_>>>()?;

        // Compute output shape
        let mut output_shape = Vec::new();
        let mut reduction_size: usize = 1;

        for (i, &dim) in data.shape.iter().enumerate() {
            if normalized_axes.contains(&i) {
                reduction_size *= dim;
                if self.keepdims {
                    output_shape.push(1);
                }
            } else {
                output_shape.push(dim);
            }
        }

        // Handle scalar output (all axes reduced, keepdims=false)
        if output_shape.is_empty() {
            output_shape.push(1);
        }

        let output_size: usize = output_shape.iter().product();
        let input_size: usize = data.shape.iter().product();

        // Compute stride for reduction indexing
        // Stride is the product of all dimensions after the first reduced axis
        // For reducing axis k in shape [d0, d1, ..., dk, ..., dn]:
        //   stride = product of [dk+1, ..., dn]
        //
        // Special cases:
        // - Reducing last axis: stride = 1 (contiguous)
        // - Reducing all axes: stride = 1 (doesn't matter)
        // - Reducing middle axis: stride = product of trailing dims
        let first_reduced_axis = normalized_axes.iter().min().copied().unwrap_or(0);
        let stride: usize = data.shape[(first_reduced_axis + 1)..]
            .iter()
            .product::<usize>()
            .max(1); // Ensure stride is at least 1

        // Allocate output buffer
        let output = ctx.create_output_tensor(&output_shape, data.dtype)?;

        // Encode immediates (must match ImmediateConstants in shader)
        let mut immediates = Vec::with_capacity(16);
        immediates.extend_from_slice(&(input_size as u32).to_le_bytes());
        immediates.extend_from_slice(&(output_size as u32).to_le_bytes());
        immediates.extend_from_slice(&(reduction_size as u32).to_le_bytes());
        immediates.extend_from_slice(&(stride as u32).to_le_bytes());

        // Compute workgroups (one per output element)
        let num_workgroups = output_size as u32;

        // Get or create pipeline
        let (pipeline, bind_group_layout) =
            ctx.get_or_create_pipeline(&self.label, &self.module, "main")?;

        // Create bind group
        let entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: data.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.buffer.as_entire_binding(),
            },
        ];

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reduce_mean_bind_group"),
            layout: &bind_group_layout,
            entries: &entries,
        });

        // Dispatch compute shader
        ctx.dispatch_compute(
            &pipeline,
            &bind_group,
            [num_workgroups, 1, 1],
            Some(&immediates),
        )?;

        Ok(vec![output])
    }
}
