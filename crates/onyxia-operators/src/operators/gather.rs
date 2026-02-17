//! Gather operator - gather elements from input tensor using index tensor.

use onyxia_core::{
    CompileCtx, DispatchCtx, Error, OpDispatch, Operator, Result, RuntimeTensor,
};
use std::collections::HashMap;

/// Shader source for the Gather operator.
const GATHER_SHADER: &str = include_str!("../../shaders/gather.wgsl");

/// Gather operator - gather elements from input tensor along a specified axis.
///
/// ONNX opset 13+:
/// - **Inputs**: data (T), indices (Tind)
/// - **Outputs**: output (T)
/// - **Attributes**: axis (int, default=0)
///
/// The operator gathers slices of the data tensor along the specified axis
/// according to the indices tensor. Negative indices are supported.
pub struct GatherOp;

/// Runtime dispatch for Gather.
struct GatherDispatch {
    /// Pre-compiled naga module for the shader.
    module: naga::Module,

    /// Axis to gather along.
    axis: i64,
}

impl OpDispatch for GatherDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> onyxia_core::Result<Vec<RuntimeTensor>> {
        let data_tensor = &inputs[0];
        let indices_tensor = &inputs[1];

        let data_shape = &data_tensor.shape;
        let indices_shape = &indices_tensor.shape;
        let rank = data_shape.len();

        // Normalize axis (handle negative values)
        let axis = if self.axis < 0 {
            (rank as i64 + self.axis) as usize
        } else {
            self.axis as usize
        };

        if axis >= rank {
            return Err(Error::Shape(format!(
                "Gather: axis {} out of bounds for rank {}",
                self.axis, rank
            )));
        }

        // Compute output shape:
        // output.shape = data.shape[:axis] + indices.shape + data.shape[axis+1:]
        let mut output_shape = Vec::new();
        output_shape.extend_from_slice(&data_shape[..axis]);
        output_shape.extend_from_slice(indices_shape);
        output_shape.extend_from_slice(&data_shape[axis + 1..]);

        let num_elements: usize = output_shape.iter().product();

        // Allocate output buffer
        let output = ctx.create_output_tensor(&output_shape, data_tensor.dtype)?;

        // Encode immediates
        let mut immediates = Vec::new();

        // Total output elements (u32)
        immediates.extend_from_slice(&(num_elements as u32).to_le_bytes());

        // Axis (u32)
        immediates.extend_from_slice(&(axis as u32).to_le_bytes());

        // Data rank (u32)
        immediates.extend_from_slice(&(rank as u32).to_le_bytes());

        // Indices rank (u32)
        immediates.extend_from_slice(&(indices_shape.len() as u32).to_le_bytes());

        // Output rank (u32)
        immediates.extend_from_slice(&(output_shape.len() as u32).to_le_bytes());

        // Data shape (up to 8 dimensions)
        for dim in data_shape.iter().take(8) {
            immediates.extend_from_slice(&(*dim as u32).to_le_bytes());
        }
        for _ in data_shape.len()..8 {
            immediates.extend_from_slice(&0u32.to_le_bytes());
        }

        // Indices shape (up to 8 dimensions)
        for dim in indices_shape.iter().take(8) {
            immediates.extend_from_slice(&(*dim as u32).to_le_bytes());
        }
        for _ in indices_shape.len()..8 {
            immediates.extend_from_slice(&0u32.to_le_bytes());
        }

        // Output shape (up to 8 dimensions)
        for dim in output_shape.iter().take(8) {
            immediates.extend_from_slice(&(*dim as u32).to_le_bytes());
        }
        for _ in output_shape.len()..8 {
            immediates.extend_from_slice(&0u32.to_le_bytes());
        }

        // Compute workgroup count
        let workgroup_size: u32 = 256;
        let num_workgroups = (num_elements as u32).div_ceil(workgroup_size);

        // Get or create pipeline
        let (pipeline, bind_group_layout) =
            ctx.get_or_create_pipeline("Gather", &self.module, "main")?;

        // Create bind group
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gather_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data_tensor.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: indices_tensor.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.buffer.as_entire_binding(),
                },
            ],
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

impl Operator for GatherOp {
    fn name(&self) -> &str {
        "Gather"
    }

    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        // Read axis attribute (default = 0)
        let axis = ctx.attr_i64("axis").unwrap_or(0);

        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), "256".to_string());

        let module = ctx.compile_shader("Gather", GATHER_SHADER, &shader_defs)?;

        Ok(Box::new(GatherDispatch { module, axis }))
    }
}
