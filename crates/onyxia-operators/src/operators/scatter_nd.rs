//! ScatterND operator - scatter updates into tensor at specified indices.

use onyxia_core::{
    CompileCtx, DispatchCtx, Error, OpDispatch, Operator, Result, RuntimeTensor,
};
use std::collections::HashMap;

/// Shader source for ScatterND copy phase (copy data to output).
const SCATTER_ND_COPY_SHADER: &str = include_str!("../../shaders/scatter_nd_copy.wgsl");

/// Shader source for ScatterND scatter phase (scatter updates to output).
const SCATTER_ND_SCATTER_SHADER: &str = include_str!("../../shaders/scatter_nd_scatter.wgsl");

/// ScatterND operator - scatter updates into output tensor at locations specified by indices.
///
/// ONNX opset 18+:
/// - **Inputs**: data (T), indices (tensor(int64)), updates (T)
/// - **Outputs**: output (T)
/// - **Attributes**: reduction (string, default="none")
///
/// The operator creates a copy of the data tensor with updates scattered at the
/// specified index locations. The indices tensor specifies where each update should go.
pub struct ScatterNDOp;

/// Runtime dispatch for ScatterND.
struct ScatterNDDispatch {
    /// Pre-compiled naga module for the copy phase.
    copy_module: naga::Module,

    /// Pre-compiled naga module for the scatter phase.
    scatter_module: naga::Module,

    /// Reduction mode as u32 (0=none, 1=add, 2=mul, 3=max, 4=min).
    reduction_mode: u32,
}

impl OpDispatch for ScatterNDDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> onyxia_core::Result<Vec<RuntimeTensor>> {
        let data_tensor = &inputs[0];
        let indices_tensor = &inputs[1];
        let updates_tensor = &inputs[2];

        let data_shape = &data_tensor.shape;
        let indices_shape = &indices_tensor.shape;
        let updates_shape = &updates_tensor.shape;

        let data_rank = data_shape.len();
        let indices_rank = indices_shape.len();

        // Validate indices shape: last dimension is the index tuple length (K)
        if indices_rank == 0 {
            return Err(Error::Shape(
                "ScatterND: indices tensor must have at least 1 dimension".into(),
            ));
        }

        let k = indices_shape[indices_rank - 1];
        if k > data_rank {
            return Err(Error::Shape(format!(
                "ScatterND: indices last dimension ({}) exceeds data rank ({})",
                k, data_rank
            )));
        }

        // Validate updates shape
        // updates.shape should be: indices.shape[:-1] + data.shape[k:]
        let expected_updates_rank = indices_rank - 1 + data_rank - k;
        if updates_shape.len() != expected_updates_rank {
            return Err(Error::Shape(format!(
                "ScatterND: expected updates rank {} but got {}",
                expected_updates_rank,
                updates_shape.len()
            )));
        }

        // Check the prefix matches indices shape (excluding last dim)
        for i in 0..(indices_rank - 1) {
            if updates_shape[i] != indices_shape[i] {
                return Err(Error::Shape(format!(
                    "ScatterND: updates dim {} is {} but expected {} (from indices)",
                    i, updates_shape[i], indices_shape[i]
                )));
            }
        }

        // Check the suffix matches data shape
        for i in 0..(data_rank - k) {
            let update_idx = indices_rank - 1 + i;
            let data_idx = k + i;
            if updates_shape[update_idx] != data_shape[data_idx] {
                return Err(Error::Shape(format!(
                    "ScatterND: updates dim {} is {} but expected {} (from data)",
                    update_idx, updates_shape[update_idx], data_shape[data_idx]
                )));
            }
        }

        // Output has the same shape as data
        let output_shape = data_shape.clone();
        let num_elements: usize = output_shape.iter().product();

        // Allocate output buffer
        let output = ctx.create_output_tensor(&output_shape, data_tensor.dtype)?;

        // Phase 1: Copy data to output (simple element-wise copy)
        {
            let mut copy_immediates = Vec::new();
            copy_immediates.extend_from_slice(&(num_elements as u32).to_le_bytes());

            let workgroup_size: u32 = 256;
            let num_workgroups = (num_elements as u32).div_ceil(workgroup_size);

            let (pipeline, bind_group_layout) =
                ctx.get_or_create_pipeline("ScatterND_Copy", &self.copy_module, "main")?;

            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scatter_nd_copy_bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: data_tensor.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output.buffer.as_entire_binding(),
                    },
                ],
            });

            ctx.dispatch_compute(
                &pipeline,
                &bind_group,
                [num_workgroups, 1, 1],
                Some(&copy_immediates),
            )?;
        }

        // Phase 2: Scatter updates to output
        {
            let updates_elements: usize = updates_shape.iter().product();
            let mut scatter_immediates = Vec::new();

            // Total updates elements (u32)
            scatter_immediates.extend_from_slice(&(updates_elements as u32).to_le_bytes());

            // K - number of indexed dimensions (u32)
            scatter_immediates.extend_from_slice(&(k as u32).to_le_bytes());

            // Data rank (u32)
            scatter_immediates.extend_from_slice(&(data_rank as u32).to_le_bytes());

            // Indices rank (u32)
            scatter_immediates.extend_from_slice(&(indices_rank as u32).to_le_bytes());

            // Updates rank (u32)
            scatter_immediates.extend_from_slice(&(updates_shape.len() as u32).to_le_bytes());

            // Reduction mode (u32): 0=none, 1=add, 2=mul, 3=max, 4=min
            scatter_immediates.extend_from_slice(&self.reduction_mode.to_le_bytes());

            // Data shape (up to 8 dimensions)
            for dim in data_shape.iter().take(8) {
                scatter_immediates.extend_from_slice(&(*dim as u32).to_le_bytes());
            }
            for _ in data_shape.len()..8 {
                scatter_immediates.extend_from_slice(&0u32.to_le_bytes());
            }

            // Indices shape (up to 8 dimensions)
            for dim in indices_shape.iter().take(8) {
                scatter_immediates.extend_from_slice(&(*dim as u32).to_le_bytes());
            }
            for _ in indices_shape.len()..8 {
                scatter_immediates.extend_from_slice(&0u32.to_le_bytes());
            }

            // Updates shape (up to 8 dimensions)
            for dim in updates_shape.iter().take(8) {
                scatter_immediates.extend_from_slice(&(*dim as u32).to_le_bytes());
            }
            for _ in updates_shape.len()..8 {
                scatter_immediates.extend_from_slice(&0u32.to_le_bytes());
            }

            let workgroup_size: u32 = 256;
            let num_workgroups = (updates_elements as u32).div_ceil(workgroup_size);

            let (pipeline, bind_group_layout) =
                ctx.get_or_create_pipeline("ScatterND_Scatter", &self.scatter_module, "main")?;

            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scatter_nd_scatter_bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: indices_tensor.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: updates_tensor.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buffer.as_entire_binding(),
                    },
                ],
            });

            ctx.dispatch_compute(
                &pipeline,
                &bind_group,
                [num_workgroups, 1, 1],
                Some(&scatter_immediates),
            )?;
        }

        Ok(vec![output])
    }
}

impl Operator for ScatterNDOp {
    fn name(&self) -> &str {
        "ScatterND"
    }

    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        // Read reduction attribute (default = "none")
        let reduction = ctx.attr_string("reduction").unwrap_or("none");

        // Convert reduction mode to u32
        let reduction_mode = match reduction {
            "none" => 0u32,
            "add" => 1u32,
            "mul" => 2u32,
            "max" => 3u32,
            "min" => 4u32,
            _ => 0u32,
        };

        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), "256".to_string());

        let copy_module =
            ctx.compile_shader("ScatterND_Copy", SCATTER_ND_COPY_SHADER, &shader_defs)?;
        let scatter_module =
            ctx.compile_shader("ScatterND_Scatter", SCATTER_ND_SCATTER_SHADER, &shader_defs)?;

        Ok(Box::new(ScatterNDDispatch {
            copy_module,
            scatter_module,
            reduction_mode,
        }))
    }
}