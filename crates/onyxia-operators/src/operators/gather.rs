//! Gather operator - gather elements from input tensor using index tensor.

use onyxia_core::{CompileCtx, DispatchCtx, Error, OpDispatch, Operator, Result, RuntimeTensor};
use std::collections::HashMap;
use std::sync::Arc;

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

        // Fallback path for non-F32 data tensors (e.g., I64 shape subgraphs).
        // GPU shader currently supports f32 data only.
        if data_tensor.dtype != onyxia_core::DataType::F32 {
            let gathered = cpu_gather(ctx, data_tensor, indices_tensor, axis, &output_shape)?;
            return Ok(vec![gathered]);
        }

        // Shader expects i32 indices. Convert i64 indices to i32 when needed.
        let indices_for_gpu = if indices_tensor.dtype == onyxia_core::DataType::I32 {
            Arc::clone(&indices_tensor.buffer)
        } else if indices_tensor.dtype == onyxia_core::DataType::I64 {
            let indices_data = ctx.download_tensor(indices_tensor)?;
            let indices_i64: &[i64] = bytemuck::cast_slice(&indices_data);
            let mut indices_i32 = Vec::with_capacity(indices_i64.len());
            for &v in indices_i64 {
                if v < i32::MIN as i64 || v > i32::MAX as i64 {
                    return Err(Error::Shape(format!(
                        "Gather index {} does not fit in i32",
                        v
                    )));
                }
                indices_i32.push(v as i32);
            }
            let indices_i32_bytes: &[u8] = bytemuck::cast_slice(&indices_i32);
            let converted = ctx.upload_tensor(
                indices_i32_bytes,
                &indices_tensor.shape,
                onyxia_core::DataType::I32,
            )?;
            Arc::clone(&converted.buffer)
        } else {
            return Err(Error::Shape(format!(
                "Gather indices must be I32 or I64, got {:?}",
                indices_tensor.dtype
            )));
        };

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

        // Compute workgroup count with 2D dispatch support for large tensors
        let workgroup_size: u32 = 256;
        let num_workgroups = (num_elements as u32).div_ceil(workgroup_size);
        let (dispatch_size, x_stride) =
            DispatchCtx::compute_dispatch_size(num_workgroups, workgroup_size);

        // Add x_stride for 2D dispatch
        immediates.extend_from_slice(&x_stride.to_le_bytes());

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
                    resource: indices_for_gpu.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader
        ctx.dispatch_compute(&pipeline, &bind_group, dispatch_size, Some(&immediates))?;

        Ok(vec![output])
    }
}

fn read_indices_i64(indices_data: &[u8], dtype: onyxia_core::DataType) -> Result<Vec<i64>> {
    match dtype {
        onyxia_core::DataType::I64 => Ok(bytemuck::cast_slice(indices_data).to_vec()),
        onyxia_core::DataType::I32 => Ok(bytemuck::cast_slice::<u8, i32>(indices_data)
            .iter()
            .map(|&v| v as i64)
            .collect()),
        _ => Err(Error::Shape(format!(
            "Gather indices must be I32 or I64, got {:?}",
            dtype
        ))),
    }
}

fn element_size(dtype: onyxia_core::DataType) -> Result<usize> {
    match dtype {
        onyxia_core::DataType::F32 | onyxia_core::DataType::I32 => Ok(4),
        onyxia_core::DataType::I64 => Ok(8),
        _ => Err(Error::Runtime(format!(
            "Gather CPU fallback does not support dtype {:?}",
            dtype
        ))),
    }
}

fn cpu_gather(
    ctx: &mut DispatchCtx,
    data_tensor: &RuntimeTensor,
    indices_tensor: &RuntimeTensor,
    axis: usize,
    output_shape: &[usize],
) -> Result<RuntimeTensor> {
    let data = ctx.download_tensor(data_tensor)?;
    let indices_bytes = ctx.download_tensor(indices_tensor)?;
    let indices = read_indices_i64(&indices_bytes, indices_tensor.dtype)?;

    let data_shape = &data_tensor.shape;
    let axis_dim = data_shape[axis] as i64;
    let outer: usize = data_shape[..axis].iter().product();
    let inner: usize = data_shape[axis + 1..].iter().product();
    let indices_count: usize = indices_tensor.shape.iter().product();
    let elem_size = element_size(data_tensor.dtype)?;

    let mut output = vec![0u8; output_shape.iter().product::<usize>() * elem_size];
    let chunk_bytes = inner * elem_size;

    for outer_idx in 0..outer {
        for (indices_pos, &raw_index) in indices.iter().enumerate().take(indices_count) {
            let mut idx = raw_index;
            if idx < 0 {
                idx += axis_dim;
            }
            if idx < 0 || idx >= axis_dim {
                return Err(Error::Shape(format!(
                    "Gather index {} is out of bounds for axis size {}",
                    raw_index, axis_dim
                )));
            }

            let src_elem_offset = ((outer_idx * axis_dim as usize) + idx as usize) * inner;
            let dst_elem_offset = (outer_idx * indices_count + indices_pos) * inner;

            let src_start = src_elem_offset * elem_size;
            let src_end = src_start + chunk_bytes;
            let dst_start = dst_elem_offset * elem_size;
            let dst_end = dst_start + chunk_bytes;

            output[dst_start..dst_end].copy_from_slice(&data[src_start..src_end]);
        }
    }

    ctx.upload_tensor(&output, output_shape, data_tensor.dtype)
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
