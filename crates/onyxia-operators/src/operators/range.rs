//! Range operator - generate a sequence of numbers.
//!
//! The Range operator generates a 1-D tensor containing a sequence of numbers
//! starting from a start value, incrementing by a delta value, and stopping
//! before reaching a limit value.

use onyxia_core::{
    CompileCtx, DataType, DispatchCtx, Error, OpDispatch, Operator, Result, RuntimeTensor,
};
use std::collections::HashMap;

/// Shader source for the Range operator.
const RANGE_SHADER: &str = include_str!("../../shaders/range.wgsl");

/// Range operator - generate a sequence of numbers.
///
/// **ONNX Specification (opset 11):**
/// - **Inputs:**
///   - start (T) - Scalar start value
///   - limit (T) - Scalar limit value (exclusive)
///   - delta (T) - Scalar step size
/// - **Outputs:**
///   - output (T) - 1-D tensor containing the sequence
/// - **Type constraints:** T = float, double, int16, int32, int64
///
/// **Behavior:**
/// - Generate sequence `[start, start+delta, start+2*delta, ...]` up to but not including limit
/// - Output length = `ceil((limit - start) / delta)`
/// - Empty range if `start >= limit` with positive delta
/// - Negative delta requires `start > limit`
pub struct RangeOp;

impl Operator for RangeOp {
    fn name(&self) -> &str {
        "Range"
    }

    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), "256".to_string());

        let module = ctx.compile_shader("Range", RANGE_SHADER, &shader_defs)?;

        Ok(Box::new(RangeDispatch {
            module,
            label: "Range".to_string(),
        }))
    }
}

/// Runtime dispatch for Range operation.
struct RangeDispatch {
    /// Pre-compiled naga module.
    module: naga::Module,

    /// Label for pipeline caching.
    label: String,
}

impl OpDispatch for RangeDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> Result<Vec<RuntimeTensor>> {
        if inputs.len() != 3 {
            return Err(Error::Runtime(format!(
                "Range: expected 3 inputs, got {}",
                inputs.len()
            )));
        }

        let start_tensor = &inputs[0];
        let limit_tensor = &inputs[1];
        let delta_tensor = &inputs[2];

        // All inputs must be scalars
        if start_tensor.shape.iter().product::<usize>() != 1
            || limit_tensor.shape.iter().product::<usize>() != 1
            || delta_tensor.shape.iter().product::<usize>() != 1
        {
            return Err(Error::Shape(
                "Range: all inputs must be scalars".to_string(),
            ));
        }

        // All inputs must have the same dtype
        if start_tensor.dtype != limit_tensor.dtype || start_tensor.dtype != delta_tensor.dtype {
            return Err(Error::Runtime(
                "Range: all inputs must have the same dtype".to_string(),
            ));
        }

        let dtype = start_tensor.dtype;

        // Download scalar values from GPU
        let start_data = ctx.download_tensor(start_tensor)?;
        let limit_data = ctx.download_tensor(limit_tensor)?;
        let delta_data = ctx.download_tensor(delta_tensor)?;

        // Parse scalars based on dtype
        let (start, _limit, delta, length) = match dtype {
            DataType::F32 => {
                let start = f32::from_le_bytes(start_data[0..4].try_into().unwrap());
                let limit = f32::from_le_bytes(limit_data[0..4].try_into().unwrap());
                let delta = f32::from_le_bytes(delta_data[0..4].try_into().unwrap());

                if delta == 0.0 {
                    return Err(Error::Runtime("Range: delta cannot be zero".to_string()));
                }

                let length = if (delta > 0.0 && start >= limit) || (delta < 0.0 && start <= limit) {
                    0
                } else {
                    ((limit - start) / delta).ceil() as usize
                };

                (start, limit, delta, length)
            }
            DataType::I64 => {
                let start = i64::from_le_bytes(start_data[0..8].try_into().unwrap()) as f32;
                let limit = i64::from_le_bytes(limit_data[0..8].try_into().unwrap()) as f32;
                let delta = i64::from_le_bytes(delta_data[0..8].try_into().unwrap()) as f32;

                if delta == 0.0 {
                    return Err(Error::Runtime("Range: delta cannot be zero".to_string()));
                }

                let length = if (delta > 0.0 && start >= limit) || (delta < 0.0 && start <= limit) {
                    0
                } else {
                    ((limit - start) / delta).ceil() as usize
                };

                (start, limit, delta, length)
            }
            DataType::I32 => {
                let start = i32::from_le_bytes(start_data[0..4].try_into().unwrap()) as f32;
                let limit = i32::from_le_bytes(limit_data[0..4].try_into().unwrap()) as f32;
                let delta = i32::from_le_bytes(delta_data[0..4].try_into().unwrap()) as f32;

                if delta == 0.0 {
                    return Err(Error::Runtime("Range: delta cannot be zero".to_string()));
                }

                let length = if (delta > 0.0 && start >= limit) || (delta < 0.0 && start <= limit) {
                    0
                } else {
                    ((limit - start) / delta).ceil() as usize
                };

                (start, limit, delta, length)
            }
            _ => {
                return Err(Error::Runtime(format!(
                    "Range: unsupported dtype {:?}",
                    dtype
                )));
            }
        };

        // Handle empty range
        if length == 0 {
            return Ok(vec![ctx.create_output_tensor(&[0], dtype)?]);
        }

        // Create output tensor
        let output = ctx.create_output_tensor(&[length], dtype)?;

        // Get or create compute pipeline
        let (pipeline, bind_group_layout) =
            ctx.get_or_create_pipeline(&self.label, &self.module, "main")?;

        // Create bind group
        let bind_group = ctx.create_bind_group(&bind_group_layout, &[&output.buffer])?;

        // Encode immediates
        let mut immediates = Vec::new();
        immediates.extend_from_slice(&(length as u32).to_le_bytes());
        immediates.extend_from_slice(&start.to_le_bytes());
        immediates.extend_from_slice(&delta.to_le_bytes());

        // Dispatch compute shader
        let workgroups_x = length.div_ceil(256) as u32;
        ctx.dispatch_compute(
            &pipeline,
            &bind_group,
            [workgroups_x, 1, 1],
            Some(&immediates),
        )?;

        Ok(vec![output])
    }
}
