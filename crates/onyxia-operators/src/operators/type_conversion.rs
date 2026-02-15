//! Type conversion operators.

use onyxia_core::{
    BindingDesc, DataType, Error, InferenceCtx, Operator, PlanCtx, Result, Step, TensorShape,
};
use std::collections::HashMap;

/// Cast operator - converts tensor elements from one data type to another.
pub struct CastOp;

impl Operator for CastOp {
    fn name(&self) -> &str {
        "Cast"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        // Cast preserves shape, only changes data type
        if ctx.input_count() == 0 {
            return Err(Error::ShapeInference(
                "Cast requires at least one input".to_string(),
            ));
        }
        let shape = ctx.input_shape(0)?;
        Ok(vec![shape.clone()])
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // Get input and output info
        let input_tensor = ctx.input_tensor(0)?;
        let output_tensor = ctx.output_tensor(0)?;

        // Get source and target data types
        let source_dtype = input_tensor.dtype;
        let target_dtype = output_tensor.dtype;

        // Calculate number of elements based on output shape
        let output_shape = ctx.static_dims(&output_tensor.shape)?;
        let num_elements: usize = output_shape.iter().product();

        // Handle no-op casts (source == target)
        if source_dtype == target_dtype {
            // For same-type casts, just copy the buffer
            let buffer_size_bytes = num_elements * source_dtype.size();
            return Ok(vec![Step::CopyBuffer {
                src: ctx.input(0)?,
                src_offset: 0,
                dst: ctx.output(0)?,
                dst_offset: 0,
                size: buffer_size_bytes as u64,
            }]);
        }

        // Determine which shader variant to use
        let (shader_label, shader_def) = match (source_dtype, target_dtype) {
            (DataType::I64, DataType::F32) => ("cast_i64_to_f32", "CAST_I64_TO_F32"),
            (DataType::I64, DataType::I32) => ("cast_i64_to_i32", "CAST_I64_TO_I32"),
            (DataType::I32, DataType::F32) => ("cast_i32_to_f32", "CAST_I32_TO_F32"),
            (DataType::F32, DataType::I32) => ("cast_f32_to_i32", "CAST_F32_TO_I32"),
            (DataType::F32, DataType::F16) => ("cast_f32_to_f16", "CAST_F32_TO_F16"),
            _ => {
                return Err(Error::Planning(format!(
                    "Cast from {:?} to {:?} is not yet supported",
                    source_dtype, target_dtype
                )));
            }
        };

        // Configure workgroup size
        let workgroup_size: u32 = 256;
        let num_workgroups = (num_elements as u32).div_ceil(workgroup_size);

        // Prepare shader definitions
        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), workgroup_size.to_string());
        shader_defs.insert(shader_def.to_string(), "true".to_string());

        // Compile shader
        let shader_index = ctx.compile_shader(
            shader_label,
            include_str!("../../shaders/elementwise/cast.wgsl"),
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
