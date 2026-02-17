//! Cast operator - type conversion between different data types.
//!
//! The Cast operator converts tensor elements from one data type to another.
//! This is essential for mixed-precision computation and type compatibility
//! between operators.

use onyxia_core::{
    CompileCtx, DataType, DispatchCtx, Error, OpDispatch, Operator, Result, RuntimeTensor,
};
use onyxia_onnx::AttributeValue;
use std::collections::HashMap;
use std::sync::Arc;

/// Shader sources for different type conversions
const CAST_F32_I32: &str = include_str!("../../shaders/cast/cast_f32_i32.wgsl");
const CAST_I32_F32: &str = include_str!("../../shaders/cast/cast_i32_f32.wgsl");
const CAST_F32_U32: &str = include_str!("../../shaders/cast/cast_f32_u32.wgsl");
const CAST_U32_F32: &str = include_str!("../../shaders/cast/cast_u32_f32.wgsl");
const CAST_F32_BOOL: &str = include_str!("../../shaders/cast/cast_f32_bool.wgsl");
const CAST_BOOL_F32: &str = include_str!("../../shaders/cast/cast_bool_f32.wgsl");
const CAST_I32_BOOL: &str = include_str!("../../shaders/cast/cast_i32_bool.wgsl");
const CAST_BOOL_I32: &str = include_str!("../../shaders/cast/cast_bool_i32.wgsl");
const CAST_I32_I64: &str = include_str!("../../shaders/cast/cast_i32_i64.wgsl");
const CAST_I64_I32: &str = include_str!("../../shaders/cast/cast_i64_i32.wgsl");
const CAST_I64_F32: &str = include_str!("../../shaders/cast/cast_i64_f32.wgsl");
const CAST_F32_I64: &str = include_str!("../../shaders/cast/cast_f32_i64.wgsl");

/// Cast operator - converts tensor elements from one data type to another.
///
/// ONNX Cast operator (opset 21):
/// - Inputs: input (T1)
/// - Outputs: output (T2)
/// - Attributes:
///   - to (int, required): Target data type (ONNX TensorProto.DataType enum)
///   - saturate (int, default=1): Whether to saturate on overflow (not yet implemented)
///
/// Conversion semantics:
/// - Float to int: Truncate towards zero (not round)
/// - Int to float: Exact for small values, may lose precision for large values
/// - Bool conversion: 0 → false, non-zero → true
pub struct CastOp;

/// Runtime dispatch for Cast operator.
struct CastDispatch {
    target_dtype: DataType,
    /// Pre-compiled shaders for each conversion type
    modules: HashMap<(DataType, DataType), naga::Module>,
}

impl OpDispatch for CastDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> Result<Vec<RuntimeTensor>> {
        let input = &inputs[0];
        let source_dtype = input.dtype;
        let target_dtype = self.target_dtype;

        // Identity cast - no conversion needed, just clone tensor
        if source_dtype == target_dtype {
            return Ok(vec![RuntimeTensor {
                buffer: Arc::clone(&input.buffer),
                shape: input.shape.clone(),
                dtype: input.dtype,
                size_bytes: input.size_bytes,
            }]);
        }

        // Get the appropriate shader module
        let module = self
            .modules
            .get(&(source_dtype, target_dtype))
            .ok_or_else(|| {
                Error::Compilation(format!(
                    "Cast: unsupported conversion from {:?} to {:?}",
                    source_dtype, target_dtype
                ))
            })?;

        let num_elements: usize = input.shape.iter().product();

        // For I64, we store as pairs of u32, so output size is different
        let output_size_bytes = if target_dtype == DataType::I64 {
            num_elements * 8 // i64 = 8 bytes = 2 u32s
        } else {
            num_elements * target_dtype.size()
        };

        // Create output buffer with correct size
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cast_output"),
            size: output_size_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let output = RuntimeTensor {
            buffer: Arc::new(output_buffer),
            shape: input.shape.clone(),
            dtype: target_dtype,
            size_bytes: output_size_bytes,
        };

        // Encode immediates
        let mut immediates = Vec::with_capacity(4);
        immediates.extend_from_slice(&(num_elements as u32).to_le_bytes());

        // Compute workgroups
        let workgroup_size: u32 = 256;
        let num_workgroups = (num_elements as u32).div_ceil(workgroup_size);

        // Get or create pipeline
        let label = format!("cast_{:?}_to_{:?}", source_dtype, target_dtype);
        let (pipeline, bind_group_layout) = ctx.get_or_create_pipeline(&label, module, "main")?;

        // Create bind group
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cast_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
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

impl Operator for CastOp {
    fn name(&self) -> &str {
        "Cast"
    }

    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        // Read target dtype from ONNX "to" attribute
        let to_attr = ctx
            .attr("to")
            .ok_or_else(|| Error::Compilation("Cast: missing 'to' attribute".to_string()))?;

        let onnx_dtype: i64 = match to_attr {
            AttributeValue::Int(v) => *v,
            _ => {
                return Err(Error::Compilation(
                    "Cast: 'to' attribute must be an int".to_string(),
                ));
            }
        };

        let target_dtype = onnx_dtype_to_datatype(onnx_dtype)?;

        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), "256".to_string());

        // Compile all supported conversion shaders
        let mut modules = HashMap::new();

        // Helper to compile a shader for a specific conversion
        let mut compile = |source: &str, from: DataType, to: DataType| -> Result<naga::Module> {
            let label = format!("cast_{:?}_to_{:?}", from, to);
            ctx.compile_shader(&label, source, &shader_defs)
        };

        // F32 conversions
        modules.insert(
            (DataType::F32, DataType::I32),
            compile(CAST_F32_I32, DataType::F32, DataType::I32)?,
        );
        modules.insert(
            (DataType::F32, DataType::U32),
            compile(CAST_F32_U32, DataType::F32, DataType::U32)?,
        );
        modules.insert(
            (DataType::F32, DataType::Bool),
            compile(CAST_F32_BOOL, DataType::F32, DataType::Bool)?,
        );
        modules.insert(
            (DataType::F32, DataType::I64),
            compile(CAST_F32_I64, DataType::F32, DataType::I64)?,
        );

        // I32 conversions
        modules.insert(
            (DataType::I32, DataType::F32),
            compile(CAST_I32_F32, DataType::I32, DataType::F32)?,
        );
        modules.insert(
            (DataType::I32, DataType::Bool),
            compile(CAST_I32_BOOL, DataType::I32, DataType::Bool)?,
        );
        modules.insert(
            (DataType::I32, DataType::I64),
            compile(CAST_I32_I64, DataType::I32, DataType::I64)?,
        );

        // U32 conversions
        modules.insert(
            (DataType::U32, DataType::F32),
            compile(CAST_U32_F32, DataType::U32, DataType::F32)?,
        );

        // Bool conversions
        modules.insert(
            (DataType::Bool, DataType::F32),
            compile(CAST_BOOL_F32, DataType::Bool, DataType::F32)?,
        );
        modules.insert(
            (DataType::Bool, DataType::I32),
            compile(CAST_BOOL_I32, DataType::Bool, DataType::I32)?,
        );

        // I64 conversions
        modules.insert(
            (DataType::I64, DataType::I32),
            compile(CAST_I64_I32, DataType::I64, DataType::I32)?,
        );
        modules.insert(
            (DataType::I64, DataType::F32),
            compile(CAST_I64_F32, DataType::I64, DataType::F32)?,
        );

        Ok(Box::new(CastDispatch {
            target_dtype,
            modules,
        }))
    }
}

/// Convert ONNX TensorProto.DataType code to internal DataType.
///
/// ONNX data type codes (from onnx.proto):
/// - FLOAT (1) → F32
/// - UINT8 (2) → U8
/// - INT32 (6) → I32
/// - INT64 (7) → I64
/// - BOOL (9) → Bool
/// - FLOAT16 (10) → F16
/// - UINT32 (12) → U32
fn onnx_dtype_to_datatype(code: i64) -> Result<DataType> {
    match code {
        1 => Ok(DataType::F32),
        2 => Ok(DataType::U8),
        6 => Ok(DataType::I32),
        7 => Ok(DataType::I64),
        9 => Ok(DataType::Bool),
        10 => Ok(DataType::F16),
        12 => Ok(DataType::U32),
        _ => Err(Error::Compilation(format!(
            "Cast: unsupported ONNX data type code: {}",
            code
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_dtype_to_datatype() {
        assert_eq!(onnx_dtype_to_datatype(1).unwrap(), DataType::F32);
        assert_eq!(onnx_dtype_to_datatype(6).unwrap(), DataType::I32);
        assert_eq!(onnx_dtype_to_datatype(7).unwrap(), DataType::I64);
        assert_eq!(onnx_dtype_to_datatype(9).unwrap(), DataType::Bool);
        assert_eq!(onnx_dtype_to_datatype(12).unwrap(), DataType::U32);

        // Unsupported type
        assert!(onnx_dtype_to_datatype(999).is_err());
    }
}
