//! Metadata and constant operators.

use onyxia_core::{
    BindingDesc, DataType, Error, InferenceCtx, Operator, PlanCtx, Result, Step, TensorData,
    TensorShape, TensorValue,
};
use std::collections::HashMap;

/// Constant tensor operator.
///
/// Constant nodes produce a tensor with fixed data that is known at plan time.
/// The data is already in the tensor's initializer, which the runtime uploads
/// during buffer allocation. Therefore, this operator emits zero GPU steps.
pub struct ConstantOp;

impl Operator for ConstantOp {
    fn name(&self) -> &str {
        "Constant"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        // Constant nodes have no inputs. The output shape is already set
        // by the parser when it extracted the tensor from the node's "value" attribute.
        // Return the shape from the graph's tensor registry
        if ctx.output_count() == 0 {
            return Err(Error::ShapeInference(
                "Constant node must have at least one output".to_string(),
            ));
        }

        // Get the output tensor ID from the node
        let output_tensor_id = ctx.node.outputs[0];
        let output_tensor = ctx.graph.tensor(output_tensor_id)?;
        Ok(vec![output_tensor.shape.clone()])
    }

    fn plan(&self, _ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // No-op: constant data is in TensorInfo.initializer,
        // which the runtime uploads during buffer allocation.
        Ok(vec![])
    }
}

/// ConstantOfShape operator - creates a tensor filled with a constant value.
pub struct ConstantOfShapeOp;

impl Operator for ConstantOfShapeOp {
    fn name(&self) -> &str {
        "ConstantOfShape"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        // ConstantOfShape has one input: the shape tensor (1D int64)
        if ctx.input_count() == 0 {
            return Err(Error::ShapeInference(
                "ConstantOfShape requires one input: shape tensor".to_string(),
            ));
        }

        // Try to get the shape from the input value (if it's a constant)
        let Some(shape_val) = ctx.input_value(0) else {
            // Shape is not a constant - error (we need compile-time shape)
            return Err(Error::ShapeInference(
                "ConstantOfShape shape input must be a compile-time constant".to_string(),
            ));
        };

        // Parse i64 shape from the value
        let target_shape = shape_val.as_i64().ok_or_else(|| {
            Error::ShapeInference("ConstantOfShape shape input must be I64".to_string())
        })?;

        // Convert target shape to output dimensions and validate
        let mut output_dims = Vec::new();
        for (i, &dim) in target_shape.iter().enumerate() {
            if dim < 0 {
                return Err(Error::ShapeInference(format!(
                    "ConstantOfShape shape dimension {} is invalid: {} (must be >= 0)",
                    i, dim
                )));
            }
            output_dims.push(dim as usize);
        }

        Ok(vec![TensorShape::Static(output_dims)])
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // Get output info
        let output_tensor = ctx.output_tensor(0)?;
        let output_shape = ctx.static_dims(&output_tensor.shape)?;
        let output_size: usize = output_shape.iter().product();

        if output_size == 0 {
            // Empty tensor - no GPU work needed
            return Ok(vec![]);
        }

        // Get the fill value from the 'value' attribute (default: 0.0)
        // Note: In ONNX, this is a tensor attribute with raw bytes
        // For simplicity, we parse common cases
        let fill_value = ctx.attr_f32("value").unwrap_or(0.0);

        // Load shader source
        let shader_source = include_str!("../../shaders/indexing/constantofshape.wgsl");

        // Compile shader with workgroup size
        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), "256".to_string());

        let shader_index = ctx.compile_shader("constantofshape", shader_source, &shader_defs)?;

        // Prepare immediates (output_size, fill_value)
        let mut immediates = Vec::new();
        immediates.extend_from_slice(&(output_size as u32).to_le_bytes());
        immediates.extend_from_slice(&fill_value.to_le_bytes());

        // Calculate workgroup dispatch
        let workgroups_x = output_size.div_ceil(256);

        // Create dispatch step
        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![BindingDesc {
                buffer: ctx.output(0)?,
                read_only: false,
            }],
            workgroups: [workgroups_x as u32, 1, 1],
            immediates: Some(immediates),
        }])
    }
}

/// Shape operator - extracts shape information from a tensor.
pub struct ShapeOp;

impl Operator for ShapeOp {
    fn name(&self) -> &str {
        "Shape"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        // Shape takes one input and produces one 1D int64 output
        if ctx.input_count() == 0 {
            return Err(Error::ShapeInference(
                "Shape requires one input".to_string(),
            ));
        }

        // Get input dimensions
        let input_dims = ctx.require_static(0)?;

        // Handle start/end attributes to slice the shape
        let start = ctx.attr_i64("start").unwrap_or(0);
        let end = ctx.attr_i64("end").unwrap_or(input_dims.len() as i64);

        // Normalize negative indices
        let rank = input_dims.len() as i64;
        let start = if start < 0 { rank + start } else { start };
        let end = if end < 0 { rank + end } else { end };

        // Calculate output length
        let output_len = (end - start).max(0) as usize;

        // Output is a 1D int64 tensor
        Ok(vec![TensorShape::Static(vec![output_len])])
    }

    fn try_fold(&self, ctx: &onyxia_core::FoldCtx) -> Result<Vec<Option<TensorValue>>> {
        // Return the input tensor's shape as constant I64 values
        if ctx.input_count() == 0 {
            return Ok(vec![None]);
        }

        let input_dims = ctx.require_static(0)?;

        // Handle start/end attributes to slice the shape
        let start = ctx.attr_i64("start").unwrap_or(0);
        let end = ctx.attr_i64("end").unwrap_or(input_dims.len() as i64);

        // Normalize negative indices
        let rank = input_dims.len() as i64;
        let start = if start < 0 {
            (rank + start).max(0) as usize
        } else {
            start.min(rank) as usize
        };
        let end = if end < 0 {
            (rank + end).max(0) as usize
        } else {
            end.min(rank) as usize
        };

        // Slice the shape dimensions
        let shape_slice = if start <= end {
            &input_dims[start..end]
        } else {
            &[]
        };

        // Convert to i64 values
        let values: Vec<i64> = shape_slice.iter().map(|&dim| dim as i64).collect();
        let result = TensorValue::new(
            TensorData::I64(values.clone()),
            vec![values.len()],
            DataType::I64,
        );
        Ok(vec![Some(result)])
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // Shape operator is always folded - return no-op
        // The runtime should never execute this since try_fold handles it
        // But if it does get called, we can just write the shape data
        let input_tensor = ctx.input_tensor(0)?;
        let shape = ctx.static_dims(&input_tensor.shape)?;

        // Handle start/end attributes
        let start = ctx.attr_i64("start").unwrap_or(0);
        let end = ctx.attr_i64("end").unwrap_or(shape.len() as i64);

        // Normalize negative indices
        let rank = shape.len() as i64;
        let start = if start < 0 {
            (rank + start).max(0) as usize
        } else {
            start.min(rank) as usize
        };
        let end = if end < 0 {
            (rank + end).max(0) as usize
        } else {
            end.min(rank) as usize
        };

        // Slice and convert to bytes
        let shape_slice = if start <= end {
            &shape[start..end]
        } else {
            &[]
        };

        let mut shape_data = Vec::new();
        for &dim in shape_slice {
            shape_data.extend_from_slice(&(dim as i64).to_le_bytes());
        }

        // Write shape data directly to output buffer
        Ok(vec![Step::WriteBuffer {
            dst: ctx.output(0)?,
            data: shape_data,
        }])
    }
}
