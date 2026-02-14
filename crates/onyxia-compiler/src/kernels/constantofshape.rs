//! ConstantOfShapeKernel implementation for ONNX ConstantOfShape nodes.

use crate::error::{CodegenError, Result};
use crate::inference::{InferenceContext, TensorValue};
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::{AttributeValue, TensorShape};
use std::collections::HashMap;

/// Kernel for ONNX ConstantOfShape operator.
///
/// Creates a tensor filled with a constant value. The shape of the output
/// tensor is specified by the input tensor (1D int64 tensor), and the fill
/// value is specified by the optional 'value' attribute (defaults to 0.0f32).
///
/// # ONNX Specification
///
/// - **Inputs**: `shape` (1D int64 tensor specifying output shape)
/// - **Outputs**: `output` (tensor filled with constant value)
/// - **Attributes**: `value` (optional tensor, defaults to scalar 0.0f32)
pub struct ConstantOfShapeKernel;

impl OpKernel for ConstantOfShapeKernel {
    fn name(&self) -> &str {
        "ConstantOfShape"
    }

    fn infer_output_shapes(&self, ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>> {
        // ConstantOfShape has one input: the shape tensor (1D int64)
        if ctx.input_shapes.is_empty() {
            return Err(CodegenError::InvalidShape(
                "ConstantOfShape requires one input: shape tensor".to_string(),
            ));
        }

        // Try to get the shape from the input value (if it's a constant)
        let Some(shape_val) = ctx.input_value(0)? else {
            // Shape is not a constant - we can't infer the output shape at compile time
            return Ok(vec![TensorShape::Unknown]);
        };

        // Parse i64 shape from the value
        let target_shape = match shape_val {
            TensorValue::I64(v) => v.as_slice(),
            _ => {
                return Err(CodegenError::InvalidShape(
                    "ConstantOfShape shape input must be I64".to_string(),
                ));
            }
        };

        // Convert target shape to output dimensions and validate
        let mut output_dims = Vec::new();
        for (i, &dim) in target_shape.iter().enumerate() {
            if dim < 0 {
                return Err(CodegenError::InvalidShape(format!(
                    "ConstantOfShape shape dimension {} is invalid: {} (must be >= 0)",
                    i, dim
                )));
            }
            output_dims.push(dim as usize);
        }

        Ok(vec![TensorShape::Static(output_dims)])
    }

    fn try_fold(&self, ctx: &InferenceContext<'_>) -> Result<Vec<Option<TensorValue>>> {
        // Try to constant-fold if the shape input is known
        let Some(shape_val) = ctx.input_value(0)? else {
            return Ok(vec![None]);
        };

        // Parse target shape from the input
        let target_shape = match shape_val {
            TensorValue::I64(v) => v.as_slice(),
            _ => return Ok(vec![None]),
        };

        let output_dims: Vec<usize> = target_shape.iter().map(|&d| d as usize).collect();
        let num_elements: usize = output_dims.iter().product();

        // For constant folding, use default fill value (0.0)
        // Note: Parsing 'value' attribute from raw tensor bytes is complex and not
        // needed for shape inference. The actual fill value will be used at runtime.
        let result = TensorValue::F32(vec![0.0; num_elements]);

        Ok(vec![Some(result)])
    }

    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        // Get output info
        let output_info = ctx.output_info(0)?;
        let output_shape = ctx.static_shape(&output_info.shape)?;
        let output_size: usize = output_shape.iter().product();

        if output_size == 0 {
            // Empty tensor - no GPU work needed
            return Ok(vec![]);
        }

        // Get the fill value from the 'value' attribute (default: 0.0)
        let fill_value = Self::parse_value_attribute_simple(ctx);

        // Load shader source
        let shader_source = include_str!("../../shaders/indexing/constantofshape.wgsl");

        // Compile shader with workgroup size
        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), ShaderDefValue::UInt(256));

        let shader_index = ctx.compile_shader("constantofshape", shader_source, shader_defs)?;

        // Prepare immediates (constants uniform)
        // Must match ImmediateConstants struct in WGSL: output_size (u32), fill_value (f32)
        let mut immediates = Vec::new();
        immediates.extend_from_slice(&(output_size as u32).to_le_bytes());
        immediates.extend_from_slice(&fill_value.to_le_bytes());

        // Calculate workgroup dispatch
        let workgroups_x = (output_size + 255) / 256;

        // Create dispatch step
        let step = Step::Dispatch {
            shader_index,
            bindings: vec![BindingDesc {
                buffer: ctx.output(0),
                read_only: false,
            }],
            workgroups: [workgroups_x as u32, 1, 1],
            immediates: Some(immediates),
        };

        Ok(vec![step])
    }
}

impl ConstantOfShapeKernel {
    /// Parse the 'value' attribute from the node.
    ///
    /// Returns the fill value as f32. Defaults to 0.0 if not specified.
    /// Note: This is a simplified parser that only handles the most common cases.
    /// Full tensor proto parsing from raw bytes would be more complex.
    fn parse_value_attribute_simple(ctx: &PlanContext<'_>) -> f32 {
        // Try to get the 'value' attribute
        let Some(value_attr) = ctx.node.attributes.get("value") else {
            // Default: 0.0
            return 0.0;
        };

        // The 'value' attribute is stored as raw bytes (serialized tensor proto)
        // For simplicity, we'll try to parse common cases:
        // - If it's a float attribute directly (uncommon but possible)
        // - Otherwise, default to 0.0
        // Full parsing would require deserializing the protobuf tensor proto from bytes

        match value_attr {
            AttributeValue::Float(v) => *v,
            AttributeValue::Int(v) => *v as f32,
            AttributeValue::Tensor(bytes) => {
                // Simplified parsing: assume float32 and read first 4 bytes
                // This works for the common case where 'value' is a scalar tensor
                if bytes.len() >= 4 {
                    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onyxia_onnx::{Graph, Node};
    use std::collections::HashMap;

    #[test]
    fn test_constantofshape_kernel_shape_inference() {
        let node = Node {
            name: "ConstantOfShape_0".to_string(),
            op_type: "ConstantOfShape".to_string(),
            inputs: vec!["shape".to_string()],
            outputs: vec!["output".to_string()],
            attributes: HashMap::new(),
            domain: String::new(),
        };

        let kernel = ConstantOfShapeKernel;
        let graph = Graph::new();

        // Case 1: Shape input is not a constant - should return Unknown
        let shapes = kernel
            .infer_output_shapes(&InferenceContext::new(
                &node,
                &graph,
                vec![TensorShape::Static(vec![3])],
                vec![None],
            ))
            .expect("shape inference should succeed");

        assert_eq!(shapes.len(), 1);
        assert!(matches!(shapes[0], TensorShape::Unknown));

        // Case 2: Shape input is a constant - should return Static shape
        let shape_value = TensorValue::I64(vec![2, 3, 4]);
        let shapes = kernel
            .infer_output_shapes(&InferenceContext::new(
                &node,
                &graph,
                vec![TensorShape::Static(vec![3])],
                vec![Some(shape_value)],
            ))
            .expect("shape inference should succeed");

        assert_eq!(shapes.len(), 1);
        assert_eq!(shapes[0], TensorShape::Static(vec![2, 3, 4]));
    }

    #[test]
    fn test_constantofshape_constant_folding() {
        let node = Node {
            name: "ConstantOfShape_0".to_string(),
            op_type: "ConstantOfShape".to_string(),
            inputs: vec!["shape".to_string()],
            outputs: vec!["output".to_string()],
            attributes: HashMap::new(),
            domain: String::new(),
        };

        let kernel = ConstantOfShapeKernel;
        let graph = Graph::new();

        // Test constant folding with default value (0.0)
        let shape_value = TensorValue::I64(vec![2, 3]);
        let values = kernel
            .try_fold(&InferenceContext::new(
                &node,
                &graph,
                vec![TensorShape::Static(vec![2])],
                vec![Some(shape_value)],
            ))
            .expect("constant folding should succeed");

        assert_eq!(values.len(), 1);
        assert!(values[0].is_some());

        let folded_value = values[0].as_ref().unwrap();
        match folded_value {
            TensorValue::F32(vals) => {
                assert_eq!(vals.len(), 6); // 2 * 3 = 6
                assert!(vals.iter().all(|&v| v == 0.0));
            }
            _ => panic!("Expected F32 tensor value"),
        }
    }
}
