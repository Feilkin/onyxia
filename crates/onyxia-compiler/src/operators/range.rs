//! RangeOperator implementation for ONNX Range nodes.

use crate::error::{CodegenError, Result};
use crate::inference::{InferenceContext, TensorValue};
use crate::operator::{Operator, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::{DataType, TensorShape};
use std::collections::HashMap;

/// Operator for ONNX Range operator.
///
/// Generates a 1D tensor containing a sequence of numbers from `start` to `limit`
/// (exclusive) with step `delta`.
///
/// # ONNX Specification
///
/// - **Inputs**:
///   - `start` (scalar): Starting value of the sequence
///   - `limit` (scalar): Ending value (exclusive) of the sequence
///   - `delta` (scalar): Step/increment value
/// - **Outputs**:
///   - `output` (1D tensor): Generated sequence
/// - **Attributes**: None
///
/// The output size is: `ceil((limit - start) / delta)`
///
/// # Examples
///
/// - Range(0, 5, 1) = [0, 1, 2, 3, 4]
/// - Range(2, 10, 2) = [2, 4, 6, 8]
/// - Range(10, 0, -2) = [10, 8, 6, 4, 2]
pub struct RangeOperator;

impl Operator for RangeOperator {
    fn name(&self) -> &str {
        "Range"
    }

    fn infer_output_shapes(&self, ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>> {
        // Range requires 3 scalar inputs: start, limit, delta
        if ctx.input_shapes.len() != 3 {
            return Err(CodegenError::InvalidShape(format!(
                "Range requires 3 inputs (start, limit, delta), got {}",
                ctx.input_shapes.len()
            )));
        }

        // Try to get the scalar values to compute output size
        let start_val = ctx.input_value(0)?;
        let limit_val = ctx.input_value(1)?;
        let delta_val = ctx.input_value(2)?;

        // If all inputs are constants, we can infer the output size
        if let (Some(start), Some(limit), Some(delta)) = (start_val, limit_val, delta_val) {
            let output_size = self.compute_output_size(start, limit, delta)?;
            Ok(vec![TensorShape::Static(vec![output_size])])
        } else {
            // If inputs are not constant, we can't infer size at compile time
            Ok(vec![TensorShape::Unknown])
        }
    }

    fn try_fold(&self, ctx: &InferenceContext<'_>) -> Result<Vec<Option<TensorValue>>> {
        // Try to constant-fold if all inputs are known
        let Some(start) = ctx.input_value(0)? else {
            return Ok(vec![None]);
        };
        let Some(limit) = ctx.input_value(1)? else {
            return Ok(vec![None]);
        };
        let Some(delta) = ctx.input_value(2)? else {
            return Ok(vec![None]);
        };

        // Compute the range for supported types
        let result = match (start, limit, delta) {
            (TensorValue::I64(s), TensorValue::I64(l), TensorValue::I64(d)) => {
                if s.len() != 1 || l.len() != 1 || d.len() != 1 {
                    return Err(CodegenError::InvalidShape(
                        "Range inputs must be scalars".to_string(),
                    ));
                }
                let (start, limit, delta) = (s[0], l[0], d[0]);
                if delta == 0 {
                    return Err(CodegenError::InvalidShape(
                        "Range delta cannot be zero".to_string(),
                    ));
                }
                let size = ((limit - start) as f64 / delta as f64).ceil() as usize;
                let values: Vec<i64> = (0..size).map(|i| start + i as i64 * delta).collect();
                TensorValue::I64(values)
            }
            (TensorValue::I32(s), TensorValue::I32(l), TensorValue::I32(d)) => {
                if s.len() != 1 || l.len() != 1 || d.len() != 1 {
                    return Err(CodegenError::InvalidShape(
                        "Range inputs must be scalars".to_string(),
                    ));
                }
                let (start, limit, delta) = (s[0], l[0], d[0]);
                if delta == 0 {
                    return Err(CodegenError::InvalidShape(
                        "Range delta cannot be zero".to_string(),
                    ));
                }
                let size = ((limit - start) as f64 / delta as f64).ceil() as usize;
                let values: Vec<i32> = (0..size).map(|i| start + i as i32 * delta).collect();
                TensorValue::I32(values)
            }
            (TensorValue::F32(s), TensorValue::F32(l), TensorValue::F32(d)) => {
                if s.len() != 1 || l.len() != 1 || d.len() != 1 {
                    return Err(CodegenError::InvalidShape(
                        "Range inputs must be scalars".to_string(),
                    ));
                }
                let (start, limit, delta) = (s[0], l[0], d[0]);
                if delta == 0.0 {
                    return Err(CodegenError::InvalidShape(
                        "Range delta cannot be zero".to_string(),
                    ));
                }
                let size = ((limit - start) / delta).ceil() as usize;
                let values: Vec<f32> = (0..size).map(|i| start + i as f32 * delta).collect();
                TensorValue::F32(values)
            }
            _ => return Ok(vec![None]), // Unsupported type combination
        };

        Ok(vec![Some(result)])
    }

    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        // Get output info
        let output_info = ctx.output_info(0)?;
        let output_shape = ctx.static_shape(&output_info.shape)?;

        if output_shape.len() != 1 {
            return Err(CodegenError::InvalidShape(format!(
                "Range output must be 1D, got shape {:?}",
                output_shape
            )));
        }

        let output_size = output_shape[0];

        if output_size == 0 {
            // Empty range - no GPU work needed
            return Ok(vec![]);
        }

        // Read scalar values from input tensors
        let (start, delta, _dtype) = self.read_scalar_inputs(ctx)?;

        // Load appropriate shader based on dtype
        let shader_source = include_str!("../../shaders/indexing/range.wgsl");

        // Compile shader with workgroup size
        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), ShaderDefValue::UInt(256));

        let shader_index = ctx.compile_shader("range", shader_source, shader_defs)?;

        // Prepare immediates (constants uniform)
        // Must match ImmediateConstants struct in WGSL: start (f32), delta (f32), size (u32)
        let mut immediates = Vec::new();
        immediates.extend_from_slice(&start.to_le_bytes());
        immediates.extend_from_slice(&delta.to_le_bytes());
        immediates.extend_from_slice(&(output_size as u32).to_le_bytes());

        // Pad to 16-byte alignment
        immediates.extend_from_slice(&0u32.to_le_bytes());

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

impl RangeOperator {
    /// Calculate the output size from start, limit, delta values.
    fn compute_output_size(
        &self,
        start: &TensorValue,
        limit: &TensorValue,
        delta: &TensorValue,
    ) -> Result<usize> {
        match (start, limit, delta) {
            (TensorValue::I64(s), TensorValue::I64(l), TensorValue::I64(d)) => {
                if s.len() != 1 || l.len() != 1 || d.len() != 1 {
                    return Err(CodegenError::InvalidShape(
                        "Range inputs must be scalars".to_string(),
                    ));
                }
                let (start, limit, delta) = (s[0], l[0], d[0]);
                if delta == 0 {
                    return Err(CodegenError::InvalidShape(
                        "Range delta cannot be zero".to_string(),
                    ));
                }
                Ok(((limit - start) as f64 / delta as f64).ceil().max(0.0) as usize)
            }
            (TensorValue::I32(s), TensorValue::I32(l), TensorValue::I32(d)) => {
                if s.len() != 1 || l.len() != 1 || d.len() != 1 {
                    return Err(CodegenError::InvalidShape(
                        "Range inputs must be scalars".to_string(),
                    ));
                }
                let (start, limit, delta) = (s[0], l[0], d[0]);
                if delta == 0 {
                    return Err(CodegenError::InvalidShape(
                        "Range delta cannot be zero".to_string(),
                    ));
                }
                Ok(((limit - start) as f64 / delta as f64).ceil().max(0.0) as usize)
            }
            (TensorValue::F32(s), TensorValue::F32(l), TensorValue::F32(d)) => {
                if s.len() != 1 || l.len() != 1 || d.len() != 1 {
                    return Err(CodegenError::InvalidShape(
                        "Range inputs must be scalars".to_string(),
                    ));
                }
                let (start, limit, delta) = (s[0], l[0], d[0]);
                if delta == 0.0 {
                    return Err(CodegenError::InvalidShape(
                        "Range delta cannot be zero".to_string(),
                    ));
                }
                Ok(((limit - start) / delta).ceil().max(0.0) as usize)
            }
            _ => Err(CodegenError::InvalidShape(
                "Range inputs must have matching types (I64, I32, or F32)".to_string(),
            )),
        }
    }

    /// Read scalar values from input tensors (from initializers).
    /// Returns (start, delta, dtype) as f32 values for shader.
    fn read_scalar_inputs(&self, ctx: &PlanContext<'_>) -> Result<(f32, f32, DataType)> {
        // Read start value (input 0)
        let start_info = ctx.input_info(0)?;
        let start_val = self.read_scalar_from_initializer(start_info)?;

        // Read limit value (input 1) - not actually needed for shader, only for size calc
        // let limit_info = ctx.input_info(1)?;
        // let limit_val = self.read_scalar_from_initializer(limit_info)?;

        // Read delta value (input 2)
        let delta_info = ctx.input_info(2)?;
        let delta_val = self.read_scalar_from_initializer(delta_info)?;

        let dtype = start_info.dtype;

        Ok((start_val, delta_val, dtype))
    }

    /// Read a scalar value from a tensor's initializer.
    fn read_scalar_from_initializer(&self, tensor_info: &onyxia_onnx::TensorInfo) -> Result<f32> {
        let Some(ref initializer) = tensor_info.initializer else {
            return Err(CodegenError::InvalidShape(format!(
                "Range input '{}' is not a constant",
                tensor_info.name
            )));
        };

        match tensor_info.dtype {
            DataType::F32 => {
                if initializer.len() != 4 {
                    return Err(CodegenError::InvalidShape(
                        "Range F32 input must be scalar (4 bytes)".to_string(),
                    ));
                }
                let bytes: [u8; 4] = initializer.as_slice().try_into().map_err(|_| {
                    CodegenError::InvalidShape("Failed to read F32 scalar".to_string())
                })?;
                Ok(f32::from_le_bytes(bytes))
            }
            DataType::I64 => {
                if initializer.len() != 8 {
                    return Err(CodegenError::InvalidShape(
                        "Range I64 input must be scalar (8 bytes)".to_string(),
                    ));
                }
                let bytes: [u8; 8] = initializer.as_slice().try_into().map_err(|_| {
                    CodegenError::InvalidShape("Failed to read I64 scalar".to_string())
                })?;
                Ok(i64::from_le_bytes(bytes) as f32)
            }
            DataType::I32 => {
                if initializer.len() != 4 {
                    return Err(CodegenError::InvalidShape(
                        "Range I32 input must be scalar (4 bytes)".to_string(),
                    ));
                }
                let bytes: [u8; 4] = initializer.as_slice().try_into().map_err(|_| {
                    CodegenError::InvalidShape("Failed to read I32 scalar".to_string())
                })?;
                Ok(i32::from_le_bytes(bytes) as f32)
            }
            _ => Err(CodegenError::UnsupportedOp(format!(
                "Range does not support dtype {:?}",
                tensor_info.dtype
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onyxia_onnx::{Graph, Node, TensorInfo, TensorKind};

    #[test]
    fn test_infer_output_shapes_integer_range() {
        let operator = RangeOperator;

        // Create a minimal graph with Range inputs as initializers
        let mut graph = Graph::new();

        // start = 0 (I64 scalar)
        graph.add_tensor(TensorInfo {
            name: "start".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![]),
            kind: TensorKind::Weight,
            initializer: Some(vec![0, 0, 0, 0, 0, 0, 0, 0]), // i64: 0
        });

        // limit = 5 (I64 scalar)
        graph.add_tensor(TensorInfo {
            name: "limit".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![]),
            kind: TensorKind::Weight,
            initializer: Some(vec![5, 0, 0, 0, 0, 0, 0, 0]), // i64: 5
        });

        // delta = 1 (I64 scalar)
        graph.add_tensor(TensorInfo {
            name: "delta".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![]),
            kind: TensorKind::Weight,
            initializer: Some(vec![1, 0, 0, 0, 0, 0, 0, 0]), // i64: 1
        });

        // Create Range node
        let mut node = Node::new("Range");
        node.name = "range_node".to_string();
        node.inputs = vec![
            "start".to_string(),
            "limit".to_string(),
            "delta".to_string(),
        ];
        node.outputs = vec!["output".to_string()];

        // Get initializer values for inference
        let start_val = TensorValue::I64(vec![0]);
        let limit_val = TensorValue::I64(vec![5]);
        let delta_val = TensorValue::I64(vec![1]);

        let ctx = InferenceContext::new(
            &node,
            &graph,
            vec![
                TensorShape::Static(vec![]),
                TensorShape::Static(vec![]),
                TensorShape::Static(vec![]),
            ],
            vec![Some(start_val), Some(limit_val), Some(delta_val)],
        );

        let output_shapes = operator
            .infer_output_shapes(&ctx)
            .expect("Shape inference should succeed");

        assert_eq!(output_shapes.len(), 1);
        assert_eq!(output_shapes[0], TensorShape::Static(vec![5]));
    }

    #[test]
    fn test_try_fold_integer_range() {
        let operator = RangeOperator;

        let graph = Graph::new();
        let mut node = Node::new("Range");
        node.name = "range_node".to_string();

        let start_val = TensorValue::I64(vec![2]);
        let limit_val = TensorValue::I64(vec![10]);
        let delta_val = TensorValue::I64(vec![2]);

        let ctx = InferenceContext::new(
            &node,
            &graph,
            vec![
                TensorShape::Static(vec![]),
                TensorShape::Static(vec![]),
                TensorShape::Static(vec![]),
            ],
            vec![Some(start_val), Some(limit_val), Some(delta_val)],
        );

        let folded = operator.try_fold(&ctx).expect("Folding should succeed");

        assert_eq!(folded.len(), 1);
        assert!(folded[0].is_some());

        if let Some(TensorValue::I64(values)) = &folded[0] {
            assert_eq!(values, &[2, 4, 6, 8]);
        } else {
            panic!("Expected I64 tensor value");
        }
    }

    #[test]
    fn test_try_fold_float_range() {
        let operator = RangeOperator;

        let graph = Graph::new();
        let mut node = Node::new("Range");
        node.name = "range_node".to_string();

        let start_val = TensorValue::F32(vec![0.0]);
        let limit_val = TensorValue::F32(vec![2.5]);
        let delta_val = TensorValue::F32(vec![0.5]);

        let ctx = InferenceContext::new(
            &node,
            &graph,
            vec![
                TensorShape::Static(vec![]),
                TensorShape::Static(vec![]),
                TensorShape::Static(vec![]),
            ],
            vec![Some(start_val), Some(limit_val), Some(delta_val)],
        );

        let folded = operator.try_fold(&ctx).expect("Folding should succeed");

        assert_eq!(folded.len(), 1);
        assert!(folded[0].is_some());

        if let Some(TensorValue::F32(values)) = &folded[0] {
            assert_eq!(values, &[0.0, 0.5, 1.0, 1.5, 2.0]);
        } else {
            panic!("Expected F32 tensor value");
        }
    }
}
