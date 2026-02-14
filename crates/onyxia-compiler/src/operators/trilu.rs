//! TriluOperator implementation for triangular matrix extraction.

use crate::error::Result;
use crate::inference::{InferenceContext, TensorValue};
use crate::operator::{Operator, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::TensorShape;
use std::collections::HashMap;

/// Operator for Trilu (ONNX Trilu operator).
///
/// Returns the upper or lower triangular part of a 2D matrix or batch of matrices.
/// For tensors with rank > 2, the operation is applied to the last two dimensions.
///
/// **Inputs:**
/// - `input` (T): Input tensor of shape [..., M, N]
/// - `k` (optional, tensor(int64)): Diagonal offset. Default is 0.
///
/// **Outputs:**
/// - `output` (T): Output tensor with same shape as input
///
/// **Attributes:**
/// - `upper` (int, default=1): If 1, extract upper triangle; if 0, extract lower triangle
///
/// **Behavior:**
/// - Upper triangle (upper=1): Keep elements where row <= col + k, zero out others
/// - Lower triangle (upper=0): Keep elements where row >= col + k, zero out others
pub struct TriluOperator;

impl Operator for TriluOperator {
    fn name(&self) -> &str {
        "Trilu"
    }

    fn infer_output_shapes(&self, ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>> {
        // Trilu output has the same shape as input
        if ctx.input_shapes.is_empty() {
            return Err(crate::error::CodegenError::InvalidShape(
                "Trilu requires at least one input".to_string(),
            ));
        }
        Ok(vec![ctx.input_shapes[0].clone()])
    }

    fn try_fold(&self, ctx: &InferenceContext<'_>) -> Result<Vec<Option<TensorValue>>> {
        // Try to constant-fold if input is known
        let Some(input) = ctx.input_value(0)? else {
            return Ok(vec![None]);
        };

        // Get input shape
        let input_shape = match &ctx.input_shapes[0] {
            TensorShape::Static(dims) => dims,
            _ => return Ok(vec![None]),
        };

        if input_shape.len() < 2 {
            return Err(crate::error::CodegenError::InvalidShape(
                "Trilu requires input with rank >= 2".to_string(),
            ));
        }

        let rank = input_shape.len();
        let m = input_shape[rank - 2];
        let n = input_shape[rank - 1];

        // Read 'upper' attribute (default=1)
        let upper: i64 = ctx.node.attr("upper").unwrap_or(1);

        // Read optional 'k' input (default=0)
        let k: i64 = if ctx.input_values.len() > 1 {
            if let Some(TensorValue::I64(k_vals)) = &ctx.input_values[1] {
                if k_vals.len() != 1 {
                    return Err(crate::error::CodegenError::InvalidShape(
                        "Trilu k input must be a scalar".to_string(),
                    ));
                }
                k_vals[0]
            } else {
                0
            }
        } else {
            0
        };

        // Only fold F32 values
        match input {
            TensorValue::F32(vals) => {
                let batch_size: usize = input_shape[..rank - 2].iter().product();
                let matrix_size = m * n;
                let mut result = vals.clone();

                for batch_idx in 0..batch_size {
                    for row in 0..m {
                        for col in 0..n {
                            let idx = batch_idx * matrix_size + row * n + col;
                            let keep = if upper == 1 {
                                (row as i64) <= (col as i64) + k
                            } else {
                                (row as i64) >= (col as i64) + k
                            };
                            if !keep {
                                result[idx] = 0.0;
                            }
                        }
                    }
                }

                Ok(vec![Some(TensorValue::F32(result))])
            }
            _ => Ok(vec![None]),
        }
    }

    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        // Get output shape and calculate dimensions
        let output_info = ctx.output_info(0)?;
        let output_shape = ctx.static_shape(&output_info.shape)?;

        if output_shape.len() < 2 {
            return Err(crate::error::CodegenError::InvalidShape(
                "Trilu requires input with rank >= 2".to_string(),
            ));
        }

        let rank = output_shape.len();
        let m = output_shape[rank - 2] as u32;
        let n = output_shape[rank - 1] as u32;
        let num_elements: usize = output_shape.iter().product();

        // Read 'upper' attribute (default=1)
        let upper: i64 = ctx.node.attr("upper").unwrap_or(1);

        // Read optional 'k' input (default=0)
        let k: i64 = if ctx.node.inputs.len() > 1 {
            // k is provided as a second input tensor (must be scalar I64)
            let k_tensor_id = ctx.input_ids[1];
            let k_info = ctx.graph.tensor_info.get(k_tensor_id).ok_or_else(|| {
                crate::error::CodegenError::InvalidShape("Trilu k tensor not found".to_string())
            })?;

            if let Some(ref initializer) = k_info.initializer {
                // Read k from initializer
                if initializer.len() != 8 {
                    return Err(crate::error::CodegenError::InvalidShape(
                        "Trilu k input must be a scalar I64".to_string(),
                    ));
                }
                let k_bytes: [u8; 8] = initializer.as_slice().try_into().map_err(|_| {
                    crate::error::CodegenError::InvalidShape("Failed to read k value".to_string())
                })?;
                i64::from_le_bytes(k_bytes)
            } else {
                // k is not constant, default to 0
                // In a full implementation, we'd need to read k from GPU buffer at runtime
                0
            }
        } else {
            0
        };

        // Configure workgroup size
        let workgroup_size: u32 = 256;
        let num_workgroups = (num_elements as u32).div_ceil(workgroup_size);

        // Prepare shader definitions
        let mut shader_defs = HashMap::new();
        shader_defs.insert(
            "WORKGROUP_SIZE".to_string(),
            ShaderDefValue::UInt(workgroup_size),
        );

        // Compile shader
        let shader_index = ctx.compile_shader(
            "trilu",
            include_str!("../../shaders/indexing/trilu.wgsl"),
            shader_defs,
        )?;

        // Encode immediate data (must match ImmediateConstants struct in shader)
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(num_elements as u32).to_le_bytes());
        immediates_data.extend_from_slice(&m.to_le_bytes());
        immediates_data.extend_from_slice(&n.to_le_bytes());
        immediates_data.extend_from_slice(&(k as i32).to_le_bytes());
        immediates_data.extend_from_slice(&(upper as u32).to_le_bytes());

        // Create dispatch step with bindings and immediates
        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0),
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(0),
                    read_only: false,
                },
            ],
            workgroups: [num_workgroups, 1, 1],
            immediates: Some(immediates_data),
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onyxia_onnx::{AttributeValue, DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};

    fn create_trilu_test_graph(upper: i64) -> Graph {
        let mut graph = Graph::new();

        // Add input tensor [3, 3]
        graph.add_tensor(TensorInfo {
            name: "x".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![3, 3]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add output tensor [3, 3]
        graph.add_tensor(TensorInfo {
            name: "y".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![3, 3]),
            kind: TensorKind::Output,
            initializer: None,
        });

        // Create Trilu node
        let mut node = Node::new("Trilu");
        node.name = "trilu_op".to_string();
        node.inputs = vec!["x".to_string()];
        node.outputs = vec!["y".to_string()];
        node.attributes
            .insert("upper".to_string(), AttributeValue::Int(upper));
        graph.add_node(node);

        // Set graph inputs and outputs
        graph.inputs = vec!["x".to_string()];
        graph.outputs = vec!["y".to_string()];

        graph
    }

    #[test]
    fn test_trilu_kernel_name() {
        let operator = TriluOperator;
        assert_eq!(operator.name(), "Trilu");
    }

    #[test]
    fn test_trilu_output_shape_inference() {
        let graph = create_trilu_test_graph(1);
        let operator = TriluOperator;

        let ctx = InferenceContext {
            node: &graph.nodes[0],
            graph: &graph,
            input_shapes: vec![TensorShape::Static(vec![3, 3])],
            input_values: vec![None],
        };

        let output_shapes = operator
            .infer_output_shapes(&ctx)
            .expect("Shape inference should succeed");

        assert_eq!(output_shapes.len(), 1);
        assert_eq!(output_shapes[0], TensorShape::Static(vec![3, 3]));
    }

    #[test]
    fn test_trilu_upper_triangle_constant_folding() {
        let graph = create_trilu_test_graph(1);
        let operator = TriluOperator;

        // Test constant folding with F32 values
        #[rustfmt::skip]
        let input_vals = vec![
            1.0f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];

        let ctx = InferenceContext {
            node: &graph.nodes[0],
            graph: &graph,
            input_shapes: vec![TensorShape::Static(vec![3, 3])],
            input_values: vec![Some(TensorValue::F32(input_vals.clone()))],
        };

        let folded = operator
            .try_fold(&ctx)
            .expect("Constant folding should succeed");

        assert_eq!(folded.len(), 1);
        let result = folded[0].as_ref().expect("Should have folded result");

        match result {
            TensorValue::F32(vals) => {
                assert_eq!(vals.len(), 9);
                // Expected upper triangle (upper=1, k=0):
                // [[1, 2, 3],
                //  [0, 5, 6],
                //  [0, 0, 9]]
                #[rustfmt::skip]
                let expected = vec![
                    1.0, 2.0, 3.0,
                    0.0, 5.0, 6.0,
                    0.0, 0.0, 9.0,
                ];
                for (i, (&actual, &expected)) in vals.iter().zip(expected.iter()).enumerate() {
                    assert_eq!(
                        actual, expected,
                        "Mismatch at index {}: got {}, expected {}",
                        i, actual, expected
                    );
                }
            }
            _ => panic!("Expected F32 tensor value"),
        }
    }

    #[test]
    fn test_trilu_lower_triangle_constant_folding() {
        let graph = create_trilu_test_graph(0);
        let operator = TriluOperator;

        // Test constant folding with F32 values
        #[rustfmt::skip]
        let input_vals = vec![
            1.0f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];

        let ctx = InferenceContext {
            node: &graph.nodes[0],
            graph: &graph,
            input_shapes: vec![TensorShape::Static(vec![3, 3])],
            input_values: vec![Some(TensorValue::F32(input_vals.clone()))],
        };

        let folded = operator
            .try_fold(&ctx)
            .expect("Constant folding should succeed");

        assert_eq!(folded.len(), 1);
        let result = folded[0].as_ref().expect("Should have folded result");

        match result {
            TensorValue::F32(vals) => {
                assert_eq!(vals.len(), 9);
                // Expected lower triangle (upper=0, k=0):
                // [[1, 0, 0],
                //  [4, 5, 0],
                //  [7, 8, 9]]
                #[rustfmt::skip]
                let expected = vec![
                    1.0, 0.0, 0.0,
                    4.0, 5.0, 0.0,
                    7.0, 8.0, 9.0,
                ];
                for (i, (&actual, &expected)) in vals.iter().zip(expected.iter()).enumerate() {
                    assert_eq!(
                        actual, expected,
                        "Mismatch at index {}: got {}, expected {}",
                        i, actual, expected
                    );
                }
            }
            _ => panic!("Expected F32 tensor value"),
        }
    }
}
