//! NegOperator implementation for element-wise negation.

use crate::error::Result;
use crate::inference::{InferenceContext, TensorValue};
use crate::operator::{Operator, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::TensorShape;
use std::collections::HashMap;

/// Operator for Neg (ONNX Neg operator).
///
/// Neg(x) = -x - element-wise negation function.
pub struct NegOperator;

impl Operator for NegOperator {
    fn name(&self) -> &str {
        "Neg"
    }

    fn infer_output_shapes(&self, ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>> {
        // Neg is a unary operation: output shape equals input shape
        if ctx.input_shapes.is_empty() {
            return Err(crate::error::CodegenError::InvalidShape(
                "Neg requires one input".to_string(),
            ));
        }
        Ok(vec![ctx.input_shapes[0].clone()])
    }

    fn try_fold(&self, ctx: &InferenceContext<'_>) -> Result<Vec<Option<TensorValue>>> {
        // Try to constant-fold if input is known
        let Some(input) = ctx.input_value(0)? else {
            return Ok(vec![None]);
        };

        // Only fold F32 values
        match input {
            TensorValue::F32(vals) => {
                let result: Vec<f32> = vals.iter().map(|x| -x).collect();
                Ok(vec![Some(TensorValue::F32(result))])
            }
            _ => Ok(vec![None]),
        }
    }

    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        // Get output shape and calculate total elements
        let output_info = ctx.output_info(0)?;
        let output_shape = ctx.static_shape(&output_info.shape)?;
        let num_elements: usize = output_shape.iter().product();

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
            "neg",
            include_str!("../../shaders/elementwise/neg.wgsl"),
            shader_defs,
        )?;

        // Encode immediate data (must match ImmediateConstants struct in shader)
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(num_elements as u32).to_le_bytes());

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
    use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};

    fn create_neg_test_graph() -> Graph {
        let mut graph = Graph::new();

        // Add input tensor
        graph.add_tensor(TensorInfo {
            name: "x".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![8]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add output tensor
        graph.add_tensor(TensorInfo {
            name: "y".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![8]),
            kind: TensorKind::Output,
            initializer: None,
        });

        // Create Neg node
        let mut node = Node::new("Neg");
        node.name = "neg_op".to_string();
        node.inputs = vec!["x".to_string()];
        node.outputs = vec!["y".to_string()];
        graph.add_node(node);

        // Set graph inputs and outputs
        graph.inputs = vec!["x".to_string()];
        graph.outputs = vec!["y".to_string()];

        graph
    }

    #[test]
    fn test_neg_kernel_name() {
        let operator = NegOperator;
        assert_eq!(operator.name(), "Neg");
    }

    #[test]
    fn test_neg_output_shape_inference() {
        let graph = create_neg_test_graph();
        let operator = NegOperator;

        let ctx = InferenceContext {
            node: &graph.nodes[0],
            graph: &graph,
            input_shapes: vec![TensorShape::Static(vec![8])],
            input_values: vec![None],
        };

        let output_shapes = operator
            .infer_output_shapes(&ctx)
            .expect("Shape inference should succeed");

        assert_eq!(output_shapes.len(), 1);
        assert_eq!(output_shapes[0], TensorShape::Static(vec![8]));
    }

    #[test]
    fn test_neg_constant_folding() {
        let graph = create_neg_test_graph();
        let operator = NegOperator;

        // Test constant folding with F32 values
        let input_vals = vec![1.0f32, -2.0, 3.0, -4.0, 0.0, 5.5, -6.5, 100.0];
        let ctx = InferenceContext {
            node: &graph.nodes[0],
            graph: &graph,
            input_shapes: vec![TensorShape::Static(vec![8])],
            input_values: vec![Some(TensorValue::F32(input_vals.clone()))],
        };

        let folded = operator
            .try_fold(&ctx)
            .expect("Constant folding should succeed");

        assert_eq!(folded.len(), 1);
        let result = folded[0].as_ref().expect("Should have folded result");

        match result {
            TensorValue::F32(vals) => {
                assert_eq!(vals.len(), 8);
                assert_eq!(vals[0], -1.0);
                assert_eq!(vals[1], 2.0);
                assert_eq!(vals[2], -3.0);
                assert_eq!(vals[3], 4.0);
                assert_eq!(vals[4], 0.0);
                assert_eq!(vals[5], -5.5);
                assert_eq!(vals[6], 6.5);
                assert_eq!(vals[7], -100.0);
            }
            _ => panic!("Expected F32 tensor value"),
        }
    }
}
