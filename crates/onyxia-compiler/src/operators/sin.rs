//! SinOperator implementation for sine activation function.

use crate::error::Result;
use crate::inference::{InferenceContext, TensorValue};
use crate::operator::{Operator, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::TensorShape;
use std::collections::HashMap;

/// Operator for Sin activation (ONNX Sin operator).
///
/// Sin(x) = sin(x) - element-wise sine function.
pub struct SinOperator;

impl Operator for SinOperator {
    fn name(&self) -> &str {
        "Sin"
    }

    fn infer_output_shapes(&self, ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>> {
        // Sin is a unary operation: output shape equals input shape
        if ctx.input_shapes.is_empty() {
            return Err(crate::error::CodegenError::InvalidShape(
                "Sin requires one input".to_string(),
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
                let result: Vec<f32> = vals.iter().map(|x| x.sin()).collect();
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
        let num_workgroups = (num_elements as u32 + workgroup_size - 1) / workgroup_size;

        // Prepare shader definitions
        let mut shader_defs = HashMap::new();
        shader_defs.insert(
            "WORKGROUP_SIZE".to_string(),
            ShaderDefValue::UInt(workgroup_size),
        );

        // Compile shader
        let shader_index = ctx.compile_shader(
            "sin",
            include_str!("../../shaders/elementwise/sin.wgsl"),
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

    fn create_sin_test_graph() -> Graph {
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

        // Add Sin node
        let mut node = Node::new("Sin");
        node.name = "sin_op".to_string();
        node.inputs = vec!["x".to_string()];
        node.outputs = vec!["y".to_string()];
        graph.add_node(node);

        graph.inputs = vec!["x".to_string()];
        graph.outputs = vec!["y".to_string()];

        graph
    }

    #[test]
    fn test_sin_kernel_plan() {
        let graph = create_sin_test_graph();

        let operator = SinOperator;

        // Set up fake context for planning
        let input_shapes = vec![TensorShape::Static(vec![8])];
        let mut input_values = vec![];
        input_values.resize(graph.nodes[0].inputs.len(), None);

        let ctx = InferenceContext::new(&graph.nodes[0], &graph, input_shapes, input_values);

        // Test shape inference
        let output_shapes = operator
            .infer_output_shapes(&ctx)
            .expect("Shape inference should succeed");
        assert_eq!(output_shapes.len(), 1);
        assert_eq!(output_shapes[0], TensorShape::Static(vec![8]));
    }

    #[test]
    fn test_sin_kernel_empty_input() {
        let operator = SinOperator;

        // Empty input shapes
        let graph = Graph::new();
        let mut node = Node::new("Sin");
        node.name = "sin_op".to_string();

        let ctx = InferenceContext::new(&node, &graph, vec![], vec![]);

        // Should fail with invalid shape error
        let result = operator.infer_output_shapes(&ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_sin_kernel_large_tensor() {
        let mut graph = Graph::new();

        // Large tensor: 1024 x 1024 = 1,048,576 elements
        let shape = vec![1024, 1024];

        graph.add_tensor(TensorInfo {
            name: "x".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(shape.clone()),
            kind: TensorKind::Input,
            initializer: None,
        });

        graph.add_tensor(TensorInfo {
            name: "y".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(shape.clone()),
            kind: TensorKind::Output,
            initializer: None,
        });

        let mut node = Node::new("Sin");
        node.name = "sin_op".to_string();
        node.inputs = vec!["x".to_string()];
        node.outputs = vec!["y".to_string()];
        graph.add_node(node);

        graph.inputs = vec!["x".to_string()];
        graph.outputs = vec!["y".to_string()];

        let operator = SinOperator;

        let input_shapes = vec![TensorShape::Static(shape.clone())];
        let mut input_values = vec![];
        input_values.resize(graph.nodes[0].inputs.len(), None);

        let ctx = InferenceContext::new(&graph.nodes[0], &graph, input_shapes, input_values);

        let output_shapes = operator
            .infer_output_shapes(&ctx)
            .expect("Shape inference should succeed");
        assert_eq!(output_shapes.len(), 1);
        assert_eq!(output_shapes[0], TensorShape::Static(shape));
    }
}
