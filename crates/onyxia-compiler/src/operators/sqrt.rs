//! SqrtOperator implementation for square root function.

use crate::error::Result;
use crate::inference::{InferenceContext, TensorValue};
use crate::operator::{OpOperator, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::TensorShape;
use std::collections::HashMap;

/// Operator for Sqrt (ONNX Sqrt operator).
///
/// Sqrt(x) = sqrt(x) - element-wise square root function.
pub struct SqrtOperator;

impl OpOperator for SqrtOperator {
    fn name(&self) -> &str {
        "Sqrt"
    }

    fn infer_output_shapes(&self, ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>> {
        // Sqrt is a unary operation: output shape equals input shape
        if ctx.input_shapes.is_empty() {
            return Err(crate::error::CodegenError::InvalidShape(
                "Sqrt requires one input".to_string(),
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
                let result: Vec<f32> = vals.iter().map(|x| x.sqrt()).collect();
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
            "sqrt",
            include_str!("../../shaders/elementwise/sqrt.wgsl"),
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
    use crate::plan::BufferRef;
    use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
    use std::collections::HashMap;

    fn create_sqrt_test_graph() -> Graph {
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

        graph.inputs = vec!["x".to_string()];
        graph.outputs = vec!["y".to_string()];

        graph
    }

    /// Test that Sqrt kernel infers output shapes correctly.
    #[test]
    fn test_sqrt_infer_shapes() {
        let operator = SqrtOperator;
        let graph = Graph::new();

        // Test with 1D tensor
        let mut node = Node::new("Sqrt");
        node.inputs = vec!["input".to_string()];
        node.outputs = vec!["output".to_string()];

        let input_shapes = vec![TensorShape::Static(vec![4])];
        let input_values = vec![None];

        let ctx = InferenceContext::new(&node, &graph, input_shapes, input_values);

        let result = operator.infer_output_shapes(&ctx).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], TensorShape::Static(vec![4]));

        // Test with 2D tensor
        let input_shapes = vec![TensorShape::Static(vec![2, 3])];
        let input_values = vec![None];

        let ctx = InferenceContext::new(&node, &graph, input_shapes, input_values);

        let result = operator.infer_output_shapes(&ctx).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], TensorShape::Static(vec![2, 3]));
    }

    /// Test that Sqrt kernel performs constant folding correctly.
    #[test]
    fn test_sqrt_constant_fold() {
        let operator = SqrtOperator;
        let graph = Graph::new();

        let mut node = Node::new("Sqrt");
        node.inputs = vec!["input".to_string()];
        node.outputs = vec!["output".to_string()];

        // Test with known values
        let input_values = vec![Some(TensorValue::F32(vec![0.0, 1.0, 4.0, 9.0, 16.0]))];
        let input_shapes = vec![TensorShape::Static(vec![5])];

        let ctx = InferenceContext::new(&node, &graph, input_shapes, input_values);

        let result = operator.try_fold(&ctx).unwrap();
        assert_eq!(result.len(), 1);

        if let Some(TensorValue::F32(vals)) = &result[0] {
            assert_eq!(vals.len(), 5);
            assert!((vals[0] - 0.0).abs() < 1e-6); // sqrt(0) = 0
            assert!((vals[1] - 1.0).abs() < 1e-6); // sqrt(1) = 1
            assert!((vals[2] - 2.0).abs() < 1e-6); // sqrt(4) = 2
            assert!((vals[3] - 3.0).abs() < 1e-6); // sqrt(9) = 3
            assert!((vals[4] - 4.0).abs() < 1e-6); // sqrt(16) = 4
        } else {
            panic!("Expected F32 tensor value");
        }
    }

    #[test]
    fn test_sqrt_kernel_plan() {
        let graph = create_sqrt_test_graph();
        let mut node = Node::new("Sqrt");
        node.inputs = vec!["x".to_string()];
        node.outputs = vec!["y".to_string()];

        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let steps = SqrtOperator.plan(&mut ctx).expect("Planning should succeed");

        // Verify we got exactly one dispatch step
        assert_eq!(steps.len(), 1);

        match &steps[0] {
            Step::Dispatch {
                shader_index,
                bindings,
                workgroups,
                immediates,
            } => {
                // Verify shader was compiled
                assert_eq!(*shader_index, 0);

                // Verify bindings: 1 read-only input + 1 read-write output
                assert_eq!(bindings.len(), 2);

                assert_eq!(bindings[0].buffer, BufferRef::Tensor(0));
                assert!(bindings[0].read_only);

                assert_eq!(bindings[1].buffer, BufferRef::Tensor(1));
                assert!(!bindings[1].read_only);

                // Verify workgroup count: ceil(8 / 256) = 1
                assert_eq!(*workgroups, [1, 1, 1]);

                // Verify immediates contain size
                let imm = immediates.as_ref().unwrap();
                assert_eq!(imm.len(), 4); // 1 u32 = 4 bytes
                let size = u32::from_le_bytes([imm[0], imm[1], imm[2], imm[3]]);
                assert_eq!(size, 8);
            }
            _ => panic!("Expected Dispatch step"),
        }

        // Verify shader was compiled
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].label, "sqrt");
        assert_eq!(shaders[0].entry_point, "main");
    }

    #[test]
    fn test_sqrt_kernel_large_tensor() {
        let mut graph = Graph::new();

        // Create large tensor (10000 elements)
        graph.add_tensor(TensorInfo {
            name: "x".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![100, 100]),
            kind: TensorKind::Input,
            initializer: None,
        });

        graph.add_tensor(TensorInfo {
            name: "y".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![100, 100]),
            kind: TensorKind::Output,
            initializer: None,
        });

        graph.inputs = vec!["x".to_string()];
        graph.outputs = vec!["y".to_string()];

        let mut node = Node::new("Sqrt");
        node.inputs = vec!["x".to_string()];
        node.outputs = vec!["y".to_string()];

        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let steps = SqrtOperator.plan(&mut ctx).expect("Planning should succeed");

        match &steps[0] {
            Step::Dispatch { workgroups, .. } => {
                // Verify workgroup count: ceil(10000 / 256) = 40
                assert_eq!(*workgroups, [40, 1, 1]);
            }
            _ => panic!("Expected Dispatch step"),
        }
    }
}
