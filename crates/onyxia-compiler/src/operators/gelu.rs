//! GeluOperator implementation for GELU activation function.

use crate::error::Result;
use crate::inference::{InferenceContext, TensorValue};
use crate::operator::{Operator, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::TensorShape;
use std::collections::HashMap;

/// Kernel for GELU activation (ONNX Gelu operator).
///
/// GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function
/// of the standard normal distribution.
/// Uses tanh approximation for efficiency.
pub struct GeluOperator;

impl Operator for GeluOperator {
    fn name(&self) -> &str {
        "Gelu"
    }

    fn infer_output_shapes(&self, ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>> {
        // Gelu is a unary operation: output shape equals input shape
        if ctx.input_shapes.is_empty() {
            return Err(crate::error::CodegenError::InvalidShape(
                "Gelu requires one input".to_string(),
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
        // GELU(x) = x * Φ(x) where Φ is the cumulative distribution function
        // Using tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        match input {
            TensorValue::F32(vals) => {
                let result: Vec<f32> = vals
                    .iter()
                    .map(|&x| {
                        let sqrt_2_over_pi = 0.79788456_f32; // sqrt(2/π)
                        let coeff = 0.044715_f32;
                        let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
                        0.5 * x * (1.0 + inner.tanh())
                    })
                    .collect();
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
            "gelu",
            include_str!("../../shaders/activation/gelu.wgsl"),
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

    fn create_gelu_test_graph() -> Graph {
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

    #[test]
    fn test_gelu_kernel_plan() {
        let graph = create_gelu_test_graph();
        let mut node = Node::new("Gelu");
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

        let steps = GeluOperator
            .plan(&mut ctx)
            .expect("Planning should succeed");

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
        assert_eq!(shaders[0].label, "gelu");
        assert_eq!(shaders[0].entry_point, "main");
    }

    #[test]
    fn test_gelu_kernel_large_tensor() {
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

        let mut node = Node::new("Gelu");
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

        let steps = GeluOperator
            .plan(&mut ctx)
            .expect("Planning should succeed");

        match &steps[0] {
            Step::Dispatch { workgroups, .. } => {
                // Verify workgroup count: ceil(10000 / 256) = 40
                assert_eq!(*workgroups, [40, 1, 1]);
            }
            _ => panic!("Expected Dispatch step"),
        }
    }
}
