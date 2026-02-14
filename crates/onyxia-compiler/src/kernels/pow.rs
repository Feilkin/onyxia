//! PowKernel implementation for elementwise power operation.

use crate::error::Result;
use crate::inference::{InferenceContext, TensorValue, infer_elementwise_broadcast};
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::TensorShape;
use std::collections::HashMap;

/// Kernel for elementwise power operation (ONNX Pow operator).
///
/// Performs Z = X ^ Y where X is the base and Y is the exponent.
/// Broadcasting is handled by the shader itself.
pub struct PowKernel;

impl OpKernel for PowKernel {
    fn name(&self) -> &str {
        "Pow"
    }

    fn infer_output_shapes(&self, ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>> {
        infer_elementwise_broadcast(ctx)
    }

    fn try_fold(&self, ctx: &InferenceContext<'_>) -> Result<Vec<Option<TensorValue>>> {
        // Try to constant-fold if both inputs are known
        let Some(x) = ctx.input_value(0)? else {
            return Ok(vec![None]);
        };
        let Some(y) = ctx.input_value(1)? else {
            return Ok(vec![None]);
        };

        // Only fold F32 values with same shape for simplicity
        match (x, y) {
            (TensorValue::F32(x_vals), TensorValue::F32(y_vals))
                if x_vals.len() == y_vals.len() =>
            {
                let result: Vec<f32> = x_vals
                    .iter()
                    .zip(y_vals.iter())
                    .map(|(x, y)| x.powf(*y))
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
        let num_workgroups = (num_elements as u32).div_ceil(workgroup_size);

        // Prepare shader definitions
        let mut shader_defs = HashMap::new();
        shader_defs.insert(
            "WORKGROUP_SIZE".to_string(),
            ShaderDefValue::UInt(workgroup_size),
        );

        // Compile shader
        let shader_index = ctx.compile_shader(
            "pow",
            include_str!("../../shaders/elementwise/pow.wgsl"),
            shader_defs,
        )?;

        // Get input shapes for immediate data
        let input_x_info = ctx.input_info(0)?;
        let input_x_shape = ctx.static_shape(&input_x_info.shape)?;
        let x_size: u32 = input_x_shape.iter().product::<usize>() as u32;

        let input_y_info = ctx.input_info(1)?;
        let input_y_shape = ctx.static_shape(&input_y_info.shape)?;
        let y_size: u32 = input_y_shape.iter().product::<usize>() as u32;

        // Encode immediate data (must match ImmediateConstants struct in shader)
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(num_elements as u32).to_le_bytes());
        immediates_data.extend_from_slice(&x_size.to_le_bytes());
        immediates_data.extend_from_slice(&y_size.to_le_bytes());

        // Create dispatch step with bindings and immediates
        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0),
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(1),
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

    fn create_pow_test_graph() -> Graph {
        let mut graph = Graph::new();

        // Add input tensors
        graph.add_tensor(TensorInfo {
            name: "x".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Input,
            initializer: None,
        });

        graph.add_tensor(TensorInfo {
            name: "y".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add output tensor
        graph.add_tensor(TensorInfo {
            name: "z".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Output,
            initializer: None,
        });

        graph.inputs = vec!["x".to_string(), "y".to_string()];
        graph.outputs = vec!["z".to_string()];

        // Create Pow operation node
        let mut node = Node::new("Pow");
        node.name = "pow_node".to_string();
        node.inputs = vec!["x".to_string(), "y".to_string()];
        node.outputs = vec!["z".to_string()];
        graph.add_node(node);

        graph
    }

    #[test]
    fn test_pow_kernel_name() {
        let kernel = PowKernel;
        assert_eq!(kernel.name(), "Pow");
    }

    #[test]
    fn test_pow_infer_output_shapes() {
        let graph = create_pow_test_graph();
        let kernel = PowKernel;

        let mut node = Node::new("Pow");
        node.inputs = vec!["x".to_string(), "y".to_string()];
        node.outputs = vec!["z".to_string()];

        let input_shapes = vec![TensorShape::Static(vec![4]), TensorShape::Static(vec![4])];
        let input_values = vec![None, None];

        let ctx = InferenceContext::new(&node, &graph, input_shapes, input_values);
        let output_shapes = kernel
            .infer_output_shapes(&ctx)
            .expect("Should infer shapes");

        assert_eq!(output_shapes.len(), 1);
        assert_eq!(output_shapes[0], TensorShape::Static(vec![4]));
    }

    #[test]
    fn test_pow_try_fold() {
        let graph = create_pow_test_graph();
        let kernel = PowKernel;

        let mut node = Node::new("Pow");
        node.inputs = vec!["x".to_string(), "y".to_string()];
        node.outputs = vec!["z".to_string()];

        // Test with known values
        let input_shapes = vec![TensorShape::Static(vec![4]), TensorShape::Static(vec![4])];
        let input_values = vec![
            Some(TensorValue::F32(vec![2.0, 3.0, 4.0, 5.0])),
            Some(TensorValue::F32(vec![2.0, 2.0, 2.0, 2.0])),
        ];

        let ctx = InferenceContext::new(&node, &graph, input_shapes, input_values);
        let folded = kernel.try_fold(&ctx).expect("Should fold");

        assert_eq!(folded.len(), 1);
        if let Some(TensorValue::F32(result)) = &folded[0] {
            assert_eq!(result.len(), 4);
            assert_eq!(result[0], 4.0); // 2^2
            assert_eq!(result[1], 9.0); // 3^2
            assert_eq!(result[2], 16.0); // 4^2
            assert_eq!(result[3], 25.0); // 5^2
        } else {
            panic!("Expected folded F32 value");
        }
    }
}
