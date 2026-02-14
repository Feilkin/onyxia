//! WhereKernel implementation for conditional element selection.

use crate::error::Result;
use crate::inference::{InferenceContext, infer_elementwise_broadcast};
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::TensorShape;
use std::collections::HashMap;

/// Kernel for conditional element selection (ONNX Where operator).
///
/// Performs output = condition ? X : Y where condition, X, and Y are tensors.
/// Broadcasting is supported for all three inputs.
pub struct WhereKernel;

impl OpKernel for WhereKernel {
    fn name(&self) -> &str {
        "Where"
    }

    fn infer_output_shapes(&self, ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>> {
        // infer_elementwise_broadcast handles any number of inputs, including 3
        infer_elementwise_broadcast(ctx)
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
            "where",
            include_str!("../../shaders/elementwise/where.wgsl"),
            shader_defs,
        )?;

        // Get input shapes for immediate data
        let input_condition_info = ctx.input_info(0)?;
        let condition_shape = ctx.static_shape(&input_condition_info.shape)?;
        let condition_size: u32 = condition_shape.iter().product::<usize>() as u32;

        let input_x_info = ctx.input_info(1)?;
        let x_shape = ctx.static_shape(&input_x_info.shape)?;
        let x_size: u32 = x_shape.iter().product::<usize>() as u32;

        let input_y_info = ctx.input_info(2)?;
        let y_shape = ctx.static_shape(&input_y_info.shape)?;
        let y_size: u32 = y_shape.iter().product::<usize>() as u32;

        // Encode immediate data (must match ImmediateConstants struct in shader)
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(num_elements as u32).to_le_bytes());
        immediates_data.extend_from_slice(&condition_size.to_le_bytes());
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
                    buffer: ctx.input(2),
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

    fn create_where_test_graph() -> Graph {
        let mut graph = Graph::new();

        // Add input tensors
        graph.add_tensor(TensorInfo {
            name: "condition".to_string(),
            dtype: DataType::I32, // Boolean represented as i32
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Input,
            initializer: None,
        });

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
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Output,
            initializer: None,
        });

        graph.inputs = vec!["condition".to_string(), "x".to_string(), "y".to_string()];
        graph.outputs = vec!["output".to_string()];

        graph
    }

    #[test]
    fn test_where_kernel_plan() {
        let graph = create_where_test_graph();
        let mut node = Node::new("Where");
        node.inputs = vec!["condition".to_string(), "x".to_string(), "y".to_string()];
        node.outputs = vec!["output".to_string()];

        let input_ids = vec![0, 1, 2];
        let output_ids = vec![3];
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

        let steps = WhereKernel.plan(&mut ctx).expect("Planning should succeed");

        // Verify we got exactly one dispatch step
        assert_eq!(steps.len(), 1);

        match &steps[0] {
            Step::Dispatch {
                shader_index,
                bindings,
                workgroups,
                ..
            } => {
                // Verify shader was compiled
                assert_eq!(*shader_index, 0);

                // Verify bindings: 3 read-only inputs + 1 read-write output
                assert_eq!(bindings.len(), 4);

                assert_eq!(bindings[0].buffer, BufferRef::Tensor(0));
                assert!(bindings[0].read_only);

                assert_eq!(bindings[1].buffer, BufferRef::Tensor(1));
                assert!(bindings[1].read_only);

                assert_eq!(bindings[2].buffer, BufferRef::Tensor(2));
                assert!(bindings[2].read_only);

                assert_eq!(bindings[3].buffer, BufferRef::Tensor(3));
                assert!(!bindings[3].read_only);

                // Verify workgroup count: ceil(4 / 256) = 1
                assert_eq!(*workgroups, [1, 1, 1]);
            }
            _ => panic!("Expected Dispatch step"),
        }

        // Verify shader was compiled
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].label, "where");
        assert_eq!(shaders[0].entry_point, "main");
    }
}
