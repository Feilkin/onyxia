//! GreaterKernel implementation for elementwise greater-than comparison.

use crate::error::Result;
use crate::inference::{InferenceContext, infer_elementwise_broadcast};
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::TensorShape;
use std::collections::HashMap;

/// Kernel for elementwise greater-than comparison (ONNX Greater operator).
///
/// Performs C = (A > B) where A and B are tensors of the same shape.
/// Broadcasting is handled by the shader itself.
///
/// ## Output Type
///
/// The ONNX spec defines Greater's output as boolean (DataType::Bool). Since WGSL
/// does not support bool storage arrays, the shader writes u32 values (0 for false,
/// 1 for true). The DataType::Bool size is defined as 4 bytes to match this GPU
/// representation, so Bool-typed outputs work correctly.
pub struct GreaterKernel;

impl OpKernel for GreaterKernel {
    fn name(&self) -> &str {
        "Greater"
    }

    fn infer_output_shapes(&self, ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>> {
        infer_elementwise_broadcast(ctx)
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
            "greater",
            include_str!("../../shaders/elementwise/greater.wgsl"),
            shader_defs,
        )?;

        // Get input shapes for immediate data
        let input_a_info = ctx.input_info(0)?;
        let input_a_shape = ctx.static_shape(&input_a_info.shape)?;
        let a_size: u32 = input_a_shape.iter().product::<usize>() as u32;

        let input_b_info = ctx.input_info(1)?;
        let input_b_shape = ctx.static_shape(&input_b_info.shape)?;
        let b_size: u32 = input_b_shape.iter().product::<usize>() as u32;

        // Encode immediate data (must match ImmediateConstants struct in shader)
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(num_elements as u32).to_le_bytes());
        immediates_data.extend_from_slice(&a_size.to_le_bytes());
        immediates_data.extend_from_slice(&b_size.to_le_bytes());

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
    use crate::inference::InferenceContext;
    use crate::plan::BufferRef;
    use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
    use std::collections::HashMap;

    fn create_greater_test_graph() -> Graph {
        let mut graph = Graph::new();

        // Add input tensors
        graph.add_tensor(TensorInfo {
            name: "a".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Input,
            initializer: None,
        });

        graph.add_tensor(TensorInfo {
            name: "b".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add output tensor (Bool type per ONNX spec)
        graph.add_tensor(TensorInfo {
            name: "c".to_string(),
            dtype: DataType::Bool,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Intermediate,
            initializer: None,
        });

        graph
    }

    #[test]
    fn test_greater_kernel_basic() {
        let kernel = GreaterKernel;
        assert_eq!(kernel.name(), "Greater");
    }

    #[test]
    fn test_greater_kernel_plan() {
        let kernel = GreaterKernel;
        let graph = create_greater_test_graph();

        let mut node = Node::new("Greater");
        node.inputs = vec!["a".to_string(), "b".to_string()];
        node.outputs = vec!["c".to_string()];

        let input_ids = vec![0, 1];
        let output_ids = vec![2];
        let dynamic_dimensions = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let steps = kernel.plan(&mut ctx).expect("Planning should succeed");

        // Verify we got a dispatch step
        assert_eq!(steps.len(), 1);
        match &steps[0] {
            Step::Dispatch {
                shader_index,
                bindings,
                workgroups,
                immediates,
            } => {
                // Should have compiled one shader
                assert_eq!(*shader_index, 0);

                // Should have 3 bindings: input A, input B, output C
                assert_eq!(bindings.len(), 3);

                // Check binding configuration
                assert_eq!(bindings[0].buffer, BufferRef::Tensor(0)); // input A
                assert!(bindings[0].read_only);
                assert_eq!(bindings[1].buffer, BufferRef::Tensor(1)); // input B
                assert!(bindings[1].read_only);
                assert_eq!(bindings[2].buffer, BufferRef::Tensor(2)); // output C
                assert!(!bindings[2].read_only);

                // Should have workgroups for 4 elements with workgroup_size 256
                assert_eq!(*workgroups, [1, 1, 1]);

                // Check immediates (size, a_size, b_size)
                let imm = immediates.as_ref().expect("Should have immediates");
                assert_eq!(imm.len(), 12); // 3 u32 values = 12 bytes
            }
            _ => panic!("Expected Dispatch step"),
        }

        // Verify shader was compiled
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].label, "greater");
        assert_eq!(shaders[0].entry_point, "main");
    }

    #[test]
    fn test_greater_kernel_infer_shapes() {
        let kernel = GreaterKernel;
        let graph = create_greater_test_graph();

        // Test case: same shapes
        let mut node = Node::new("Greater");
        node.inputs = vec!["a".to_string(), "b".to_string()];
        node.outputs = vec!["c".to_string()];

        let input_shapes = vec![
            TensorShape::Static(vec![2, 3]),
            TensorShape::Static(vec![2, 3]),
        ];
        let input_values = vec![None, None];

        let ctx = InferenceContext::new(&node, &graph, input_shapes, input_values);

        let result = kernel
            .infer_output_shapes(&ctx)
            .expect("Should infer shapes");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], TensorShape::Static(vec![2, 3]));

        // Test case: broadcasting - scalar and tensor
        let input_shapes_broadcast =
            vec![TensorShape::Static(vec![1]), TensorShape::Static(vec![4])];
        let input_values_broadcast = vec![None, None];

        let ctx_broadcast = InferenceContext::new(
            &node,
            &graph,
            input_shapes_broadcast,
            input_values_broadcast,
        );

        let result_broadcast = kernel
            .infer_output_shapes(&ctx_broadcast)
            .expect("Should infer broadcast shapes");
        assert_eq!(result_broadcast.len(), 1);
        assert_eq!(result_broadcast[0], TensorShape::Static(vec![4]));
    }
}
