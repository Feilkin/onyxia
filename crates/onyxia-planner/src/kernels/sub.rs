//! SubKernel implementation for elementwise subtraction.

use crate::error::Result;
use crate::inference::{InferenceContext, infer_elementwise_broadcast};
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::TensorShape;
use std::collections::HashMap;

/// Kernel for elementwise subtraction (ONNX Sub operator).
///
/// Performs C = A - B where A and B are tensors of the same shape.
/// Broadcasting is handled by the shader itself.
pub struct SubKernel;

impl OpKernel for SubKernel {
    fn name(&self) -> &str {
        "Sub"
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
            "sub",
            include_str!("../../shaders/elementwise/sub.wgsl"),
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
    use crate::plan::BufferRef;
    use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
    use std::collections::HashMap;

    fn create_sub_test_graph() -> Graph {
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

        // Add output tensor
        graph.add_tensor(TensorInfo {
            name: "c".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Output,
            initializer: None,
        });

        graph.inputs = vec!["a".to_string(), "b".to_string()];
        graph.outputs = vec!["c".to_string()];

        graph
    }

    #[test]
    fn test_sub_kernel_plan() {
        let graph = create_sub_test_graph();
        let mut node = Node::new("Sub");
        node.inputs = vec!["a".to_string(), "b".to_string()];
        node.outputs = vec!["c".to_string()];

        let input_ids = vec![0, 1];
        let output_ids = vec![2];
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

        let steps = SubKernel.plan(&mut ctx).expect("Planning should succeed");

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

                // Verify bindings: 2 read-only inputs + 1 read-write output
                assert_eq!(bindings.len(), 3);

                assert_eq!(bindings[0].buffer, BufferRef::Tensor(0));
                assert!(bindings[0].read_only);

                assert_eq!(bindings[1].buffer, BufferRef::Tensor(1));
                assert!(bindings[1].read_only);

                assert_eq!(bindings[2].buffer, BufferRef::Tensor(2));
                assert!(!bindings[2].read_only);

                // Verify workgroup count: ceil(4 / 256) = 1
                assert_eq!(*workgroups, [1, 1, 1]);
            }
            _ => panic!("Expected Dispatch step"),
        }

        // Verify shader was compiled
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].label, "sub");
        assert_eq!(shaders[0].entry_point, "main");
    }

    #[test]
    fn test_sub_kernel_broadcasting() {
        let mut graph = Graph::new();

        // Create tensors with broadcasting: scalar [1] - tensor [4]
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

        graph.add_tensor(TensorInfo {
            name: "c".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Output,
            initializer: None,
        });

        let mut node = Node::new("Sub");
        node.inputs = vec!["a".to_string(), "b".to_string()];
        node.outputs = vec!["c".to_string()];

        let input_ids = vec![0, 1];
        let output_ids = vec![2];
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

        let steps = SubKernel.plan(&mut ctx).expect("Planning should succeed");

        // Verify the immediate data includes shape information for broadcasting
        match &steps[0] {
            Step::Dispatch { immediates, .. } => {
                assert!(immediates.is_some());
                let imm_data = immediates.as_ref().unwrap();
                // Should have 3 u32 values: num_elements, a_size, b_size
                assert_eq!(imm_data.len(), 12); // 3 * 4 bytes
            }
            _ => panic!("Expected Dispatch step"),
        }
    }

    #[test]
    fn test_sub_kernel_larger_tensor() {
        let mut graph = Graph::new();

        // Create larger tensors (1024 elements)
        graph.add_tensor(TensorInfo {
            name: "a".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![32, 32]),
            kind: TensorKind::Input,
            initializer: None,
        });

        graph.add_tensor(TensorInfo {
            name: "b".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![32, 32]),
            kind: TensorKind::Input,
            initializer: None,
        });

        graph.add_tensor(TensorInfo {
            name: "c".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![32, 32]),
            kind: TensorKind::Output,
            initializer: None,
        });

        let mut node = Node::new("Sub");
        node.inputs = vec!["a".to_string(), "b".to_string()];
        node.outputs = vec!["c".to_string()];

        let input_ids = vec![0, 1];
        let output_ids = vec![2];
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

        let steps = SubKernel.plan(&mut ctx).expect("Planning should succeed");

        match &steps[0] {
            Step::Dispatch { workgroups, .. } => {
                // 1024 elements / 256 = 4 workgroups
                assert_eq!(*workgroups, [4, 1, 1]);
            }
            _ => panic!("Expected Dispatch step"),
        }
    }

    #[test]
    fn test_sub_kernel_non_divisible_size() {
        let mut graph = Graph::new();

        // Create tensors with size that doesn't divide evenly by workgroup size
        // 100 elements: ceil(100 / 256) = 1 workgroup
        graph.add_tensor(TensorInfo {
            name: "a".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![100]),
            kind: TensorKind::Input,
            initializer: None,
        });

        graph.add_tensor(TensorInfo {
            name: "b".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![100]),
            kind: TensorKind::Input,
            initializer: None,
        });

        graph.add_tensor(TensorInfo {
            name: "c".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![100]),
            kind: TensorKind::Output,
            initializer: None,
        });

        let mut node = Node::new("Sub");
        node.inputs = vec!["a".to_string(), "b".to_string()];
        node.outputs = vec!["c".to_string()];

        let input_ids = vec![0, 1];
        let output_ids = vec![2];
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

        let steps = SubKernel.plan(&mut ctx).expect("Planning should succeed");

        match &steps[0] {
            Step::Dispatch { workgroups, .. } => {
                // ceil(100 / 256) = 1
                assert_eq!(*workgroups, [1, 1, 1]);
            }
            _ => panic!("Expected Dispatch step"),
        }
    }

    #[test]
    fn test_sub_kernel_shader_deduplication() {
        let graph = create_sub_test_graph();
        let mut node = Node::new("Sub");
        node.inputs = vec!["a".to_string(), "b".to_string()];
        node.outputs = vec!["c".to_string()];

        let input_ids = vec![0, 1];
        let output_ids = vec![2];
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
        let mut shaders = Vec::new();

        // Plan twice with same context
        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let _steps1 = SubKernel
            .plan(&mut ctx)
            .expect("First planning should succeed");
        let steps2 = SubKernel
            .plan(&mut ctx)
            .expect("Second planning should succeed");

        // Both should reference the same shader (deduplicated)
        match &steps2[0] {
            Step::Dispatch { shader_index, .. } => {
                assert_eq!(*shader_index, 0);
            }
            _ => panic!("Expected Dispatch step"),
        }

        // Should only have one compiled shader
        assert_eq!(shaders.len(), 1);
    }
}
