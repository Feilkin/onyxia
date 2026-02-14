//! SoftmaxKernel implementation for softmax activation function.

use crate::error::{CodegenError, Result};
use crate::inference::InferenceContext;
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::TensorShape;
use std::collections::HashMap;

/// Kernel for Softmax operation (ONNX Softmax operator).
///
/// Computes softmax activation along a specified axis:
///   output[i] = exp(input[i]) / sum(exp(input[j]) for all j in slice)
///
/// Uses numerically stable implementation:
///   max_val = max(input along axis)
///   output[i] = exp(input[i] - max_val) / sum(exp(input[j] - max_val))
pub struct SoftmaxKernel;

impl OpKernel for SoftmaxKernel {
    fn name(&self) -> &str {
        "Softmax"
    }

    fn infer_output_shapes(&self, ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>> {
        // Softmax: output shape equals input shape
        if ctx.input_shapes.is_empty() {
            return Err(CodegenError::InvalidShape(
                "Softmax requires one input".to_string(),
            ));
        }
        Ok(vec![ctx.input_shapes[0].clone()])
    }

    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        // Get input shape
        let input_info = ctx.input_info(0)?;
        let input_shape = ctx.static_shape(&input_info.shape)?;
        let rank = input_shape.len();

        // Get axis attribute (default to -1, which means last axis)
        let axis: i64 = ctx.node.attr("axis").unwrap_or(-1);
        let normalized_axis = if axis < 0 {
            (rank as i64 + axis) as usize
        } else {
            axis as usize
        };

        if normalized_axis >= rank {
            return Err(CodegenError::InvalidShape(format!(
                "Softmax axis {} is out of bounds for rank {}",
                axis, rank
            )));
        }

        // Calculate dimensions
        let axis_dim = input_shape[normalized_axis];
        let outer_size: usize = input_shape[..normalized_axis].iter().product();
        let outer_size = if outer_size == 0 { 1 } else { outer_size };
        let num_elements: usize = input_shape.iter().product();

        // Configure workgroup size
        let workgroup_size: u32 = 256;
        let num_workgroups = (outer_size as u32).div_ceil(workgroup_size);

        // Prepare shader definitions
        let mut shader_defs = HashMap::new();
        shader_defs.insert(
            "WORKGROUP_SIZE".to_string(),
            ShaderDefValue::UInt(workgroup_size),
        );

        // Compile shader
        let shader_index = ctx.compile_shader(
            "softmax",
            include_str!("../../shaders/activation/softmax.wgsl"),
            shader_defs,
        )?;

        // Encode immediate data
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(num_elements as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(outer_size as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(axis_dim as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(normalized_axis as u32).to_le_bytes());

        // Create dispatch step
        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0), // input
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(0), // output
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

    #[test]
    fn test_softmax_shape_inference() {
        let kernel = SoftmaxKernel;
        let mut graph = Graph::new();

        graph.add_tensor(TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![2, 4, 8]),
            kind: TensorKind::Input,
            initializer: None,
        });

        let mut node = Node::new("Softmax");
        node.inputs = vec!["input".to_string()];

        let input_shapes = vec![TensorShape::Static(vec![2, 4, 8])];
        let input_values = vec![None];

        let ctx = InferenceContext::new(&node, &graph, input_shapes, input_values);

        let shapes = kernel
            .infer_output_shapes(&ctx)
            .expect("Shape inference should succeed");
        assert_eq!(shapes.len(), 1);
        assert_eq!(shapes[0], TensorShape::Static(vec![2, 4, 8]));
    }

    #[test]
    fn test_softmax_kernel_plan() {
        let mut graph = Graph::new();

        graph.add_tensor(TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![2, 32]),
            kind: TensorKind::Input,
            initializer: None,
        });

        graph.add_tensor(TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![2, 32]),
            kind: TensorKind::Output,
            initializer: None,
        });

        let mut node = Node::new("Softmax");
        node.inputs = vec!["input".to_string()];
        node.outputs = vec!["output".to_string()];
        // Default axis = -1 (last axis)

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

        let steps = SoftmaxKernel
            .plan(&mut ctx)
            .expect("Planning should succeed");

        assert_eq!(steps.len(), 1);

        match &steps[0] {
            Step::Dispatch {
                shader_index,
                bindings,
                workgroups: _,
                immediates,
            } => {
                assert_eq!(*shader_index, 0);
                assert_eq!(bindings.len(), 2);
                assert_eq!(bindings[0].buffer, BufferRef::Tensor(0));
                assert!(bindings[0].read_only);
                assert_eq!(bindings[1].buffer, BufferRef::Tensor(1));
                assert!(!bindings[1].read_only);
                assert!(immediates.is_some());
            }
            _ => panic!("Expected Dispatch step"),
        }
    }
}
