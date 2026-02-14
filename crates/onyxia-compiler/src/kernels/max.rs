//! MaxKernel implementation for elementwise maximum.

use crate::error::{CodegenError, Result};
use crate::inference::{InferenceContext, TensorValue, broadcast_shapes};
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::{BindingDesc, ScratchBufferDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::TensorShape;
use std::collections::HashMap;

/// Kernel for elementwise maximum (ONNX Max operator).
///
/// Performs max = max(A, B, C, ...) where inputs are tensors.
/// Broadcasting is handled by the shader itself.
/// For >2 inputs, chains pairwise max operations.
pub struct MaxKernel;

impl OpKernel for MaxKernel {
    fn name(&self) -> &str {
        "Max"
    }

    fn infer_output_shapes(&self, ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>> {
        // Handle variadic inputs with broadcasting
        if ctx.input_shapes.is_empty() {
            return Err(CodegenError::UnsupportedOp(
                "Max requires at least one input".to_string(),
            ));
        }

        // Collect all non-Absent input shapes
        let mut static_shapes = Vec::new();
        for shape in &ctx.input_shapes {
            match shape {
                TensorShape::Static(dims) => static_shapes.push(dims.as_slice()),
                TensorShape::Unknown => return Ok(vec![TensorShape::Unknown]),
                TensorShape::Absent => continue,
                TensorShape::Dynamic(_) => {
                    return Err(CodegenError::InvalidShape(
                        "Dynamic shapes should have been resolved by Phase 1".to_string(),
                    ));
                }
            }
        }

        if static_shapes.is_empty() {
            return Ok(vec![TensorShape::Unknown]);
        }

        let result_dims = broadcast_shapes(&static_shapes)?;
        Ok(vec![TensorShape::Static(result_dims)])
    }

    fn try_fold(&self, ctx: &InferenceContext<'_>) -> Result<Vec<Option<TensorValue>>> {
        // Try to constant-fold if all inputs are known
        let mut input_values = Vec::new();
        for i in 0..ctx.input_shapes.len() {
            match ctx.input_value(i)? {
                Some(val) => input_values.push(val),
                None => return Ok(vec![None]),
            }
        }

        // Only fold F32 values with same shape for simplicity
        let first_len = match &input_values[0] {
            TensorValue::F32(vals) => vals.len(),
            _ => return Ok(vec![None]),
        };

        // Check all inputs are F32 with same length
        for val in &input_values {
            match val {
                TensorValue::F32(vals) if vals.len() == first_len => {}
                _ => return Ok(vec![None]),
            }
        }

        // Compute element-wise maximum
        let mut result = vec![f32::NEG_INFINITY; first_len];
        for val in input_values {
            if let TensorValue::F32(vals) = val {
                for (i, &v) in vals.iter().enumerate() {
                    result[i] = result[i].max(v);
                }
            }
        }

        Ok(vec![Some(TensorValue::F32(result))])
    }

    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        let num_inputs = ctx.node.inputs.len();

        if num_inputs == 0 {
            return Err(CodegenError::UnsupportedOp(
                "Max requires at least one input".to_string(),
            ));
        }

        // Handle single input - copy input to output using CopyBuffer
        if num_inputs == 1 {
            let input_info = ctx.input_info(0)?;
            let input_shape = ctx.static_shape(&input_info.shape)?;
            let size_bytes =
                (input_shape.iter().product::<usize>() * std::mem::size_of::<f32>()) as u64;

            return Ok(vec![Step::CopyBuffer {
                src: ctx.input(0),
                src_offset: 0,
                dst: ctx.output(0),
                dst_offset: 0,
                size: size_bytes,
            }]);
        }

        // For 2+ inputs, chain binary max operations
        let mut steps = Vec::new();
        let mut current_input = ctx.input(0);

        for i in 1..num_inputs {
            let is_last = i == num_inputs - 1;
            let output = if is_last {
                ctx.output(0)
            } else {
                // Allocate scratch buffer for intermediate result
                let output_info = ctx.output_info(0)?;
                let output_shape = ctx.static_shape(&output_info.shape)?;
                let size_bytes =
                    (output_shape.iter().product::<usize>() * std::mem::size_of::<f32>()) as u64;

                ctx.alloc_scratch(ScratchBufferDesc {
                    size: size_bytes,
                    label: format!("max_intermediate_{}", i),
                })
            };

            // Get output shape for this operation
            let output_shape = if is_last {
                let output_info = ctx.output_info(0)?;
                ctx.static_shape(&output_info.shape)?
            } else {
                // For intermediate results, compute broadcast shape
                let a_info = if i == 1 {
                    ctx.input_info(0)?
                } else {
                    // Would need to track intermediate shapes - for now use output shape
                    ctx.output_info(0)?
                };
                let b_info = ctx.input_info(i)?;
                let a_shape = ctx.static_shape(&a_info.shape)?;
                let b_shape = ctx.static_shape(&b_info.shape)?;
                let result_dims = broadcast_shapes(&[&a_shape, &b_shape])?;
                result_dims
            };

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
                "max",
                include_str!("../../shaders/elementwise/max.wgsl"),
                shader_defs,
            )?;

            // Get input sizes
            let a_info = if i == 1 {
                ctx.input_info(0)?
            } else {
                // For intermediate inputs, use the calculated size
                // This is simplified - ideally we'd track intermediate shapes
                ctx.output_info(0)?
            };
            let b_info = ctx.input_info(i)?;

            let a_shape = ctx.static_shape(&a_info.shape)?;
            let b_shape = ctx.static_shape(&b_info.shape)?;

            let a_size: u32 = a_shape.iter().product::<usize>() as u32;
            let b_size: u32 = b_shape.iter().product::<usize>() as u32;

            // Encode immediate data (must match ImmediateConstants struct in shader)
            let mut immediates_data = Vec::new();
            immediates_data.extend_from_slice(&(num_elements as u32).to_le_bytes());
            immediates_data.extend_from_slice(&a_size.to_le_bytes());
            immediates_data.extend_from_slice(&b_size.to_le_bytes());

            // Create dispatch step
            steps.push(Step::Dispatch {
                shader_index,
                bindings: vec![
                    BindingDesc {
                        buffer: current_input,
                        read_only: true,
                    },
                    BindingDesc {
                        buffer: ctx.input(i),
                        read_only: true,
                    },
                    BindingDesc {
                        buffer: output.clone(),
                        read_only: false,
                    },
                ],
                workgroups: [num_workgroups, 1, 1],
                immediates: Some(immediates_data),
            });

            current_input = output;
        }

        Ok(steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::BufferRef;
    use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
    use std::collections::HashMap;

    fn create_max_test_graph_2inputs() -> Graph {
        let mut graph = Graph::new();

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

        graph.inputs = vec!["a".to_string(), "b".to_string()];
        graph.outputs = vec!["c".to_string()];

        let mut node = Node::new("Max");
        node.inputs = vec!["a".to_string(), "b".to_string()];
        node.outputs = vec!["c".to_string()];
        graph.add_node(node);

        graph
    }

    #[test]
    fn test_max_kernel_name() {
        let kernel = MaxKernel;
        assert_eq!(kernel.name(), "Max");
    }

    #[test]
    fn test_max_infer_output_shapes_2inputs() {
        let graph = create_max_test_graph_2inputs();
        let kernel = MaxKernel;

        let mut node = Node::new("Max");
        node.inputs = vec!["a".to_string(), "b".to_string()];
        node.outputs = vec!["c".to_string()];

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
    fn test_max_infer_output_shapes_broadcasting() {
        let graph = create_max_test_graph_2inputs();
        let kernel = MaxKernel;

        let mut node = Node::new("Max");
        node.inputs = vec!["a".to_string(), "b".to_string()];
        node.outputs = vec!["c".to_string()];

        // Test [4] + [1] -> [4]
        let input_shapes = vec![TensorShape::Static(vec![4]), TensorShape::Static(vec![1])];
        let input_values = vec![None, None];

        let ctx = InferenceContext::new(&node, &graph, input_shapes, input_values);
        let output_shapes = kernel
            .infer_output_shapes(&ctx)
            .expect("Should infer shapes");

        assert_eq!(output_shapes.len(), 1);
        assert_eq!(output_shapes[0], TensorShape::Static(vec![4]));
    }

    #[test]
    fn test_max_kernel_plan() {
        let graph = create_max_test_graph_2inputs();
        let mut node = Node::new("Max");
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

        let steps = MaxKernel.plan(&mut ctx).expect("Planning should succeed");

        // Should have one dispatch step for 2 inputs
        assert_eq!(steps.len(), 1);

        match &steps[0] {
            Step::Dispatch {
                shader_index,
                bindings,
                workgroups,
                immediates,
            } => {
                assert_eq!(*shader_index, 0);
                assert_eq!(bindings.len(), 3);
                assert_eq!(bindings[0].buffer, BufferRef::Tensor(0));
                assert_eq!(bindings[1].buffer, BufferRef::Tensor(1));
                assert_eq!(bindings[2].buffer, BufferRef::Tensor(2));
                assert!(immediates.is_some());
                assert_eq!(workgroups[0], 1); // ceil(4 / 256) = 1
            }
            _ => panic!("Expected Dispatch step"),
        }

        // Verify shader was compiled
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].label, "max");
        assert_eq!(shaders[0].entry_point, "main");
    }

    #[test]
    fn test_max_try_fold() {
        let graph = create_max_test_graph_2inputs();
        let kernel = MaxKernel;

        let mut node = Node::new("Max");
        node.inputs = vec!["a".to_string(), "b".to_string()];
        node.outputs = vec!["c".to_string()];

        let input_shapes = vec![TensorShape::Static(vec![4]), TensorShape::Static(vec![4])];
        let input_values = vec![
            Some(TensorValue::F32(vec![1.0, 5.0, 2.0, 8.0])),
            Some(TensorValue::F32(vec![3.0, 2.0, 6.0, 4.0])),
        ];

        let ctx = InferenceContext::new(&node, &graph, input_shapes, input_values);
        let folded = kernel.try_fold(&ctx).expect("Should fold");

        assert_eq!(folded.len(), 1);
        match &folded[0] {
            Some(TensorValue::F32(vals)) => {
                assert_eq!(vals, &[3.0, 5.0, 6.0, 8.0]);
            }
            _ => panic!("Expected F32 tensor value"),
        }
    }
}
