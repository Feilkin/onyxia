//! ScatterNDOperator implementation for scattered tensor updates.

use crate::error::Result;
use crate::inference::InferenceContext;
use crate::operator::{OpOperator, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::TensorShape;
use std::collections::HashMap;

/// Operator for ScatterND operation (ONNX ScatterND operator).
///
/// Scatters updates into a tensor at specified indices:
///   output = copy(data)
///   for each index in indices:
///     output[index] = update (with reduction)
///
/// Inputs:
/// - data: Tensor to copy and update
/// - indices: Indices to scatter updates into output
/// - updates: Values to scatter into output
///
/// Attributes:
/// - reduction: "none" (default), "add", "mul", "max", "min"
pub struct ScatterNDOperator;

impl OpOperator for ScatterNDOperator {
    fn name(&self) -> &str {
        "ScatterND"
    }

    fn infer_output_shapes(&self, ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>> {
        if ctx.input_shapes.len() < 3 {
            return Err(crate::error::CodegenError::InvalidShape(
                "ScatterND requires 3 inputs (data, indices, updates)".to_string(),
            ));
        }

        // Output has same shape as data input
        Ok(vec![ctx.input_shapes[0].clone()])
    }

    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        // Get reduction attribute (defaults to "none")
        let reduction: String = ctx.node.attr("reduction").unwrap_or("none".to_string());

        let reduction_mode = match reduction.as_str() {
            "none" => 0u32,
            "add" => 1u32,
            "mul" => 2u32,
            "max" => 3u32,
            "min" => 4u32,
            other => {
                return Err(crate::error::CodegenError::InvalidShape(format!(
                    "ScatterND: unsupported reduction mode '{}'",
                    other
                )));
            }
        };

        // Get tensor shapes
        let data_info = ctx.input_info(0)?;
        let data_shape = ctx.static_shape(&data_info.shape)?;

        let indices_info = ctx.input_info(1)?;
        let indices_shape = ctx.static_shape(&indices_info.shape)?;

        let _updates_info = ctx.input_info(2)?;
        let _updates_shape = ctx.static_shape(&_updates_info.shape)?;

        // Calculate sizes
        let data_size: usize = data_shape.iter().product();

        // indices last dimension is the rank of indices
        let indices_last_dim = *indices_shape.last().ok_or_else(|| {
            crate::error::CodegenError::InvalidShape(
                "ScatterND: indices tensor must have at least 1 dimension".to_string(),
            )
        })?;

        // Number of updates: product of all dimensions except last in indices
        let num_updates: usize = if indices_shape.len() > 1 {
            indices_shape[..indices_shape.len() - 1].iter().product()
        } else {
            1
        };

        // Calculate strides for output tensor (for index computation)
        let mut output_strides = vec![0u32; 8]; // Max 8 dimensions
        if !data_shape.is_empty() {
            let mut stride = 1usize;
            for i in (0..data_shape.len()).rev() {
                output_strides[i] = stride as u32;
                stride *= data_shape[i];
            }
        }

        // Configure workgroup size
        let workgroup_size: u32 = 256;

        // Prepare shader definitions
        let mut shader_defs = HashMap::new();
        shader_defs.insert(
            "WORKGROUP_SIZE".to_string(),
            ShaderDefValue::UInt(workgroup_size),
        );

        // Compile shader
        let shader_index = ctx.compile_shader(
            "scatternd",
            include_str!("../../shaders/indexing/scatternd.wgsl"),
            shader_defs,
        )?;

        // Create two dispatch steps: one to copy data, one to scatter updates
        let mut steps = Vec::new();

        // Step 1: Copy data to output
        let copy_workgroups = (data_size as u32 + workgroup_size - 1) / workgroup_size;

        // Unified immediate structure: use indices_last_dim=0 to signal copy mode
        let mut copy_immediates_data = Vec::new();
        copy_immediates_data.extend_from_slice(&(data_size as u32).to_le_bytes());
        copy_immediates_data.extend_from_slice(&0u32.to_le_bytes()); // indices_last_dim=0 for copy mode
        copy_immediates_data.extend_from_slice(&0u32.to_le_bytes()); // reduction (unused)
        copy_immediates_data.extend_from_slice(&0u32.to_le_bytes()); // padding
        // Add 8 zero strides (unused in copy mode)
        for _ in 0..8 {
            copy_immediates_data.extend_from_slice(&0u32.to_le_bytes());
        }

        steps.push(Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0), // data
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(1), // indices (unused in copy pass)
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(2), // updates (unused in copy pass)
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(0), // output
                    read_only: false,
                },
            ],
            workgroups: [copy_workgroups, 1, 1],
            immediates: Some(copy_immediates_data),
        });

        // Step 2: Scatter updates
        let scatter_workgroups = (num_updates as u32 + workgroup_size - 1) / workgroup_size;

        let mut scatter_immediates_data = Vec::new();
        scatter_immediates_data.extend_from_slice(&(num_updates as u32).to_le_bytes());
        scatter_immediates_data.extend_from_slice(&(indices_last_dim as u32).to_le_bytes());
        scatter_immediates_data.extend_from_slice(&reduction_mode.to_le_bytes());
        scatter_immediates_data.extend_from_slice(&0u32.to_le_bytes()); // padding for alignment
        // Add strides (8 u32 values)
        for stride in &output_strides {
            scatter_immediates_data.extend_from_slice(&stride.to_le_bytes());
        }

        steps.push(Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0), // data (unused in scatter pass)
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(1), // indices
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(2), // updates
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(0), // output
                    read_only: false,
                },
            ],
            workgroups: [scatter_workgroups, 1, 1],
            immediates: Some(scatter_immediates_data),
        });

        Ok(steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::InferenceContext;
    use crate::plan::BufferRef;
    use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
    use std::collections::HashMap;

    fn create_scatternd_test_graph() -> Graph {
        let mut graph = Graph::new();

        // Add data input tensor
        graph.add_tensor(TensorInfo {
            name: "data".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![3, 3]), // 3x3 matrix
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add indices input tensor (I64)
        graph.add_tensor(TensorInfo {
            name: "indices".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![2, 2]), // 2 indices of rank 2
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add updates input tensor
        graph.add_tensor(TensorInfo {
            name: "updates".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![2]), // 2 update values
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add output tensor
        graph.add_tensor(TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![3, 3]), // Same as data
            kind: TensorKind::Output,
            initializer: None,
        });

        graph.inputs = vec![
            "data".to_string(),
            "indices".to_string(),
            "updates".to_string(),
        ];
        graph.outputs = vec!["output".to_string()];

        graph
    }

    #[test]
    fn test_scatternd_kernel_infer_output_shapes() {
        let graph = create_scatternd_test_graph();
        let mut node = Node::new("ScatterND");
        node.inputs = vec![
            "data".to_string(),
            "indices".to_string(),
            "updates".to_string(),
        ];
        node.outputs = vec!["output".to_string()];

        let input_shapes = vec![
            TensorShape::Static(vec![3, 3]),
            TensorShape::Static(vec![2, 2]),
            TensorShape::Static(vec![2]),
        ];
        let input_values = vec![None, None, None];

        let ctx = InferenceContext::new(&node, &graph, input_shapes, input_values);
        let output_shapes = ScatterNDOperator
            .infer_output_shapes(&ctx)
            .expect("Shape inference should succeed");

        // Output shape should match data shape
        assert_eq!(output_shapes.len(), 1);
        assert_eq!(output_shapes[0], TensorShape::Static(vec![3, 3]));
    }

    #[test]
    fn test_scatternd_kernel_plan() {
        let graph = create_scatternd_test_graph();
        let mut node = Node::new("ScatterND");
        node.inputs = vec![
            "data".to_string(),
            "indices".to_string(),
            "updates".to_string(),
        ];
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

        let steps = ScatterNDOperator
            .plan(&mut ctx)
            .expect("Planning should succeed");

        // Verify we got two dispatch steps (copy + scatter)
        assert_eq!(steps.len(), 2);

        // Both should be dispatch steps
        for step in &steps {
            match step {
                Step::Dispatch {
                    shader_index,
                    bindings,
                    workgroups: _,
                    immediates,
                } => {
                    // Verify shader was compiled
                    assert_eq!(*shader_index, 0);

                    // Verify bindings: 3 inputs + 1 output
                    assert_eq!(bindings.len(), 4);
                    assert_eq!(bindings[0].buffer, BufferRef::Tensor(0)); // data
                    assert_eq!(bindings[1].buffer, BufferRef::Tensor(1)); // indices
                    assert_eq!(bindings[2].buffer, BufferRef::Tensor(2)); // updates
                    assert_eq!(bindings[3].buffer, BufferRef::Tensor(3)); // output
                    assert!(!bindings[3].read_only);

                    // Verify immediates exist
                    assert!(immediates.is_some());
                }
                _ => panic!("Expected Dispatch step"),
            }
        }
    }
}
