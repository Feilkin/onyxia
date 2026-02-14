//! GatherKernel implementation for embedding lookup and tensor indexing.

use crate::error::Result;
use crate::inference::{InferenceContext, TensorValue};
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::TensorShape;
use std::collections::HashMap;

/// Kernel for Gather operation (ONNX Gather operator).
///
/// Performs indexed selection from a data tensor:
///   output[i][j][k] = data[indices[i][j]][k]  (for axis=0)
///
/// Commonly used for embedding lookup where:
/// - data: embedding table [vocab_size, hidden_dim]
/// - indices: token IDs [batch, seq]
/// - output: embeddings [batch, seq, hidden_dim]
pub struct GatherKernel;

impl OpKernel for GatherKernel {
    fn name(&self) -> &str {
        "Gather"
    }

    fn infer_output_shapes(&self, ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>> {
        if ctx.input_shapes.len() < 2 {
            return Err(crate::error::CodegenError::InvalidShape(
                "Gather requires 2 inputs (data and indices)".to_string(),
            ));
        }

        // Get axis (defaults to 0 per ONNX spec)
        let axis: i64 = ctx.node.attr("axis").unwrap_or(0);

        // Extract static dimensions (Phase 1 already resolved Dynamic dims)
        let data_shape = match &ctx.input_shapes[0] {
            TensorShape::Static(dims) => dims,
            TensorShape::Unknown | TensorShape::Absent => return Ok(vec![TensorShape::Unknown]),
            TensorShape::Dynamic(_) => {
                return Err(crate::error::CodegenError::InvalidShape(
                    "Unexpected Dynamic shape after dimension resolution".to_string(),
                ));
            }
        };

        let indices_shape = match &ctx.input_shapes[1] {
            TensorShape::Static(dims) => dims,
            TensorShape::Unknown | TensorShape::Absent => return Ok(vec![TensorShape::Unknown]),
            TensorShape::Dynamic(_) => {
                return Err(crate::error::CodegenError::InvalidShape(
                    "Unexpected Dynamic shape after dimension resolution".to_string(),
                ));
            }
        };

        // Normalize negative axis
        let normalized_axis = if axis < 0 {
            (data_shape.len() as i64 + axis) as usize
        } else {
            axis as usize
        };

        if normalized_axis >= data_shape.len() {
            return Err(crate::error::CodegenError::InvalidShape(format!(
                "Gather axis {} is out of bounds for data shape {:?}",
                axis, data_shape
            )));
        }

        // Output shape: data[:axis] + indices + data[axis+1:]
        let mut output_shape = Vec::new();
        output_shape.extend_from_slice(&data_shape[..normalized_axis]);
        output_shape.extend_from_slice(indices_shape);
        output_shape.extend_from_slice(&data_shape[normalized_axis + 1..]);

        Ok(vec![TensorShape::Static(output_shape)])
    }

    fn try_fold(&self, ctx: &InferenceContext<'_>) -> Result<Vec<Option<TensorValue>>> {
        // If both data and indices are known, gather at compile time
        let Some(data_val) = ctx.input_value(0)? else {
            return Ok(vec![None]);
        };
        let Some(indices_val) = ctx.input_value(1)? else {
            return Ok(vec![None]);
        };

        let axis: i64 = ctx.node.attr("axis").unwrap_or(0);

        // For axis=0 and 1D data, simple indexing
        if axis == 0 {
            let TensorShape::Static(ref data_shape) = ctx.input_shapes[0] else {
                return Ok(vec![None]);
            };

            // Only support 1D data for now (e.g., selecting scalars from a list)
            if data_shape.len() == 1 {
                let result = match (data_val, indices_val) {
                    (TensorValue::I64(data), TensorValue::I64(indices)) => {
                        let mut result = Vec::new();
                        for &idx in indices {
                            let idx_usize = idx as usize;
                            if idx_usize >= data.len() {
                                return Err(crate::error::CodegenError::InvalidShape(format!(
                                    "Gather index {} out of bounds for data length {}",
                                    idx,
                                    data.len()
                                )));
                            }
                            result.push(data[idx_usize]);
                        }
                        TensorValue::I64(result)
                    }
                    (TensorValue::I32(data), TensorValue::I64(indices)) => {
                        let mut result = Vec::new();
                        for &idx in indices {
                            let idx_usize = idx as usize;
                            if idx_usize >= data.len() {
                                return Err(crate::error::CodegenError::InvalidShape(format!(
                                    "Gather index {} out of bounds for data length {}",
                                    idx,
                                    data.len()
                                )));
                            }
                            result.push(data[idx_usize]);
                        }
                        TensorValue::I32(result)
                    }
                    _ => return Ok(vec![None]),
                };
                return Ok(vec![Some(result)]);
            }
        }

        Ok(vec![None])
    }

    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        // Get axis attribute (default to 0)
        let axis: i64 = ctx.node.attr("axis").unwrap_or(0);

        // Get data shape
        let data_info = ctx.input_info(0)?;
        let data_shape = ctx.static_shape(&data_info.shape)?;

        // Normalize negative axis
        let normalized_axis = if axis < 0 {
            (data_shape.len() as i64 + axis) as usize
        } else {
            axis as usize
        };

        // Get output shape and calculate total elements
        let output_info = ctx.output_info(0)?;
        let output_shape = ctx.static_shape(&output_info.shape)?;
        let num_elements: usize = output_shape.iter().product();

        // Calculate inner_dim: product of data.shape[axis+1:]
        let inner_dim: usize = data_shape[normalized_axis + 1..].iter().product();
        let inner_dim = if inner_dim == 0 { 1 } else { inner_dim };

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
            "gather",
            include_str!("../../shaders/indexing/gather.wgsl"),
            shader_defs,
        )?;

        // Encode immediate data (must match ImmediateConstants struct in shader)
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(num_elements as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(inner_dim as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(normalized_axis as u32).to_le_bytes());

        // Create dispatch step with bindings and immediates
        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0), // data
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(1), // indices (I64 as u32 pairs)
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
    use crate::inference::InferenceContext;
    use crate::plan::BufferRef;
    use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
    use std::collections::HashMap;

    fn create_gather_test_graph() -> Graph {
        let mut graph = Graph::new();

        // Add data input tensor (embedding table)
        graph.add_tensor(TensorInfo {
            name: "data".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4, 3]), // 4 rows, 3 columns
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add indices input tensor (I64)
        graph.add_tensor(TensorInfo {
            name: "indices".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![2]), // 2 indices
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add output tensor
        graph.add_tensor(TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![2, 3]), // 2 gathered rows of 3 elements
            kind: TensorKind::Output,
            initializer: None,
        });

        graph.inputs = vec!["data".to_string(), "indices".to_string()];
        graph.outputs = vec!["output".to_string()];

        graph
    }

    #[test]
    fn test_gather_kernel_plan_axis0() {
        let graph = create_gather_test_graph();
        let mut node = Node::new("Gather");
        node.inputs = vec!["data".to_string(), "indices".to_string()];
        node.outputs = vec!["output".to_string()];
        // axis=0 is default, no need to set

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

        let steps = GatherKernel
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

                // Verify bindings: 2 read-only inputs + 1 read-write output
                assert_eq!(bindings.len(), 3);

                assert_eq!(bindings[0].buffer, BufferRef::Tensor(0)); // data
                assert!(bindings[0].read_only);

                assert_eq!(bindings[1].buffer, BufferRef::Tensor(1)); // indices
                assert!(bindings[1].read_only);

                assert_eq!(bindings[2].buffer, BufferRef::Tensor(2)); // output
                assert!(!bindings[2].read_only);

                // Verify workgroup count: ceil(6 / 256) = 1
                // (2 indices × 3 inner_dim = 6 output elements)
                assert_eq!(*workgroups, [1, 1, 1]);

                // Verify immediates: [num_elements=6, inner_dim=3, axis=0]
                let imm = immediates.as_ref().expect("Should have immediates");
                assert_eq!(imm.len(), 12); // 3 u32 values × 4 bytes

                let num_elements = u32::from_le_bytes(imm[0..4].try_into().unwrap());
                let inner_dim = u32::from_le_bytes(imm[4..8].try_into().unwrap());
                let axis = u32::from_le_bytes(imm[8..12].try_into().unwrap());

                assert_eq!(num_elements, 6); // 2 × 3
                assert_eq!(inner_dim, 3);
                assert_eq!(axis, 0);
            }
            _ => panic!("Expected Dispatch step"),
        }

        // Verify shader was compiled
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].label, "gather");
        assert_eq!(shaders[0].entry_point, "main");
    }

    #[test]
    fn test_gather_kernel_output_shape_inference() {
        let _graph = create_gather_test_graph();
        let node = Node::new("Gather");

        let input_shapes = vec![
            TensorShape::Static(vec![4, 3]), // data
            TensorShape::Static(vec![2]),    // indices
        ];
        let input_values = vec![None; input_shapes.len()];

        let graph = onyxia_onnx::Graph::new();
        let ctx = InferenceContext::new(&node, &graph, input_shapes, input_values);
        let output_shapes = GatherKernel
            .infer_output_shapes(&ctx)
            .expect("Shape inference should succeed");

        assert_eq!(output_shapes.len(), 1);
        assert_eq!(output_shapes[0], TensorShape::Static(vec![2, 3]));
    }

    #[test]
    fn test_gather_kernel_embedding_shape() {
        // Simulate Gemma embedding: [262144, 640] data with [8, 32] token indices
        let node = Node::new("Gather");

        let input_shapes = vec![
            TensorShape::Static(vec![262144, 640]), // vocab × hidden
            TensorShape::Static(vec![8, 32]),       // batch × seq
        ];
        let input_values = vec![None; input_shapes.len()];

        let graph = onyxia_onnx::Graph::new();
        let ctx = InferenceContext::new(&node, &graph, input_shapes, input_values);
        let output_shapes = GatherKernel
            .infer_output_shapes(&ctx)
            .expect("Shape inference should succeed");

        assert_eq!(output_shapes.len(), 1);
        // Output: [batch, seq, hidden] = [8, 32, 640]
        assert_eq!(output_shapes[0], TensorShape::Static(vec![8, 32, 640]));
    }

    #[test]
    fn test_gather_kernel_negative_axis() {
        let mut node_with_axis = Node::new("Gather");
        node_with_axis
            .attributes
            .insert("axis".to_string(), onyxia_onnx::AttributeValue::Int(-2));

        let input_shapes = vec![
            TensorShape::Static(vec![4, 5, 3]), // data
            TensorShape::Static(vec![2]),       // indices
        ];
        let input_values = vec![None; input_shapes.len()];

        let graph = onyxia_onnx::Graph::new();
        let ctx = InferenceContext::new(&node_with_axis, &graph, input_shapes, input_values);
        let output_shapes = GatherKernel
            .infer_output_shapes(&ctx)
            .expect("Shape inference should succeed");

        // axis=-2 → axis=1 (for rank 3)
        // Output: data[:1] + indices + data[2:] = [4] + [2] + [3] = [4, 2, 3]
        assert_eq!(output_shapes.len(), 1);
        assert_eq!(output_shapes[0], TensorShape::Static(vec![4, 2, 3]));
    }

    #[test]
    fn test_gather_kernel_immediate_data() {
        let graph = create_gather_test_graph();
        let mut node = Node::new("Gather");
        node.inputs = vec!["data".to_string(), "indices".to_string()];
        node.outputs = vec!["output".to_string()];

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

        let steps = GatherKernel
            .plan(&mut ctx)
            .expect("Planning should succeed");

        match &steps[0] {
            Step::Dispatch { immediates, .. } => {
                let imm = immediates.as_ref().expect("Should have immediates");

                let num_elements = u32::from_le_bytes(imm[0..4].try_into().unwrap());
                let inner_dim = u32::from_le_bytes(imm[4..8].try_into().unwrap());
                let axis = u32::from_le_bytes(imm[8..12].try_into().unwrap());

                // data: [4, 3], indices: [2], axis: 0
                // output: [2, 3] → 6 elements
                // inner_dim: product of data.shape[1:] = 3
                assert_eq!(num_elements, 6);
                assert_eq!(inner_dim, 3);
                assert_eq!(axis, 0);
            }
            _ => panic!("Expected Dispatch step"),
        }
    }
}
