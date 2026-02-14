//! ExpandOperator implementation for tensor broadcasting.

use crate::error::{CodegenError, Result};
use crate::inference::{InferenceContext, TensorValue};
use crate::operator::{Operator, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::TensorShape;
use std::collections::HashMap;

/// Operator for Expand operator (broadcasts input to target shape).
///
/// Expand replicates the input tensor along dimensions of size 1 to match
/// a specified target shape. Broadcasting follows ONNX semantics:
/// - Dimensions of size 1 can be expanded to larger sizes
/// - New leading dimensions can be added
/// - Target shape must be compatible with input shape
pub struct ExpandOperator;

impl Operator for ExpandOperator {
    fn name(&self) -> &str {
        "Expand"
    }

    fn infer_output_shapes(&self, ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>> {
        // Expand has 2 inputs: data (input 0) and shape (input 1)
        if ctx.input_shapes.len() < 2 {
            return Err(CodegenError::InvalidShape(
                "Expand requires 2 inputs: data and shape".to_string(),
            ));
        }

        // Try to get the target shape from the second input value first.
        // The output shape is determined by the target shape, not the input data shape.
        let Some(target_shape_val) = ctx.input_value(1)? else {
            // Shape is not a constant - we can't infer the output shape
            return Ok(vec![TensorShape::Unknown]);
        };

        // Parse i64 shape from the value
        let target_shape = match target_shape_val {
            TensorValue::I64(v) => v.as_slice(),
            _ => {
                return Err(CodegenError::InvalidShape(
                    "Expand shape input must be I64".to_string(),
                ));
            }
        };

        // Convert target shape to output dimensions and validate
        let mut output_dims = Vec::new();
        for (i, &target_dim) in target_shape.iter().enumerate() {
            if target_dim <= 0 {
                return Err(CodegenError::InvalidShape(format!(
                    "Expand target shape dimension {} is invalid: {}",
                    i, target_dim
                )));
            }
            output_dims.push(target_dim as usize);
        }

        // If the input data shape is known (Static), validate compatibility
        if let TensorShape::Static(data_shape) = &ctx.input_shapes[0] {
            let input_rank = data_shape.len();
            let output_rank = output_dims.len();

            if output_rank < input_rank {
                return Err(CodegenError::InvalidShape(format!(
                    "Expand target shape rank {} is less than input rank {}",
                    output_rank, input_rank
                )));
            }

            // Align shapes from the right and validate compatibility
            let offset = output_rank - input_rank;
            for i in 0..input_rank {
                let input_dim = data_shape[i];
                let output_dim = output_dims[offset + i];
                if input_dim != 1 && input_dim != output_dim {
                    return Err(CodegenError::InvalidShape(format!(
                        "Expand: incompatible dimensions at position {}: input={}, target={}",
                        offset + i,
                        input_dim,
                        output_dim
                    )));
                }
            }
        }
        // If input shape is Unknown or Absent, we can still return the target shape
        // Runtime validation will ensure compatibility

        Ok(vec![TensorShape::Static(output_dims)])
    }

    fn try_fold(&self, ctx: &InferenceContext<'_>) -> Result<Vec<Option<TensorValue>>> {
        // Try to constant-fold if data and shape inputs are known
        let Some(data_val) = ctx.input_value(0)? else {
            return Ok(vec![None]);
        };
        let Some(shape_val) = ctx.input_value(1)? else {
            return Ok(vec![None]);
        };

        // Get input shape
        let input_shape = match &ctx.input_shapes[0] {
            TensorShape::Static(dims) => dims,
            _ => return Ok(vec![None]),
        };

        // Parse target shape from the second input
        let target_shape = match shape_val {
            TensorValue::I64(v) => v.as_slice(),
            _ => return Ok(vec![None]),
        };

        let output_dims: Vec<usize> = target_shape.iter().map(|&d| d as usize).collect();

        // Helper function to expand (broadcast) values
        fn expand_values<T: Clone>(
            values: &[T],
            input_shape: &[usize],
            output_shape: &[usize],
        ) -> Vec<T> {
            let input_rank = input_shape.len();
            let output_rank = output_shape.len();
            let num_elements: usize = output_shape.iter().product();

            // Align shapes from the right (broadcasting rules)
            let offset = output_rank - input_rank;

            // Compute strides
            let mut input_strides = vec![1; input_rank];
            let mut output_strides = vec![1; output_rank];
            for i in (0..input_rank - 1).rev() {
                input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
            }
            for i in (0..output_rank - 1).rev() {
                output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
            }

            let mut result = Vec::with_capacity(num_elements);

            // For each element in the output
            for out_idx in 0..num_elements {
                // Compute multi-dimensional index in output
                let mut out_coords = vec![0; output_rank];
                let mut temp = out_idx;
                for i in 0..output_rank {
                    out_coords[i] = temp / output_strides[i];
                    temp %= output_strides[i];
                }

                // Map to input coordinates (with broadcasting)
                let mut in_idx = 0;
                for i in 0..input_rank {
                    let out_dim_idx = offset + i;
                    let coord = if input_shape[i] == 1 {
                        0 // broadcast this dimension
                    } else {
                        out_coords[out_dim_idx]
                    };
                    in_idx += coord * input_strides[i];
                }

                result.push(values[in_idx].clone());
            }

            result
        }

        // Apply expand based on value type
        let result = match data_val {
            TensorValue::F32(vals) => {
                TensorValue::F32(expand_values(vals, input_shape, &output_dims))
            }
            TensorValue::I64(vals) => {
                TensorValue::I64(expand_values(vals, input_shape, &output_dims))
            }
            TensorValue::I32(vals) => {
                TensorValue::I32(expand_values(vals, input_shape, &output_dims))
            }
            TensorValue::Bool(vals) => {
                TensorValue::Bool(expand_values(vals, input_shape, &output_dims))
            }
            TensorValue::U8(vals) => {
                TensorValue::U8(expand_values(vals, input_shape, &output_dims))
            }
        };

        Ok(vec![Some(result)])
    }

    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        // Get input and output shapes
        let input_info = ctx.input_info(0)?;
        let input_shape = ctx.static_shape(&input_info.shape)?;
        let input_rank = input_shape.len();

        let output_info = ctx.output_info(0)?;
        let output_shape = ctx.static_shape(&output_info.shape)?;
        let output_rank = output_shape.len();
        let output_size: usize = output_shape.iter().product();

        if output_rank > 6 {
            return Err(CodegenError::InvalidShape(format!(
                "Expand: output rank {} exceeds maximum of 6",
                output_rank
            )));
        }

        // Calculate strides for input and output (row-major order)
        let mut input_strides = vec![0u32; 6];
        let mut output_strides = vec![0u32; 6];
        let mut input_shape_padded = vec![1u32; 6];
        let mut output_shape_padded = vec![1u32; 6];

        // Fill output strides and shape
        let mut stride = 1usize;
        for i in (0..output_rank).rev() {
            output_strides[i] = stride as u32;
            output_shape_padded[i] = output_shape[i] as u32;
            stride *= output_shape[i];
        }

        // Fill input strides and shape (aligned from the right)
        let offset = output_rank - input_rank;
        stride = 1;
        for i in (0..input_rank).rev() {
            input_strides[offset + i] = stride as u32;
            input_shape_padded[offset + i] = input_shape[i] as u32;
            stride *= input_shape[i];
        }

        // Configure workgroup size
        let workgroup_size: u32 = 256;
        let num_workgroups = (output_size as u32).div_ceil(workgroup_size);

        // Prepare shader definitions
        let mut shader_defs = HashMap::new();
        shader_defs.insert(
            "WORKGROUP_SIZE".to_string(),
            ShaderDefValue::UInt(workgroup_size),
        );

        // Compile shader
        let shader_index = ctx.compile_shader(
            "expand",
            include_str!("../../shaders/indexing/expand.wgsl"),
            shader_defs,
        )?;

        // Encode immediate data (must match ImmediateConstants struct in shader)
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(output_rank as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(output_size as u32).to_le_bytes());

        // Add input strides (6 u32 values)
        for &stride in &input_strides {
            immediates_data.extend_from_slice(&stride.to_le_bytes());
        }

        // Add output strides (6 u32 values)
        for &stride in &output_strides {
            immediates_data.extend_from_slice(&stride.to_le_bytes());
        }

        // Add input shape (6 u32 values)
        for &dim in &input_shape_padded {
            immediates_data.extend_from_slice(&dim.to_le_bytes());
        }

        // Add output shape (6 u32 values)
        for &dim in &output_shape_padded {
            immediates_data.extend_from_slice(&dim.to_le_bytes());
        }

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
    use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind};
    use std::collections::HashMap;

    fn create_expand_test_graph(
        input_shape: Vec<usize>,
        target_shape: Vec<i64>,
        output_shape: Vec<usize>,
    ) -> Graph {
        let mut graph = Graph::new();

        // Add input tensor
        graph.add_tensor(TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(input_shape),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add shape tensor as initializer (constant)
        let mut shape_data = Vec::new();
        for &dim in &target_shape {
            shape_data.extend_from_slice(&dim.to_le_bytes());
        }
        graph.add_tensor(TensorInfo {
            name: "shape".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![target_shape.len()]),
            kind: TensorKind::Weight,
            initializer: Some(shape_data),
        });

        // Add output tensor
        graph.add_tensor(TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(output_shape),
            kind: TensorKind::Output,
            initializer: None,
        });

        graph.inputs = vec!["input".to_string()];
        graph.outputs = vec!["output".to_string()];

        graph
    }

    #[test]
    fn test_expand_kernel_infer_output_shapes() {
        // Test: expand [3, 1] to [3, 5]
        let graph = create_expand_test_graph(vec![3, 1], vec![3, 5], vec![3, 5]);
        let node = Node {
            name: "expand_node".to_string(),
            op_type: "Expand".to_string(),
            inputs: vec!["input".to_string(), "shape".to_string()],
            outputs: vec!["output".to_string()],
            attributes: HashMap::new(),
            domain: String::new(),
        };

        let input_shapes = vec![
            TensorShape::Static(vec![3, 1]),
            TensorShape::Static(vec![2]),
        ];

        // Get the shape tensor value from the graph
        let shape_tensor_info = graph.tensor(*graph.tensors.get("shape").unwrap()).unwrap();
        let shape_value = TensorValue::from_initializer(shape_tensor_info)
            .unwrap()
            .unwrap();

        let input_values = vec![None, Some(shape_value)];

        let ctx = InferenceContext::new(&node, &graph, input_shapes, input_values);

        let operator = ExpandOperator;
        let output_shapes = operator.infer_output_shapes(&ctx).unwrap();

        assert_eq!(output_shapes.len(), 1);
        if let TensorShape::Static(dims) = &output_shapes[0] {
            assert_eq!(dims, &vec![3, 5]);
        } else {
            panic!("Expected static shape");
        }
    }

    #[test]
    fn test_expand_kernel_infer_output_shapes_unknown_input() {
        // Test: when input shape is Unknown but target shape is constant,
        // we should still be able to infer the output shape
        let graph = create_expand_test_graph(vec![3, 1], vec![2, 3, 4], vec![2, 3, 4]);
        let node = Node {
            name: "expand_node".to_string(),
            op_type: "Expand".to_string(),
            inputs: vec!["input".to_string(), "shape".to_string()],
            outputs: vec!["output".to_string()],
            attributes: HashMap::new(),
            domain: String::new(),
        };

        // Input shape is Unknown (common in dynamic models like Gemma)
        let input_shapes = vec![TensorShape::Unknown, TensorShape::Static(vec![3])];

        // Get the shape tensor value from the graph
        let shape_tensor_info = graph.tensor(*graph.tensors.get("shape").unwrap()).unwrap();
        let shape_value = TensorValue::from_initializer(shape_tensor_info)
            .unwrap()
            .unwrap();

        let input_values = vec![None, Some(shape_value)];

        let ctx = InferenceContext::new(&node, &graph, input_shapes, input_values);

        let operator = ExpandOperator;
        let output_shapes = operator.infer_output_shapes(&ctx).unwrap();

        // Should infer output shape from target shape constant
        assert_eq!(output_shapes.len(), 1);
        if let TensorShape::Static(dims) = &output_shapes[0] {
            assert_eq!(dims, &vec![2, 3, 4]);
        } else {
            panic!("Expected static shape even with unknown input");
        }
    }

    #[test]
    fn test_expand_kernel_plan() {
        // Test: plan for expanding [2, 1, 3] to [2, 4, 3]
        let graph = create_expand_test_graph(vec![2, 1, 3], vec![2, 4, 3], vec![2, 4, 3]);
        let node = Node {
            name: "expand_node".to_string(),
            op_type: "Expand".to_string(),
            inputs: vec!["input".to_string(), "shape".to_string()],
            outputs: vec!["output".to_string()],
            attributes: HashMap::new(),
            domain: String::new(),
        };

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

        let operator = ExpandOperator;
        let steps = operator.plan(&mut ctx).unwrap();

        assert_eq!(steps.len(), 1);
        match &steps[0] {
            Step::Dispatch {
                bindings,
                workgroups,
                immediates,
                ..
            } => {
                assert_eq!(bindings.len(), 2);
                assert!(workgroups[0] > 0);
                assert!(immediates.is_some());
            }
            _ => panic!("Expected Dispatch step"),
        }
    }
}
