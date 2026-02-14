//! ConcatKernel implementation for tensor concatenation.

use crate::error::Result;
use crate::inference::{InferenceContext, TensorValue};
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::Step;
use onyxia_onnx::TensorShape;

/// Kernel for concatenating multiple tensors along an axis (ONNX Concat operator).
///
/// Concatenates N input tensors end-to-end along a specified axis.
/// Currently optimized for axis=0 concatenation using buffer copies.
pub struct ConcatKernel;

impl OpKernel for ConcatKernel {
    fn name(&self) -> &str {
        "Concat"
    }

    fn infer_output_shapes(&self, ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>> {
        if ctx.input_shapes.is_empty() {
            return Err(crate::error::CodegenError::InvalidShape(
                "Concat requires at least one input".to_string(),
            ));
        }

        // Get the axis attribute (defaults to 0 if not specified)
        let axis: i64 = ctx.node.attr("axis").unwrap_or(0);

        // For now, we only support axis=0 concatenation
        if axis != 0 {
            return Err(crate::error::CodegenError::UnsupportedOp(format!(
                "Concat only supports axis=0, got axis={}",
                axis
            )));
        }

        // All inputs must have Static shapes (Phase 1 resolved Dynamic dims)
        let mut first_dims = Vec::new();
        let mut concat_dim_sum: usize = 0;

        for (i, shape) in ctx.input_shapes.iter().enumerate() {
            let dims = match shape {
                TensorShape::Static(dims) => dims,
                TensorShape::Unknown => {
                    return Ok(vec![TensorShape::Unknown]);
                }
                TensorShape::Absent => {
                    return Err(crate::error::CodegenError::InvalidShape(format!(
                        "Concat input {} is absent (optional input not provided)",
                        i
                    )));
                }
                TensorShape::Dynamic(_) => {
                    return Err(crate::error::CodegenError::InvalidShape(
                        "Unexpected Dynamic shape after dimension resolution".to_string(),
                    ));
                }
            };

            if dims.is_empty() {
                return Err(crate::error::CodegenError::InvalidShape(
                    "Concat inputs must have at least one dimension".to_string(),
                ));
            }

            // For axis=0, accumulate first dimension, verify other dimensions match
            if i == 0 {
                first_dims = dims.clone();
                concat_dim_sum = dims[0];
            } else {
                // Verify all dimensions except axis 0 match
                if dims.len() != first_dims.len() {
                    return Err(crate::error::CodegenError::InvalidShape(format!(
                        "Concat input {} has {} dimensions, expected {}",
                        i,
                        dims.len(),
                        first_dims.len()
                    )));
                }

                for (dim_idx, (d1, d2)) in first_dims.iter().zip(dims.iter()).enumerate() {
                    if dim_idx != 0 && d1 != d2 {
                        return Err(crate::error::CodegenError::InvalidShape(format!(
                            "Concat input {} dimension {} mismatch: {} vs {}",
                            i, dim_idx, d2, d1
                        )));
                    }
                }

                concat_dim_sum += dims[0];
            }
        }

        // Output shape: same as first input, but with concatenated first dimension
        let mut output_dims = first_dims;
        output_dims[0] = concat_dim_sum;

        Ok(vec![TensorShape::Static(output_dims)])
    }

    fn try_fold(&self, ctx: &InferenceContext<'_>) -> Result<Vec<Option<TensorValue>>> {
        // If all input values are known, concatenate them
        let axis: i64 = ctx.node.attr("axis").unwrap_or(0);

        // Only support axis=0 for now
        if axis != 0 {
            return Ok(vec![None]);
        }

        // Check if all inputs have values
        let mut all_values = Vec::new();
        for i in 0..ctx.input_values.len() {
            let Some(val) = ctx.input_value(i)? else {
                return Ok(vec![None]);
            };
            all_values.push(val);
        }

        if all_values.is_empty() {
            return Ok(vec![None]);
        }

        // Concatenate values based on type
        let result = match all_values[0] {
            TensorValue::I64(_) => {
                let mut result = Vec::new();
                for val in all_values {
                    let Some(slice) = val.as_i64() else {
                        return Err(crate::error::CodegenError::InvalidShape(
                            "Concat: input types mismatch".to_string(),
                        ));
                    };
                    result.extend_from_slice(slice);
                }
                TensorValue::I64(result)
            }
            TensorValue::I32(_) => {
                let mut result = Vec::new();
                for val in all_values {
                    let Some(slice) = val.as_i32() else {
                        return Err(crate::error::CodegenError::InvalidShape(
                            "Concat: input types mismatch".to_string(),
                        ));
                    };
                    result.extend_from_slice(slice);
                }
                TensorValue::I32(result)
            }
            TensorValue::F32(_) => {
                let mut result = Vec::new();
                for val in all_values {
                    let Some(slice) = val.as_f32() else {
                        return Err(crate::error::CodegenError::InvalidShape(
                            "Concat: input types mismatch".to_string(),
                        ));
                    };
                    result.extend_from_slice(slice);
                }
                TensorValue::F32(result)
            }
            _ => return Ok(vec![None]), // Other types not supported for constant folding
        };

        Ok(vec![Some(result)])
    }

    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        // Get the axis attribute
        let axis: i64 = ctx.node.attr("axis").unwrap_or(0);

        if axis != 0 {
            return Err(crate::error::CodegenError::UnsupportedOp(format!(
                "Concat only supports axis=0, got axis={}",
                axis
            )));
        }

        let num_inputs = ctx.node.inputs.len();
        if num_inputs == 0 {
            return Err(crate::error::CodegenError::InvalidShape(
                "Concat requires at least one input".to_string(),
            ));
        }

        // Generate one CopyBuffer step per input
        // Each copies from the input buffer to the correct offset in the output buffer
        let mut steps = Vec::new();
        let mut dst_offset: u64 = 0;

        for i in 0..num_inputs {
            let input_info = ctx.input_info(i)?;
            let input_shape = ctx.static_shape(&input_info.shape)?;
            let element_count: usize = input_shape.iter().product();
            let bytes = (element_count * input_info.dtype.size()) as u64;

            steps.push(Step::CopyBuffer {
                src: ctx.input(i),
                src_offset: 0,
                dst: ctx.output(0),
                dst_offset,
                size: bytes,
            });

            dst_offset += bytes;
        }

        Ok(steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::InferenceContext;
    use crate::plan::BufferRef;
    use onyxia_onnx::{AttributeValue, DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
    use std::collections::HashMap;

    #[test]
    fn test_concat_kernel_two_inputs() {
        let mut graph = Graph::new();

        // Input 1: [3] elements
        graph.add_tensor(TensorInfo {
            name: "a".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![3]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Input 2: [4] elements
        graph.add_tensor(TensorInfo {
            name: "b".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Output: [7] elements (3 + 4)
        graph.add_tensor(TensorInfo {
            name: "c".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![7]),
            kind: TensorKind::Output,
            initializer: None,
        });

        let mut node = Node::new("Concat");
        node.attributes
            .insert("axis".to_string(), AttributeValue::Int(0i64));
        node.inputs = vec!["a".to_string(), "b".to_string()];
        node.outputs = vec!["c".to_string()];

        let input_ids = vec![0, 1];
        let output_ids = vec![2];
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = crate::kernel::PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let steps = ConcatKernel
            .plan(&mut ctx)
            .expect("Planning should succeed");

        // Verify we got two CopyBuffer steps
        assert_eq!(steps.len(), 2);

        // First copy: input_a (3 elements = 12 bytes) to output offset 0
        match &steps[0] {
            Step::CopyBuffer {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } => {
                assert_eq!(*src, BufferRef::Tensor(0)); // input 0
                assert_eq!(*src_offset, 0);
                assert_eq!(*dst, BufferRef::Tensor(2)); // output 0
                assert_eq!(*dst_offset, 0);
                assert_eq!(*size, 12); // 3 * 4 bytes
            }
            _ => panic!("Expected CopyBuffer step"),
        }

        // Second copy: input_b (4 elements = 16 bytes) to output offset 12
        match &steps[1] {
            Step::CopyBuffer {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } => {
                assert_eq!(*src, BufferRef::Tensor(1)); // input 1
                assert_eq!(*src_offset, 0);
                assert_eq!(*dst, BufferRef::Tensor(2)); // output 0
                assert_eq!(*dst_offset, 12); // Offset by first input's size
                assert_eq!(*size, 16); // 4 * 4 bytes
            }
            _ => panic!("Expected CopyBuffer step"),
        }

        // No shaders should be compiled
        assert_eq!(shaders.len(), 0);
    }

    #[test]
    fn test_concat_kernel_three_inputs() {
        let mut graph = Graph::new();

        // Three inputs: [2], [3], [4]
        for (name, size) in [("a", 2), ("b", 3), ("c", 4)] {
            graph.add_tensor(TensorInfo {
                name: name.to_string(),
                dtype: DataType::F32,
                shape: TensorShape::Static(vec![size]),
                kind: TensorKind::Input,
                initializer: None,
            });
        }

        // Output: [9] elements (2 + 3 + 4)
        graph.add_tensor(TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![9]),
            kind: TensorKind::Output,
            initializer: None,
        });

        let mut node = Node::new("Concat");
        node.attributes
            .insert("axis".to_string(), AttributeValue::Int(0i64));
        node.inputs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        node.outputs = vec!["output".to_string()];

        let input_ids = vec![0, 1, 2];
        let output_ids = vec![3];
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = crate::kernel::PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let steps = ConcatKernel
            .plan(&mut ctx)
            .expect("Planning should succeed");

        // Verify we got three CopyBuffer steps with correct offsets
        assert_eq!(steps.len(), 3);

        let expected_offsets = vec![0, 8, 20]; // 0, 2*4, (2+3)*4
        let expected_sizes = vec![8, 12, 16]; // 2*4, 3*4, 4*4

        for (i, step) in steps.iter().enumerate() {
            match step {
                Step::CopyBuffer {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => {
                    assert_eq!(*src, BufferRef::Tensor(i));
                    assert_eq!(*src_offset, 0);
                    assert_eq!(*dst, BufferRef::Tensor(3)); // output
                    assert_eq!(*dst_offset, expected_offsets[i]);
                    assert_eq!(*size, expected_sizes[i]);
                }
                _ => panic!("Expected CopyBuffer step"),
            }
        }
    }

    #[test]
    fn test_concat_kernel_infer_shapes() {
        let mut node = Node::new("Concat");
        node.attributes
            .insert("axis".to_string(), AttributeValue::Int(0i64));

        let input_shapes = vec![TensorShape::Static(vec![3]), TensorShape::Static(vec![4])];
        let input_values = vec![None; input_shapes.len()];

        let graph = onyxia_onnx::Graph::new();
        let ctx = InferenceContext::new(&node, &graph, input_shapes, input_values);
        let output_shapes = ConcatKernel
            .infer_output_shapes(&ctx)
            .expect("Shape inference should succeed");

        assert_eq!(output_shapes.len(), 1);
        assert_eq!(output_shapes[0], TensorShape::Static(vec![7]));
    }

    #[test]
    fn test_concat_kernel_2d_tensors() {
        let mut node = Node::new("Concat");
        node.attributes
            .insert("axis".to_string(), AttributeValue::Int(0i64));

        // [2, 5] concat [3, 5] = [5, 5]
        let input_shapes = vec![
            TensorShape::Static(vec![2, 5]),
            TensorShape::Static(vec![3, 5]),
        ];
        let input_values = vec![None; input_shapes.len()];

        let graph = onyxia_onnx::Graph::new();
        let ctx = InferenceContext::new(&node, &graph, input_shapes, input_values);
        let output_shapes = ConcatKernel
            .infer_output_shapes(&ctx)
            .expect("Shape inference should succeed");

        assert_eq!(output_shapes.len(), 1);
        assert_eq!(output_shapes[0], TensorShape::Static(vec![5, 5]));
    }

    #[test]
    fn test_concat_kernel_dimension_mismatch() {
        let mut node = Node::new("Concat");
        node.attributes
            .insert("axis".to_string(), AttributeValue::Int(0i64));

        // [2, 5] concat [3, 7] should fail (dimension 1 doesn't match)
        let input_shapes = vec![
            TensorShape::Static(vec![2, 5]),
            TensorShape::Static(vec![3, 7]),
        ];
        let input_values = vec![None; input_shapes.len()];

        let graph = onyxia_onnx::Graph::new();
        let ctx = InferenceContext::new(&node, &graph, input_shapes, input_values);
        let result = ConcatKernel.infer_output_shapes(&ctx);

        assert!(result.is_err());
    }
}
