//! ReduceSumKernel implementation for sum reduction along axes.

use crate::error::Result;
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::{Dimension, TensorShape};
use std::collections::HashMap;

/// Kernel for sum-reducing a tensor along specified axes (ONNX ReduceSum operator).
///
/// Reduces the input tensor by summing along the specified axes.
/// Supports `keepdims` attribute to retain reduced dimensions as size 1.
pub struct ReduceSumKernel;

impl OpKernel for ReduceSumKernel {
    fn name(&self) -> &str {
        "ReduceSum"
    }

    fn infer_output_shapes(
        &self,
        node: &onyxia_onnx::Node,
        input_shapes: &[TensorShape],
        dynamic_dimensions: &HashMap<String, usize>,
    ) -> Result<Vec<TensorShape>> {
        if input_shapes.is_empty() {
            return Err(crate::error::CodegenError::InvalidShape(
                "ReduceSum requires at least one input".to_string(),
            ));
        }

        // Get input shape
        let input_dims = match &input_shapes[0] {
            TensorShape::Static(dims) => dims.clone(),
            TensorShape::Dynamic(dims) => {
                let mut resolved = Vec::new();
                for dim in dims {
                    let resolved_dim = match dim {
                        Dimension::Static(s) => *s,
                        Dimension::Named(name) => dynamic_dimensions
                            .get(name)
                            .copied()
                            .ok_or_else(|| {
                                crate::error::CodegenError::InvalidShape(format!(
                                    "Cannot resolve dynamic dimension '{}'",
                                    name
                                ))
                            })?,
                    };
                    resolved.push(resolved_dim);
                }
                resolved
            }
            TensorShape::Unknown => return Ok(vec![TensorShape::Unknown]),
            TensorShape::Absent => {
                return Err(crate::error::CodegenError::InvalidShape(
                    "ReduceSum input is absent".to_string(),
                ))
            }
        };

        // Get axes to reduce over
        // For ONNX opset 13+, axes can be:
        // - From second input (optional, for opset 18+)
        // - From "axes" attribute (for older opsets)
        // - Empty means reduce all axes
        let axes: Vec<i64> = if input_shapes.len() > 1 {
            // Axes from second input - for now, we don't support this at plan time
            // (would need to read initializer data)
            return Err(crate::error::CodegenError::UnsupportedOp(
                "ReduceSum with axes as second input not yet supported".to_string(),
            ));
        } else {
            // Try to get axes from attribute
            node.attr("axes").unwrap_or_else(|_| {
                // Default: reduce all axes
                (0..input_dims.len() as i64).collect()
            })
        };

        // Get keepdims (defaults to 1 in ONNX)
        let keepdims: i64 = node.attr("keepdims").unwrap_or(1);

        // Compute output shape
        let mut output_dims = Vec::new();
        for (i, &dim) in input_dims.iter().enumerate() {
            if axes.contains(&(i as i64)) || axes.contains(&(i as i64 - input_dims.len() as i64)) {
                // This axis is reduced
                if keepdims == 1 {
                    output_dims.push(1);
                }
                // If keepdims == 0, don't include this dimension
            } else {
                // This axis is preserved
                output_dims.push(dim);
            }
        }

        // If all axes reduced and keepdims=0, output is a scalar (shape [])
        if output_dims.is_empty() {
            output_dims.push(1); // Represent scalar as [1]
        }

        Ok(vec![TensorShape::Static(output_dims)])
    }

    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        // For MVP, only support single-axis reduction
        let axes: Vec<i64> = ctx.node.attr("axes").unwrap_or_else(|_| vec![1]);

        if axes.len() != 1 {
            return Err(crate::error::CodegenError::UnsupportedOp(format!(
                "ReduceSum currently only supports single-axis reduction, got {} axes",
                axes.len()
            )));
        }

        let axis = axes[0];

        // Get input and output shapes
        let input_info = ctx.input_info(0)?;
        let input_shape = ctx.resolve_shape(&input_info.shape)?;

        let output_info = ctx.output_info(0)?;
        let output_shape = ctx.resolve_shape(&output_info.shape)?;

        // Normalize negative axis
        let rank = input_shape.len() as i64;
        let normalized_axis = if axis < 0 { rank + axis } else { axis } as usize;

        if normalized_axis >= input_shape.len() {
            return Err(crate::error::CodegenError::InvalidShape(format!(
                "Reduction axis {} out of bounds for rank {}",
                axis, rank
            )));
        }

        // Calculate dimension sizes
        let outer_size: usize = input_shape[..normalized_axis].iter().product();
        let reduce_size: usize = input_shape[normalized_axis];
        let inner_size: usize = input_shape[normalized_axis + 1..].iter().product();

        let input_size: usize = input_shape.iter().product();
        let output_size: usize = output_shape.iter().product();

        // Configure workgroup size
        let workgroup_size: u32 = 256;

        // Number of workgroups = number of output elements
        // Each workgroup computes one output element by summing along the reduction axis
        let num_workgroups = output_size as u32;

        // Prepare shader definitions
        let mut shader_defs = HashMap::new();
        shader_defs.insert(
            "WORKGROUP_SIZE".to_string(),
            ShaderDefValue::UInt(workgroup_size),
        );

        // Compile shader
        let shader_index = ctx.compile_shader(
            "reducesum",
            include_str!("../../shaders/reduction/reducesum.wgsl"),
            shader_defs,
        )?;

        // Encode immediate data (must match ImmediateConstants struct in shader)
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(input_size as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(output_size as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(reduce_size as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(outer_size as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(inner_size as u32).to_le_bytes());

        // Create dispatch step
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
    use crate::plan::BufferRef;
    use onyxia_onnx::{AttributeValue, DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
    use std::collections::HashMap;

    #[test]
    fn test_reducesum_kernel_plan() {
        let mut graph = Graph::new();

        // Input: [2, 8, 3] - reduce along axis 1
        graph.add_tensor(TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![2, 8, 3]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Output: [2, 1, 3] (keepdims=1)
        graph.add_tensor(TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![2, 1, 3]),
            kind: TensorKind::Output,
            initializer: None,
        });

        let mut node = Node::new("ReduceSum");
        node.attributes
            .insert("axes".to_string(), AttributeValue::Ints(vec![1i64]));
        node.attributes
            .insert("keepdims".to_string(), AttributeValue::Int(1i64));
        node.inputs = vec!["input".to_string()];
        node.outputs = vec!["output".to_string()];

        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = crate::kernel::PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let steps = ReduceSumKernel
            .plan(&mut ctx)
            .expect("Planning should succeed");

        // Verify we got one dispatch step
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

                // Verify bindings: 1 read-only input + 1 read-write output
                assert_eq!(bindings.len(), 2);
                assert_eq!(bindings[0].buffer, BufferRef::Tensor(0));
                assert!(bindings[0].read_only);
                assert_eq!(bindings[1].buffer, BufferRef::Tensor(1));
                assert!(!bindings[1].read_only);

                // Verify workgroup count: output has 2 * 1 * 3 = 6 elements
                assert_eq!(*workgroups, [6, 1, 1]);

                // Verify immediates are present
                assert!(immediates.is_some());
            }
            _ => panic!("Expected Dispatch step"),
        }

        // Verify shader was compiled
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].label, "reducesum");
    }

    #[test]
    fn test_reducesum_kernel_keepdims() {
        let mut node = Node::new("ReduceSum");
        node.attributes
            .insert("axes".to_string(), AttributeValue::Ints(vec![1i64]));
        node.attributes
            .insert("keepdims".to_string(), AttributeValue::Int(1i64));

        let input_shapes = vec![TensorShape::Static(vec![4, 8])];
        let dynamic_dimensions = HashMap::new();

        let output_shapes = ReduceSumKernel
            .infer_output_shapes(&node, &input_shapes, &dynamic_dimensions)
            .expect("Shape inference should succeed");

        assert_eq!(output_shapes.len(), 1);
        assert_eq!(output_shapes[0], TensorShape::Static(vec![4, 1]));
    }

    #[test]
    fn test_reducesum_kernel_no_keepdims() {
        let mut node = Node::new("ReduceSum");
        node.attributes
            .insert("axes".to_string(), AttributeValue::Ints(vec![1i64]));
        node.attributes
            .insert("keepdims".to_string(), AttributeValue::Int(0i64));

        let input_shapes = vec![TensorShape::Static(vec![4, 8])];
        let dynamic_dimensions = HashMap::new();

        let output_shapes = ReduceSumKernel
            .infer_output_shapes(&node, &input_shapes, &dynamic_dimensions)
            .expect("Shape inference should succeed");

        assert_eq!(output_shapes.len(), 1);
        assert_eq!(output_shapes[0], TensorShape::Static(vec![4]));
    }

    #[test]
    fn test_reducesum_kernel_3d() {
        let mut node = Node::new("ReduceSum");
        node.attributes
            .insert("axes".to_string(), AttributeValue::Ints(vec![1i64]));
        node.attributes
            .insert("keepdims".to_string(), AttributeValue::Int(1i64));

        let input_shapes = vec![TensorShape::Static(vec![2, 8, 3])];
        let dynamic_dimensions = HashMap::new();

        let output_shapes = ReduceSumKernel
            .infer_output_shapes(&node, &input_shapes, &dynamic_dimensions)
            .expect("Shape inference should succeed");

        assert_eq!(output_shapes.len(), 1);
        assert_eq!(output_shapes[0], TensorShape::Static(vec![2, 1, 3]));
    }
}
