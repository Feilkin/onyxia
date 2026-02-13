//! UnsqueezeKernel implementation for buffer-copy unsqueeze operations.

use crate::error::Result;
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::Step;
use onyxia_onnx::{Node, TensorShape};

/// Kernel for Unsqueeze operator (buffer copy).
///
/// Unsqueeze adds size-1 dimensions to the input tensor. Like Reshape,
/// the underlying data doesn't change - only the tensor metadata.
/// Since our runtime allocates separate buffers per tensor, we emit a CopyBuffer step.
///
/// Future optimization: buffer aliasing to avoid copies entirely.
pub struct UnsqueezeKernel;

impl OpKernel for UnsqueezeKernel {
    fn name(&self) -> &str {
        "Unsqueeze"
    }

    fn infer_output_shapes(
        &self,
        graph: &onyxia_onnx::Graph,
        node: &Node,
        input_shapes: &[TensorShape],
    ) -> Result<Vec<TensorShape>> {
        // Unsqueeze has 1 or 2 inputs depending on opset:
        // - Opset < 13: 1 input (data) + axes attribute
        // - Opset >= 13: 2 inputs (data, axes tensor)
        if input_shapes.is_empty() {
            return Err(crate::error::CodegenError::InvalidShape(
                "Unsqueeze requires at least one input".to_string(),
            ));
        }

        // Get input data shape
        let data_shape = match &input_shapes[0] {
            TensorShape::Static(dims) => dims,
            TensorShape::Unknown => return Ok(vec![TensorShape::Unknown]),
            TensorShape::Absent => {
                return Err(crate::error::CodegenError::InvalidShape(
                    "Unsqueeze data input is absent".to_string(),
                ));
            }
            TensorShape::Dynamic(_) => {
                return Err(crate::error::CodegenError::InvalidShape(
                    "Unexpected Dynamic shape after dimension resolution".to_string(),
                ));
            }
        };

        // Get axes - try attribute first (opset < 13), then second input (opset >= 13)
        let axes: Vec<i64> = if node.has_attr("axes") {
            // Opset < 13: axes from attribute
            node.attr("axes").unwrap_or_else(|_| vec![])
        } else if input_shapes.len() >= 2 && node.inputs.len() >= 2 {
            // Opset >= 13: axes from second input (must be initializer)
            let axes_tensor_name = &node.inputs[1];
            if let Some(&axes_tensor_id) = graph.tensors.get(axes_tensor_name) {
                if let Ok(axes_info) = graph.tensor(axes_tensor_id) {
                    if let Some(initializer) = &axes_info.initializer {
                        // Parse i64 axes from raw bytes
                        parse_i64_array(initializer)
                    } else {
                        // Axes tensor exists but is not a constant - can't infer shape yet
                        return Ok(vec![TensorShape::Unknown]);
                    }
                } else {
                    // Axes tensor not in graph - can't infer shape yet
                    return Ok(vec![TensorShape::Unknown]);
                }
            } else {
                // Axes tensor not in graph - can't infer shape yet
                return Ok(vec![TensorShape::Unknown]);
            }
        } else {
            // No axes provided - can't infer shape yet
            return Ok(vec![TensorShape::Unknown]);
        };

        // Compute output shape by inserting 1s at specified axes
        let output_rank = data_shape.len() + axes.len();
        let mut output_shape = Vec::new();
        
        // Convert negative axes to positive and sort
        let mut normalized_axes: Vec<usize> = axes
            .iter()
            .map(|&axis| {
                if axis < 0 {
                    (output_rank as i64 + axis) as usize
                } else {
                    axis as usize
                }
            })
            .collect();
        normalized_axes.sort_unstable();

        let mut data_idx = 0;
        let mut axes_idx = 0;
        
        for out_idx in 0..output_rank {
            if axes_idx < normalized_axes.len() && out_idx == normalized_axes[axes_idx] {
                // Insert a 1 at this position
                output_shape.push(1);
                axes_idx += 1;
            } else {
                // Copy from input shape
                if data_idx < data_shape.len() {
                    output_shape.push(data_shape[data_idx]);
                    data_idx += 1;
                }
            }
        }

        Ok(vec![TensorShape::Static(output_shape)])
    }

    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        // Unsqueeze: copy input 0 (data) to output 0
        // Input 1 (axes) is only used at plan time, not runtime

        let input_info = ctx.input_info(0)?;
        let shape = ctx.static_shape(&input_info.shape)?;
        let element_count: usize = shape.iter().product();
        let bytes = element_count * input_info.dtype.size();

        Ok(vec![Step::CopyBuffer {
            src: ctx.input(0),
            src_offset: 0,
            dst: ctx.output(0),
            dst_offset: 0,
            size: bytes as u64,
        }])
    }
}

/// Helper function to parse i64 array from raw bytes (little-endian).
fn parse_i64_array(bytes: &[u8]) -> Vec<i64> {
    bytes
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::BufferRef;
    use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
    use std::collections::HashMap;

    fn create_unsqueeze_test_graph() -> Graph {
        let mut graph = Graph::new();

        // Add input tensor (4,)
        graph.add_tensor(TensorInfo {
            name: "data".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add axes tensor (constant)
        graph.add_tensor(TensorInfo {
            name: "axes".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![1]),
            kind: TensorKind::Weight,
            initializer: Some(vec![0u8; 8]), // axes=[0]
        });

        // Add output tensor (1, 4)
        graph.add_tensor(TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 4]),
            kind: TensorKind::Output,
            initializer: None,
        });

        graph.inputs = vec!["data".to_string()];
        graph.outputs = vec!["output".to_string()];

        graph
    }

    #[test]
    fn test_unsqueeze_kernel_plan() {
        let graph = create_unsqueeze_test_graph();
        let mut node = Node::new("Unsqueeze");
        node.inputs = vec!["data".to_string(), "axes".to_string()];
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

        let steps = UnsqueezeKernel
            .plan(&mut ctx)
            .expect("Planning should succeed");

        // Verify we got exactly one CopyBuffer step
        assert_eq!(steps.len(), 1);

        match &steps[0] {
            Step::CopyBuffer {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } => {
                assert_eq!(*src, BufferRef::Tensor(0)); // input 0 (data)
                assert_eq!(*src_offset, 0);
                assert_eq!(*dst, BufferRef::Tensor(2)); // output 0
                assert_eq!(*dst_offset, 0);
                // 4 elements * 4 bytes per F32 = 16 bytes
                assert_eq!(*size, 16);
            }
            _ => panic!("Expected CopyBuffer step"),
        }

        // No shaders should be compiled for buffer copy
        assert_eq!(shaders.len(), 0);
    }

    #[test]
    fn test_unsqueeze_kernel_different_dtypes() {
        // Test F32 (4 bytes)
        {
            let mut graph = Graph::new();
            graph.add_tensor(TensorInfo {
                name: "data".to_string(),
                dtype: DataType::F32,
                shape: TensorShape::Static(vec![8]),
                kind: TensorKind::Input,
                initializer: None,
            });
            graph.add_tensor(TensorInfo {
                name: "axes".to_string(),
                dtype: DataType::I64,
                shape: TensorShape::Static(vec![1]),
                kind: TensorKind::Weight,
                initializer: Some(vec![0u8; 8]),
            });
            graph.add_tensor(TensorInfo {
                name: "output".to_string(),
                dtype: DataType::F32,
                shape: TensorShape::Static(vec![1, 8]),
                kind: TensorKind::Output,
                initializer: None,
            });

            let node = Node::new("Unsqueeze");
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

            let steps = UnsqueezeKernel
                .plan(&mut ctx)
                .expect("Planning should succeed");
            match &steps[0] {
                Step::CopyBuffer { size, .. } => {
                    assert_eq!(*size, 32); // 8 * 4 bytes
                }
                _ => panic!("Expected CopyBuffer"),
            }
        }

        // Test I64 (8 bytes)
        {
            let mut graph = Graph::new();
            graph.add_tensor(TensorInfo {
                name: "data".to_string(),
                dtype: DataType::I64,
                shape: TensorShape::Static(vec![3]),
                kind: TensorKind::Input,
                initializer: None,
            });
            graph.add_tensor(TensorInfo {
                name: "axes".to_string(),
                dtype: DataType::I64,
                shape: TensorShape::Static(vec![1]),
                kind: TensorKind::Weight,
                initializer: Some(vec![0u8; 8]),
            });
            graph.add_tensor(TensorInfo {
                name: "output".to_string(),
                dtype: DataType::I64,
                shape: TensorShape::Static(vec![1, 3]),
                kind: TensorKind::Output,
                initializer: None,
            });

            let node = Node::new("Unsqueeze");
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

            let steps = UnsqueezeKernel
                .plan(&mut ctx)
                .expect("Planning should succeed");
            match &steps[0] {
                Step::CopyBuffer { size, .. } => {
                    assert_eq!(*size, 24); // 3 * 8 bytes
                }
                _ => panic!("Expected CopyBuffer"),
            }
        }
    }

    #[test]
    fn test_unsqueeze_kernel_shape_inference() {
        let kernel = UnsqueezeKernel;
        let graph = onyxia_onnx::Graph::new();
        
        // Test with axes from attribute (opset < 13 pattern)
        let mut node = Node::new("Unsqueeze");
        node.attributes.insert(
            "axes".to_string(),
            onyxia_onnx::AttributeValue::Ints(vec![0i64, 2i64]),
        );
        
        let input_shapes = vec![TensorShape::Static(vec![4])];

        let output_shapes = kernel
            .infer_output_shapes(&graph, &node, &input_shapes)
            .expect("Shape inference should succeed");

        assert_eq!(output_shapes.len(), 1);
        // Input [4] with axes [0, 2] should produce [1, 4, 1]
        assert_eq!(output_shapes[0], TensorShape::Static(vec![1, 4, 1]));
    }

    #[test]
    fn test_unsqueeze_kernel_shape_inference_with_initializer() {
        // Test with axes from initializer (opset >= 13 pattern)
        let mut graph = onyxia_onnx::Graph::new();
        
        // Add data tensor
        graph.add_tensor(TensorInfo {
            name: "data".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![3, 4]),
            kind: TensorKind::Input,
            initializer: None,
        });
        
        // Add axes tensor as constant/initializer
        let axes_data: Vec<u8> = vec![1i64]
            .into_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        graph.add_tensor(TensorInfo {
            name: "axes".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![1]),
            kind: TensorKind::Weight,
            initializer: Some(axes_data),
        });
        
        let mut node = Node::new("Unsqueeze");
        node.inputs = vec!["data".to_string(), "axes".to_string()];
        
        let kernel = UnsqueezeKernel;
        let input_shapes = vec![
            TensorShape::Static(vec![3, 4]),
            TensorShape::Static(vec![1]),
        ];

        let output_shapes = kernel
            .infer_output_shapes(&graph, &node, &input_shapes)
            .expect("Shape inference should succeed");

        assert_eq!(output_shapes.len(), 1);
        // Input [3, 4] with axes [1] should produce [3, 1, 4]
        assert_eq!(output_shapes[0], TensorShape::Static(vec![3, 1, 4]));
    }

    #[test]
    fn test_unsqueeze_kernel_single_input() {
        // Test older opset with only 1 input (axes in attributes)
        let mut graph = Graph::new();
        graph.add_tensor(TensorInfo {
            name: "data".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![2, 3]),
            kind: TensorKind::Input,
            initializer: None,
        });
        graph.add_tensor(TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 2, 3]),
            kind: TensorKind::Output,
            initializer: None,
        });

        let mut node = Node::new("Unsqueeze");
        node.inputs = vec!["data".to_string()];
        node.outputs = vec!["output".to_string()];

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

        let steps = UnsqueezeKernel
            .plan(&mut ctx)
            .expect("Planning should succeed");

        assert_eq!(steps.len(), 1);
        match &steps[0] {
            Step::CopyBuffer {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } => {
                assert_eq!(*src, BufferRef::Tensor(0));
                assert_eq!(*src_offset, 0);
                assert_eq!(*dst, BufferRef::Tensor(1));
                assert_eq!(*dst_offset, 0);
                // 6 elements * 4 bytes = 24 bytes
                assert_eq!(*size, 24);
            }
            _ => panic!("Expected CopyBuffer"),
        }
    }
}
