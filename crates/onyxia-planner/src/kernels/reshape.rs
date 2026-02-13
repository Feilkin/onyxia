//! ReshapeKernel implementation for buffer-copy reshape operations.

use crate::error::Result;
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::Step;
use onyxia_onnx::{Node, TensorShape};

/// Kernel for Reshape operator (buffer copy).
///
/// Reshape reinterprets the input buffer with a new shape. Since GPU buffers
/// are flat arrays, no data movement is needed in theory, but our runtime
/// allocates separate buffers per tensor, so we emit a CopyBuffer step.
///
/// Future optimization: buffer aliasing to avoid copies entirely.
pub struct ReshapeKernel;

impl OpKernel for ReshapeKernel {
    fn name(&self) -> &str {
        "Reshape"
    }

    fn infer_output_shapes(
        &self,
        graph: &onyxia_onnx::Graph,
        node: &Node,
        input_shapes: &[TensorShape],
    ) -> Result<Vec<TensorShape>> {
        // Reshape has 2 inputs: data (input 0) and shape (input 1)
        // Shape input should be a constant tensor or have static shape
        if input_shapes.len() < 2 {
            return Err(crate::error::CodegenError::InvalidShape(
                "Reshape requires 2 inputs: data and shape".to_string(),
            ));
        }

        // Get input data shape
        let data_shape = match &input_shapes[0] {
            TensorShape::Static(dims) => dims,
            TensorShape::Unknown => return Ok(vec![TensorShape::Unknown]),
            TensorShape::Absent => {
                return Err(crate::error::CodegenError::InvalidShape(
                    "Reshape data input is absent".to_string(),
                ));
            }
            TensorShape::Dynamic(_) => {
                return Err(crate::error::CodegenError::InvalidShape(
                    "Unexpected Dynamic shape after dimension resolution".to_string(),
                ));
            }
        };

        // Check if node has shape input
        if node.inputs.len() < 2 {
            // Shape not provided - can't infer
            return Ok(vec![TensorShape::Unknown]);
        }

        // Get the target shape from the second input (must be initializer)
        let shape_tensor_name = &node.inputs[1];
        let shape_tensor_id = match graph.tensors.get(shape_tensor_name) {
            Some(&id) => id,
            None => {
                // Shape tensor not found - might be computed later
                return Ok(vec![TensorShape::Unknown]);
            }
        };

        let shape_info = match graph.tensor(shape_tensor_id) {
            Ok(info) => info,
            Err(_) => return Ok(vec![TensorShape::Unknown]),
        };

        let initializer = match &shape_info.initializer {
            Some(init) => init,
            None => {
                // Shape is not a constant - we can't infer the output shape
                return Ok(vec![TensorShape::Unknown]);
            }
        };

        // Parse i64 shape from raw bytes
        let target_shape = parse_i64_array(initializer);
                    
                    // Handle -1 dimension (infer from total size)
                    let total_elements: usize = data_shape.iter().product();
                    let mut output_shape = Vec::new();
                    let mut infer_dim = None;
                    let mut known_product: i64 = 1;
                    
                    for (idx, &dim) in target_shape.iter().enumerate() {
                        if dim == -1 {
                            if infer_dim.is_some() {
                                return Err(crate::error::CodegenError::InvalidShape(
                                    "Reshape can have at most one -1 dimension".to_string(),
                                ));
                            }
                            infer_dim = Some(idx);
                            output_shape.push(0); // Placeholder
                        } else if dim == 0 {
                            // 0 means "copy from input shape"
                            if idx < data_shape.len() {
                                output_shape.push(data_shape[idx]);
                                known_product *= data_shape[idx] as i64;
                            } else {
                                return Err(crate::error::CodegenError::InvalidShape(
                                    format!("Reshape: dimension 0 at index {} out of range", idx),
                                ));
                            }
                        } else if dim > 0 {
                            output_shape.push(dim as usize);
                            known_product *= dim;
                        } else {
                            return Err(crate::error::CodegenError::InvalidShape(
                                format!("Invalid reshape dimension: {}", dim),
                            ));
                        }
                    }
                    
                    // Compute inferred dimension
                    if let Some(idx) = infer_dim {
                        let inferred = total_elements as i64 / known_product;
                        if inferred * known_product != total_elements as i64 {
                            return Err(crate::error::CodegenError::InvalidShape(
                                format!("Cannot reshape {} elements into {:?}", total_elements, target_shape),
                            ));
                        }
                        output_shape[idx] = inferred as usize;
                    }
                    
                    Ok(vec![TensorShape::Static(output_shape)])
    }

    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        // Reshape: copy input 0 (data) to output 0
        // Input 1 (shape) is only used at plan time, not runtime

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

    fn create_reshape_test_graph() -> Graph {
        let mut graph = Graph::new();

        // Add input tensor (2x3)
        graph.add_tensor(TensorInfo {
            name: "data".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![2, 3]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add shape tensor (constant)
        graph.add_tensor(TensorInfo {
            name: "shape".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![1]),
            kind: TensorKind::Weight,
            initializer: Some(vec![6u8; 8]), // [6] as i64
        });

        // Add output tensor (6)
        graph.add_tensor(TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![6]),
            kind: TensorKind::Output,
            initializer: None,
        });

        graph.inputs = vec!["data".to_string()];
        graph.outputs = vec!["output".to_string()];

        graph
    }

    #[test]
    fn test_reshape_kernel_plan() {
        let graph = create_reshape_test_graph();
        let mut node = Node::new("Reshape");
        node.inputs = vec!["data".to_string(), "shape".to_string()];
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

        let steps = ReshapeKernel
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
                // 6 elements * 4 bytes per F32 = 24 bytes
                assert_eq!(*size, 24);
            }
            _ => panic!("Expected CopyBuffer step"),
        }

        // No shaders should be compiled for buffer copy
        assert_eq!(shaders.len(), 0);
    }

    #[test]
    fn test_reshape_kernel_different_dtypes() {
        // Test F32 (4 bytes)
        {
            let mut graph = Graph::new();
            graph.add_tensor(TensorInfo {
                name: "data".to_string(),
                dtype: DataType::F32,
                shape: TensorShape::Static(vec![10]),
                kind: TensorKind::Input,
                initializer: None,
            });
            graph.add_tensor(TensorInfo {
                name: "shape".to_string(),
                dtype: DataType::I64,
                shape: TensorShape::Static(vec![2]),
                kind: TensorKind::Weight,
                initializer: Some(vec![2u8, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0]),
            });
            graph.add_tensor(TensorInfo {
                name: "output".to_string(),
                dtype: DataType::F32,
                shape: TensorShape::Static(vec![2, 5]),
                kind: TensorKind::Output,
                initializer: None,
            });

            let node = Node::new("Reshape");
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

            let steps = ReshapeKernel
                .plan(&mut ctx)
                .expect("Planning should succeed");
            match &steps[0] {
                Step::CopyBuffer { size, .. } => {
                    assert_eq!(*size, 40); // 10 * 4 bytes
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
                shape: TensorShape::Static(vec![5]),
                kind: TensorKind::Input,
                initializer: None,
            });
            graph.add_tensor(TensorInfo {
                name: "shape".to_string(),
                dtype: DataType::I64,
                shape: TensorShape::Static(vec![1]),
                kind: TensorKind::Weight,
                initializer: Some(vec![5u8; 8]),
            });
            graph.add_tensor(TensorInfo {
                name: "output".to_string(),
                dtype: DataType::I64,
                shape: TensorShape::Static(vec![5]),
                kind: TensorKind::Output,
                initializer: None,
            });

            let node = Node::new("Reshape");
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

            let steps = ReshapeKernel
                .plan(&mut ctx)
                .expect("Planning should succeed");
            match &steps[0] {
                Step::CopyBuffer { size, .. } => {
                    assert_eq!(*size, 40); // 5 * 8 bytes
                }
                _ => panic!("Expected CopyBuffer"),
            }
        }
    }

    #[test]
    fn test_reshape_kernel_shape_inference() {
        let kernel = ReshapeKernel;
        let graph = Graph::new();  // Empty test graph
        let node = Node::new("Reshape");
        let input_shapes = vec![
            TensorShape::Static(vec![2, 3]),
            TensorShape::Static(vec![1]), // shape input
        ];

        let output_shapes = kernel
            .infer_output_shapes(&graph, &node, &input_shapes)
            .expect("Shape inference should succeed");

        assert_eq!(output_shapes.len(), 1);
        // Shape inference returns Unknown - actual shape is determined by global pass
        assert_eq!(output_shapes[0], TensorShape::Unknown);
    }
}
