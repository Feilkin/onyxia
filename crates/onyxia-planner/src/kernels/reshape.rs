//! ReshapeKernel implementation for buffer-copy reshape operations.

use crate::error::Result;
use crate::inference::{InferenceContext, TensorValue};
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::Step;
use onyxia_onnx::TensorShape;

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

    fn infer_output_shapes(&self, ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>> {
        // Reshape has 2 inputs: data (input 0) and shape (input 1)
        // Shape input should be a constant tensor or have static shape
        if ctx.input_shapes.len() < 2 {
            return Err(crate::error::CodegenError::InvalidShape(
                "Reshape requires 2 inputs: data and shape".to_string(),
            ));
        }

        // Get input data shape
        let data_shape = match &ctx.input_shapes[0] {
            TensorShape::Static(dims) => dims,
            TensorShape::Unknown => {
                return Ok(vec![TensorShape::Unknown]);
            }
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

        // Get the target shape from the second input value
        let Some(target_shape_val) = ctx.input_value(1)? else {
            // Shape is not a constant - we can't infer the output shape
            return Ok(vec![TensorShape::Unknown]);
        };

        // Parse i64 shape from the value
        let target_shape = match target_shape_val {
            TensorValue::I64(v) => v.as_slice(),
            _ => {
                return Err(crate::error::CodegenError::InvalidShape(
                    "Reshape shape input must be I64".to_string(),
                ));
            }
        };

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
                    return Err(crate::error::CodegenError::InvalidShape(format!(
                        "Reshape: dimension 0 at index {} out of range",
                        idx
                    )));
                }
            } else if dim > 0 {
                output_shape.push(dim as usize);
                known_product *= dim;
            } else {
                return Err(crate::error::CodegenError::InvalidShape(format!(
                    "Invalid reshape dimension: {}",
                    dim
                )));
            }
        }

        // Compute inferred dimension
        if let Some(idx) = infer_dim {
            let inferred = total_elements as i64 / known_product;
            if inferred * known_product != total_elements as i64 {
                return Err(crate::error::CodegenError::InvalidShape(format!(
                    "Cannot reshape {} elements into {:?}",
                    total_elements, target_shape
                )));
            }
            output_shape[idx] = inferred as usize;
        }

        Ok(vec![TensorShape::Static(output_shape)])
    }

    fn try_fold(&self, ctx: &InferenceContext<'_>) -> Result<Vec<Option<TensorValue>>> {
        // Reshape doesn't change values, only shape metadata
        // Return the input value unchanged if available
        let value = ctx.input_value(0)?.cloned();
        Ok(vec![value])
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::InferenceContext;
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
        let mut graph = Graph::new();

        // Add data tensor
        graph.add_tensor(TensorInfo {
            name: "data".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![2, 3]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add shape tensor (constant with target shape [3, 2])
        let shape_data: Vec<u8> = vec![3i64, 2i64]
            .into_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        graph.add_tensor(TensorInfo {
            name: "target_shape".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![2]),
            kind: TensorKind::Weight,
            initializer: Some(shape_data),
        });

        let mut node = Node::new("Reshape");
        node.inputs = vec!["data".to_string(), "target_shape".to_string()];

        let kernel = ReshapeKernel;
        let input_shapes = vec![
            TensorShape::Static(vec![2, 3]),
            TensorShape::Static(vec![2]), // shape input
        ];

        // Load constant values from initializers
        let shape_tensor_id = *graph.tensors.get("target_shape").unwrap();
        let shape_tensor = &graph.tensor_info[shape_tensor_id];
        let shape_value = TensorValue::from_initializer(shape_tensor).unwrap();
        let input_values = vec![None, shape_value];
        
        let ctx = InferenceContext::new(&node, &graph, input_shapes.clone(), input_values);
        let output_shapes = kernel
            .infer_output_shapes(&ctx)
            .expect("Shape inference should succeed");

        assert_eq!(output_shapes.len(), 1);
        assert_eq!(output_shapes[0], TensorShape::Static(vec![3, 2]));
    }

    /// Test case reproducing the Gemma model pattern where Reshape's target shape
    /// is computed by a chain of operations: Shape → Gather → Unsqueeze → Concat → Reshape
    ///
    /// This pattern is currently unsupported and returns Unknown, but could be handled
    /// with constant folding or symbolic execution in the future.
    #[test]
    fn test_reshape_with_computed_shape_gemma_pattern() {
        let mut graph = Graph::new();

        // Input: position_ids [batch_size, sequence_length]
        graph.add_tensor(TensorInfo {
            name: "input_ids".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![1, 64]),
            kind: TensorKind::Input,
            initializer: None,
        });

        graph.add_tensor(TensorInfo {
            name: "position_ids".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![1, 64]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Constant: index 1 (to extract sequence_length)
        graph.add_tensor(TensorInfo {
            name: "index_1".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![]),
            kind: TensorKind::Weight,
            initializer: Some(1i64.to_le_bytes().to_vec()),
        });

        // Constant: -1 (for reshape)
        graph.add_tensor(TensorInfo {
            name: "neg_one".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![1]),
            kind: TensorKind::Weight,
            initializer: Some((-1i64).to_le_bytes().to_vec()),
        });

        // Constant: axes for unsqueeze
        graph.add_tensor(TensorInfo {
            name: "axes_0".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![1]),
            kind: TensorKind::Weight,
            initializer: Some(0i64.to_le_bytes().to_vec()),
        });

        // Intermediate: Shape/output_0 - shape of input_ids as int64 tensor
        graph.add_tensor(TensorInfo {
            name: "shape_output".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Unknown, // Shape op output - can't infer without execution
            kind: TensorKind::Intermediate,
            initializer: None,
        });

        // Intermediate: Gather/output_0 - extracted dimension (sequence_length)
        graph.add_tensor(TensorInfo {
            name: "gather_output".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Unknown, // Scalar value 64
            kind: TensorKind::Intermediate,
            initializer: None,
        });

        // Intermediate: Unsqueeze/output_0 - [64]
        graph.add_tensor(TensorInfo {
            name: "unsqueeze_output".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Unknown,
            kind: TensorKind::Intermediate,
            initializer: None,
        });

        // Intermediate: Concat/output_0 - [-1, 64]
        graph.add_tensor(TensorInfo {
            name: "concat_output".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Unknown,
            kind: TensorKind::Intermediate,
            initializer: None,
        });

        // Output: Reshape/output_0
        graph.add_tensor(TensorInfo {
            name: "reshape_output".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Unknown,
            kind: TensorKind::Output,
            initializer: None,
        });

        // Build the chain of nodes
        let mut shape_node = Node::new("Shape");
        shape_node.inputs = vec!["input_ids".to_string()];
        shape_node.outputs = vec!["shape_output".to_string()];

        let mut gather_node = Node::new("Gather");
        gather_node.inputs = vec!["shape_output".to_string(), "index_1".to_string()];
        gather_node.outputs = vec!["gather_output".to_string()];

        let mut unsqueeze_node = Node::new("Unsqueeze");
        unsqueeze_node.inputs = vec!["gather_output".to_string(), "axes_0".to_string()];
        unsqueeze_node.outputs = vec!["unsqueeze_output".to_string()];

        let mut concat_node = Node::new("Concat");
        concat_node
            .attributes
            .insert("axis".to_string(), onyxia_onnx::AttributeValue::Int(0));
        concat_node.inputs = vec!["neg_one".to_string(), "unsqueeze_output".to_string()];
        concat_node.outputs = vec!["concat_output".to_string()];

        let mut reshape_node = Node::new("Reshape");
        reshape_node.inputs = vec!["position_ids".to_string(), "concat_output".to_string()];
        reshape_node.outputs = vec!["reshape_output".to_string()];

        graph.nodes.push(shape_node);
        graph.nodes.push(gather_node);
        graph.nodes.push(unsqueeze_node);
        graph.nodes.push(concat_node);
        graph.nodes.push(reshape_node.clone());

        graph.inputs = vec!["input_ids".to_string(), "position_ids".to_string()];
        graph.outputs = vec!["reshape_output".to_string()];

        // Test shape inference on the Reshape node using the full shape inference pipeline
        // This will perform constant folding through Shape → Gather → Unsqueeze → Concat chain
        use crate::kernel::KernelRegistry;
        use crate::shape_inference::infer_shapes;
        
        let registry = KernelRegistry::default();
        infer_shapes(&mut graph, &registry).expect("Shape inference should succeed");
        
        // Check the final reshape output shape
        let reshape_output = graph.tensors.get("reshape_output").unwrap();
        let output_shape = &graph.tensor_info[*reshape_output].shape;

        // Expected output shape: position_ids is [1, 64] = 64 elements
        // Target shape [-1, 64] with 64 elements means: -1 = 64/64 = 1
        // So output should be [1, 64]
        //
        // With constant folding, the Shape → Gather → Unsqueeze → Concat chain
        // should evaluate to [-1, 64] at compile time.
        assert_eq!(output_shape, &TensorShape::Static(vec![1, 64]));
    }
}
