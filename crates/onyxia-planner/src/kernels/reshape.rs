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
        _node: &Node,
        input_shapes: &[TensorShape],
    ) -> Result<Vec<TensorShape>> {
        // Reshape has 2 inputs: data (input 0) and shape (input 1)
        // Shape input should be a constant tensor or have static shape
        if input_shapes.is_empty() {
            return Err(crate::error::CodegenError::InvalidShape(
                "Reshape requires at least one input".to_string(),
            ));
        }

        // Get the shape tensor to determine output shape
        if input_shapes.len() < 2 {
            return Err(crate::error::CodegenError::InvalidShape(
                "Reshape requires 2 inputs: data and shape".to_string(),
            ));
        }

        // For now, we need shape inference to be handled elsewhere
        // The output shape should already be inferred by the shape inference pass
        // Here we just validate and return Unknown if we can't determine it

        // Try to get the shape from the node's output tensor info if available
        // This is set during the global shape inference pass
        Ok(vec![TensorShape::Unknown])
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
        let node = Node::new("Reshape");
        let input_shapes = vec![
            TensorShape::Static(vec![2, 3]),
            TensorShape::Static(vec![1]), // shape input
        ];

        let output_shapes = kernel
            .infer_output_shapes(&node, &input_shapes)
            .expect("Shape inference should succeed");

        assert_eq!(output_shapes.len(), 1);
        // Shape inference returns Unknown - actual shape is determined by global pass
        assert_eq!(output_shapes[0], TensorShape::Unknown);
    }
}
