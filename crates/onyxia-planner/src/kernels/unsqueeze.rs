//! UnsqueezeKernel implementation for buffer-copy unsqueeze operations.

use crate::error::Result;
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::Step;
use onyxia_onnx::{Node, TensorShape};
use std::collections::HashMap;

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
        _node: &Node,
        input_shapes: &[TensorShape],
        _dynamic_dimensions: &HashMap<String, usize>,
    ) -> Result<Vec<TensorShape>> {
        // Unsqueeze has 1 or 2 inputs depending on opset:
        // - Opset < 13: 1 input (data) + axes attribute
        // - Opset >= 13: 2 inputs (data, axes tensor)
        if input_shapes.is_empty() {
            return Err(crate::error::CodegenError::InvalidShape(
                "Unsqueeze requires at least one input".to_string(),
            ));
        }

        // For now, we rely on the global shape inference pass to determine output shape
        // The actual shape computation with axes is handled elsewhere
        Ok(vec![TensorShape::Unknown])
    }

    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        // Unsqueeze: copy input 0 (data) to output 0
        // Input 1 (axes) is only used at plan time, not runtime

        let input_info = ctx.input_info(0)?;
        let shape = ctx.resolve_shape(&input_info.shape)?;
        let element_count: usize = shape.iter().product();
        let bytes = element_count * input_info.dtype.size();

        Ok(vec![Step::CopyBuffer {
            src: ctx.input(0),
            dst: ctx.output(0),
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
        let dynamic_dimensions = HashMap::new();
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
            Step::CopyBuffer { src, dst, size } => {
                assert_eq!(*src, BufferRef::Tensor(0)); // input 0 (data)
                assert_eq!(*dst, BufferRef::Tensor(2)); // output 0
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
            let dynamic_dimensions = HashMap::new();
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
            let dynamic_dimensions = HashMap::new();
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
        let node = Node::new("Unsqueeze");
        let input_shapes = vec![
            TensorShape::Static(vec![4]),
            TensorShape::Static(vec![1]), // axes input
        ];
        let dynamic_dimensions = HashMap::new();

        let output_shapes = kernel
            .infer_output_shapes(&node, &input_shapes, &dynamic_dimensions)
            .expect("Shape inference should succeed");

        assert_eq!(output_shapes.len(), 1);
        // Shape inference returns Unknown - actual shape is determined by global pass
        assert_eq!(output_shapes[0], TensorShape::Unknown);
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
        let dynamic_dimensions = HashMap::new();
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
            Step::CopyBuffer { src, dst, size } => {
                assert_eq!(*src, BufferRef::Tensor(0));
                assert_eq!(*dst, BufferRef::Tensor(1));
                // 6 elements * 4 bytes = 24 bytes
                assert_eq!(*size, 24);
            }
            _ => panic!("Expected CopyBuffer"),
        }
    }
}
