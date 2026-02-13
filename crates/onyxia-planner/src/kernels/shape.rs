//! ShapeKernel implementation for ONNX Shape nodes.

use crate::error::{CodegenError, Result};
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::Step;
use onyxia_onnx::TensorShape;

/// Kernel for ONNX Shape operator.
///
/// Shape nodes output the dimensions of their input tensor as a 1D int64 array.
/// Since all shapes are resolved at plan time, this data is known statically and
/// can be written directly via a WriteBuffer step.
///
/// Optional ONNX attributes:
/// - `start`: Starting dimension index (default: 0)
/// - `end`: Ending dimension index (default: rank of input)
pub struct ShapeKernel;

impl OpKernel for ShapeKernel {
    fn name(&self) -> &str {
        "Shape"
    }

    fn infer_output_shapes(
        &self,
        node: &onyxia_onnx::Node,
        input_shapes: &[TensorShape],
    ) -> Result<Vec<TensorShape>> {
        // Shape takes one input and produces one 1D int64 output
        if input_shapes.is_empty() {
            return Err(CodegenError::InvalidShape(
                "Shape requires one input".to_string(),
            ));
        }

        // Extract static dimensions (Phase 1 already resolved Dynamic dims)
        let resolved_dims = match &input_shapes[0] {
            TensorShape::Static(dims) => dims,
            TensorShape::Unknown => {
                return Err(CodegenError::InvalidShape(
                    "Cannot infer Shape output for unknown input shape".to_string(),
                ));
            }
            TensorShape::Absent => {
                return Err(CodegenError::InvalidShape(
                    "Cannot get shape of absent optional input".to_string(),
                ));
            }
            TensorShape::Dynamic(_) => {
                return Err(CodegenError::InvalidShape(
                    "Unexpected Dynamic shape after dimension resolution".to_string(),
                ));
            }
        };

        // Handle start/end attributes to slice the shape
        let start: i64 = node
            .attributes
            .get("start")
            .and_then(|attr| {
                if let onyxia_onnx::AttributeValue::Int(v) = attr {
                    Some(*v)
                } else {
                    None
                }
            })
            .unwrap_or(0);

        let end: i64 = node
            .attributes
            .get("end")
            .and_then(|attr| {
                if let onyxia_onnx::AttributeValue::Int(v) = attr {
                    Some(*v)
                } else {
                    None
                }
            })
            .unwrap_or(resolved_dims.len() as i64);

        // Normalize negative indices
        let rank = resolved_dims.len() as i64;
        let start = if start < 0 { rank + start } else { start };
        let end = if end < 0 { rank + end } else { end };

        // Calculate output length
        let output_len = (end - start).max(0) as usize;

        // Output is a 1D int64 tensor with length = number of dimensions (or slice thereof)
        Ok(vec![TensorShape::Static(vec![output_len])])
    }

    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        let input_info = ctx.input_info(0)?;
        let shape = ctx.static_shape(&input_info.shape)?;

        // Handle start/end attributes to slice the shape
        let start: i64 = ctx
            .node
            .attributes
            .get("start")
            .and_then(|attr| {
                if let onyxia_onnx::AttributeValue::Int(v) = attr {
                    Some(*v)
                } else {
                    None
                }
            })
            .unwrap_or(0);

        let end: i64 = ctx
            .node
            .attributes
            .get("end")
            .and_then(|attr| {
                if let onyxia_onnx::AttributeValue::Int(v) = attr {
                    Some(*v)
                } else {
                    None
                }
            })
            .unwrap_or(shape.len() as i64);

        // Normalize negative indices
        let rank = shape.len() as i64;
        let start = if start < 0 {
            (rank + start).max(0) as usize
        } else {
            start.min(rank) as usize
        };
        let end = if end < 0 {
            (rank + end).max(0) as usize
        } else {
            end.min(rank) as usize
        };

        // Slice the shape dimensions
        let shape_slice = if start <= end {
            &shape[start..end]
        } else {
            &[]
        };

        // Encode dimensions as I64 little-endian bytes
        let mut data = Vec::with_capacity(shape_slice.len() * 8);
        for &dim in shape_slice {
            data.extend_from_slice(&(dim as i64).to_le_bytes());
        }

        Ok(vec![Step::WriteBuffer {
            dst: ctx.output(0),
            data,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onyxia_onnx::{AttributeValue, DataType, Graph, Node, TensorInfo, TensorKind};
    use std::collections::HashMap;

    #[test]
    fn test_shape_kernel_static() {
        // Create a graph with a tensor of shape [3, 4, 5]
        let mut graph = Graph::new();

        let input_id = graph.add_tensor(TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![3, 4, 5]),
            kind: TensorKind::Input,
            initializer: None,
        });

        let output_id = graph.add_tensor(TensorInfo {
            name: "shape_out".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![3]), // 1D tensor with 3 elements
            kind: TensorKind::Intermediate,
            initializer: None,
        });

        let node = Node {
            name: "Shape_0".to_string(),
            op_type: "Shape".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["shape_out".to_string()],
            attributes: HashMap::new(),
            domain: String::new(),
        };

        let mut shaders = Vec::new();
        let input_ids = [input_id];
        let output_ids = [output_id];
        let dynamic_dims: HashMap<String, usize> = HashMap::new();
        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dims,
            &mut shaders,
        );

        let kernel = ShapeKernel;
        let steps = kernel.plan(&mut ctx).expect("plan should succeed");

        // Verify WriteBuffer step with correct data
        assert_eq!(
            steps.len(),
            1,
            "ShapeKernel should emit one WriteBuffer step"
        );

        if let Step::WriteBuffer { dst, data } = &steps[0] {
            // Verify it writes to the output
            assert_eq!(
                *dst,
                crate::plan::BufferRef::Tensor(output_id),
                "Should write to output buffer"
            );

            // Verify data: [3i64, 4i64, 5i64] in little-endian
            assert_eq!(data.len(), 24, "Should have 24 bytes (3 * 8)");

            let dim0 = i64::from_le_bytes(data[0..8].try_into().unwrap());
            let dim1 = i64::from_le_bytes(data[8..16].try_into().unwrap());
            let dim2 = i64::from_le_bytes(data[16..24].try_into().unwrap());

            assert_eq!(dim0, 3, "First dimension should be 3");
            assert_eq!(dim1, 4, "Second dimension should be 4");
            assert_eq!(dim2, 5, "Third dimension should be 5");
        } else {
            panic!("Expected WriteBuffer step, got {:?}", steps[0]);
        }
    }

    #[test]
    fn test_shape_kernel_with_start_end() {
        // Test Shape with start=1, end=3 to get middle dimensions
        let mut graph = Graph::new();

        let input_id = graph.add_tensor(TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![10, 20, 30, 40]),
            kind: TensorKind::Input,
            initializer: None,
        });

        let output_id = graph.add_tensor(TensorInfo {
            name: "shape_out".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![2]), // dimensions [1:3] = [20, 30]
            kind: TensorKind::Intermediate,
            initializer: None,
        });

        let mut attributes = HashMap::new();
        attributes.insert("start".to_string(), AttributeValue::Int(1));
        attributes.insert("end".to_string(), AttributeValue::Int(3));

        let node = Node {
            name: "Shape_0".to_string(),
            op_type: "Shape".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["shape_out".to_string()],
            attributes,
            domain: String::new(),
        };

        let mut shaders = Vec::new();
        let input_ids = [input_id];
        let output_ids = [output_id];
        let dynamic_dims: HashMap<String, usize> = HashMap::new();
        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dims,
            &mut shaders,
        );

        let kernel = ShapeKernel;
        let steps = kernel.plan(&mut ctx).expect("plan should succeed");

        if let Step::WriteBuffer { data, .. } = &steps[0] {
            assert_eq!(data.len(), 16, "Should have 16 bytes (2 * 8)");

            let dim0 = i64::from_le_bytes(data[0..8].try_into().unwrap());
            let dim1 = i64::from_le_bytes(data[8..16].try_into().unwrap());

            assert_eq!(dim0, 20, "Should get dimension at index 1");
            assert_eq!(dim1, 30, "Should get dimension at index 2");
        }
    }

    #[test]
    fn test_shape_kernel_inference() {
        let node = Node {
            name: "Shape_0".to_string(),
            op_type: "Shape".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["shape_out".to_string()],
            attributes: HashMap::new(),
            domain: String::new(),
        };

        let input_shapes = vec![TensorShape::Static(vec![2, 3, 4])];

        let kernel = ShapeKernel;
        let output_shapes = kernel
            .infer_output_shapes(&node, &input_shapes)
            .expect("shape inference should succeed");

        assert_eq!(output_shapes.len(), 1);
        assert_eq!(
            output_shapes[0],
            TensorShape::Static(vec![3]),
            "Output should be 1D tensor with 3 elements"
        );
    }
}
