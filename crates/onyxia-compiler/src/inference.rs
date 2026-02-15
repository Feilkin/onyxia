//! Shape and value inference types and utilities.
//!
//! This module defines the types and functions used for compile-time shape
//! inference and constant folding (value propagation). It enables
//! data-dependent shape inference like Reshape nodes that read their target
//! shape from upstream Shape/Gather/Concat chains.

use crate::error::{CodegenError, Result};
pub use onyxia_core::types::{TensorData, TensorValue};
use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorShape};

/// Parse a TensorValue from ONNX initializer bytes.
pub fn parse_tensor_value_from_initializer(
    tensor_info: &TensorInfo,
) -> Result<Option<TensorValue>> {
    let Some(ref bytes) = tensor_info.initializer else {
        return Ok(None);
    };

    let TensorShape::Static(ref dims) = tensor_info.shape else {
        return Ok(None);
    };

    let element_count: usize = dims.iter().product();
    if element_count == 0 {
        // Empty tensor — return empty vec of appropriate type
        let data = match tensor_info.dtype {
            DataType::I64 => TensorData::I64(vec![]),
            DataType::I32 => TensorData::I32(vec![]),
            DataType::F32 => TensorData::F32(vec![]),
            DataType::Bool => TensorData::Bool(vec![]),
            DataType::U8 => TensorData::U8(vec![]),
            _ => return Ok(None), // Other types not needed for shape inference
        };
        return Ok(Some(TensorValue::new(
            data,
            dims.clone(),
            tensor_info.dtype,
        )));
    }

    let data = match tensor_info.dtype {
        DataType::I64 => {
            if bytes.len() != element_count * 8 {
                return Err(CodegenError::InvalidShape(format!(
                    "I64 initializer for {} has {} bytes, expected {}",
                    tensor_info.name,
                    bytes.len(),
                    element_count * 8
                )));
            }
            let values: Vec<i64> = bytes
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            TensorData::I64(values)
        }
        DataType::I32 => {
            if bytes.len() != element_count * 4 {
                return Err(CodegenError::InvalidShape(format!(
                    "I32 initializer for {} has {} bytes, expected {}",
                    tensor_info.name,
                    bytes.len(),
                    element_count * 4
                )));
            }
            let values: Vec<i32> = bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            TensorData::I32(values)
        }
        DataType::F32 => {
            if bytes.len() != element_count * 4 {
                return Err(CodegenError::InvalidShape(format!(
                    "F32 initializer for {} has {} bytes, expected {}",
                    tensor_info.name,
                    bytes.len(),
                    element_count * 4
                )));
            }
            let values: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            TensorData::F32(values)
        }
        DataType::Bool => {
            if bytes.len() != element_count {
                return Err(CodegenError::InvalidShape(format!(
                    "Bool initializer for {} has {} bytes, expected {}",
                    tensor_info.name,
                    bytes.len(),
                    element_count
                )));
            }
            let values: Vec<bool> = bytes.iter().map(|&b| b != 0).collect();
            TensorData::Bool(values)
        }
        DataType::U8 => {
            if bytes.len() != element_count {
                return Err(CodegenError::InvalidShape(format!(
                    "U8 initializer for {} has {} bytes, expected {}",
                    tensor_info.name,
                    bytes.len(),
                    element_count
                )));
            }
            TensorData::U8(bytes.clone())
        }
        _ => return Ok(None), // Other types not needed for shape inference
    };

    Ok(Some(TensorValue::new(
        data,
        dims.clone(),
        tensor_info.dtype,
    )))
}

/// Context provided to Operator during shape and value inference.
///
/// Gives operators read-only access to everything they need:
/// input shapes, input values, node attributes, and the graph.
pub struct InferenceContext<'a> {
    /// The ONNX node being analyzed.
    pub node: &'a Node,

    /// Shapes of input tensors.
    /// Guaranteed: no `Dynamic` variants (Phase 1 already resolved them).
    /// May contain: `Static`, `Unknown`, `Absent`.
    pub input_shapes: Vec<TensorShape>,

    /// Constant-folded values of input tensors, if known.
    /// `None` means the value is not statically known (e.g., user input).
    /// `Some(v)` means the tensor was folded to a constant by an upstream
    /// operator's `try_fold`, or is an initializer.
    pub input_values: Vec<Option<TensorValue>>,

    /// The full graph (for accessing initializer data, tensor metadata).
    pub graph: &'a Graph,
}

impl<'a> InferenceContext<'a> {
    /// Create a new InferenceContext.
    pub fn new(
        node: &'a Node,
        graph: &'a Graph,
        input_shapes: Vec<TensorShape>,
        input_values: Vec<Option<TensorValue>>,
    ) -> Self {
        Self {
            node,
            input_shapes,
            input_values,
            graph,
        }
    }

    /// Get the shape of the nth input tensor.
    pub fn input_shape(&self, n: usize) -> Result<&TensorShape> {
        self.input_shapes.get(n).ok_or_else(|| {
            CodegenError::InvalidShape(format!(
                "Node {} ({}): input {} does not exist (has {} inputs)",
                self.node.name,
                self.node.op_type,
                n,
                self.input_shapes.len()
            ))
        })
    }

    /// Get the constant-folded value of the nth input tensor, if available.
    pub fn input_value(&self, n: usize) -> Result<Option<&TensorValue>> {
        self.input_values
            .get(n)
            .map(|opt| opt.as_ref())
            .ok_or_else(|| {
                CodegenError::InvalidShape(format!(
                    "Node {} ({}): input {} does not exist (has {} inputs)",
                    self.node.name,
                    self.node.op_type,
                    n,
                    self.input_values.len()
                ))
            })
    }

    /// Get the constant-folded value of the nth input as i64 slice.
    pub fn input_value_as_i64(&self, n: usize) -> Result<Option<&[i64]>> {
        match self.input_value(n)? {
            Some(val) => Ok(val.as_i64()),
            None => Ok(None),
        }
    }

    /// Get the constant-folded value of the nth input as i32 slice.
    pub fn input_value_as_i32(&self, n: usize) -> Result<Option<&[i32]>> {
        match self.input_value(n)? {
            Some(val) => Ok(val.as_i32()),
            None => Ok(None),
        }
    }

    /// Get static dimensions from a tensor shape.
    ///
    /// Returns an error if the shape is not Static — by the planning phase,
    /// all shapes should be resolved.
    pub fn static_shape(&self, n: usize) -> Result<&[usize]> {
        match self.input_shape(n)? {
            TensorShape::Static(dims) => Ok(dims),
            TensorShape::Absent => Err(CodegenError::InvalidShape(format!(
                "Node {} ({}): input {} is Absent (optional input not provided)",
                self.node.name, self.node.op_type, n
            ))),
            TensorShape::Unknown => Err(CodegenError::InvalidShape(format!(
                "Node {} ({}): input {} shape is Unknown — shape inference failed",
                self.node.name, self.node.op_type, n
            ))),
            TensorShape::Dynamic(_) => Err(CodegenError::InvalidShape(format!(
                "Node {} ({}): input {} shape is still Dynamic — Phase 1 should have resolved it",
                self.node.name, self.node.op_type, n
            ))),
        }
    }
}

/// Compute the broadcast-result shape of two or more tensors.
///
/// Implements ONNX multidirectional broadcasting (NumPy semantics):
/// shapes are right-aligned, and each dimension pair must be equal or
/// one of them must be 1.
///
/// # Examples
///
/// ```
/// use onyxia_compiler::inference::broadcast_shapes;
///
/// assert_eq!(broadcast_shapes(&[&[2,3,4,5], &[5]]).unwrap(),      vec![2,3,4,5]);
/// assert_eq!(broadcast_shapes(&[&[1,4,5], &[2,3,1,1]]).unwrap(),  vec![2,3,4,5]);
/// ```
pub fn broadcast_shapes(shapes: &[&[usize]]) -> Result<Vec<usize>> {
    if shapes.is_empty() {
        return Ok(vec![]);
    }

    if shapes.len() == 1 {
        return Ok(shapes[0].to_vec());
    }

    // Find the maximum rank
    let max_rank = shapes.iter().map(|s| s.len()).max().unwrap();

    // Build the result shape dimension by dimension (right-aligned)
    let mut result = Vec::with_capacity(max_rank);

    for i in 0..max_rank {
        let mut dim = 1;
        for shape in shapes {
            let rank = shape.len();
            if i < max_rank - rank {
                // This shape doesn't have a dimension at this position (implicit 1)
                continue;
            }
            let shape_dim = shape[rank - (max_rank - i)];

            if dim == 1 {
                dim = shape_dim;
            } else if shape_dim != 1 && shape_dim != dim {
                return Err(CodegenError::InvalidShape(format!(
                    "Cannot broadcast shapes: dimension mismatch at position {} (expected {} or 1, got {})",
                    i, dim, shape_dim
                )));
            }
        }
        result.push(dim);
    }

    Ok(result)
}

/// Helper for elementwise operations: infer output shape via broadcasting.
pub fn infer_elementwise_broadcast(ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>> {
    // Collect all non-Absent input shapes
    let mut static_shapes = Vec::new();
    for shape in &ctx.input_shapes {
        match shape {
            TensorShape::Static(dims) => static_shapes.push(dims.as_slice()),
            TensorShape::Unknown => return Ok(vec![TensorShape::Unknown]),
            TensorShape::Absent => continue, // Skip absent (optional) inputs
            TensorShape::Dynamic(_) => {
                return Err(CodegenError::InvalidShape(
                    "Dynamic shapes should have been resolved by Phase 1".to_string(),
                ));
            }
        }
    }

    if static_shapes.is_empty() {
        return Ok(vec![TensorShape::Unknown]);
    }

    let result_dims = broadcast_shapes(&static_shapes)?;
    Ok(vec![TensorShape::Static(result_dims)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shapes_same_rank() {
        // [2, 3, 4] and [2, 3, 4] -> [2, 3, 4]
        assert_eq!(
            broadcast_shapes(&[&[2, 3, 4], &[2, 3, 4]]).unwrap(),
            vec![2, 3, 4]
        );

        // [2, 3, 4] and [2, 1, 4] -> [2, 3, 4]
        assert_eq!(
            broadcast_shapes(&[&[2, 3, 4], &[2, 1, 4]]).unwrap(),
            vec![2, 3, 4]
        );

        // [2, 3, 4] and [1, 3, 1] -> [2, 3, 4]
        assert_eq!(
            broadcast_shapes(&[&[2, 3, 4], &[1, 3, 1]]).unwrap(),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn test_broadcast_shapes_different_rank() {
        // [2, 3, 4, 5] and [5] -> [2, 3, 4, 5]
        assert_eq!(
            broadcast_shapes(&[&[2, 3, 4, 5], &[5]]).unwrap(),
            vec![2, 3, 4, 5]
        );

        // [1, 4, 5] and [2, 3, 1, 1] -> [2, 3, 4, 5]
        assert_eq!(
            broadcast_shapes(&[&[1, 4, 5], &[2, 3, 1, 1]]).unwrap(),
            vec![2, 3, 4, 5]
        );

        // [8, 1, 6, 1] and [7, 1, 5] -> [8, 7, 6, 5]
        assert_eq!(
            broadcast_shapes(&[&[8, 1, 6, 1], &[7, 1, 5]]).unwrap(),
            vec![8, 7, 6, 5]
        );
    }

    #[test]
    fn test_broadcast_shapes_scalar() {
        // [] and [2, 3] -> [2, 3] (scalar broadcasts to any shape)
        assert_eq!(broadcast_shapes(&[&[], &[2, 3]]).unwrap(), vec![2, 3]);

        // [2, 3] and [] -> [2, 3]
        assert_eq!(broadcast_shapes(&[&[2, 3], &[]]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn test_broadcast_shapes_incompatible() {
        // [3] and [4] -> error (neither is 1)
        assert!(broadcast_shapes(&[&[3], &[4]]).is_err());

        // [2, 3] and [2, 4] -> error
        assert!(broadcast_shapes(&[&[2, 3], &[2, 4]]).is_err());
    }

    #[test]
    fn test_broadcast_shapes_multiple_inputs() {
        // [2, 3, 4], [3, 4], [4] -> [2, 3, 4]
        assert_eq!(
            broadcast_shapes(&[&[2, 3, 4], &[3, 4], &[4]]).unwrap(),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn test_tensor_value_from_initializer_i64() {
        let tensor_info = TensorInfo {
            name: "test".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![2]),
            kind: onyxia_onnx::TensorKind::Weight,
            initializer: Some(vec![1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]),
        };

        let value = parse_tensor_value_from_initializer(&tensor_info)
            .unwrap()
            .unwrap();
        let expected = TensorValue::new(TensorData::I64(vec![1, 2]), vec![2], DataType::I64);
        assert_eq!(value.as_i64(), expected.as_i64());
        assert_eq!(value.shape, expected.shape);
    }

    #[test]
    fn test_tensor_value_cast() {
        let value = TensorValue::new(TensorData::I64(vec![1, 2, 3]), vec![3], DataType::I64);
        let casted = value.cast(DataType::F32).unwrap();
        let expected =
            TensorValue::new(TensorData::F32(vec![1.0, 2.0, 3.0]), vec![3], DataType::F32);
        assert_eq!(casted.as_f32(), expected.as_f32());
        assert_eq!(casted.shape, expected.shape);
    }
}
