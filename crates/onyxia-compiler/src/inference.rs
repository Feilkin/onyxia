//! Shape and value inference types and utilities.
//!
//! This module defines the types and functions used for compile-time shape
//! inference and constant folding (value propagation). It enables
//! data-dependent shape inference like Reshape nodes that read their target
//! shape from upstream Shape/Gather/Concat chains.

use crate::error::{CodegenError, Result};
use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorShape};

/// A tensor value known at compile time (for value propagation).
///
/// Used to enable data-dependent shape inference: e.g., Reshape reads its
/// target shape from an upstream Concat whose values were propagated from
/// Shape and Gather.
///
/// Only small tensors are stored (shape-metadata, indices, axes).
/// Large weight tensors should NOT be wrapped in TensorValue.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorValue {
    I64(Vec<i64>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    U8(Vec<u8>),
}

impl TensorValue {
    /// Get the number of elements in this tensor value.
    pub fn len(&self) -> usize {
        match self {
            TensorValue::I64(v) => v.len(),
            TensorValue::I32(v) => v.len(),
            TensorValue::F32(v) => v.len(),
            TensorValue::Bool(v) => v.len(),
            TensorValue::U8(v) => v.len(),
        }
    }

    /// Check if this tensor value is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Try to get as i64 slice.
    pub fn as_i64(&self) -> Option<&[i64]> {
        match self {
            TensorValue::I64(v) => Some(v),
            _ => None,
        }
    }

    /// Try to get as i32 slice.
    pub fn as_i32(&self) -> Option<&[i32]> {
        match self {
            TensorValue::I32(v) => Some(v),
            _ => None,
        }
    }

    /// Try to get as f32 slice.
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            TensorValue::F32(v) => Some(v),
            _ => None,
        }
    }

    /// Cast this value to a different type.
    pub fn cast(&self, target_dtype: DataType) -> Result<TensorValue> {
        match (self, target_dtype) {
            (TensorValue::I64(v), DataType::I64) => Ok(TensorValue::I64(v.clone())),
            (TensorValue::I64(v), DataType::I32) => {
                Ok(TensorValue::I32(v.iter().map(|&x| x as i32).collect()))
            }
            (TensorValue::I64(v), DataType::F32) => {
                Ok(TensorValue::F32(v.iter().map(|&x| x as f32).collect()))
            }
            (TensorValue::I32(v), DataType::I32) => Ok(TensorValue::I32(v.clone())),
            (TensorValue::I32(v), DataType::I64) => {
                Ok(TensorValue::I64(v.iter().map(|&x| x as i64).collect()))
            }
            (TensorValue::I32(v), DataType::F32) => {
                Ok(TensorValue::F32(v.iter().map(|&x| x as f32).collect()))
            }
            (TensorValue::F32(v), DataType::F32) => Ok(TensorValue::F32(v.clone())),
            (TensorValue::F32(v), DataType::I32) => {
                Ok(TensorValue::I32(v.iter().map(|&x| x as i32).collect()))
            }
            (TensorValue::F32(v), DataType::I64) => {
                Ok(TensorValue::I64(v.iter().map(|&x| x as i64).collect()))
            }
            _ => Err(CodegenError::UnsupportedOp(format!(
                "Cast from {:?} to {:?} not supported in constant folding",
                self, target_dtype
            ))),
        }
    }

    /// Parse a TensorValue from initializer bytes.
    pub fn from_initializer(tensor_info: &TensorInfo) -> Result<Option<Self>> {
        let Some(ref bytes) = tensor_info.initializer else {
            return Ok(None);
        };

        let TensorShape::Static(ref dims) = tensor_info.shape else {
            return Ok(None);
        };

        let element_count: usize = dims.iter().product();
        if element_count == 0 {
            // Empty tensor — return empty vec of appropriate type
            return Ok(Some(match tensor_info.dtype {
                DataType::I64 => TensorValue::I64(vec![]),
                DataType::I32 => TensorValue::I32(vec![]),
                DataType::F32 => TensorValue::F32(vec![]),
                DataType::Bool => TensorValue::Bool(vec![]),
                DataType::U8 => TensorValue::U8(vec![]),
                _ => return Ok(None), // Other types not needed for shape inference
            }));
        }

        match tensor_info.dtype {
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
                Ok(Some(TensorValue::I64(values)))
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
                Ok(Some(TensorValue::I32(values)))
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
                Ok(Some(TensorValue::F32(values)))
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
                Ok(Some(TensorValue::Bool(values)))
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
                Ok(Some(TensorValue::U8(bytes.clone())))
            }
            _ => Ok(None), // Other types not needed for shape inference
        }
    }
}

/// Context provided to OpOperator during shape and value inference.
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
            Some(TensorValue::I64(v)) => Ok(Some(v)),
            Some(_) => Err(CodegenError::InvalidShape(format!(
                "Node {} ({}): input {} is not I64",
                self.node.name, self.node.op_type, n
            ))),
            None => Ok(None),
        }
    }

    /// Get the constant-folded value of the nth input as i32 slice.
    pub fn input_value_as_i32(&self, n: usize) -> Result<Option<&[i32]>> {
        match self.input_value(n)? {
            Some(TensorValue::I32(v)) => Ok(Some(v)),
            Some(_) => Err(CodegenError::InvalidShape(format!(
                "Node {} ({}): input {} is not I32",
                self.node.name, self.node.op_type, n
            ))),
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

        let value = TensorValue::from_initializer(&tensor_info)
            .unwrap()
            .unwrap();
        assert_eq!(value, TensorValue::I64(vec![1, 2]));
    }

    #[test]
    fn test_tensor_value_cast() {
        let value = TensorValue::I64(vec![1, 2, 3]);
        let casted = value.cast(DataType::F32).unwrap();
        assert_eq!(casted, TensorValue::F32(vec![1.0, 2.0, 3.0]));
    }
}
