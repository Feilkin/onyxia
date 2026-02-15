//! Core types for tensor shapes, values, and metadata.

use crate::symbolic_expr::SymbolicExpr;
use crate::{Error, Result};

// Re-export types from onyxia-onnx
pub use onyxia_onnx::{DataType, TensorKind};

/// Tensor shape with support for static, symbolic, and absent shapes.
///
/// Unlike the ONNX `TensorShape`, this version removes the `Unknown` variant
/// to force operators to return errors when shapes cannot be determined,
/// rather than silently propagating unknown shapes that crash later.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorShape {
    /// All dimensions are known at compile time.
    Static(Vec<usize>),

    /// Mix of static and symbolic dimensions (e.g., `[batch_size, 128, seq_length * 8]`).
    Symbolic(Vec<SymbolicDim>),

    /// Optional input that is absent (ONNX empty string).
    Absent,
}

impl TensorShape {
    /// Check if the shape is fully static.
    pub fn is_static(&self) -> bool {
        matches!(self, TensorShape::Static(_))
    }

    /// Check if the shape is absent.
    pub fn is_absent(&self) -> bool {
        matches!(self, TensorShape::Absent)
    }

    /// Get static dimensions if available.
    pub fn as_static(&self) -> Option<&[usize]> {
        match self {
            TensorShape::Static(dims) => Some(dims),
            _ => None,
        }
    }

    /// Number of dimensions, if known.
    pub fn ndim(&self) -> Option<usize> {
        match self {
            TensorShape::Static(dims) => Some(dims.len()),
            TensorShape::Symbolic(dims) => Some(dims.len()),
            TensorShape::Absent => None,
        }
    }

    /// Convert from ONNX TensorShape to core TensorShape.
    ///
    /// Maps ONNX `Unknown` to an error instead of silently propagating it.
    pub fn from_onnx(onnx_shape: &onyxia_onnx::TensorShape) -> Result<Self> {
        match onnx_shape {
            onyxia_onnx::TensorShape::Static(dims) => Ok(TensorShape::Static(dims.clone())),
            onyxia_onnx::TensorShape::Dynamic(dims) => {
                let symbolic_dims = dims
                    .iter()
                    .map(|dim| match dim {
                        onyxia_onnx::Dimension::Static(n) => SymbolicDim::Fixed(*n),
                        onyxia_onnx::Dimension::Named(name) => {
                            // Try to parse as expression, fall back to variable
                            match crate::symbolic_expr::parse_expr(name) {
                                Ok(expr) => SymbolicDim::Expr(expr),
                                Err(_) => {
                                    // If parsing fails, treat as a simple variable
                                    SymbolicDim::Expr(SymbolicExpr::Variable(name.clone()))
                                }
                            }
                        }
                    })
                    .collect();
                Ok(TensorShape::Symbolic(symbolic_dims))
            }
            onyxia_onnx::TensorShape::Absent => Ok(TensorShape::Absent),
            onyxia_onnx::TensorShape::Unknown => {
                Err(Error::ShapeInference("Shape is unknown".to_string()))
            }
        }
    }
}

/// A single dimension in a symbolic tensor shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolicDim {
    /// Compile-time constant dimension.
    Fixed(usize),

    /// Symbolic expression (e.g., `sequence_length * num_attention_heads`).
    Expr(SymbolicExpr),
}

impl SymbolicDim {
    /// Check if this dimension is fixed.
    pub fn is_fixed(&self) -> bool {
        matches!(self, SymbolicDim::Fixed(_))
    }

    /// Get the fixed value if available.
    pub fn as_fixed(&self) -> Option<usize> {
        match self {
            SymbolicDim::Fixed(n) => Some(*n),
            _ => None,
        }
    }
}

/// A tensor value known at compile time (for constant folding).
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

    /// Try to get as bool slice.
    pub fn as_bool(&self) -> Option<&[bool]> {
        match self {
            TensorValue::Bool(v) => Some(v),
            _ => None,
        }
    }

    /// Try to get as u8 slice.
    pub fn as_u8(&self) -> Option<&[u8]> {
        match self {
            TensorValue::U8(v) => Some(v),
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
            (TensorValue::Bool(v), DataType::Bool) => Ok(TensorValue::Bool(v.clone())),
            (TensorValue::U8(v), DataType::U8) => Ok(TensorValue::U8(v.clone())),
            _ => Err(Error::ConstantFolding(format!(
                "Cast from {:?} to {:?} not supported in constant folding",
                self, target_dtype
            ))),
        }
    }

    /// Parse a TensorValue from initializer bytes.
    ///
    /// # Arguments
    /// * `bytes` - Raw tensor data in native endian format
    /// * `dtype` - Data type of the tensor
    /// * `shape` - Shape of the tensor (must be static)
    pub fn from_bytes(bytes: &[u8], dtype: DataType, shape: &[usize]) -> Result<Self> {
        let numel: usize = shape.iter().product();

        match dtype {
            DataType::I64 => {
                if bytes.len() != numel * 8 {
                    return Err(Error::ConstantFolding(format!(
                        "Invalid byte length for I64 tensor: expected {}, got {}",
                        numel * 8,
                        bytes.len()
                    )));
                }
                let values = bytes
                    .chunks_exact(8)
                    .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                Ok(TensorValue::I64(values))
            }
            DataType::I32 => {
                if bytes.len() != numel * 4 {
                    return Err(Error::ConstantFolding(format!(
                        "Invalid byte length for I32 tensor: expected {}, got {}",
                        numel * 4,
                        bytes.len()
                    )));
                }
                let values = bytes
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                Ok(TensorValue::I32(values))
            }
            DataType::F32 => {
                if bytes.len() != numel * 4 {
                    return Err(Error::ConstantFolding(format!(
                        "Invalid byte length for F32 tensor: expected {}, got {}",
                        numel * 4,
                        bytes.len()
                    )));
                }
                let values = bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                Ok(TensorValue::F32(values))
            }
            DataType::U8 => Ok(TensorValue::U8(bytes.to_vec())),
            DataType::Bool => {
                // Bool stored as u32 in GPU memory (4 bytes per bool)
                if bytes.len() != numel * 4 {
                    return Err(Error::ConstantFolding(format!(
                        "Invalid byte length for Bool tensor: expected {}, got {}",
                        numel * 4,
                        bytes.len()
                    )));
                }
                let values = bytes
                    .chunks_exact(4)
                    .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()) != 0)
                    .collect();
                Ok(TensorValue::Bool(values))
            }
            _ => Err(Error::ConstantFolding(format!(
                "from_bytes not implemented for {:?}",
                dtype
            ))),
        }
    }

    /// Get the data type of this value.
    pub fn dtype(&self) -> DataType {
        match self {
            TensorValue::I64(_) => DataType::I64,
            TensorValue::I32(_) => DataType::I32,
            TensorValue::F32(_) => DataType::F32,
            TensorValue::Bool(_) => DataType::Bool,
            TensorValue::U8(_) => DataType::U8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_value_cast() {
        let i64_val = TensorValue::I64(vec![1, 2, 3]);
        let i32_val = i64_val.cast(DataType::I32).unwrap();
        assert_eq!(i32_val.as_i32(), Some(&[1, 2, 3][..]));

        let f32_val = i32_val.cast(DataType::F32).unwrap();
        assert_eq!(f32_val.as_f32(), Some(&[1.0, 2.0, 3.0][..]));
    }

    #[test]
    fn test_tensor_value_from_bytes() {
        // Test I32
        let bytes = vec![1u8, 0, 0, 0, 2, 0, 0, 0];
        let value = TensorValue::from_bytes(&bytes, DataType::I32, &[2]).unwrap();
        assert_eq!(value.as_i32(), Some(&[1, 2][..]));

        // Test F32
        let bytes = vec![0, 0, 128, 63, 0, 0, 0, 64]; // 1.0f32, 2.0f32 in little-endian
        let value = TensorValue::from_bytes(&bytes, DataType::F32, &[2]).unwrap();
        assert_eq!(value.as_f32(), Some(&[1.0, 2.0][..]));
    }

    #[test]
    fn test_tensor_shape_is_static() {
        let static_shape = TensorShape::Static(vec![1, 2, 3]);
        assert!(static_shape.is_static());
        assert_eq!(static_shape.ndim(), Some(3));

        let symbolic_shape = TensorShape::Symbolic(vec![
            SymbolicDim::Fixed(1),
            SymbolicDim::Expr(SymbolicExpr::Variable("batch".to_string())),
        ]);
        assert!(!symbolic_shape.is_static());
        assert_eq!(symbolic_shape.ndim(), Some(2));

        let absent_shape = TensorShape::Absent;
        assert!(!absent_shape.is_static());
        assert!(absent_shape.is_absent());
        assert_eq!(absent_shape.ndim(), None);
    }
}
