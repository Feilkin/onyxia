//! Core types for tensor shapes, values, and metadata.

use crate::symbolic_expr::SymbolicExpr;
use crate::{Error, Result};

// Re-export types from onyxia-onnx
pub use onyxia_onnx::DataType;

/// Tensor shape with support for static, symbolic, absent, and unknown shapes.
///
/// Unknown shapes represent tensors whose shapes have not yet been inferred.
/// These are allowed to exist during graph construction, but must be resolved
/// by the shape inference pass before planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorShape {
    /// All dimensions are known at compile time.
    Static(Vec<usize>),

    /// Mix of static and symbolic dimensions (e.g., `[batch_size, 128, seq_length * 8]`).
    Symbolic(Vec<SymbolicDim>),

    /// Optional input that is absent (ONNX empty string).
    Absent,

    /// Shape has not yet been inferred (must be resolved before planning).
    Unknown,
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

    /// Check if the shape is unknown (not yet inferred).
    pub fn is_unknown(&self) -> bool {
        matches!(self, TensorShape::Unknown)
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
            TensorShape::Absent | TensorShape::Unknown => None,
        }
    }

    /// Convert from ONNX TensorShape to core TensorShape.
    ///
    /// Maps all ONNX shape variants to their core equivalents, including Unknown.
    /// Unknown shapes must be resolved by the shape inference pass before planning.
    pub fn from_onnx(onnx_shape: &onyxia_onnx::TensorShape) -> Self {
        match onnx_shape {
            onyxia_onnx::TensorShape::Static(dims) => TensorShape::Static(dims.clone()),
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
                TensorShape::Symbolic(symbolic_dims)
            }
            onyxia_onnx::TensorShape::Absent => TensorShape::Absent,
            onyxia_onnx::TensorShape::Unknown => TensorShape::Unknown,
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

/// Raw tensor data for constant folding.
///
/// Separated from metadata (shape, dtype) to enable flexible tensor operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorData {
    I64(Vec<i64>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    U8(Vec<u8>),
}

impl TensorData {
    /// Get the number of elements in this tensor data.
    pub fn len(&self) -> usize {
        match self {
            TensorData::I64(v) => v.len(),
            TensorData::I32(v) => v.len(),
            TensorData::F32(v) => v.len(),
            TensorData::Bool(v) => v.len(),
            TensorData::U8(v) => v.len(),
        }
    }

    /// Check if this tensor data is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Try to get as i64 slice.
    pub fn as_i64(&self) -> Option<&[i64]> {
        match self {
            TensorData::I64(v) => Some(v),
            _ => None,
        }
    }

    /// Try to get as i32 slice.
    pub fn as_i32(&self) -> Option<&[i32]> {
        match self {
            TensorData::I32(v) => Some(v),
            _ => None,
        }
    }

    /// Try to get as f32 slice.
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            TensorData::F32(v) => Some(v),
            _ => None,
        }
    }

    /// Try to get as bool slice.
    pub fn as_bool(&self) -> Option<&[bool]> {
        match self {
            TensorData::Bool(v) => Some(v),
            _ => None,
        }
    }

    /// Try to get as u8 slice.
    pub fn as_u8(&self) -> Option<&[u8]> {
        match self {
            TensorData::U8(v) => Some(v),
            _ => None,
        }
    }

    /// Get the inferred data type from this tensor data.
    pub fn dtype(&self) -> DataType {
        match self {
            TensorData::I64(_) => DataType::I64,
            TensorData::I32(_) => DataType::I32,
            TensorData::F32(_) => DataType::F32,
            TensorData::Bool(_) => DataType::Bool,
            TensorData::U8(_) => DataType::U8,
        }
    }
}

/// A tensor value known at compile time (for constant folding).
///
/// Bundles data, shape, and dtype together to enable correct constant folding
/// for shape-transforming operators like Reshape and Transpose.
///
/// Used to enable data-dependent shape inference: e.g., Reshape reads its
/// target shape from an upstream Concat whose values were propagated from
/// Shape and Gather.
///
/// Only small tensors are stored (shape-metadata, indices, axes).
/// Large weight tensors should NOT be wrapped in TensorValue.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorValue {
    /// The raw tensor data.
    pub data: TensorData,

    /// The shape of the tensor (dimensions).
    pub shape: Vec<usize>,

    /// The data type of the tensor.
    pub dtype: DataType,
}

impl TensorValue {
    /// Create a new TensorValue with data, shape, and dtype.
    ///
    /// # Panics
    ///
    /// Panics if the data length doesn't match the shape product.
    pub fn new(data: TensorData, shape: Vec<usize>, dtype: DataType) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} doesn't match shape {:?} (product = {})",
            data.len(),
            shape,
            expected_len
        );
        assert_eq!(
            data.dtype(),
            dtype,
            "Data type {:?} doesn't match declared dtype {:?}",
            data.dtype(),
            dtype
        );
        Self { data, shape, dtype }
    }

    /// Create a scalar TensorValue (shape = []).
    pub fn scalar(data: TensorData, dtype: DataType) -> Self {
        Self::new(data, vec![], dtype)
    }

    /// Get the number of elements in this tensor value.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if this tensor value is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the total number of elements based on shape.
    pub fn total_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Try to get as i64 slice.
    pub fn as_i64(&self) -> Option<&[i64]> {
        self.data.as_i64()
    }

    /// Try to get as i32 slice.
    pub fn as_i32(&self) -> Option<&[i32]> {
        self.data.as_i32()
    }

    /// Try to get as f32 slice.
    pub fn as_f32(&self) -> Option<&[f32]> {
        self.data.as_f32()
    }

    /// Try to get as bool slice.
    pub fn as_bool(&self) -> Option<&[bool]> {
        self.data.as_bool()
    }

    /// Try to get as u8 slice.
    pub fn as_u8(&self) -> Option<&[u8]> {
        self.data.as_u8()
    }

    /// Create a new TensorValue with a different shape (data unchanged).
    ///
    /// # Panics
    ///
    /// Panics if the new shape product doesn't match the data length.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        Self::new(self.data.clone(), new_shape, self.dtype)
    }

    /// Cast this value to a different type.
    pub fn cast(&self, target_dtype: DataType) -> Result<TensorValue> {
        if self.dtype == target_dtype {
            return Ok(self.clone());
        }

        let new_data = match (&self.data, target_dtype) {
            (TensorData::I64(v), DataType::I32) => {
                TensorData::I32(v.iter().map(|&x| x as i32).collect())
            }
            (TensorData::I64(v), DataType::F32) => {
                TensorData::F32(v.iter().map(|&x| x as f32).collect())
            }
            (TensorData::I32(v), DataType::I64) => {
                TensorData::I64(v.iter().map(|&x| x as i64).collect())
            }
            (TensorData::I32(v), DataType::F32) => {
                TensorData::F32(v.iter().map(|&x| x as f32).collect())
            }
            (TensorData::F32(v), DataType::I32) => {
                TensorData::I32(v.iter().map(|&x| x as i32).collect())
            }
            (TensorData::F32(v), DataType::I64) => {
                TensorData::I64(v.iter().map(|&x| x as i64).collect())
            }
            _ => {
                return Err(Error::ConstantFolding(format!(
                    "Cast from {:?} to {:?} not supported in constant folding",
                    self.dtype, target_dtype
                )));
            }
        };

        Ok(TensorValue::new(new_data, self.shape.clone(), target_dtype))
    }

    /// Parse a TensorValue from initializer bytes.
    ///
    /// # Arguments
    /// * `bytes` - Raw tensor data in native endian format
    /// * `dtype` - Data type of the tensor
    /// * `shape` - Shape of the tensor (must be static)
    pub fn from_bytes(bytes: &[u8], dtype: DataType, shape: &[usize]) -> Result<Self> {
        let numel: usize = shape.iter().product();

        let data = match dtype {
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
                TensorData::I64(values)
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
                TensorData::I32(values)
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
                TensorData::F32(values)
            }
            DataType::U8 => TensorData::U8(bytes.to_vec()),
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
                TensorData::Bool(values)
            }
            _ => {
                return Err(Error::ConstantFolding(format!(
                    "from_bytes not implemented for {:?}",
                    dtype
                )));
            }
        };

        Ok(TensorValue::new(data, shape.to_vec(), dtype))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_value_cast() {
        let i64_val = TensorValue::new(TensorData::I64(vec![1, 2, 3]), vec![3], DataType::I64);
        let i32_val = i64_val.cast(DataType::I32).unwrap();
        assert_eq!(i32_val.as_i32(), Some(&[1, 2, 3][..]));
        assert_eq!(i32_val.shape, vec![3]);

        let f32_val = i32_val.cast(DataType::F32).unwrap();
        assert_eq!(f32_val.as_f32(), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(f32_val.shape, vec![3]);
    }

    #[test]
    fn test_tensor_value_from_bytes() {
        // Test I32
        let bytes = vec![1u8, 0, 0, 0, 2, 0, 0, 0];
        let value = TensorValue::from_bytes(&bytes, DataType::I32, &[2]).unwrap();
        assert_eq!(value.as_i32(), Some(&[1, 2][..]));
        assert_eq!(value.shape, vec![2]);
        assert_eq!(value.dtype, DataType::I32);

        // Test F32
        let bytes = vec![0, 0, 128, 63, 0, 0, 0, 64]; // 1.0f32, 2.0f32 in little-endian
        let value = TensorValue::from_bytes(&bytes, DataType::F32, &[2]).unwrap();
        assert_eq!(value.as_f32(), Some(&[1.0, 2.0][..]));
        assert_eq!(value.shape, vec![2]);
        assert_eq!(value.dtype, DataType::F32);
    }

    #[test]
    fn test_tensor_value_reshape() {
        let value = TensorValue::new(
            TensorData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![2, 3],
            DataType::F32,
        );
        assert_eq!(value.shape, vec![2, 3]);

        let reshaped = value.reshape(vec![3, 2]);
        assert_eq!(reshaped.shape, vec![3, 2]);
        assert_eq!(reshaped.as_f32(), Some(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0][..]));
    }

    #[test]
    #[should_panic(expected = "doesn't match shape")]
    fn test_tensor_value_new_validates_shape() {
        // 6 elements but shape product is 8
        TensorValue::new(
            TensorData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![2, 4],
            DataType::F32,
        );
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

        let unknown_shape = TensorShape::Unknown;
        assert!(!unknown_shape.is_static());
        assert!(unknown_shape.is_unknown());
        assert_eq!(unknown_shape.ndim(), None);
    }
}
