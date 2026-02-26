//! Core types for tensor shapes, values, and metadata.

use crate::{Error, Result};

// Re-export types from onyxia-onnx
pub use onyxia_onnx::DataType;

/// Raw tensor data.
///
/// A single dimension that may be symbolic.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dim {
    /// Statically known at compile time.
    Fixed(usize),
    /// Symbolic — references an input dimension. Resolved before dispatch
    /// from input tensor shapes. E.g., `"batch_size"` or `"seq_len"`.
    Named(String),
}

impl Dim {
    /// Returns `true` if this dimension is a known static value.
    pub fn is_fixed(&self) -> bool {
        matches!(self, Dim::Fixed(_))
    }

    /// Returns the static size if this dimension is [`Dim::Fixed`], otherwise `None`.
    pub fn as_fixed(&self) -> Option<usize> {
        match self {
            Dim::Fixed(n) => Some(*n),
            Dim::Named(_) => None,
        }
    }
}

/// A shape where some or all dimensions may be symbolic.
///
/// This is the result of compile-time shape inference. Some dimensions are
/// known statically (e.g., weight shapes, constant shapes), others are
/// symbolic references resolved from runtime input shapes before dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolicShape {
    /// All dimensions are known or symbolic (but rank is known).
    Ranked(Vec<Dim>),
    /// Rank is unknown — shape inference could not determine anything.
    Unranked,
}

impl SymbolicShape {
    /// Create a fully-static `Ranked` shape from concrete dimensions.
    ///
    /// Convenience constructor for tests and places that have concrete
    /// `usize` dimensions available.
    ///
    /// # Example
    /// ```
    /// use onyxia_core::SymbolicShape;
    /// let shape = SymbolicShape::fixed(&[2, 3, 4]);
    /// assert!(shape.is_fully_static());
    /// ```
    pub fn fixed(dims: &[usize]) -> Self {
        SymbolicShape::Ranked(dims.iter().map(|&d| Dim::Fixed(d)).collect())
    }

    /// Convert an ONNX [`onyxia_onnx::TensorShape`] to a `SymbolicShape`,
    /// preserving named (symbolic) dimensions.
    pub fn from_onnx(onnx: &onyxia_onnx::TensorShape) -> Self {
        match onnx {
            onyxia_onnx::TensorShape::Static(dims) => {
                SymbolicShape::Ranked(dims.iter().map(|&d| Dim::Fixed(d)).collect())
            }
            onyxia_onnx::TensorShape::Dynamic(dims) => SymbolicShape::Ranked(
                dims.iter()
                    .map(|d| match d {
                        onyxia_onnx::Dimension::Static(n) => Dim::Fixed(*n),
                        onyxia_onnx::Dimension::Named(name) => Dim::Named(name.clone()),
                    })
                    .collect(),
            ),
            onyxia_onnx::TensorShape::Absent | onyxia_onnx::TensorShape::Unknown => {
                SymbolicShape::Unranked
            }
        }
    }

    /// Returns `true` if every dimension in a `Ranked` shape is [`Dim::Fixed`].
    ///
    /// Returns `false` for `Unranked` shapes.
    pub fn is_fully_static(&self) -> bool {
        match self {
            SymbolicShape::Ranked(dims) => dims.iter().all(Dim::is_fixed),
            SymbolicShape::Unranked => false,
        }
    }

    /// Returns the static dimensions if this shape is fully static, otherwise `None`.
    pub fn as_static(&self) -> Option<Vec<usize>> {
        match self {
            SymbolicShape::Ranked(dims) => {
                dims.iter().map(Dim::as_fixed).collect::<Option<Vec<_>>>()
            }
            SymbolicShape::Unranked => None,
        }
    }

    /// Returns the rank (number of dimensions) if known, otherwise `None`.
    pub fn rank(&self) -> Option<usize> {
        match self {
            SymbolicShape::Ranked(dims) => Some(dims.len()),
            SymbolicShape::Unranked => None,
        }
    }
}

/// Raw tensor data.
///
/// Separated from metadata (shape, dtype) to enable flexible tensor operations.
/// Used for compile-time constants and initializer data.
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

/// A tensor value known at compile time.
///
/// Bundles data, shape, and dtype together for operations that need
/// concrete tensor values. Used for initializers and compile-time constants.
///
/// Note: Currently preserved for backward compatibility. In the dispatch-based
/// execution model, most compile-time evaluation has been removed.
///
/// Only small tensors should be stored (shape-metadata, indices, axes).
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
                return Err(Error::Compilation(format!(
                    "Cast from {:?} to {:?} not supported",
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
                    return Err(Error::Compilation(format!(
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
                    return Err(Error::Compilation(format!(
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
                    return Err(Error::Compilation(format!(
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
                    return Err(Error::Compilation(format!(
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
                return Err(Error::Compilation(format!(
                    "from_bytes not implemented for {:?}",
                    dtype
                )));
            }
        };

        Ok(TensorValue::new(data, shape.to_vec(), dtype))
    }

    /// Serialize this tensor value to raw little-endian bytes.
    ///
    /// This is the inverse of [`from_bytes`](Self::from_bytes). The output is
    /// suitable for direct upload to a GPU buffer via `queue.write_buffer()`.
    ///
    /// Bool values are serialized as `u32` (4 bytes each) to match GPU
    /// alignment requirements (same convention as `from_bytes`).
    pub fn to_bytes(&self) -> Vec<u8> {
        match &self.data {
            TensorData::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorData::I32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorData::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorData::U8(v) => v.clone(),
            TensorData::Bool(v) => v.iter().flat_map(|&b| (b as u32).to_le_bytes()).collect(),
        }
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
    fn test_tensor_value_to_bytes_roundtrip() {
        // F32 roundtrip
        let original = TensorValue::new(
            TensorData::F32(vec![1.0, 2.5, -3.0]),
            vec![3],
            DataType::F32,
        );
        let bytes = original.to_bytes();
        let restored = TensorValue::from_bytes(&bytes, DataType::F32, &[3]).unwrap();
        assert_eq!(restored.as_f32(), original.as_f32());

        // I32 roundtrip
        let original = TensorValue::new(TensorData::I32(vec![10, -20, 30]), vec![3], DataType::I32);
        let bytes = original.to_bytes();
        let restored = TensorValue::from_bytes(&bytes, DataType::I32, &[3]).unwrap();
        assert_eq!(restored.as_i32(), original.as_i32());

        // I64 roundtrip
        let original = TensorValue::new(TensorData::I64(vec![100, -200]), vec![2], DataType::I64);
        let bytes = original.to_bytes();
        let restored = TensorValue::from_bytes(&bytes, DataType::I64, &[2]).unwrap();
        assert_eq!(restored.as_i64(), original.as_i64());

        // U8 roundtrip
        let original = TensorValue::new(TensorData::U8(vec![1, 2, 255]), vec![3], DataType::U8);
        let bytes = original.to_bytes();
        assert_eq!(bytes, vec![1, 2, 255]);

        // Bool roundtrip
        let original = TensorValue::new(
            TensorData::Bool(vec![true, false, true]),
            vec![3],
            DataType::Bool,
        );
        let bytes = original.to_bytes();
        let restored = TensorValue::from_bytes(&bytes, DataType::Bool, &[3]).unwrap();
        assert_eq!(restored.as_bool(), original.as_bool());
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
    fn test_dim_fixed_and_named() {
        let fixed = Dim::Fixed(42);
        let named = Dim::Named("batch_size".to_string());

        assert!(fixed.is_fixed());
        assert_eq!(fixed.as_fixed(), Some(42));

        assert!(!named.is_fixed());
        assert_eq!(named.as_fixed(), None);

        // Equality / hash
        assert_eq!(fixed, Dim::Fixed(42));
        assert_ne!(fixed, named);

        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Dim::Fixed(1));
        set.insert(Dim::Named("seq".to_string()));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_symbolic_shape_fixed() {
        let ss = SymbolicShape::fixed(&[2, 3, 4]);
        assert_eq!(
            ss,
            SymbolicShape::Ranked(vec![Dim::Fixed(2), Dim::Fixed(3), Dim::Fixed(4)])
        );
        assert!(ss.is_fully_static());
        assert_eq!(ss.as_static(), Some(vec![2, 3, 4]));
        assert_eq!(ss.rank(), Some(3));
    }

    #[test]
    fn test_symbolic_shape_from_onnx_preserves_named() {
        let onnx_shape = onyxia_onnx::TensorShape::Dynamic(vec![
            onyxia_onnx::Dimension::Named("batch_size".to_string()),
            onyxia_onnx::Dimension::Static(128),
            onyxia_onnx::Dimension::Named("seq_len".to_string()),
        ]);
        let ss = SymbolicShape::from_onnx(&onnx_shape);
        assert_eq!(
            ss,
            SymbolicShape::Ranked(vec![
                Dim::Named("batch_size".to_string()),
                Dim::Fixed(128),
                Dim::Named("seq_len".to_string()),
            ])
        );
        assert!(!ss.is_fully_static());
        assert_eq!(ss.as_static(), None);
        assert_eq!(ss.rank(), Some(3));
    }

    #[test]
    fn test_symbolic_shape_is_fully_static() {
        let fully_static = SymbolicShape::Ranked(vec![Dim::Fixed(1), Dim::Fixed(2)]);
        assert!(fully_static.is_fully_static());

        let mixed = SymbolicShape::Ranked(vec![Dim::Fixed(1), Dim::Named("n".to_string())]);
        assert!(!mixed.is_fully_static());

        let unranked = SymbolicShape::Unranked;
        assert!(!unranked.is_fully_static());
    }
}
