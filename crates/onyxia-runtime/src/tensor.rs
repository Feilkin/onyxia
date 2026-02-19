//! User-facing tensor type for CPU/GPU data interchange.

use crate::error::{Result, RuntimeError};
use bytemuck::Pod;
use onyxia_onnx::DataType;

/// User-facing tensor for input/output data.
///
/// Tensors can hold data on CPU or reference GPU buffers. For basic usage,
/// create from Vec and extract to Vec.
#[derive(Debug, Clone)]
pub struct Tensor {
    data: TensorData,
    shape: Vec<usize>,
    dtype: DataType,
}

impl Tensor {
    /// Create a tensor from a vector with a given shape.
    ///
    /// # Example
    /// ```no_run
    /// # use onyxia_runtime::Tensor;
    /// let data = vec![1.0f32, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::from_vec(data, &[2, 2]);
    /// ```
    pub fn from_vec<T: Pod>(data: Vec<T>, shape: &[usize]) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_len
        );

        let dtype = Self::infer_dtype::<T>();
        let bytes = bytemuck::cast_slice(&data).to_vec();

        Self {
            data: TensorData::Cpu(bytes),
            shape: shape.to_vec(),
            dtype,
        }
    }

    /// Create a tensor from raw bytes.
    pub(crate) fn from_raw(data: Vec<u8>, shape: &[usize], dtype: DataType) -> Self {
        Self {
            data: TensorData::Cpu(data),
            shape: shape.to_vec(),
            dtype,
        }
    }

    /// Get a slice view of the tensor data.
    ///
    /// # Errors
    /// Returns an error if type doesn't match.
    pub fn as_slice<T: Pod>(&self) -> Result<&[T]> {
        let TensorData::Cpu(bytes) = &self.data;
        if std::mem::size_of::<T>() * self.len() != bytes.len() {
            return Err(RuntimeError::TensorError("Type size mismatch".to_string()));
        }
        Ok(bytemuck::cast_slice(bytes))
    }

    /// Convert tensor to a Vec.
    ///
    /// # Errors
    /// Returns an error if type doesn't match.
    pub fn to_vec<T: Pod>(&self) -> Result<Vec<T>> {
        Ok(self.as_slice::<T>()?.to_vec())
    }

    /// Get raw bytes of the tensor data.
    pub(crate) fn raw_data(&self) -> Result<&[u8]> {
        let TensorData::Cpu(bytes) = &self.data;
        Ok(bytes)
    }

    /// Get raw bytes of the tensor data.
    ///
    /// This is used by the dispatch executor to upload tensors to GPU.
    pub fn as_bytes(&self) -> Result<&[u8]> {
        self.raw_data()
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the data type of the tensor.
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Get the total number of elements in the tensor.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Infer DataType from Rust type.
    fn infer_dtype<T: Pod>() -> DataType {
        let type_name = std::any::type_name::<T>();
        if type_name.contains("f32") {
            DataType::F32
        } else if type_name.contains("f16") {
            DataType::F16
        } else if type_name.contains("i32") {
            DataType::I32
        } else if type_name.contains("i64") {
            DataType::I64
        } else if type_name.contains("u32") {
            DataType::U32
        } else if type_name.contains("u8") {
            DataType::U8
        } else if type_name.contains("bool") {
            DataType::Bool
        } else {
            // Default to F32 for unknown types
            DataType::F32
        }
    }
}

/// Internal tensor data representation.
#[derive(Debug, Clone)]
pub(crate) enum TensorData {
    /// Data on CPU (host memory).
    Cpu(Vec<u8>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_from_vec() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data.clone(), &[2, 2]);

        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.dtype(), DataType::F32);
        assert_eq!(tensor.len(), 4);
        assert!(!tensor.is_empty());
    }

    #[test]
    fn test_tensor_as_slice() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data.clone(), &[2, 2]);

        let slice = tensor.as_slice::<f32>().unwrap();
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tensor_to_vec() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data.clone(), &[2, 2]);

        let result = tensor.to_vec::<f32>().unwrap();
        assert_eq!(result, data);
    }

    #[test]
    #[should_panic(expected = "doesn't match shape")]
    fn test_tensor_shape_mismatch() {
        let data = vec![1.0f32, 2.0, 3.0];
        Tensor::from_vec(data, &[2, 2]); // Should panic
    }
}
