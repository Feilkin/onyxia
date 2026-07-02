//! Element data types and tensor types.

use crate::dim::SymbolicShape;

/// Element data type of a tensor.
///
/// Sub-byte types (`U4`, `I4`) carry a *logical* element count in the tensor
/// shape while their storage is bit-packed, two elements per byte, padded to
/// a whole byte per innermost row. Only [`Prim::Dequantize`](crate::prim::Prim)
/// and [`Prim::Cast`](crate::prim::Prim) interpret the packing; all other
/// code treats such buffers as opaque bytes (pinned decision 2 in
/// `doc/ir-implementation-plan.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    F32,
    F16,
    I64,
    I32,
    U32,
    U8,
    I8,
    Bool,
    /// Unsigned 4-bit integer (ONNX `UINT4`), bit-packed storage.
    U4,
    /// Signed 4-bit integer (ONNX `INT4`), bit-packed storage.
    I4,
}

impl DataType {
    /// Size of one element in bits.
    pub fn bits(self) -> usize {
        match self {
            DataType::F32 | DataType::I32 | DataType::U32 => 32,
            DataType::F16 => 16,
            DataType::I64 => 64,
            DataType::U8 | DataType::I8 | DataType::Bool => 8,
            DataType::U4 | DataType::I4 => 4,
        }
    }

    /// Whether this is a floating-point type.
    pub fn is_float(self) -> bool {
        matches!(self, DataType::F32 | DataType::F16)
    }

    /// Whether this is an integer type (signed or unsigned, any width).
    pub fn is_int(self) -> bool {
        matches!(
            self,
            DataType::I64
                | DataType::I32
                | DataType::U32
                | DataType::U8
                | DataType::I8
                | DataType::U4
                | DataType::I4
        )
    }

    /// Whether this type's storage is bit-packed (sub-byte elements).
    pub fn is_packed(self) -> bool {
        self.bits() < 8
    }

    /// Storage size in bytes for `num_elements` elements.
    ///
    /// For whole-byte types this is `num_elements * bytes_per_element`. For
    /// packed types the total is rounded up to a whole byte.
    ///
    /// The bit count is computed in u64: on wasm32 (32-bit usize) the
    /// intermediate `num_elements * bits` wraps for tensors ≥ 512 MiB —
    /// Gemma's 671 MB f32 embed table hit exactly this. A byte size that
    /// itself exceeds the address space panics instead of wrapping.
    pub fn storage_bytes(self, num_elements: usize) -> usize {
        let bytes = (num_elements as u64 * self.bits() as u64).div_ceil(8);
        usize::try_from(bytes).expect("tensor byte size exceeds address space")
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            DataType::F32 => "f32",
            DataType::F16 => "f16",
            DataType::I64 => "i64",
            DataType::I32 => "i32",
            DataType::U32 => "u32",
            DataType::U8 => "u8",
            DataType::I8 => "i8",
            DataType::Bool => "bool",
            DataType::U4 => "u4",
            DataType::I4 => "i4",
        };
        f.write_str(s)
    }
}

/// The type of a value: element dtype plus (possibly symbolic) shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorType {
    /// Element data type.
    pub dtype: DataType,
    /// Shape; dimensions may be symbolic expressions.
    pub shape: SymbolicShape,
}

impl TensorType {
    /// Create a tensor type.
    pub fn new(dtype: DataType, shape: SymbolicShape) -> Self {
        Self { dtype, shape }
    }

    /// Create a tensor type with a fully static shape.
    pub fn of(dtype: DataType, dims: &[u64]) -> Self {
        Self {
            dtype,
            shape: SymbolicShape::fixed(dims),
        }
    }
}

impl std::fmt::Display for TensorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.dtype, self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_bytes_whole_byte_types() {
        assert_eq!(DataType::F32.storage_bytes(3), 12);
        assert_eq!(DataType::F16.storage_bytes(3), 6);
        assert_eq!(DataType::I64.storage_bytes(2), 16);
        assert_eq!(DataType::U8.storage_bytes(5), 5);
        assert_eq!(DataType::Bool.storage_bytes(4), 4);
    }

    #[test]
    fn storage_bytes_large_tensor_no_32bit_overflow() {
        // Gemma 3 270m's embed table: f32[262144, 640]. numel × 32 bits
        // exceeds 2^32 — on wasm32 the old usize math wrapped to a bogus
        // 134217728 and every big-weight model failed to lower in the
        // browser. The math must be 64-bit regardless of pointer width.
        assert_eq!(DataType::F32.storage_bytes(262144 * 640), 671_088_640);
    }

    #[test]
    fn storage_bytes_packed_types_round_up() {
        assert_eq!(DataType::U4.storage_bytes(0), 0);
        assert_eq!(DataType::U4.storage_bytes(1), 1);
        assert_eq!(DataType::U4.storage_bytes(2), 1);
        assert_eq!(DataType::U4.storage_bytes(3), 2);
        assert_eq!(DataType::I4.storage_bytes(32), 16);
    }

    #[test]
    fn classification() {
        assert!(DataType::F16.is_float());
        assert!(!DataType::F16.is_int());
        assert!(DataType::U4.is_int());
        assert!(DataType::U4.is_packed());
        assert!(!DataType::U8.is_packed());
        assert!(!DataType::Bool.is_int());
        assert!(!DataType::Bool.is_float());
    }
}
