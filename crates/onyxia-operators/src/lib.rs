//! Core operator implementations for Onyxia.
//!
//! This crate provides the standard ONNX operator set with optimized implementations
//! using collapsed operator families to eliminate code duplication.
//!
//! # Operator Families
//!
//! - **Binary elementwise**: Add, Sub, Mul, Div, Pow, Max
//! - **Unary elementwise**: Cos, Sin, Sqrt, Neg, Tanh
//! - **Comparison**: Equal, Greater
//! - **Reduction**: ReduceSum, ReduceMean
//!
//! # Individual Operators
//!
//! Complex operators that don't fit into families:
//! - Activation (Gelu, Softmax, Tanh)
//! - Normalization (RmsNorm)
//! - Matrix operations (MatMulF32, MatMulNBits)
//! - Shape manipulation (Reshape, Unsqueeze, Transpose, Concat, Expand)
//! - Indexing (Gather, Slice, ScatterND, Range, Trilu)
//! - Metadata (Constant, ConstantOfShape, Shape)
//! - Type conversion (Cast)
//! - Conditional (Where)
//! - Attention (RotaryEmbedding, GroupQueryAttention)

pub mod families;
pub mod operators;
pub mod passes;

mod helpers;
mod registry;

// Re-export operator types
pub use families::{BinaryElementwiseOp, ComparisonOp, ReductionOp, UnaryElementwiseOp};
pub use registry::core_operator_registry;

/// Result type for operator operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for operator operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Operator error: {0}")]
    Operator(String),

    #[error(transparent)]
    Core(#[from] onyxia_core::Error),
}
