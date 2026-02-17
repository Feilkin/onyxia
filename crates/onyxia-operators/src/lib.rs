//! Core operator implementations for Onyxia.
//!
//! This crate provides a minimal operator set for validating the new dispatch architecture.
//!
//! # Operator Families
//!
//! - **Binary elementwise**: Add, Mul, Div, Sub, Pow
//! - **Comparison**: Equal, Greater, Less, GreaterOrEqual, LessOrEqual
//!
//! # Individual Operators
//!
//! - Shape manipulation (Reshape, Concat, Expand, Transpose, Unsqueeze)

pub mod families;
pub mod operators;
pub mod passes;

mod helpers;
mod registry;

// Re-export operator types
pub use families::{BinaryElementwiseOp, ComparisonOp};
pub use operators::{
    CastOp, ConcatOp, ConstantOp, ExpandOp, ReshapeOp, SimplifiedLayerNormOp, SliceOp, TransposeOp,
    UnsqueezeOp,
};
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
