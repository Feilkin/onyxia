//! Collapsed operator families that eliminate code duplication.
//!
//! These families group similar operators together, implementing shared logic once
//! and parameterizing only the differences (shader source, fold function).

pub mod binary_elementwise;
pub mod comparison;
pub mod max_elementwise;
pub mod unary_math;

pub use binary_elementwise::BinaryElementwiseOp;
pub use comparison::ComparisonOp;
pub use max_elementwise::MaxOp;
pub use unary_math::UnaryMathOp;
