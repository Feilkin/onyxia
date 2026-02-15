//! Collapsed operator families that eliminate code duplication.
//!
//! These families group similar operators together, implementing shared logic once
//! and parameterizing only the differences (shader source, fold function).

pub mod binary_elementwise;
pub mod comparison;
pub mod reduction;
pub mod unary_elementwise;

pub use binary_elementwise::BinaryElementwiseOp;
pub use comparison::ComparisonOp;
pub use reduction::ReductionOp;
pub use unary_elementwise::UnaryElementwiseOp;
