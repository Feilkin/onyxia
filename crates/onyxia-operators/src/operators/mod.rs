//! Individual operator implementations that don't fit into families.
//!
//! These operators have unique logic that doesn't generalize well into families.

pub mod matmul;
pub mod reduce_mean;
pub mod shape;

// Re-export all operators
pub use matmul::MatMulOp;
pub use reduce_mean::ReduceMeanOp;
pub use shape::{ConcatOp, ExpandOp, ReshapeOp, SliceOp, TransposeOp, UnsqueezeOp};
