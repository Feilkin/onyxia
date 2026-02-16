//! Individual operator implementations that don't fit into families.
//!
//! These operators have unique logic that doesn't generalize well into families.

pub mod activation;
pub mod attention;
pub mod conditional;
pub mod indexing;
pub mod matmul;
pub mod metadata;
pub mod normalization;
pub mod shape;
pub mod type_conversion;

// Re-export all operators
pub use activation::{GeluOp, SoftmaxOp};
pub use attention::{
    GemmaRotaryEmbeddingOp, GroupQueryAttentionOp, MicrosoftRotaryEmbeddingOp, RotaryEmbeddingOp,
};
pub use conditional::WhereOp;
pub use indexing::{GatherOp, RangeOp, ScatterNDOp, SliceOp, TriluOp};
pub use matmul::{MatMulF32Op, MatMulNBitsOp};
pub use metadata::{ConstantOfShapeOp, ConstantOp, ShapeOp};
pub use normalization::RmsNormOp;
pub use shape::{ConcatOp, ExpandOp, ReshapeOp, TransposeOp, UnsqueezeOp};
pub use type_conversion::CastOp;
