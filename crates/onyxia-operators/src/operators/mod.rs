//! Individual operator implementations that don't fit into families.
//!
//! These operators have unique logic that doesn't generalize well into families.

pub mod cast;
pub mod constant;
pub mod gather;
pub mod group_query_attention;
pub mod matmul;
pub mod matmul_nbits;
pub mod range;
pub mod reduce_mean;
pub mod reduce_sum;
pub mod rotary_embedding;
pub mod scatter_nd;
pub mod shape;
pub mod simplified_layer_norm;
pub mod softmax;
pub mod trilu;
pub mod where_op;

// Re-export all operators
pub use cast::CastOp;
pub use constant::ConstantOp;
pub use gather::GatherOp;
pub use group_query_attention::GroupQueryAttentionOp;
pub use matmul::MatMulOp;
pub use matmul_nbits::MatMulNBitsOp;
pub use range::RangeOp;
pub use reduce_mean::ReduceMeanOp;
pub use reduce_sum::ReduceSumOp;
pub use rotary_embedding::RotaryEmbeddingOp;
pub use scatter_nd::ScatterNDOp;
pub use shape::{
    ConcatOp, ConstantOfShapeOp, ExpandOp, ReshapeOp, ShapeOp, SliceOp, TransposeOp, UnsqueezeOp,
};
pub use simplified_layer_norm::SimplifiedLayerNormOp;
pub use softmax::SoftmaxOp;
pub use trilu::TriluOp;
pub use where_op::WhereOp;
