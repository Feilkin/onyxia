//! Attention mechanism operators.

use onyxia_core::{InferenceCtx, Operator, PlanCtx, Result, Step, TensorShape};

/// Rotary positional embedding operator.
pub struct RotaryEmbeddingOp;

impl Operator for RotaryEmbeddingOp {
    fn name(&self) -> &str {
        "RotaryEmbedding"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for RotaryEmbedding - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for RotaryEmbedding - will be implemented in Tasks 024/025")
    }
}

/// Grouped Query Attention operator.
pub struct GroupQueryAttentionOp;

impl Operator for GroupQueryAttentionOp {
    fn name(&self) -> &str {
        "GroupQueryAttention"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for GroupQueryAttention - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for GroupQueryAttention - will be implemented in Tasks 024/025")
    }
}
