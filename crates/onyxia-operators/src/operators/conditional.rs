//! Conditional operators.

use onyxia_core::{InferenceCtx, Operator, PlanCtx, Result, Step, TensorShape};

/// Where operator - selects elements from two tensors based on a condition.
pub struct WhereOp;

impl Operator for WhereOp {
    fn name(&self) -> &str {
        "Where"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for Where - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for Where - will be implemented in Tasks 024/025")
    }
}
