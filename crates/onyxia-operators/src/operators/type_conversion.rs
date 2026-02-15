//! Type conversion operators.

use onyxia_core::{InferenceCtx, Operator, PlanCtx, Result, Step, TensorShape};

/// Cast operator - converts tensor elements from one data type to another.
pub struct CastOp;

impl Operator for CastOp {
    fn name(&self) -> &str {
        "Cast"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for Cast - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for Cast - will be implemented in Tasks 024/025")
    }
}
