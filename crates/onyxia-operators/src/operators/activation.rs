//! Activation operators.

use onyxia_core::{InferenceCtx, Operator, PlanCtx, Result, Step, TensorShape};

/// GELU (Gaussian Error Linear Unit) activation operator.
pub struct GeluOp;

impl Operator for GeluOp {
    fn name(&self) -> &str {
        "Gelu"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for Gelu - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for Gelu - will be implemented in Tasks 024/025")
    }
}

/// Softmax activation operator.
pub struct SoftmaxOp;

impl Operator for SoftmaxOp {
    fn name(&self) -> &str {
        "Softmax"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for Softmax - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for Softmax - will be implemented in Tasks 024/025")
    }
}
