//! Normalization operators.

use onyxia_core::{InferenceCtx, Operator, PlanCtx, Result, Step, TensorShape};

/// RMS Normalization operator (SimplifiedLayerNormalization in ONNX).
pub struct RmsNormOp;

impl Operator for RmsNormOp {
    fn name(&self) -> &str {
        "SimplifiedLayerNormalization"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for RmsNorm - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for RmsNorm - will be implemented in Tasks 024/025")
    }
}
