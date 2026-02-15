//! Matrix multiplication operators.

use onyxia_core::{InferenceCtx, Operator, PlanCtx, Result, Step, TensorShape};

/// F32 matrix multiplication operator.
pub struct MatMulF32Op;

impl Operator for MatMulF32Op {
    fn name(&self) -> &str {
        "MatMul"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for MatMul - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for MatMul - will be implemented in Tasks 024/025")
    }
}

/// Quantized (4-bit) matrix multiplication operator.
pub struct MatMulNBitsOp;

impl Operator for MatMulNBitsOp {
    fn name(&self) -> &str {
        "MatMulNBits"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for MatMulNBits - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for MatMulNBits - will be implemented in Tasks 024/025")
    }
}
