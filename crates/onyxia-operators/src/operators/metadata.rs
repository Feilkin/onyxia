//! Metadata and constant operators.

use onyxia_core::{InferenceCtx, Operator, PlanCtx, Result, Step, TensorShape};

/// Constant tensor operator.
pub struct ConstantOp;

impl Operator for ConstantOp {
    fn name(&self) -> &str {
        "Constant"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for Constant - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for Constant - will be implemented in Tasks 024/025")
    }
}

/// ConstantOfShape operator - creates a tensor filled with a constant value.
pub struct ConstantOfShapeOp;

impl Operator for ConstantOfShapeOp {
    fn name(&self) -> &str {
        "ConstantOfShape"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for ConstantOfShape - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for ConstantOfShape - will be implemented in Tasks 024/025")
    }
}

/// Shape operator - extracts shape information from a tensor.
pub struct ShapeOp;

impl Operator for ShapeOp {
    fn name(&self) -> &str {
        "Shape"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for Shape - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for Shape - will be implemented in Tasks 024/025")
    }
}
