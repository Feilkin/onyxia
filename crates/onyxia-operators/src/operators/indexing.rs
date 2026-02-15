//! Indexing and data access operators.

use onyxia_core::{InferenceCtx, Operator, PlanCtx, Result, Step, TensorShape};

/// Gather operator - selects elements from input along an axis using indices.
pub struct GatherOp;

impl Operator for GatherOp {
    fn name(&self) -> &str {
        "Gather"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for Gather - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for Gather - will be implemented in Tasks 024/025")
    }
}

/// Slice operator - extracts a slice from a tensor.
pub struct SliceOp;

impl Operator for SliceOp {
    fn name(&self) -> &str {
        "Slice"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for Slice - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for Slice - will be implemented in Tasks 024/025")
    }
}

/// ScatterND operator - writes updates to a tensor at specified indices.
pub struct ScatterNDOp;

impl Operator for ScatterNDOp {
    fn name(&self) -> &str {
        "ScatterND"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for ScatterND - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for ScatterND - will be implemented in Tasks 024/025")
    }
}

/// Range operator - generates a sequence of numbers.
pub struct RangeOp;

impl Operator for RangeOp {
    fn name(&self) -> &str {
        "Range"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for Range - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for Range - will be implemented in Tasks 024/025")
    }
}

/// Trilu operator - returns upper or lower triangular part of a matrix.
pub struct TriluOp;

impl Operator for TriluOp {
    fn name(&self) -> &str {
        "Trilu"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for Trilu - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for Trilu - will be implemented in Tasks 024/025")
    }
}
