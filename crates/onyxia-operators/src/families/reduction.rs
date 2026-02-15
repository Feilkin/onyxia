//! Reduction operator family.
//!
//! Covers: ReduceSum, ReduceMean

use onyxia_core::{InferenceCtx, Operator, PlanCtx, Result, Step, TensorShape};

/// Reduction operator family.
///
/// All reduction operations share the same structure:
/// - Axis-based shape inference (reduce dimensions specified by axes attribute)
/// - Reduction computation for constant folding
/// - WGSL shader dispatch for GPU execution
///
/// The only differences are:
/// - Shader source code (sum vs mean computation)
/// - Fold function (sum vs mean)
pub struct ReductionOp {
    name: &'static str,
    shader_source: &'static str,
    fold_fn: fn(&[f32]) -> f32,
}

impl ReductionOp {
    /// Create a ReduceSum operator.
    pub fn reduce_sum() -> Self {
        Self {
            name: "ReduceSum",
            shader_source: include_str!("../../shaders/reduction/reducesum.wgsl"),
            fold_fn: |values| values.iter().sum(),
        }
    }

    /// Create a ReduceMean operator.
    pub fn reduce_mean() -> Self {
        Self {
            name: "ReduceMean",
            shader_source: include_str!("../../shaders/reduction/reducemean.wgsl"),
            fold_fn: |values| {
                let sum: f32 = values.iter().sum();
                sum / values.len() as f32
            },
        }
    }
}

impl Operator for ReductionOp {
    fn name(&self) -> &str {
        self.name
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        // Reduction operations reduce dimensions specified by axes attribute
        // keepdims attribute controls whether reduced dimensions are kept (size 1) or removed
        // TODO: Implement in Tasks 024/025
        let _ = (ctx, self.shader_source, self.fold_fn);
        todo!(
            "Shape inference for {} - will be implemented in Tasks 024/025",
            self.name
        )
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // TODO: Implement planning logic in Tasks 024/025
        // This will:
        // 1. Get reduction axes and output shape
        // 2. Compute workgroup sizing for reduction
        // 3. Encode immediates (shape metadata, axes)
        // 4. Create binding layout (1 read-only input + 1 output)
        // 5. Compile WGSL shader
        // 6. Emit dispatch step(s) (may need multiple passes for large reductions)
        let _ = (ctx, self.shader_source, self.fold_fn);
        todo!(
            "Planning for {} - will be implemented in Tasks 024/025",
            self.name
        )
    }
}
