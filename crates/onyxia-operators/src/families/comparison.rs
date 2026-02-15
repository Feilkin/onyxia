//! Comparison operator family.
//!
//! Covers: Equal, Greater

use onyxia_core::{InferenceCtx, Operator, PlanCtx, Result, Step, TensorShape};

/// Comparison operator family.
///
/// All comparison operations share the same structure:
/// - NumPy-style broadcasting for shape inference
/// - Element-by-element comparison for constant folding
/// - WGSL shader dispatch for GPU execution
/// - Output dtype is always Bool
///
/// The only differences are:
/// - Shader source code (which comparison function to call)
/// - Fold function (which CPU comparison to perform)
pub struct ComparisonOp {
    name: &'static str,
    shader_source: &'static str,
    fold_fn: fn(f32, f32) -> bool,
}

impl ComparisonOp {
    /// Create an Equal operator.
    pub fn equal() -> Self {
        Self {
            name: "Equal",
            shader_source: include_str!("../../shaders/elementwise/equal.wgsl"),
            fold_fn: |a, b| a == b,
        }
    }

    /// Create a Greater operator.
    pub fn greater() -> Self {
        Self {
            name: "Greater",
            shader_source: include_str!("../../shaders/elementwise/greater.wgsl"),
            fold_fn: |a, b| a > b,
        }
    }
}

impl Operator for ComparisonOp {
    fn name(&self) -> &str {
        self.name
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        // Comparison operations use NumPy-style broadcasting
        // Output dtype is Bool
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
        // 1. Get output shape and compute workgroup sizing
        // 2. Encode immediates (shape metadata)
        // 3. Create binding layout (2 read-only inputs + 1 output)
        // 4. Compile WGSL shader
        // 5. Emit dispatch step
        let _ = (ctx, self.shader_source, self.fold_fn);
        todo!(
            "Planning for {} - will be implemented in Tasks 024/025",
            self.name
        )
    }
}
