//! Binary elementwise operator family.
//!
//! Covers: Add, Sub, Mul, Div, Pow, Max

use onyxia_core::{InferenceCtx, Operator, PlanCtx, Result, Step, TensorShape};

/// Binary elementwise operator family.
///
/// All binary elementwise operations share the same structure:
/// - NumPy-style broadcasting for shape inference
/// - Element-by-element computation for constant folding
/// - WGSL shader dispatch for GPU execution
///
/// The only differences are:
/// - Shader source code (which WGSL function to call)
/// - Fold function (which CPU operation to perform)
pub struct BinaryElementwiseOp {
    name: &'static str,
    shader_source: &'static str,
    fold_fn: fn(f32, f32) -> f32,
}

impl BinaryElementwiseOp {
    /// Create an Add operator.
    pub fn add() -> Self {
        Self {
            name: "Add",
            shader_source: include_str!("../../shaders/elementwise/add.wgsl"),
            fold_fn: |a, b| a + b,
        }
    }

    /// Create a Sub operator.
    pub fn sub() -> Self {
        Self {
            name: "Sub",
            shader_source: include_str!("../../shaders/elementwise/sub.wgsl"),
            fold_fn: |a, b| a - b,
        }
    }

    /// Create a Mul operator.
    pub fn mul() -> Self {
        Self {
            name: "Mul",
            shader_source: include_str!("../../shaders/elementwise/mul.wgsl"),
            fold_fn: |a, b| a * b,
        }
    }

    /// Create a Div operator.
    pub fn div() -> Self {
        Self {
            name: "Div",
            shader_source: include_str!("../../shaders/elementwise/div.wgsl"),
            fold_fn: |a, b| a / b,
        }
    }

    /// Create a Pow operator.
    pub fn pow() -> Self {
        Self {
            name: "Pow",
            shader_source: include_str!("../../shaders/elementwise/pow.wgsl"),
            fold_fn: |a, b| a.powf(b),
        }
    }

    /// Create a Max operator.
    pub fn max() -> Self {
        Self {
            name: "Max",
            shader_source: include_str!("../../shaders/elementwise/max.wgsl"),
            fold_fn: |a, b| a.max(b),
        }
    }
}

impl Operator for BinaryElementwiseOp {
    fn name(&self) -> &str {
        self.name
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        // Binary elementwise operations use NumPy-style broadcasting
        // TODO: Implement full broadcasting logic in Tasks 024/025
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
