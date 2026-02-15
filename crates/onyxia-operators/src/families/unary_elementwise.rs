//! Unary elementwise operator family.
//!
//! Covers: Cos, Sin, Sqrt, Neg, Tanh

use onyxia_core::{InferenceCtx, Operator, PlanCtx, Result, Step, TensorShape};

/// Unary elementwise operator family.
///
/// All unary elementwise operations share the same structure:
/// - Identity shape inference (output shape = input shape)
/// - Element-by-element computation for constant folding
/// - WGSL shader dispatch for GPU execution
///
/// The only differences are:
/// - Shader source code (which WGSL function to call)
/// - Fold function (which CPU operation to perform)
pub struct UnaryElementwiseOp {
    name: &'static str,
    shader_source: &'static str,
    fold_fn: fn(f32) -> f32,
}

impl UnaryElementwiseOp {
    /// Create a Cos operator.
    pub fn cos() -> Self {
        Self {
            name: "Cos",
            shader_source: include_str!("../../shaders/elementwise/cos.wgsl"),
            fold_fn: f32::cos,
        }
    }

    /// Create a Sin operator.
    pub fn sin() -> Self {
        Self {
            name: "Sin",
            shader_source: include_str!("../../shaders/elementwise/sin.wgsl"),
            fold_fn: f32::sin,
        }
    }

    /// Create a Sqrt operator.
    pub fn sqrt() -> Self {
        Self {
            name: "Sqrt",
            shader_source: include_str!("../../shaders/elementwise/sqrt.wgsl"),
            fold_fn: f32::sqrt,
        }
    }

    /// Create a Neg operator.
    pub fn neg() -> Self {
        Self {
            name: "Neg",
            shader_source: include_str!("../../shaders/elementwise/neg.wgsl"),
            fold_fn: |x| -x,
        }
    }

    /// Create a Tanh operator.
    pub fn tanh() -> Self {
        Self {
            name: "Tanh",
            shader_source: include_str!("../../shaders/activation/tanh.wgsl"),
            fold_fn: f32::tanh,
        }
    }
}

impl Operator for UnaryElementwiseOp {
    fn name(&self) -> &str {
        self.name
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        // Unary elementwise operations preserve input shape
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
        // 3. Create binding layout (1 read-only input + 1 output)
        // 4. Compile WGSL shader
        // 5. Emit dispatch step
        let _ = (ctx, self.shader_source, self.fold_fn);
        todo!(
            "Planning for {} - will be implemented in Tasks 024/025",
            self.name
        )
    }
}
