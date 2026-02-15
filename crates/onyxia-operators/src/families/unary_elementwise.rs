//! Unary elementwise operator family.
//!
//! Covers: Cos, Sin, Sqrt, Neg, Tanh

use onyxia_core::{
    BindingDesc, FoldCtx, InferenceCtx, Operator, PlanCtx, Result, Step, TensorShape,
};
use std::collections::HashMap;

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
        let shape = ctx.input_shape(0)?;
        Ok(vec![shape.clone()])
    }

    fn try_fold(&self, ctx: &FoldCtx) -> Result<Vec<Option<onyxia_core::TensorValue>>> {
        // Use the helper from FoldCtx to fold unary F32 operations
        ctx.unary_fold_f32(self.fold_fn)
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // Get output tensor and shape
        let output_tensor = ctx.output_tensor(0)?;
        let output_shape = ctx.static_dims(&output_tensor.shape)?;
        let num_elements: usize = output_shape.iter().product();

        // Configure workgroup size
        let workgroup_size: u32 = 256;
        let num_workgroups = (num_elements as u32 + workgroup_size - 1) / workgroup_size;

        // Prepare shader definitions
        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), workgroup_size.to_string());

        // Compile shader
        let shader_index = ctx.compile_shader(self.name, self.shader_source, &shader_defs)?;

        // Encode immediate data (must match ImmediateConstants struct in shader)
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(num_elements as u32).to_le_bytes());

        // Create dispatch step with bindings and immediates
        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0)?,
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(0)?,
                    read_only: false,
                },
            ],
            workgroups: [num_workgroups, 1, 1],
            immediates: Some(immediates_data),
        }])
    }
}
