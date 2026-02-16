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
/// - Fold functions (which CPU operations to perform, per type)
///
/// Per the ONNX spec:
/// - Cos/Sin/Sqrt/Tanh: T → T where T ∈ {float16, float, double, bfloat16}
/// - Neg: T → T where T ∈ {float, int8, int16, int32, int64, float16, double, bfloat16}
pub struct UnaryElementwiseOp {
    name: &'static str,
    shader_source: &'static str,
    fold_fn_f32: fn(f32) -> f32,
    fold_fn_i64: Option<fn(i64) -> i64>,
    fold_fn_i32: Option<fn(i32) -> i32>,
}

impl UnaryElementwiseOp {
    /// Create a Cos operator.
    pub fn cos() -> Self {
        Self {
            name: "Cos",
            shader_source: include_str!("../../shaders/elementwise/cos.wgsl"),
            fold_fn_f32: f32::cos,
            fold_fn_i64: None,
            fold_fn_i32: None,
        }
    }

    /// Create a Sin operator.
    pub fn sin() -> Self {
        Self {
            name: "Sin",
            shader_source: include_str!("../../shaders/elementwise/sin.wgsl"),
            fold_fn_f32: f32::sin,
            fold_fn_i64: None,
            fold_fn_i32: None,
        }
    }

    /// Create a Sqrt operator.
    pub fn sqrt() -> Self {
        Self {
            name: "Sqrt",
            shader_source: include_str!("../../shaders/elementwise/sqrt.wgsl"),
            fold_fn_f32: f32::sqrt,
            fold_fn_i64: None,
            fold_fn_i32: None,
        }
    }

    /// Create a Neg operator.
    pub fn neg() -> Self {
        Self {
            name: "Neg",
            shader_source: include_str!("../../shaders/elementwise/neg.wgsl"),
            fold_fn_f32: |x| -x,
            fold_fn_i64: Some(|x| -x),
            fold_fn_i32: Some(|x| -x),
        }
    }

    /// Create a Tanh operator.
    pub fn tanh() -> Self {
        Self {
            name: "Tanh",
            shader_source: include_str!("../../shaders/activation/tanh.wgsl"),
            fold_fn_f32: f32::tanh,
            fold_fn_i64: None,
            fold_fn_i32: None,
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
        // Try f32 folding first
        let result = ctx.unary_fold_f32(self.fold_fn_f32)?;
        if result.first().is_some_and(|v| v.is_some()) {
            return Ok(result);
        }

        // Try i64 folding (e.g. Neg supports integer types)
        if let Some(fold_fn) = self.fold_fn_i64 {
            let val = ctx.input_value(0);
            if let Some(val) = val {
                if let Some(input) = val.as_i64() {
                    let result_data: Vec<i64> = input.iter().map(|&x| fold_fn(x)).collect();
                    return Ok(vec![Some(onyxia_core::TensorValue::new(
                        onyxia_core::TensorData::I64(result_data),
                        val.shape.clone(),
                        onyxia_core::DataType::I64,
                    ))]);
                }
            }
        }

        // Try i32 folding
        if let Some(fold_fn) = self.fold_fn_i32 {
            let val = ctx.input_value(0);
            if let Some(val) = val {
                if let Some(input) = val.as_i32() {
                    let result_data: Vec<i32> = input.iter().map(|&x| fold_fn(x)).collect();
                    return Ok(vec![Some(onyxia_core::TensorValue::new(
                        onyxia_core::TensorData::I32(result_data),
                        val.shape.clone(),
                        onyxia_core::DataType::I32,
                    ))]);
                }
            }
        }

        Ok(vec![None])
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // Get output tensor and shape
        let output_tensor = ctx.output_tensor(0)?;
        let output_shape = ctx.static_dims(&output_tensor.shape)?;
        let num_elements: usize = output_shape.iter().product();

        // Configure workgroup size
        let workgroup_size: u32 = 256;
        let num_workgroups = (num_elements as u32).div_ceil(workgroup_size);

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
