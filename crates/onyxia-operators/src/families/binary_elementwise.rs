//! Binary elementwise operator family.
//!
//! Covers: Add, Sub, Mul, Div, Pow, Max

use onyxia_core::{
    BindingDesc, FoldCtx, InferenceCtx, Operator, PlanCtx, Result, Step, TensorShape,
};
use std::collections::HashMap;

use crate::helpers::infer_elementwise_broadcast;

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
        infer_elementwise_broadcast(ctx)
    }

    fn try_fold(&self, ctx: &FoldCtx) -> Result<Vec<Option<onyxia_core::TensorValue>>> {
        // Use the helper from FoldCtx to fold binary F32 operations
        ctx.binary_fold_f32(self.fold_fn)
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

        // Get input shapes for immediate data
        let input_a_tensor = ctx.input_tensor(0)?;
        let input_a_shape = ctx.static_dims(&input_a_tensor.shape)?;
        let a_size: u32 = input_a_shape.iter().product::<usize>() as u32;

        let input_b_tensor = ctx.input_tensor(1)?;
        let input_b_shape = ctx.static_dims(&input_b_tensor.shape)?;
        let b_size: u32 = input_b_shape.iter().product::<usize>() as u32;

        // Encode immediate data (must match ImmediateConstants struct in shader)
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(num_elements as u32).to_le_bytes());
        immediates_data.extend_from_slice(&a_size.to_le_bytes());
        immediates_data.extend_from_slice(&b_size.to_le_bytes());

        // Create dispatch step with bindings and immediates
        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0)?,
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(1)?,
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
