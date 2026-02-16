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
/// - Fold functions (which CPU operations to perform, per type)
///
/// Per the ONNX spec, Add/Sub/Mul/Div/Max support T → T where
/// T ∈ {uint8..uint64, int8..int64, float16, float, double, bfloat16}.
/// Pow has (T, T1) → T with a wider exponent constraint.
pub struct BinaryElementwiseOp {
    name: &'static str,
    shader_source: &'static str,
    fold_fn_f32: fn(f32, f32) -> f32,
    fold_fn_i64: Option<fn(i64, i64) -> i64>,
    fold_fn_i32: Option<fn(i32, i32) -> i32>,
}

impl BinaryElementwiseOp {
    /// Create an Add operator.
    pub fn add() -> Self {
        Self {
            name: "Add",
            shader_source: include_str!("../../shaders/elementwise/add.wgsl"),
            fold_fn_f32: |a, b| a + b,
            fold_fn_i64: Some(|a, b| a + b),
            fold_fn_i32: Some(|a, b| a + b),
        }
    }

    /// Create a Sub operator.
    pub fn sub() -> Self {
        Self {
            name: "Sub",
            shader_source: include_str!("../../shaders/elementwise/sub.wgsl"),
            fold_fn_f32: |a, b| a - b,
            fold_fn_i64: Some(|a, b| a - b),
            fold_fn_i32: Some(|a, b| a - b),
        }
    }

    /// Create a Mul operator.
    pub fn mul() -> Self {
        Self {
            name: "Mul",
            shader_source: include_str!("../../shaders/elementwise/mul.wgsl"),
            fold_fn_f32: |a, b| a * b,
            fold_fn_i64: Some(|a, b| a * b),
            fold_fn_i32: Some(|a, b| a * b),
        }
    }

    /// Create a Div operator.
    pub fn div() -> Self {
        Self {
            name: "Div",
            shader_source: include_str!("../../shaders/elementwise/div.wgsl"),
            fold_fn_f32: |a, b| a / b,
            fold_fn_i64: Some(|a, b| if b != 0 { a / b } else { 0 }),
            fold_fn_i32: Some(|a, b| if b != 0 { a / b } else { 0 }),
        }
    }

    /// Create a Pow operator.
    pub fn pow() -> Self {
        Self {
            name: "Pow",
            shader_source: include_str!("../../shaders/elementwise/pow.wgsl"),
            fold_fn_f32: |a, b| a.powf(b),
            fold_fn_i64: Some(|a, b| (a as f64).powi(b as i32) as i64),
            fold_fn_i32: Some(|a, b| (a as f64).powi(b) as i32),
        }
    }

    /// Create a Max operator.
    pub fn max() -> Self {
        Self {
            name: "Max",
            shader_source: include_str!("../../shaders/elementwise/max.wgsl"),
            fold_fn_f32: |a, b| a.max(b),
            fold_fn_i64: Some(|a, b| a.max(b)),
            fold_fn_i32: Some(|a, b| a.max(b)),
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
        // Try f32 folding first
        let result = ctx.binary_fold_f32(self.fold_fn_f32)?;
        if result.first().is_some_and(|v| v.is_some()) {
            return Ok(result);
        }

        // Try i64 folding
        if let Some(fold_fn) = self.fold_fn_i64 {
            let result = ctx.binary_fold_i64(fold_fn)?;
            if result.first().is_some_and(|v| v.is_some()) {
                return Ok(result);
            }
        }

        // Try i32 folding
        if let Some(fold_fn) = self.fold_fn_i32 {
            let result = ctx.binary_fold_i32(fold_fn)?;
            if result.first().is_some_and(|v| v.is_some()) {
                return Ok(result);
            }
        }

        Ok(vec![None])
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let num_inputs = ctx.input_count();

        // Handle variadic inputs for operators that support them (Max, Add, Mul)
        if num_inputs > 2 && (self.name == "Max" || self.name == "Add" || self.name == "Mul") {
            return self.plan_variadic(ctx);
        }

        // Standard binary operation
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

impl BinaryElementwiseOp {
    /// Plan execution for variadic inputs (more than 2 inputs).
    ///
    /// Chains multiple binary operations: op(op(a, b), c) for 3 inputs, etc.
    /// Uses scratch buffers for intermediate results.
    fn plan_variadic(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let num_inputs = ctx.input_count();
        let mut steps = Vec::new();

        // Get output shape and dtype (save them to avoid borrow issues later)
        let output_shape = {
            let output_tensor = ctx.output_tensor(0)?;
            ctx.static_dims(&output_tensor.shape)?.to_vec()
        };
        let num_elements: usize = output_shape.iter().product();
        let element_size = {
            let output_tensor = ctx.output_tensor(0)?;
            output_tensor.dtype.size()
        };
        let buffer_size = (num_elements * element_size) as u64;

        // Configure workgroup size
        let workgroup_size: u32 = 256;
        let num_workgroups = (num_elements as u32).div_ceil(workgroup_size);

        // Prepare shader definitions
        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), workgroup_size.to_string());

        // Compile shader
        let shader_index = ctx.compile_shader(self.name, self.shader_source, &shader_defs)?;

        // Process inputs pairwise: first op(input[0], input[1]), then chain the rest
        for i in 0..(num_inputs - 1) {
            let input_a = if i == 0 {
                ctx.input(0)?
            } else {
                // Use the scratch buffer from the previous iteration
                onyxia_core::BufferRef::Scratch(i - 1)
            };

            let input_b = ctx.input(i + 1)?;

            let output = if i == num_inputs - 2 {
                // Last iteration: write to the actual output
                ctx.output(0)?
            } else {
                // Intermediate iteration: write to a scratch buffer
                ctx.alloc_scratch(buffer_size, format!("{}_temp_{}", self.name, i))
            };

            // Get input shapes for immediate data
            let a_size: u32 = if i == 0 {
                let input_a_tensor = ctx.input_tensor(0)?;
                let input_a_shape = ctx.static_dims(&input_a_tensor.shape)?;
                input_a_shape.iter().product::<usize>() as u32
            } else {
                // For intermediate results, use the output shape
                num_elements as u32
            };

            let input_b_tensor = ctx.input_tensor(i + 1)?;
            let input_b_shape = ctx.static_dims(&input_b_tensor.shape)?;
            let b_size: u32 = input_b_shape.iter().product::<usize>() as u32;

            // Encode immediate data
            let mut immediates_data = Vec::new();
            immediates_data.extend_from_slice(&(num_elements as u32).to_le_bytes());
            immediates_data.extend_from_slice(&a_size.to_le_bytes());
            immediates_data.extend_from_slice(&b_size.to_le_bytes());

            // Create dispatch step
            steps.push(Step::Dispatch {
                shader_index,
                bindings: vec![
                    BindingDesc {
                        buffer: input_a,
                        read_only: true,
                    },
                    BindingDesc {
                        buffer: input_b,
                        read_only: true,
                    },
                    BindingDesc {
                        buffer: output,
                        read_only: false,
                    },
                ],
                workgroups: [num_workgroups, 1, 1],
                immediates: Some(immediates_data),
            });
        }

        Ok(steps)
    }
}
