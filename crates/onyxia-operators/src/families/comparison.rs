//! Comparison operator family.
//!
//! Covers: Equal, Greater

use onyxia_core::{
    BindingDesc, DataType, FoldCtx, InferenceCtx, Operator, PlanCtx, Result, Step, TensorData,
    TensorShape, TensorValue,
};
use std::collections::HashMap;

use crate::helpers::infer_elementwise_broadcast;

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
    fold_fn_f32: fn(f32, f32) -> bool,
    fold_fn_i64: fn(i64, i64) -> bool,
    fold_fn_i32: fn(i32, i32) -> bool,
}

impl ComparisonOp {
    /// Create an Equal operator.
    pub fn equal() -> Self {
        Self {
            name: "Equal",
            shader_source: include_str!("../../shaders/elementwise/equal.wgsl"),
            fold_fn_f32: |a, b| a == b,
            fold_fn_i64: |a, b| a == b,
            fold_fn_i32: |a, b| a == b,
        }
    }

    /// Create a Greater operator.
    pub fn greater() -> Self {
        Self {
            name: "Greater",
            shader_source: include_str!("../../shaders/elementwise/greater.wgsl"),
            fold_fn_f32: |a, b| a > b,
            fold_fn_i64: |a, b| a > b,
            fold_fn_i32: |a, b| a > b,
        }
    }
}

impl Operator for ComparisonOp {
    fn name(&self) -> &str {
        self.name
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        // Comparison operations use NumPy-style broadcasting
        infer_elementwise_broadcast(ctx)
    }

    fn try_fold(&self, ctx: &FoldCtx) -> Result<Vec<Option<TensorValue>>> {
        // Try to constant-fold if both inputs are known
        if !ctx.all_inputs_have_values() {
            return Ok(vec![None]);
        }

        let a = ctx.input_value(0);
        let b = ctx.input_value(1);

        // Only fold values with same shape for simplicity
        let result = match (a, b) {
            (Some(a_val), Some(b_val)) if a_val.len() == b_val.len() => {
                match (&a_val.data, &b_val.data) {
                    (TensorData::F32(a_vals), TensorData::F32(b_vals)) => {
                        let result: Vec<bool> = a_vals
                            .iter()
                            .zip(b_vals.iter())
                            .map(|(&a, &b)| (self.fold_fn_f32)(a, b))
                            .collect();
                        Some(TensorValue::new(
                            TensorData::Bool(result),
                            a_val.shape.clone(),
                            DataType::Bool,
                        ))
                    }
                    (TensorData::I64(a_vals), TensorData::I64(b_vals)) => {
                        let result: Vec<bool> = a_vals
                            .iter()
                            .zip(b_vals.iter())
                            .map(|(&a, &b)| (self.fold_fn_i64)(a, b))
                            .collect();
                        Some(TensorValue::new(
                            TensorData::Bool(result),
                            a_val.shape.clone(),
                            DataType::Bool,
                        ))
                    }
                    (TensorData::I32(a_vals), TensorData::I32(b_vals)) => {
                        let result: Vec<bool> = a_vals
                            .iter()
                            .zip(b_vals.iter())
                            .map(|(&a, &b)| (self.fold_fn_i32)(a, b))
                            .collect();
                        Some(TensorValue::new(
                            TensorData::Bool(result),
                            a_val.shape.clone(),
                            DataType::Bool,
                        ))
                    }
                    _ => None,
                }
            }
            _ => None,
        };

        Ok(vec![result])
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
