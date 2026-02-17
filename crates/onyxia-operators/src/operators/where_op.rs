//! Where operator - conditional element selection.
//!
//! The Where operator selects elements from two input tensors based on a
//! boolean condition, with support for NumPy-style broadcasting.

use onyxia_core::{
    CompileCtx, DataType, DispatchCtx, Error, OpDispatch, Operator, Result, RuntimeTensor,
};
use std::collections::HashMap;

/// Shader source for the Where operator.
const WHERE_SHADER: &str = include_str!("../../shaders/where.wgsl");

/// Where operator - conditional element selection.
///
/// **ONNX Specification (opset 16):**
/// - **Inputs:**
///   - condition (B) - Boolean mask
///   - X (T) - Values when condition is true
///   - Y (T) - Values when condition is false
/// - **Outputs:**
///   - output (T) - Selected values
/// - **Type constraints:**
///   - B = bool
///   - T = all types
///
/// **Behavior:**
/// - `output[i] = X[i] if condition[i] else Y[i]`
/// - Supports NumPy broadcasting across all inputs
/// - Output shape is the broadcast result of all three input shapes
pub struct WhereOp;

impl Operator for WhereOp {
    fn name(&self) -> &str {
        "Where"
    }

    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), "256".to_string());

        let module = ctx.compile_shader("Where", WHERE_SHADER, &shader_defs)?;

        Ok(Box::new(WhereDispatch {
            module,
            label: "Where".to_string(),
        }))
    }
}

/// Runtime dispatch for Where operation.
struct WhereDispatch {
    /// Pre-compiled naga module.
    module: naga::Module,

    /// Label for pipeline caching.
    label: String,
}

impl OpDispatch for WhereDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> Result<Vec<RuntimeTensor>> {
        if inputs.len() != 3 {
            return Err(Error::Runtime(format!(
                "Where: expected 3 inputs, got {}",
                inputs.len()
            )));
        }

        let condition = &inputs[0];
        let x = &inputs[1];
        let y = &inputs[2];

        // Condition must be bool
        if condition.dtype != DataType::Bool {
            return Err(Error::Runtime(format!(
                "Where: condition must be bool, got {:?}",
                condition.dtype
            )));
        }

        // X and Y must have the same dtype
        if x.dtype != y.dtype {
            return Err(Error::Runtime(format!(
                "Where: X and Y must have same dtype (got {:?} and {:?})",
                x.dtype, y.dtype
            )));
        }

        // Compute broadcast output shape
        let output_shape =
            broadcast_shape(&broadcast_shape(&condition.shape, &x.shape)?, &y.shape)?;

        let num_elements: usize = output_shape.iter().product();

        // Create output tensor
        let output = ctx.create_output_tensor(&output_shape, x.dtype)?;

        // Compute broadcast strides for each input
        let cond_size: usize = condition.shape.iter().product();
        let x_size: usize = x.shape.iter().product();
        let y_size: usize = y.shape.iter().product();

        // Get or create compute pipeline
        let (pipeline, bind_group_layout) =
            ctx.get_or_create_pipeline(&self.label, &self.module, "main")?;

        // Create bind group
        let bind_group = ctx.create_bind_group(
            &bind_group_layout,
            &[&condition.buffer, &x.buffer, &y.buffer, &output.buffer],
        )?;

        // Encode immediates
        let mut immediates = Vec::new();
        immediates.extend_from_slice(&(num_elements as u32).to_le_bytes());
        immediates.extend_from_slice(&(cond_size as u32).to_le_bytes());
        immediates.extend_from_slice(&(x_size as u32).to_le_bytes());
        immediates.extend_from_slice(&(y_size as u32).to_le_bytes());

        // Dispatch compute shader
        let workgroups_x = num_elements.div_ceil(256) as u32;
        ctx.dispatch_compute(
            &pipeline,
            &bind_group,
            [workgroups_x, 1, 1],
            Some(&immediates),
        )?;

        Ok(vec![output])
    }
}

/// Compute the broadcast shape of two shapes.
///
/// NumPy broadcasting rules:
/// - Dimensions are aligned from the right
/// - Each dimension must be either equal or one of them must be 1
/// - The output dimension is the maximum of the two
fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let a_dim = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let b_dim = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        if a_dim == b_dim || a_dim == 1 {
            result.push(b_dim);
        } else if b_dim == 1 {
            result.push(a_dim);
        } else {
            return Err(Error::Shape(format!(
                "Cannot broadcast shapes {:?} and {:?}",
                a, b
            )));
        }
    }

    result.reverse();
    Ok(result)
}
