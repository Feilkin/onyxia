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

        // Prepare shape immediates for broadcasting
        let rank = output_shape.len();
        if rank == 0 || rank > 8 {
            return Err(Error::Shape(format!(
                "Where: unsupported output rank {} (must be 1..=8)",
                rank
            )));
        }

        // Left-align shapes into arrays of length `rank`. For inputs with fewer
        // dims we pad with leading 1s so they are right-aligned with broadcasting rules.
        let mut padded_output = Vec::with_capacity(8);
        let mut padded_cond = Vec::with_capacity(8);
        let mut padded_x = Vec::with_capacity(8);
        let mut padded_y = Vec::with_capacity(8);

        for i in 0..rank {
            padded_output.push(output_shape[i] as u32);
        }

        let pad_and_copy = |src: &[usize], rank: usize| -> Vec<u32> {
            let mut v = Vec::with_capacity(rank);
            let src_len = src.len();
            let leading = rank.saturating_sub(src_len);
            for _ in 0..leading {
                v.push(1u32);
            }
            for &d in src.iter() {
                v.push(d as u32);
            }
            v
        };

        padded_cond = pad_and_copy(&condition.shape, rank);
        padded_x = pad_and_copy(&x.shape, rank);
        padded_y = pad_and_copy(&y.shape, rank);

        // Total elements
        let num_elements_u32 = num_elements as u32;

        // Get or create compute pipeline
        let (pipeline, bind_group_layout) =
            ctx.get_or_create_pipeline(&self.label, &self.module, "main")?;

        // Create bind group
        let bind_group = ctx.create_bind_group(
            &bind_group_layout,
            &[&condition.buffer, &x.buffer, &y.buffer, &output.buffer],
        )?;

        // Encode immediates matching ImmediateConstants in shader
        // Layout:
        // output_rank: u32
        // total_elements: u32
        // padding: u32
        // output_shape: 8 * u32
        // cond_shape: 8 * u32
        // x_shape: 8 * u32
        // y_shape: 8 * u32
        let mut immediates = Vec::with_capacity(4 * (3 + 8 * 4));
        immediates.extend_from_slice(&(rank as u32).to_le_bytes());
        immediates.extend_from_slice(&num_elements_u32.to_le_bytes());
        immediates.extend_from_slice(&0u32.to_le_bytes()); // padding

        // Helper to write an array of length 8, filling trailing slots with 1s
        let write_array8 = |buf: &mut Vec<u8>, vals: &Vec<u32>| {
            // Write `rank` entries first, then fill remaining up to 8 with 1s
            for i in 0..8 {
                let v = if i < vals.len() { vals[i] } else { 1u32 };
                buf.extend_from_slice(&v.to_le_bytes());
            }
        };

        // Write shapes arrays
        write_array8(&mut immediates, &padded_output);
        write_array8(&mut immediates, &padded_cond);
        write_array8(&mut immediates, &padded_x);
        write_array8(&mut immediates, &padded_y);

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
