//! Trilu operator - extract upper or lower triangular part of matrices.
//!
//! The Trilu operator returns the upper or lower triangular part of a matrix
//! (or batch of matrices), with elements outside the triangle set to zero.

use onyxia_core::{
    CompileCtx, DataType, DispatchCtx, Error, OpDispatch, Operator, Result, RuntimeTensor,
};
use onyxia_onnx::AttributeValue;
use std::collections::HashMap;

/// Shader source for the Trilu operator.
const TRILU_SHADER: &str = include_str!("../../shaders/trilu.wgsl");

/// Trilu operator - extract upper or lower triangular part of matrices.
///
/// **ONNX Specification (opset 14):**
/// - **Inputs:**
///   - input (T) - Input tensor (>=2D)
///   - k (optional, tensor(int64)) - Diagonal offset (default: 0)
/// - **Outputs:**
///   - output (T) - Output tensor (same shape as input)
/// - **Attributes:**
///   - upper (int, default=1) - 1 for upper triangular, 0 for lower
/// - **Type constraints:** T = all numeric types, bool, string
///
/// **Behavior:**
/// - Extract upper/lower triangular part of last 2 dimensions
/// - Elements outside triangle are set to 0
/// - k controls diagonal offset:
///   - k=0: Main diagonal
///   - k>0: Above main diagonal (more elements kept for upper)
///   - k<0: Below main diagonal (more elements kept for lower)
/// - Applies independently to each batch
pub struct TriluOp;

impl Operator for TriluOp {
    fn name(&self) -> &str {
        "Trilu"
    }

    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        // Read upper attribute (default=1, meaning upper triangular)
        let upper = match ctx.attr("upper") {
            Some(AttributeValue::Int(v)) => *v != 0,
            _ => true, // default is upper triangular
        };

        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), "256".to_string());

        let module = ctx.compile_shader("Trilu", TRILU_SHADER, &shader_defs)?;

        Ok(Box::new(TriluDispatch {
            module,
            label: "Trilu".to_string(),
            upper,
        }))
    }
}

/// Runtime dispatch for Trilu operation.
struct TriluDispatch {
    /// Pre-compiled naga module.
    module: naga::Module,

    /// Label for pipeline caching.
    label: String,

    /// Whether to extract upper (true) or lower (false) triangular part.
    upper: bool,
}

impl OpDispatch for TriluDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> Result<Vec<RuntimeTensor>> {
        if inputs.is_empty() || inputs.len() > 2 {
            return Err(Error::Runtime(format!(
                "Trilu: expected 1 or 2 inputs, got {}",
                inputs.len()
            )));
        }

        let input = &inputs[0];

        // Input must be at least 2D
        if input.shape.len() < 2 {
            return Err(Error::Shape(format!(
                "Trilu: input must be at least 2D, got rank {}",
                input.shape.len()
            )));
        }

        // Read optional k parameter (diagonal offset)
        let k: i32 = if inputs.len() > 1 {
            let k_tensor = &inputs[1];

            // k must be a scalar
            if k_tensor.shape.iter().product::<usize>() != 1 {
                return Err(Error::Shape("Trilu: k must be a scalar".to_string()));
            }

            // Download k value
            let k_data = ctx.download_tensor(k_tensor)?;

            match k_tensor.dtype {
                DataType::I64 => i64::from_le_bytes(k_data[0..8].try_into().unwrap()) as i32,
                DataType::I32 => i32::from_le_bytes(k_data[0..4].try_into().unwrap()),
                _ => {
                    return Err(Error::Runtime(format!(
                        "Trilu: k must be int64 or int32, got {:?}",
                        k_tensor.dtype
                    )));
                }
            }
        } else {
            0 // default k = 0 (main diagonal)
        };

        // Extract matrix dimensions (last 2 dimensions)
        let rows = input.shape[input.shape.len() - 2];
        let cols = input.shape[input.shape.len() - 1];

        // Total number of elements
        let num_elements: usize = input.shape.iter().product();

        // Create output tensor (same shape as input)
        let output = ctx.create_output_tensor(&input.shape, input.dtype)?;

        // Get or create compute pipeline
        let (pipeline, bind_group_layout) =
            ctx.get_or_create_pipeline(&self.label, &self.module, "main")?;

        // Create bind group
        let bind_group =
            ctx.create_bind_group(&bind_group_layout, &[&input.buffer, &output.buffer])?;

        // Encode immediates
        let mut immediates = Vec::new();
        immediates.extend_from_slice(&(num_elements as u32).to_le_bytes());
        immediates.extend_from_slice(&(rows as u32).to_le_bytes());
        immediates.extend_from_slice(&(cols as u32).to_le_bytes());
        immediates.extend_from_slice(&k.to_le_bytes());
        immediates.extend_from_slice(&(if self.upper { 1u32 } else { 0u32 }).to_le_bytes());

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
