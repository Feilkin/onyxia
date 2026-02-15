//! Matrix multiplication operators.

use onyxia_core::{BindingDesc, InferenceCtx, Operator, PlanCtx, Result, Step, TensorShape};
use std::collections::HashMap;

/// F32 matrix multiplication operator.
///
/// Computes C = A × B where:
/// - A: [M, K]
/// - B: [K, N]
/// - C: [M, N]
///
/// Uses tiled algorithm with shared memory for efficiency.
pub struct MatMulF32Op;

impl Operator for MatMulF32Op {
    fn name(&self) -> &str {
        "MatMul"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        // MatMul: [M, K] × [K, N] -> [M, N]
        if ctx.input_count() < 2 {
            return Err(onyxia_core::Error::ShapeInference(
                "MatMul requires two inputs".to_string(),
            ));
        }

        // Extract static dimensions
        let a_dims = match ctx.input_shape(0)? {
            TensorShape::Static(dims) => dims,
            TensorShape::Symbolic(_) => {
                return Err(onyxia_core::Error::ShapeInference(
                    "Symbolic shapes should be resolved before shape inference".to_string(),
                ));
            }
            TensorShape::Absent => {
                return Err(onyxia_core::Error::ShapeInference(
                    "MatMul input A is absent".to_string(),
                ));
            }
        };

        let b_dims = match ctx.input_shape(1)? {
            TensorShape::Static(dims) => dims,
            TensorShape::Symbolic(_) => {
                return Err(onyxia_core::Error::ShapeInference(
                    "Symbolic shapes should be resolved before shape inference".to_string(),
                ));
            }
            TensorShape::Absent => {
                return Err(onyxia_core::Error::ShapeInference(
                    "MatMul input B is absent".to_string(),
                ));
            }
        };

        if a_dims.len() < 2 || b_dims.len() < 2 {
            return Err(onyxia_core::Error::ShapeInference(format!(
                "MatMul requires at least 2D tensors, got A: {:?}, B: {:?}",
                a_dims, b_dims
            )));
        }

        // Output shape: preserve batch dimensions from A, replace last two dims with [M, N]
        // A: [...batch..., M, K]  B: [K, N]  ->  Output: [...batch..., M, N]
        let m = a_dims[a_dims.len() - 2];
        let n = b_dims[b_dims.len() - 1];

        let mut output_dims = a_dims[..a_dims.len() - 2].to_vec();
        output_dims.push(m);
        output_dims.push(n);

        Ok(vec![TensorShape::Static(output_dims)])
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // Get input shapes
        let a_tensor = ctx.input_tensor(0)?;
        let a_shape = ctx.static_dims(&a_tensor.shape)?;
        let b_tensor = ctx.input_tensor(1)?;
        let b_shape = ctx.static_dims(&b_tensor.shape)?;

        // Validate shapes
        if a_shape.len() < 2 || b_shape.len() < 2 {
            return Err(onyxia_core::Error::Planning(format!(
                "MatMul expects 2D or higher dimensional inputs, got A: {:?}, B: {:?}",
                a_shape, b_shape
            )));
        }

        // Extract matrix dimensions (handle batch dimensions later if needed)
        // For now, assume 2D matrices
        let m = a_shape[a_shape.len() - 2];
        let k_a = a_shape[a_shape.len() - 1];
        let k_b = b_shape[b_shape.len() - 2];
        let n = b_shape[b_shape.len() - 1];

        // Validate K dimensions match
        if k_a != k_b {
            return Err(onyxia_core::Error::Planning(format!(
                "MatMul K dimensions must match, got A: [.., {}, {}], B: [.., {}, {}]",
                m, k_a, k_b, n
            )));
        }
        let k = k_a;

        // Tile sizes for workgroup computation
        let tile_m: u32 = 16;
        let tile_n: u32 = 16;
        let tile_k: u32 = 16;

        // Prepare shader definitions
        let mut shader_defs = HashMap::new();
        shader_defs.insert("TILE_M".to_string(), tile_m.to_string());
        shader_defs.insert("TILE_N".to_string(), tile_n.to_string());
        shader_defs.insert("TILE_K".to_string(), tile_k.to_string());

        // Compile shader
        let shader_index = ctx.compile_shader(
            "matmul_f32",
            include_str!("../../shaders/matmul/matmul_f32.wgsl"),
            &shader_defs,
        )?;

        // Calculate workgroup dimensions
        // Each workgroup computes a TILE_M × TILE_N tile of the output
        let workgroups_x = (n as u32 + tile_n - 1) / tile_n;
        let workgroups_y = (m as u32 + tile_m - 1) / tile_m;

        // Encode immediate data (must match ImmediateConstants struct in shader)
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(m as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(n as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(k as u32).to_le_bytes());

        // Create dispatch step with bindings and immediates
        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0)?, // matrix A
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(1)?, // matrix B
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(0)?, // matrix C
                    read_only: false,
                },
            ],
            workgroups: [workgroups_x, workgroups_y, 1],
            immediates: Some(immediates_data),
        }])
    }
}

/// Quantized (4-bit) matrix multiplication operator.
///
/// Performs matrix multiplication with quantized weights (typically 4-bit).
pub struct MatMulNBitsOp;

impl Operator for MatMulNBitsOp {
    fn name(&self) -> &str {
        "MatMulNBits"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        if ctx.input_count() == 0 {
            return Err(onyxia_core::Error::ShapeInference(
                "MatMulNBits requires at least one input (activations)".to_string(),
            ));
        }

        let a_dims = match ctx.input_shape(0)? {
            TensorShape::Static(dims) => dims,
            _ => {
                return Err(onyxia_core::Error::ShapeInference(
                    "MatMulNBits requires static input shape".to_string(),
                ));
            }
        };

        if a_dims.len() < 2 {
            return Err(onyxia_core::Error::ShapeInference(format!(
                "MatMulNBits requires at least 2D activation tensor, got: {:?}",
                a_dims
            )));
        }

        // Read N from attributes (output dimension)
        let n = ctx.attr_i64("N")? as usize;

        // Output shape: preserve batch dimensions from A, replace last two dims with [M, N]
        let m = a_dims[a_dims.len() - 2];

        let mut output_dims = a_dims[..a_dims.len() - 2].to_vec();
        output_dims.push(m);
        output_dims.push(n);

        Ok(vec![TensorShape::Static(output_dims)])
    }

    fn plan(&self, _ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // TODO: Implement MatMulNBits GPU planning
        Err(onyxia_core::Error::Planning(
            "MatMulNBits GPU planning not yet implemented".to_string(),
        ))
    }
}
