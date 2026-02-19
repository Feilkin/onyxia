//! MatMulNBits operator - Quantized matrix multiplication with N-bit weights.
//!
//! This operator performs matrix multiplication where the right-hand-side matrix (weights)
//! is quantized to N bits (2-8 bits). It fuses two operations:
//! 1. Linear dequantization of quantized weights using per-block scale and zero-point
//! 2. Matrix multiplication between input and dequantized weights
//!
//! The weight matrix is quantized block-wise along the K dimension with configurable block size.

use onyxia_core::{
    CompileCtx, DataType, DispatchCtx, Error, OpDispatch, Operator, Result, RuntimeTensor,
};
use std::collections::HashMap;

/// Shader source for the MatMulNBits operator.
const MATMUL_NBITS_SHADER: &str = include_str!("../../shaders/matmul_nbits.wgsl");

/// MatMulNBits operator - Quantized matrix multiplication with N-bit weights.
///
/// Computs Y = A × dequantized(B) + bias (optional) where:
/// - A: [..., M, K] - Input tensor (float)
/// - B: [N, k_blocks, blob_size] - Packed quantized weights (uint8)
/// - scales: [N, k_blocks] - Per-block scaling factors (float)
/// - zero_points: [N, k_blocks] or [N, ceil(k_blocks * bits / 8)] - Per-block zero points (optional)
/// - bias: [N] - Bias to add to output (optional)
/// - Y: [..., M, N] - Output tensor (float)
///
/// The dequantization formula is:
/// dequantized_weight = (quantized_weight - zero_point) * scale
///
/// Weights are quantized block-wise along the K dimension.
pub struct MatMulNBitsOp;

/// Runtime dispatch for MatMulNBits.
struct MatMulNBitsDispatch {
    /// Pre-compiled naga module for the shader.
    module: naga::Module,
    /// Operator configuration.
    k: usize,
    n: usize,
    bits: u32,
    block_size: usize,
}

impl OpDispatch for MatMulNBitsDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> Result<Vec<RuntimeTensor>> {
        // Parse inputs
        if inputs.len() < 3 || inputs.len() > 6 {
            return Err(Error::Compilation(format!(
                "MatMulNBits: expected 3-6 inputs, got {}",
                inputs.len()
            )));
        }

        let a = &inputs[0]; // Input tensor
        let b = &inputs[1]; // Packed quantized weights
        let scales = &inputs[2]; // Per-block scaling factors
        let zero_points = inputs.get(3); // Optional zero points
        let _g_idx = inputs.get(4); // Deprecated group index (ignored)
        let bias = inputs.get(5); // Optional bias

        // Validate input A
        if a.shape.is_empty() {
            return Err(Error::Shape(
                "MatMulNBits: input A must have at least 1 dimension".to_string(),
            ));
        }

        // Validate input A data type
        if a.dtype != DataType::F32 {
            return Err(Error::Compilation(format!(
                "MatMulNBits: input A must be float32, got {:?}",
                a.dtype
            )));
        }

        // Extract dimensions from input A: [..., M, K]
        let k_a = a.shape[a.shape.len() - 1];
        let m = if a.shape.len() > 1 {
            a.shape[a.shape.len() - 2]
        } else {
            1
        };

        // Validate K dimension matches
        if k_a != self.k {
            return Err(Error::Shape(format!(
                "MatMulNBits: input A has K={}, but operator expects K={}",
                k_a, self.k
            )));
        }

        // Compute k_blocks
        let k_blocks = self.k.div_ceil(self.block_size);
        let blob_size = (self.block_size * self.bits as usize).div_ceil(8);

        // Validate input B shape: [N, k_blocks, blob_size]
        if b.shape.len() != 3
            || b.shape[0] != self.n
            || b.shape[1] != k_blocks
            || b.shape[2] != blob_size
        {
            return Err(Error::Shape(format!(
                "MatMulNBits: input B has shape {:?}, expected [{}, {}, {}]",
                b.shape, self.n, k_blocks, blob_size
            )));
        }

        // Validate input B data type
        if b.dtype != DataType::U8 {
            return Err(Error::Compilation(format!(
                "MatMulNBits: input B must be uint8, got {:?}",
                b.dtype
            )));
        }

        // Validate and reshape scales: accept [N, k_blocks] or [N * k_blocks]
        let expected_scale_elems = self.n * k_blocks;
        let scale_elems: usize = scales.shape.iter().product();

        if scale_elems != expected_scale_elems {
            return Err(Error::Shape(format!(
                "MatMulNBits: scales has {} elements, expected {}",
                scale_elems, expected_scale_elems
            )));
        }

        // Validate scales data type matches input A
        if scales.dtype != a.dtype {
            return Err(Error::Compilation(format!(
                "MatMulNBits: scales must have the same type as input A ({:?}), got {:?}",
                a.dtype, scales.dtype
            )));
        }

        // Reshape scales to [N, k_blocks] if it's 1D
        let scales_reshaped = if scales.shape.len() == 1 {
            RuntimeTensor {
                buffer: scales.buffer.clone(),
                shape: vec![self.n, k_blocks],
                dtype: scales.dtype,
                size_bytes: scales.size_bytes,
            }
        } else {
            scales.clone()
        };

        // Validate and reshape zero_points if provided
        let has_zero_points = zero_points.is_some();
        let zero_points_packed;
        let zero_points_reshaped = if let Some(zp) = zero_points {
            // Check if packed (uint8) or unpacked (same type as A)
            if zp.dtype == DataType::U8 {
                // Packed format: [N, ceil(k_blocks * bits / 8)] or [N * packed_size]
                let packed_size = (k_blocks * self.bits as usize).div_ceil(8);
                let expected_zp_elems = self.n * packed_size;
                let zp_elems: usize = zp.shape.iter().product();

                if zp_elems != expected_zp_elems {
                    return Err(Error::Shape(format!(
                        "MatMulNBits: packed zero_points has {} elements, expected {}",
                        zp_elems, expected_zp_elems
                    )));
                }

                zero_points_packed = true;

                // Reshape to [N, packed_size] if it's 1D
                if zp.shape.len() == 1 {
                    Some(RuntimeTensor {
                        buffer: zp.buffer.clone(),
                        shape: vec![self.n, packed_size],
                        dtype: zp.dtype,
                        size_bytes: zp.size_bytes,
                    })
                } else {
                    Some(zp.clone())
                }
            } else if zp.dtype == a.dtype {
                // Unpacked format: [N, k_blocks] or [N * k_blocks]
                let expected_zp_elems = self.n * k_blocks;
                let zp_elems: usize = zp.shape.iter().product();

                if zp_elems != expected_zp_elems {
                    return Err(Error::Shape(format!(
                        "MatMulNBits: unpacked zero_points has {} elements, expected {}",
                        zp_elems, expected_zp_elems
                    )));
                }

                zero_points_packed = false;

                // Reshape to [N, k_blocks] if it's 1D
                if zp.shape.len() == 1 {
                    Some(RuntimeTensor {
                        buffer: zp.buffer.clone(),
                        shape: vec![self.n, k_blocks],
                        dtype: zp.dtype,
                        size_bytes: zp.size_bytes,
                    })
                } else {
                    Some(zp.clone())
                }
            } else {
                return Err(Error::Compilation(format!(
                    "MatMulNBits: zero_points must be uint8 or {:?}, got {:?}",
                    a.dtype, zp.dtype
                )));
            }
        } else {
            zero_points_packed = false;
            None
        };

        // Validate bias if provided
        if let Some(b_tensor) = bias {
            if b_tensor.shape.len() != 1 || b_tensor.shape[0] != self.n {
                return Err(Error::Shape(format!(
                    "MatMulNBits: bias has shape {:?}, expected [{}]",
                    b_tensor.shape, self.n
                )));
            }
            if b_tensor.dtype != a.dtype {
                return Err(Error::Compilation(format!(
                    "MatMulNBits: bias must have the same type as input A ({:?}), got {:?}",
                    a.dtype, b_tensor.dtype
                )));
            }
        }

        // Compute batch dimensions
        let batch_dims = if a.shape.len() > 2 {
            &a.shape[..a.shape.len() - 2]
        } else {
            &[]
        };
        let batch_size: usize = if batch_dims.is_empty() {
            1
        } else {
            batch_dims.iter().product()
        };

        // Build output shape: [...batch..., M, N]
        let mut output_shape = batch_dims.to_vec();
        output_shape.push(m);
        output_shape.push(self.n);

        // Allocate output buffer
        let output = ctx.create_output_tensor(&output_shape, a.dtype)?;

        // Encode immediates (must match ImmediateConstants struct in shader)
        let mut immediates = Vec::with_capacity(64);
        immediates.extend_from_slice(&(m as u32).to_le_bytes());
        immediates.extend_from_slice(&(self.n as u32).to_le_bytes());
        immediates.extend_from_slice(&(self.k as u32).to_le_bytes());
        immediates.extend_from_slice(&(batch_size as u32).to_le_bytes());
        immediates.extend_from_slice(&(self.bits as u32).to_le_bytes());
        immediates.extend_from_slice(&(self.block_size as u32).to_le_bytes());
        immediates.extend_from_slice(&(k_blocks as u32).to_le_bytes());
        immediates.extend_from_slice(&(blob_size as u32).to_le_bytes());
        immediates.extend_from_slice(&(has_zero_points as u32).to_le_bytes());
        immediates.extend_from_slice(&(zero_points_packed as u32).to_le_bytes());
        immediates.extend_from_slice(&(bias.is_some() as u32).to_le_bytes());
        immediates.extend_from_slice(&0u32.to_le_bytes()); // Padding for alignment

        // Compute workgroups: (N/16, M/16, batch_size)
        const TILE_SIZE: u32 = 16;
        let workgroups_x = (self.n as u32).div_ceil(TILE_SIZE);
        let workgroups_y = (m as u32).div_ceil(TILE_SIZE);
        let workgroups_z = batch_size as u32;

        // Get or create pipeline
        let (pipeline, bind_group_layout) =
            ctx.get_or_create_pipeline("MatMulNBits", &self.module, "main")?;

        // Create dummy tensors if needed (must stay alive for bind group creation)
        let dummy_zp = if zero_points_reshaped.is_none() {
            Some(ctx.create_output_tensor(&[1, 1], DataType::F32)?)
        } else {
            None
        };

        let dummy_bias = if bias.is_none() {
            Some(ctx.create_output_tensor(&[1], DataType::F32)?)
        } else {
            None
        };

        // Create bind group
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: scales_reshaped.buffer.as_entire_binding(),
            },
        ];

        // Add zero_points binding
        if let Some(zp) = &zero_points_reshaped {
            entries.push(wgpu::BindGroupEntry {
                binding: 3,
                resource: zp.buffer.as_entire_binding(),
            });
        } else {
            entries.push(wgpu::BindGroupEntry {
                binding: 3,
                resource: dummy_zp.as_ref().unwrap().buffer.as_entire_binding(),
            });
        }

        // Add bias binding
        if let Some(b_tensor) = bias {
            entries.push(wgpu::BindGroupEntry {
                binding: 4,
                resource: b_tensor.buffer.as_entire_binding(),
            });
        } else {
            entries.push(wgpu::BindGroupEntry {
                binding: 4,
                resource: dummy_bias.as_ref().unwrap().buffer.as_entire_binding(),
            });
        }

        entries.push(wgpu::BindGroupEntry {
            binding: 5,
            resource: output.buffer.as_entire_binding(),
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_nbits_bind_group"),
            layout: &bind_group_layout,
            entries: &entries,
        });

        // Dispatch compute shader
        ctx.dispatch_compute(
            &pipeline,
            &bind_group,
            [workgroups_x, workgroups_y, workgroups_z],
            Some(&immediates),
        )?;

        Ok(vec![output])
    }
}

impl Operator for MatMulNBitsOp {
    fn name(&self) -> &str {
        "com.microsoft::MatMulNBits"
    }

    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        // Read required attributes
        let k = match ctx.attr("K") {
            Some(onyxia_onnx::AttributeValue::Int(v)) => *v as usize,
            _ => {
                return Err(Error::Attribute(
                    "MatMulNBits: missing or invalid 'K' attribute".into(),
                ));
            }
        };

        let n = match ctx.attr("N") {
            Some(onyxia_onnx::AttributeValue::Int(v)) => *v as usize,
            _ => {
                return Err(Error::Attribute(
                    "MatMulNBits: missing or invalid 'N' attribute".into(),
                ));
            }
        };

        let bits = match ctx.attr("bits") {
            Some(onyxia_onnx::AttributeValue::Int(v)) => *v as u32,
            _ => {
                return Err(Error::Attribute(
                    "MatMulNBits: missing or invalid 'bits' attribute".into(),
                ));
            }
        };

        if !(2..=8).contains(&bits) {
            return Err(Error::Attribute(format!(
                "MatMulNBits: 'bits' must be in range 2-8, got {}",
                bits
            )));
        }

        let block_size = match ctx.attr("block_size") {
            Some(onyxia_onnx::AttributeValue::Int(v)) => *v as usize,
            _ => {
                return Err(Error::Attribute(
                    "MatMulNBits: missing or invalid 'block_size' attribute".into(),
                ));
            }
        };

        // Validate block_size is power of 2 and >= 16
        if block_size < 16 || !block_size.is_power_of_two() {
            return Err(Error::Attribute(format!(
                "MatMulNBits: 'block_size' must be a power of 2 and >= 16, got {}",
                block_size
            )));
        }

        // Compile shader
        let shader_defs = HashMap::new();
        let module = ctx.compile_shader("MatMulNBits", MATMUL_NBITS_SHADER, &shader_defs)?;

        Ok(Box::new(MatMulNBitsDispatch {
            module,
            k,
            n,
            bits,
            block_size,
        }))
    }
}
