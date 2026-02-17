//! Shape manipulation operators.

use onyxia_core::{
    CompileCtx, DataType, DispatchCtx, Error, OpDispatch, Operator, Result, RuntimeTensor,
};
use onyxia_onnx::AttributeValue;
use std::collections::HashMap;
use std::sync::Arc;

/// Shader source for the Concat operator.
const CONCAT_SHADER: &str = include_str!("../../shaders/shape/concat.wgsl");

/// Shader source for the Expand operator.
const EXPAND_SHADER: &str = include_str!("../../shaders/shape/expand.wgsl");

/// Shader source for the Transpose operator.
const TRANSPOSE_SHADER: &str = include_str!("../../shaders/shape/transpose.wgsl");

// ══════════════════════════════════════════════════════════════════════════════
// Reshape operator
// ══════════════════════════════════════════════════════════════════════════════

/// Reshape operator - changes tensor shape without copying data.
///
/// Since GPU buffers are flat arrays, no data movement is needed in theory.
/// However, our runtime allocates separate buffers per tensor, so we emit
/// a CopyBuffer step. Future optimization: buffer aliasing to avoid copies.
pub struct ReshapeOp;

/// Runtime dispatch for Reshape — reinterprets buffer with new shape.
struct ReshapeDispatch;

impl OpDispatch for ReshapeDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> onyxia_core::Result<Vec<RuntimeTensor>> {
        let data_tensor = &inputs[0];

        // Read the target shape from the second input.
        // For Reshape, the shape tensor is typically a small 1-D i64 tensor
        // containing the target dimensions. If it's a weight/constant, the
        // data is already on GPU — we need to read it back.
        let shape_tensor = &inputs[1];

        // Download shape values from GPU (small tensor, so roundtrip is okay)
        let shape_data = ctx.download_tensor(shape_tensor)?;
        let target_shape =
            parse_reshape_shape(&shape_data, shape_tensor.dtype, &data_tensor.shape)?;

        // Validate element count matches
        let input_elements: usize = data_tensor.shape.iter().product();
        let output_elements: usize = target_shape.iter().product();
        if input_elements != output_elements {
            return Err(Error::Shape(format!(
                "Reshape: input has {input_elements} elements but target shape \
                 {target_shape:?} has {output_elements} elements"
            )));
        }

        // Reshape is a zero-copy operation — same buffer, new shape
        Ok(vec![RuntimeTensor {
            buffer: Arc::clone(&data_tensor.buffer),
            shape: target_shape,
            dtype: data_tensor.dtype,
            size_bytes: data_tensor.size_bytes,
        }])
    }
}

/// Parse target shape from a shape tensor, handling -1 (infer) dimensions.
fn parse_reshape_shape(data: &[u8], dtype: DataType, input_shape: &[usize]) -> Result<Vec<usize>> {
    let raw_dims: Vec<i64> = match dtype {
        DataType::I64 => bytemuck::cast_slice(data).to_vec(),
        DataType::I32 => bytemuck::cast_slice::<u8, i32>(data)
            .iter()
            .map(|&v| v as i64)
            .collect(),
        _ => {
            return Err(Error::Shape(format!(
                "Reshape shape tensor has unsupported dtype: {dtype:?}"
            )));
        }
    };

    let input_elements: usize = input_shape.iter().product();
    let mut inferred_idx = None;
    let mut known_product: usize = 1;
    let mut result = Vec::with_capacity(raw_dims.len());

    for (i, &dim) in raw_dims.iter().enumerate() {
        if dim == -1 {
            if inferred_idx.is_some() {
                return Err(Error::Shape("Reshape: multiple -1 dimensions".into()));
            }
            inferred_idx = Some(i);
            result.push(0); // placeholder
        } else if dim == 0 {
            // 0 means "copy from input"
            let input_dim = input_shape.get(i).copied().unwrap_or(1);
            result.push(input_dim);
            known_product *= input_dim;
        } else {
            result.push(dim as usize);
            known_product *= dim as usize;
        }
    }

    if let Some(idx) = inferred_idx {
        result[idx] = input_elements / known_product;
    }

    Ok(result)
}

impl Operator for ReshapeOp {
    fn name(&self) -> &str {
        "Reshape"
    }

    fn create_dispatch(&self, _ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        // Reshape needs no pre-compiled shader — it's a zero-copy shape change
        Ok(Box::new(ReshapeDispatch))
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Concat operator
// ══════════════════════════════════════════════════════════════════════════════

/// Concat operator - concatenate tensors along a specified axis.
///
/// ONNX opset 13+:
/// - **Inputs**: inputs (variadic, T)
/// - **Outputs**: concat_result (T)
/// - **Attributes**: axis (int, required)
pub struct ConcatOp;

/// Runtime dispatch for Concat.
struct ConcatDispatch {
    /// Pre-compiled naga module for the shader.
    module: naga::Module,

    /// Axis to concatenate along.
    axis: i64,
}

impl OpDispatch for ConcatDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> onyxia_core::Result<Vec<RuntimeTensor>> {
        if inputs.is_empty() {
            return Err(Error::Runtime("Concat: no inputs provided".into()));
        }

        if inputs.len() > 4 {
            return Err(Error::Runtime(
                "Concat: supports up to 4 inputs only".into(),
            ));
        }

        // All inputs must have the same rank
        let rank = inputs[0].shape.len();
        for (i, tensor) in inputs.iter().enumerate().skip(1) {
            if tensor.shape.len() != rank {
                return Err(Error::Shape(format!(
                    "Concat: input {} has rank {} but input 0 has rank {}",
                    i,
                    tensor.shape.len(),
                    rank
                )));
            }
        }

        // Normalize axis (handle negative values)
        let axis = if self.axis < 0 {
            (rank as i64 + self.axis) as usize
        } else {
            self.axis as usize
        };

        if axis >= rank {
            return Err(Error::Shape(format!(
                "Concat: axis {} out of bounds for rank {}",
                self.axis, rank
            )));
        }

        // Compute output shape and validate input shapes
        let mut output_shape = inputs[0].shape.clone();
        let mut axis_size = inputs[0].shape[axis];

        for (i, tensor) in inputs.iter().enumerate().skip(1) {
            // Check all dimensions except axis match
            for (dim_idx, (&dim, &expected)) in
                tensor.shape.iter().zip(&inputs[0].shape).enumerate()
            {
                if dim_idx != axis && dim != expected {
                    return Err(Error::Shape(format!(
                        "Concat: dimension {} of input {} is {} but expected {}",
                        dim_idx, i, dim, expected
                    )));
                }
            }
            axis_size += tensor.shape[axis];
        }

        output_shape[axis] = axis_size;

        // Allocate output buffer
        let num_elements: usize = output_shape.iter().product();
        let output = ctx.create_output_tensor(&output_shape, inputs[0].dtype)?;

        // Encode immediates: output shape, axis, number of inputs, per-input axis sizes
        let mut immediates = Vec::new();

        // Rank (u32)
        immediates.extend_from_slice(&(rank as u32).to_le_bytes());

        // Axis (u32)
        immediates.extend_from_slice(&(axis as u32).to_le_bytes());

        // Number of inputs (u32)
        immediates.extend_from_slice(&(inputs.len() as u32).to_le_bytes());

        // Total output elements (u32)
        immediates.extend_from_slice(&(num_elements as u32).to_le_bytes());

        // Output shape (up to 8 dimensions)
        for dim in output_shape.iter().take(8) {
            immediates.extend_from_slice(&(*dim as u32).to_le_bytes());
        }
        for _ in output_shape.len()..8 {
            immediates.extend_from_slice(&0u32.to_le_bytes());
        }

        // Per-input axis sizes (up to 4 inputs)
        for tensor in inputs.iter().take(4) {
            immediates.extend_from_slice(&(tensor.shape[axis] as u32).to_le_bytes());
        }
        for _ in inputs.len()..4 {
            immediates.extend_from_slice(&0u32.to_le_bytes());
        }

        // Compute workgroup count
        let workgroup_size: u32 = 256;
        let num_workgroups = (num_elements as u32).div_ceil(workgroup_size);

        // Get or create pipeline
        let (pipeline, bind_group_layout) =
            ctx.get_or_create_pipeline("Concat", &self.module, "main")?;

        // Create a dummy buffer for unused input slots (shader expects 4 inputs + 1 output)
        // Use the output buffer size for simplicity (it's large enough to satisfy any binding requirements)
        let dummy_buffer = if inputs.len() < 4 {
            Some(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("concat_dummy_buffer"),
                size: output.size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        // Create bind group entries: 4 input slots (real or dummy) + 1 output
        let mut entries = Vec::new();
        for i in 0..4 {
            if i < inputs.len() {
                entries.push(wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: inputs[i].buffer.as_entire_binding(),
                });
            } else if let Some(ref dummy) = dummy_buffer {
                entries.push(wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: dummy.as_entire_binding(),
                });
            }
        }
        entries.push(wgpu::BindGroupEntry {
            binding: 4,
            resource: output.buffer.as_entire_binding(),
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("concat_bind_group"),
            layout: &bind_group_layout,
            entries: &entries,
        });

        // Dispatch compute shader
        ctx.dispatch_compute(
            &pipeline,
            &bind_group,
            [num_workgroups, 1, 1],
            Some(&immediates),
        )?;

        Ok(vec![output])
    }
}

impl Operator for ConcatOp {
    fn name(&self) -> &str {
        "Concat"
    }

    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        // Read axis attribute from node
        let axis = ctx.attr_i64("axis")?;

        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), "256".to_string());

        let module = ctx.compile_shader("Concat", CONCAT_SHADER, &shader_defs)?;

        Ok(Box::new(ConcatDispatch { module, axis }))
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Expand operator
// ══════════════════════════════════════════════════════════════════════════════

/// Expand operator - broadcast a tensor to a new shape.
///
/// ONNX opset 13+:
/// - **Inputs**: input (T), shape (tensor(int64))
/// - **Outputs**: output (T)
///
/// Broadcasting rules:
/// - Input shape must be broadcastable to target shape
/// - Dimensions can only go from 1 → N (singleton expansion)
pub struct ExpandOp;

/// Runtime dispatch for Expand.
struct ExpandDispatch {
    /// Pre-compiled naga module for the shader.
    module: naga::Module,
}

impl OpDispatch for ExpandDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> onyxia_core::Result<Vec<RuntimeTensor>> {
        let data_tensor = &inputs[0];

        // Read target shape from second input (similar to Reshape)
        let shape_tensor = &inputs[1];
        let shape_data = ctx.download_tensor(shape_tensor)?;
        let target_shape = parse_shape_tensor(&shape_data, shape_tensor.dtype)?;

        // Validate broadcastability
        let input_shape = &data_tensor.shape;
        let input_rank = input_shape.len();
        let output_rank = target_shape.len();

        if input_rank > output_rank {
            return Err(Error::Shape(format!(
                "Expand: input rank {} exceeds target rank {}",
                input_rank, output_rank
            )));
        }

        // Check broadcasting rules: aligned to the right
        for (i, &input_dim) in input_shape.iter().rev().enumerate() {
            let target_dim = target_shape[output_rank - 1 - i];
            if input_dim != 1 && input_dim != target_dim {
                return Err(Error::Shape(format!(
                    "Expand: dimension {} of input ({}) is not broadcastable to target ({})",
                    input_rank - 1 - i,
                    input_dim,
                    target_dim
                )));
            }
        }

        // Allocate output buffer
        let num_elements: usize = target_shape.iter().product();
        let output = ctx.create_output_tensor(&target_shape, data_tensor.dtype)?;

        // Encode immediates: input shape, output shape, rank
        let mut immediates = Vec::new();

        // Output rank (u32)
        immediates.extend_from_slice(&(output_rank as u32).to_le_bytes());

        // Input rank (u32)
        immediates.extend_from_slice(&(input_rank as u32).to_le_bytes());

        // Total output elements (u32)
        immediates.extend_from_slice(&(num_elements as u32).to_le_bytes());

        // Padding (u32)
        immediates.extend_from_slice(&0u32.to_le_bytes());

        // Output shape (up to 8 dimensions)
        for &dim in target_shape.iter().take(8) {
            immediates.extend_from_slice(&(dim as u32).to_le_bytes());
        }
        for _ in output_rank..8 {
            immediates.extend_from_slice(&0u32.to_le_bytes());
        }

        // Input shape (up to 8 dimensions, right-aligned)
        for _ in 0..(output_rank - input_rank) {
            immediates.extend_from_slice(&1u32.to_le_bytes());
        }
        for &dim in input_shape.iter().take(8) {
            immediates.extend_from_slice(&(dim as u32).to_le_bytes());
        }
        for _ in output_rank..8 {
            immediates.extend_from_slice(&0u32.to_le_bytes());
        }

        // Compute workgroup count
        let workgroup_size: u32 = 256;
        let num_workgroups = (num_elements as u32).div_ceil(workgroup_size);

        // Get or create pipeline
        let (pipeline, bind_group_layout) =
            ctx.get_or_create_pipeline("Expand", &self.module, "main")?;

        // Create bind group
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("expand_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data_tensor.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader
        ctx.dispatch_compute(
            &pipeline,
            &bind_group,
            [num_workgroups, 1, 1],
            Some(&immediates),
        )?;

        Ok(vec![output])
    }
}

impl Operator for ExpandOp {
    fn name(&self) -> &str {
        "Expand"
    }

    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), "256".to_string());

        let module = ctx.compile_shader("Expand", EXPAND_SHADER, &shader_defs)?;

        Ok(Box::new(ExpandDispatch { module }))
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Transpose operator
// ══════════════════════════════════════════════════════════════════════════════

/// Transpose operator - permute tensor dimensions.
///
/// ONNX opset 21+:
/// - **Inputs**: data (T)
/// - **Outputs**: transposed (T)
/// - **Attributes**: perm (list of ints, optional - default: reverse dimensions)
pub struct TransposeOp;

/// Runtime dispatch for Transpose.
struct TransposeDispatch {
    /// Pre-compiled naga module for the shader.
    module: naga::Module,

    /// Permutation vector (or None for default reverse).
    perm: Option<Vec<i64>>,
}

impl OpDispatch for TransposeDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> onyxia_core::Result<Vec<RuntimeTensor>> {
        let input = &inputs[0];
        let rank = input.shape.len();

        // Determine permutation: use provided perm or default to reverse
        let perm: Vec<usize> = if let Some(ref p) = self.perm {
            if p.len() != rank {
                return Err(Error::Shape(format!(
                    "Transpose: perm has length {} but input has rank {}",
                    p.len(),
                    rank
                )));
            }
            p.iter().map(|&i| i as usize).collect()
        } else {
            // Default: reverse dimensions
            (0..rank).rev().collect()
        };

        // Validate permutation
        let mut seen = vec![false; rank];
        for &p in &perm {
            if p >= rank {
                return Err(Error::Shape(format!(
                    "Transpose: permutation index {} out of bounds for rank {}",
                    p, rank
                )));
            }
            if seen[p] {
                return Err(Error::Shape(format!(
                    "Transpose: duplicate index {} in permutation",
                    p
                )));
            }
            seen[p] = true;
        }

        // Compute output shape
        let output_shape: Vec<usize> = perm.iter().map(|&i| input.shape[i]).collect();
        let num_elements: usize = output_shape.iter().product();

        // Allocate output buffer
        let output = ctx.create_output_tensor(&output_shape, input.dtype)?;

        // Encode immediates: input shape, permutation, rank
        let mut immediates = Vec::new();

        // Rank (u32)
        immediates.extend_from_slice(&(rank as u32).to_le_bytes());

        // Total elements (u32)
        immediates.extend_from_slice(&(num_elements as u32).to_le_bytes());

        // Input shape (up to 8 dimensions)
        for &dim in input.shape.iter().take(8) {
            immediates.extend_from_slice(&(dim as u32).to_le_bytes());
        }
        for _ in rank..8 {
            immediates.extend_from_slice(&0u32.to_le_bytes());
        }

        // Permutation (up to 8 dimensions)
        for &p in perm.iter().take(8) {
            immediates.extend_from_slice(&(p as u32).to_le_bytes());
        }
        for _ in rank..8 {
            immediates.extend_from_slice(&0u32.to_le_bytes());
        }

        // Compute workgroup count
        let workgroup_size: u32 = 256;
        let num_workgroups = (num_elements as u32).div_ceil(workgroup_size);

        // Get or create pipeline
        let (pipeline, bind_group_layout) =
            ctx.get_or_create_pipeline("Transpose", &self.module, "main")?;

        // Create bind group
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transpose_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader
        ctx.dispatch_compute(
            &pipeline,
            &bind_group,
            [num_workgroups, 1, 1],
            Some(&immediates),
        )?;

        Ok(vec![output])
    }
}

impl Operator for TransposeOp {
    fn name(&self) -> &str {
        "Transpose"
    }

    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        // Read perm attribute from node (optional)
        let perm = ctx.attr("perm").and_then(|v| match v {
            AttributeValue::Ints(ints) => Some(ints.clone()),
            _ => None,
        });

        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), "256".to_string());

        let module = ctx.compile_shader("Transpose", TRANSPOSE_SHADER, &shader_defs)?;

        Ok(Box::new(TransposeDispatch { module, perm }))
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Unsqueeze operator
// ══════════════════════════════════════════════════════════════════════════════

/// Unsqueeze operator - insert singleton dimensions at specified axes.
///
/// ONNX opset 21+:
/// - **Inputs**: data (T), axes (tensor(int64))
/// - **Outputs**: expanded (T)
///
/// This is a zero-copy operation (like Reshape) — just shape metadata change.
pub struct UnsqueezeOp;

/// Runtime dispatch for Unsqueeze — zero-copy shape change.
struct UnsqueezeDispatch;

impl OpDispatch for UnsqueezeDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> onyxia_core::Result<Vec<RuntimeTensor>> {
        let data_tensor = &inputs[0];

        // Read axes from second input
        let axes_tensor = &inputs[1];
        let axes_data = ctx.download_tensor(axes_tensor)?;
        let mut axes = parse_axes_tensor(&axes_data, axes_tensor.dtype)?;

        // Sort axes for processing
        axes.sort_unstable();

        // Validate axes
        let output_rank = data_tensor.shape.len() + axes.len();
        for &axis in &axes {
            if axis < 0 || axis as usize >= output_rank {
                return Err(Error::Shape(format!(
                    "Unsqueeze: axis {} out of bounds for output rank {}",
                    axis, output_rank
                )));
            }
        }

        // Check for duplicates
        for i in 1..axes.len() {
            if axes[i] == axes[i - 1] {
                return Err(Error::Shape(format!(
                    "Unsqueeze: duplicate axis {}",
                    axes[i]
                )));
            }
        }

        // Construct output shape by inserting 1s at specified axes
        let mut output_shape = Vec::with_capacity(output_rank);
        let mut input_idx = 0;

        for output_idx in 0..output_rank {
            if axes.contains(&(output_idx as i64)) {
                // Insert singleton dimension
                output_shape.push(1);
            } else {
                // Copy dimension from input
                if input_idx < data_tensor.shape.len() {
                    output_shape.push(data_tensor.shape[input_idx]);
                    input_idx += 1;
                } else {
                    return Err(Error::Shape("Unsqueeze: axis positions invalid".into()));
                }
            }
        }

        // Unsqueeze is a zero-copy operation — same buffer, new shape
        Ok(vec![RuntimeTensor {
            buffer: Arc::clone(&data_tensor.buffer),
            shape: output_shape,
            dtype: data_tensor.dtype,
            size_bytes: data_tensor.size_bytes,
        }])
    }
}

impl Operator for UnsqueezeOp {
    fn name(&self) -> &str {
        "Unsqueeze"
    }

    fn create_dispatch(&self, _ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        // Unsqueeze needs no pre-compiled shader — it's a zero-copy shape change
        Ok(Box::new(UnsqueezeDispatch))
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Helper functions
// ══════════════════════════════════════════════════════════════════════════════

/// Parse a shape tensor (1-D i64 or i32) into a Vec<usize>.
fn parse_shape_tensor(data: &[u8], dtype: DataType) -> Result<Vec<usize>> {
    let raw_dims: Vec<i64> = match dtype {
        DataType::I64 => bytemuck::cast_slice(data).to_vec(),
        DataType::I32 => bytemuck::cast_slice::<u8, i32>(data)
            .iter()
            .map(|&v| v as i64)
            .collect(),
        _ => {
            return Err(Error::Shape(format!(
                "Shape tensor has unsupported dtype: {dtype:?}"
            )));
        }
    };

    // Convert to usize, checking for negative values
    raw_dims
        .into_iter()
        .map(|dim| {
            if dim < 0 {
                Err(Error::Shape(format!(
                    "Negative dimension in shape: {}",
                    dim
                )))
            } else {
                Ok(dim as usize)
            }
        })
        .collect()
}

/// Parse an axes tensor (1-D i64 or i32) into a Vec<i64>.
fn parse_axes_tensor(data: &[u8], dtype: DataType) -> Result<Vec<i64>> {
    match dtype {
        DataType::I64 => Ok(bytemuck::cast_slice(data).to_vec()),
        DataType::I32 => Ok(bytemuck::cast_slice::<u8, i32>(data)
            .iter()
            .map(|&v| v as i64)
            .collect()),
        _ => Err(Error::Shape(format!(
            "Axes tensor has unsupported dtype: {dtype:?}"
        ))),
    }
}
