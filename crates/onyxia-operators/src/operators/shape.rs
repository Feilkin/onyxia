//! Shape manipulation operators.

use onyxia_core::{
    DataType, FoldCtx, InferenceCtx, Operator, PlanCtx, Result, Step, TensorShape, TensorValue,
};

/// Reshape operator - changes tensor shape without copying data.
///
/// Since GPU buffers are flat arrays, no data movement is needed in theory.
/// However, our runtime allocates separate buffers per tensor, so we emit
/// a CopyBuffer step. Future optimization: buffer aliasing to avoid copies.
pub struct ReshapeOp;

impl Operator for ReshapeOp {
    fn name(&self) -> &str {
        "Reshape"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        // Reshape has 2 inputs: data (input 0) and shape (input 1)
        if ctx.input_count() < 2 {
            return Err(onyxia_core::Error::ShapeInference(
                "Reshape requires 2 inputs: data and shape".to_string(),
            ));
        }

        // Get input data shape
        let data_shape = match ctx.input_shape(0)? {
            TensorShape::Static(dims) => dims,
            TensorShape::Symbolic(_) => {
                return Err(onyxia_core::Error::ShapeInference(
                    "Reshape data input has symbolic shape - should be resolved".to_string(),
                ));
            }
            TensorShape::Absent => {
                return Err(onyxia_core::Error::ShapeInference(
                    "Reshape data input is absent".to_string(),
                ));
            }
        };

        // Get the target shape from the second input value
        let Some(target_shape_val) = ctx.input_value(1) else {
            // Shape is not a constant - we can't infer the output shape
            return Err(onyxia_core::Error::ShapeInference(
                "Reshape shape input must be a constant".to_string(),
            ));
        };

        // Parse i64 shape from the value
        let target_shape = match target_shape_val {
            TensorValue::I64(v) => v.as_slice(),
            _ => {
                return Err(onyxia_core::Error::ShapeInference(
                    "Reshape shape input must be I64".to_string(),
                ));
            }
        };

        // Handle -1 dimension (infer from total size)
        let total_elements: usize = data_shape.iter().product();
        let mut output_shape = Vec::new();
        let mut infer_dim = None;
        let mut known_product: i64 = 1;

        for (idx, &dim) in target_shape.iter().enumerate() {
            if dim == -1 {
                if infer_dim.is_some() {
                    return Err(onyxia_core::Error::ShapeInference(
                        "Reshape can have at most one -1 dimension".to_string(),
                    ));
                }
                infer_dim = Some(idx);
                output_shape.push(0); // Placeholder
            } else if dim == 0 {
                // 0 means "copy from input shape"
                if idx < data_shape.len() {
                    output_shape.push(data_shape[idx]);
                    known_product *= data_shape[idx] as i64;
                } else {
                    return Err(onyxia_core::Error::ShapeInference(format!(
                        "Reshape: dimension 0 at index {} out of range",
                        idx
                    )));
                }
            } else if dim > 0 {
                output_shape.push(dim as usize);
                known_product *= dim;
            } else {
                return Err(onyxia_core::Error::ShapeInference(format!(
                    "Invalid reshape dimension: {}",
                    dim
                )));
            }
        }

        // Compute inferred dimension
        if let Some(idx) = infer_dim {
            let inferred = total_elements as i64 / known_product;
            if inferred * known_product != total_elements as i64 {
                return Err(onyxia_core::Error::ShapeInference(format!(
                    "Cannot reshape {} elements into {:?}",
                    total_elements, target_shape
                )));
            }
            output_shape[idx] = inferred as usize;
        }

        Ok(vec![TensorShape::Static(output_shape)])
    }

    fn try_fold(&self, ctx: &FoldCtx) -> Result<Vec<Option<TensorValue>>> {
        // Reshape doesn't change values, only shape metadata
        // Return the input value unchanged if available
        let value = ctx.input_value(0).cloned();
        Ok(vec![value])
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // Reshape: copy input 0 (data) to output 0
        // Input 1 (shape) is only used at plan time, not runtime

        let input_tensor = ctx.input_tensor(0)?;
        let shape = ctx.static_dims(&input_tensor.shape)?;
        let element_count: usize = shape.iter().product();
        let bytes = element_count * dtype_size(input_tensor.dtype);

        Ok(vec![Step::CopyBuffer {
            src: ctx.input(0)?,
            src_offset: 0,
            dst: ctx.output(0)?,
            dst_offset: 0,
            size: bytes as u64,
        }])
    }
}

/// Get the size in bytes of a data type.
fn dtype_size(dtype: DataType) -> usize {
    match dtype {
        DataType::F32 | DataType::I32 | DataType::U32 | DataType::Bool => 4,
        DataType::F16 => 2,
        DataType::I64 => 8,
        DataType::U8 | DataType::Q4 | DataType::Q8 => 1,
    }
}

/// Unsqueeze operator - adds dimensions of size 1.
///
/// Unsqueeze has 1 or 2 inputs depending on opset:
/// - Opset < 13: 1 input (data) + axes attribute
/// - Opset >= 13: 2 inputs (data, axes tensor)
pub struct UnsqueezeOp;

impl Operator for UnsqueezeOp {
    fn name(&self) -> &str {
        "Unsqueeze"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        if ctx.input_count() == 0 {
            return Err(onyxia_core::Error::ShapeInference(
                "Unsqueeze requires at least one input".to_string(),
            ));
        }

        // Get input data shape
        let data_shape = match ctx.input_shape(0)? {
            TensorShape::Static(dims) => dims,
            TensorShape::Symbolic(_) => {
                return Err(onyxia_core::Error::ShapeInference(
                    "Unsqueeze data input has symbolic shape".to_string(),
                ));
            }
            TensorShape::Absent => {
                return Err(onyxia_core::Error::ShapeInference(
                    "Unsqueeze data input is absent".to_string(),
                ));
            }
        };

        // Get axes - try attribute first (opset < 13), then second input (opset >= 13)
        let axes: Vec<i64> = if ctx.has_attr("axes") {
            // Opset < 13: axes from attribute
            ctx.attr_ints("axes").unwrap_or_else(|_| &[]).to_vec()
        } else if ctx.input_count() >= 2 {
            // Opset >= 13: axes from second input (must be constant)
            let Some(axes_val) = ctx.input_value(1) else {
                return Err(onyxia_core::Error::ShapeInference(
                    "Unsqueeze axes tensor must be a constant".to_string(),
                ));
            };

            match axes_val {
                TensorValue::I64(v) => v.clone(),
                _ => {
                    return Err(onyxia_core::Error::ShapeInference(
                        "Unsqueeze axes input must be I64".to_string(),
                    ));
                }
            }
        } else {
            return Err(onyxia_core::Error::ShapeInference(
                "Unsqueeze: no axes provided".to_string(),
            ));
        };

        // Compute output shape by inserting 1s at specified axes
        let output_rank = data_shape.len() + axes.len();
        let mut output_shape = Vec::new();

        // Convert negative axes to positive and sort
        let mut normalized_axes: Vec<usize> = axes
            .iter()
            .map(|&axis| {
                if axis < 0 {
                    (output_rank as i64 + axis) as usize
                } else {
                    axis as usize
                }
            })
            .collect();
        normalized_axes.sort_unstable();

        let mut data_idx = 0;
        let mut axes_idx = 0;

        for out_idx in 0..output_rank {
            if axes_idx < normalized_axes.len() && out_idx == normalized_axes[axes_idx] {
                // Insert a 1 at this position
                output_shape.push(1);
                axes_idx += 1;
            } else {
                // Copy from input shape
                if data_idx < data_shape.len() {
                    output_shape.push(data_shape[data_idx]);
                    data_idx += 1;
                }
            }
        }

        Ok(vec![TensorShape::Static(output_shape)])
    }

    fn try_fold(&self, ctx: &FoldCtx) -> Result<Vec<Option<TensorValue>>> {
        // Unsqueeze doesn't change values, only shape metadata
        let value = ctx.input_value(0).cloned();
        Ok(vec![value])
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // Unsqueeze: copy input 0 (data) to output 0
        // Axes are only used at compile time, not runtime

        let input_tensor = ctx.input_tensor(0)?;
        let shape = ctx.static_dims(&input_tensor.shape)?;
        let element_count: usize = shape.iter().product();
        let bytes = element_count * dtype_size(input_tensor.dtype);

        Ok(vec![Step::CopyBuffer {
            src: ctx.input(0)?,
            src_offset: 0,
            dst: ctx.output(0)?,
            dst_offset: 0,
            size: bytes as u64,
        }])
    }
}

/// Transpose operator - permutes tensor dimensions.
///
/// If perm is not specified, defaults to reversing all dimensions.
pub struct TransposeOp;

impl Operator for TransposeOp {
    fn name(&self) -> &str {
        "Transpose"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        if ctx.input_count() == 0 {
            return Err(onyxia_core::Error::ShapeInference(
                "Transpose requires at least one input".to_string(),
            ));
        }

        let input_dims = match ctx.input_shape(0)? {
            TensorShape::Static(dims) => dims,
            TensorShape::Symbolic(_) | TensorShape::Absent => {
                return Err(onyxia_core::Error::ShapeInference(
                    "Transpose requires static input shape".to_string(),
                ));
            }
        };

        let rank = input_dims.len();

        // Read perm attribute (optional)
        let perm: Vec<i64> = ctx
            .attr_ints("perm")
            .map(|p| p.to_vec())
            .unwrap_or_else(|_| (0..rank as i64).rev().collect());

        // Validate perm
        if perm.len() != rank {
            return Err(onyxia_core::Error::ShapeInference(format!(
                "Transpose perm length {} does not match input rank {}",
                perm.len(),
                rank
            )));
        }

        // Compute output shape by permuting input dimensions
        let output_dims: Vec<usize> = perm.iter().map(|&p| input_dims[p as usize]).collect();

        Ok(vec![TensorShape::Static(output_dims)])
    }

    fn try_fold(&self, ctx: &FoldCtx) -> Result<Vec<Option<TensorValue>>> {
        let Some(input) = ctx.input_value(0) else {
            return Ok(vec![None]);
        };

        let input_shape = match ctx.input_shape(0)? {
            TensorShape::Static(dims) => dims,
            _ => return Ok(vec![None]),
        };

        let rank = input_shape.len();
        let perm: Vec<i64> = ctx
            .attr_ints("perm")
            .map(|p| p.to_vec())
            .unwrap_or_else(|_| (0..rank as i64).rev().collect());

        // Helper function to transpose values
        fn transpose_values<T: Clone>(values: &[T], input_shape: &[usize], perm: &[i64]) -> Vec<T> {
            let rank = input_shape.len();
            let num_elements: usize = input_shape.iter().product();
            let output_shape: Vec<usize> = perm.iter().map(|&p| input_shape[p as usize]).collect();

            let mut input_strides = vec![1; rank];
            let mut output_strides = vec![1; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
                output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
            }

            let mut result = vec![values[0].clone(); num_elements];

            for out_idx in 0..num_elements {
                let mut out_coords = vec![0; rank];
                let mut temp = out_idx;
                for i in 0..rank {
                    out_coords[i] = temp / output_strides[i];
                    temp %= output_strides[i];
                }

                let mut in_coords = vec![0; rank];
                for i in 0..rank {
                    in_coords[perm[i] as usize] = out_coords[i];
                }

                let in_idx: usize = in_coords
                    .iter()
                    .zip(input_strides.iter())
                    .map(|(c, s)| c * s)
                    .sum();
                result[out_idx] = values[in_idx].clone();
            }

            result
        }

        let result = match input {
            TensorValue::F32(vals) => TensorValue::F32(transpose_values(vals, input_shape, &perm)),
            TensorValue::I64(vals) => TensorValue::I64(transpose_values(vals, input_shape, &perm)),
            TensorValue::I32(vals) => TensorValue::I32(transpose_values(vals, input_shape, &perm)),
            TensorValue::Bool(vals) => {
                TensorValue::Bool(transpose_values(vals, input_shape, &perm))
            }
            TensorValue::U8(vals) => TensorValue::U8(transpose_values(vals, input_shape, &perm)),
        };

        Ok(vec![Some(result)])
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let input_tensor = ctx.input_tensor(0)?;
        let input_shape = ctx.static_dims(&input_tensor.shape)?;
        let rank = input_shape.len();

        let output_tensor = ctx.output_tensor(0)?;
        let output_shape = ctx.static_dims(&output_tensor.shape)?;

        let perm: Vec<i64> = ctx
            .attr_ints("perm")
            .unwrap_or_else(|_| (0..rank as i64).rev().collect());

        let num_elements: usize = output_shape.iter().product();

        // Compute strides
        let mut input_strides = vec![1usize; rank];
        let mut output_strides = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
        }

        let workgroup_size: u32 = 256;
        let total_workgroups = (num_elements as u32 + workgroup_size - 1) / workgroup_size;

        const MAX_DISPATCH_DIM: u32 = 65535;
        let (workgroups_x, workgroups_y, workgroups_z) = if total_workgroups <= MAX_DISPATCH_DIM {
            (total_workgroups, 1, 1)
        } else {
            let workgroups_y = (total_workgroups + MAX_DISPATCH_DIM - 1) / MAX_DISPATCH_DIM;
            if workgroups_y <= MAX_DISPATCH_DIM {
                (MAX_DISPATCH_DIM, workgroups_y, 1)
            } else {
                let workgroups_z = (workgroups_y + MAX_DISPATCH_DIM - 1) / MAX_DISPATCH_DIM;
                (MAX_DISPATCH_DIM, MAX_DISPATCH_DIM, workgroups_z)
            }
        };

        let mut shader_defs = std::collections::HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), workgroup_size.to_string());

        let shader_index = ctx.compile_shader(
            "transpose",
            include_str!("../../shaders/indexing/transpose.wgsl"),
            &shader_defs,
        )?;

        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(rank as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(num_elements as u32).to_le_bytes());
        immediates_data.extend_from_slice(&workgroups_x.to_le_bytes());

        for i in 0..6 {
            let stride = if i < rank { input_strides[i] as u32 } else { 0 };
            immediates_data.extend_from_slice(&stride.to_le_bytes());
        }

        for i in 0..6 {
            let stride = if i < rank {
                output_strides[i] as u32
            } else {
                0
            };
            immediates_data.extend_from_slice(&stride.to_le_bytes());
        }

        for i in 0..6 {
            let p = if i < rank { perm[i] as u32 } else { 0 };
            immediates_data.extend_from_slice(&p.to_le_bytes());
        }

        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![
                onyxia_core::BindingDesc {
                    buffer: ctx.input(0)?,
                    read_only: true,
                },
                onyxia_core::BindingDesc {
                    buffer: ctx.output(0)?,
                    read_only: false,
                },
            ],
            workgroups: [workgroups_x, workgroups_y, workgroups_z],
            immediates: Some(immediates_data),
        }])
    }
}

/// Concat operator - concatenates tensors along a dimension.
///
/// axis=0 uses efficient buffer copies, other axes use compute shader.
pub struct ConcatOp;

impl Operator for ConcatOp {
    fn name(&self) -> &str {
        "Concat"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        if ctx.input_count() == 0 {
            return Err(onyxia_core::Error::ShapeInference(
                "Concat requires at least one input".to_string(),
            ));
        }

        let axis: i64 = ctx.attr_i64("axis").unwrap_or(0);

        let rank = match ctx.input_shape(0)? {
            TensorShape::Static(dims) => dims.len() as i64,
            _ => {
                return Err(onyxia_core::Error::ShapeInference(
                    "Cannot determine rank".to_string(),
                ));
            }
        };

        let normalized_axis = if axis < 0 { rank + axis } else { axis } as usize;

        if normalized_axis >= rank as usize {
            return Err(onyxia_core::Error::ShapeInference(format!(
                "Concat axis {} out of bounds for rank {}",
                axis, rank
            )));
        }

        let mut first_dims = Vec::new();
        let mut concat_dim_sum: usize = 0;

        for (i, _) in (0..ctx.input_count()).enumerate() {
            let dims = match ctx.input_shape(i)? {
                TensorShape::Static(dims) => dims,
                _ => {
                    return Err(onyxia_core::Error::ShapeInference(
                        "Non-static shape".to_string(),
                    ));
                }
            };

            if i == 0 {
                first_dims = dims.to_vec();
                concat_dim_sum = dims[normalized_axis];
            } else {
                if dims.len() != first_dims.len() {
                    return Err(onyxia_core::Error::ShapeInference(format!(
                        "Concat input {} rank mismatch",
                        i
                    )));
                }
                for (dim_idx, (d1, d2)) in first_dims.iter().zip(dims.iter()).enumerate() {
                    if dim_idx != normalized_axis && d1 != d2 {
                        return Err(onyxia_core::Error::ShapeInference(format!(
                            "Concat input {} dimension mismatch at dim {}",
                            i, dim_idx
                        )));
                    }
                }
                concat_dim_sum += dims[normalized_axis];
            }
        }

        let mut output_dims = first_dims;
        output_dims[normalized_axis] = concat_dim_sum;

        Ok(vec![TensorShape::Static(output_dims)])
    }

    fn try_fold(&self, ctx: &FoldCtx) -> Result<Vec<Option<TensorValue>>> {
        let axis: i64 = ctx.attr_i64("axis").unwrap_or(0);

        let rank = match ctx.input_shape(0)? {
            TensorShape::Static(dims) => dims.len() as i64,
            _ => return Ok(vec![None]),
        };

        let normalized_axis = if axis < 0 { rank + axis } else { axis };

        if normalized_axis != 0 {
            return Ok(vec![None]); // Only fold for axis=0
        }

        let mut all_values = Vec::new();
        for i in 0..ctx.input_count() {
            let Some(val) = ctx.input_value(i) else {
                return Ok(vec![None]);
            };
            all_values.push(val);
        }

        if all_values.is_empty() {
            return Ok(vec![None]);
        }

        let result = match all_values[0] {
            TensorValue::I64(_) => {
                let mut concat_vals = Vec::new();
                for val in all_values {
                    if let TensorValue::I64(v) = val {
                        concat_vals.extend_from_slice(v);
                    }
                }
                TensorValue::I64(concat_vals)
            }
            TensorValue::F32(_) => {
                let mut concat_vals = Vec::new();
                for val in all_values {
                    if let TensorValue::F32(v) = val {
                        concat_vals.extend_from_slice(v);
                    }
                }
                TensorValue::F32(concat_vals)
            }
            _ => return Ok(vec![None]),
        };

        Ok(vec![Some(result)])
    }

    fn plan(&self, _ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // TODO: Implement Concat GPU planning (needs shader and immediates encoding)
        Err(onyxia_core::Error::Planning(
            "Concat GPU planning not yet implemented".to_string(),
        ))
    }
}

/// Expand operator - broadcasts a tensor to a larger shape.
///
/// Expand replicates input along dimensions of size 1 to match target shape.
pub struct ExpandOp;

impl Operator for ExpandOp {
    fn name(&self) -> &str {
        "Expand"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        if ctx.input_count() < 2 {
            return Err(onyxia_core::Error::ShapeInference(
                "Expand requires 2 inputs: data and shape".to_string(),
            ));
        }

        let Some(target_shape_val) = ctx.input_value(1) else {
            return Err(onyxia_core::Error::ShapeInference(
                "Expand shape input must be a constant".to_string(),
            ));
        };

        let target_shape = match target_shape_val {
            TensorValue::I64(v) => v.as_slice(),
            _ => {
                return Err(onyxia_core::Error::ShapeInference(
                    "Expand shape input must be I64".to_string(),
                ));
            }
        };

        let mut output_dims = Vec::new();
        for (i, &target_dim) in target_shape.iter().enumerate() {
            if target_dim <= 0 {
                return Err(onyxia_core::Error::ShapeInference(format!(
                    "Expand target dimension {} invalid: {}",
                    i, target_dim
                )));
            }
            output_dims.push(target_dim as usize);
        }

        if let TensorShape::Static(data_shape) = ctx.input_shape(0)? {
            let input_rank = data_shape.len();
            let output_rank = output_dims.len();

            if output_rank < input_rank {
                return Err(onyxia_core::Error::ShapeInference(format!(
                    "Expand target rank {} < input rank {}",
                    output_rank, input_rank
                )));
            }

            let offset = output_rank - input_rank;
            for i in 0..input_rank {
                let input_dim = data_shape[i];
                let output_dim = output_dims[offset + i];
                if input_dim != 1 && input_dim != output_dim {
                    return Err(onyxia_core::Error::ShapeInference(format!(
                        "Expand: incompatible dimensions at {}: input={}, target={}",
                        offset + i,
                        input_dim,
                        output_dim
                    )));
                }
            }
        }

        Ok(vec![TensorShape::Static(output_dims)])
    }

    fn plan(&self, _ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // TODO: Implement Expand GPU planning
        Err(onyxia_core::Error::Planning(
            "Expand GPU planning not yet implemented".to_string(),
        ))
    }
}
