//! Shape manipulation operators.

use onyxia_core::{
    DataType, FoldCtx, InferenceCtx, Operator, PlanCtx, Result, Step, TensorData, TensorShape,
    TensorValue,
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
            return Err(ctx.shape_error("Reshape requires 2 inputs: data and shape"));
        }

        // Get input data shape
        let data_shape = match ctx.input_shape(0)? {
            TensorShape::Static(dims) => dims,
            TensorShape::Symbolic(_) => {
                return Err(
                    ctx.shape_error("Reshape data input has symbolic shape - should be resolved")
                );
            }
            TensorShape::Absent => {
                return Err(ctx.shape_error("Reshape data input is absent"));
            }
            TensorShape::Unknown => {
                return Err(ctx.shape_error("Reshape data input has unknown shape"));
            }
        };

        // Get the target shape from the second input value
        let Some(target_shape_val) = ctx.input_value(1) else {
            // Shape is not a constant - we can't infer the output shape
            return Err(ctx.shape_error("Reshape shape input must be a constant"));
        };

        // Parse i64 shape from the value
        let target_shape = match &target_shape_val.data {
            TensorData::I64(v) => v.as_slice(),
            _ => {
                return Err(ctx.shape_error("Reshape shape input must be I64"));
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
                    return Err(ctx.shape_error("Reshape can have at most one -1 dimension"));
                }
                infer_dim = Some(idx);
                output_shape.push(0); // Placeholder
            } else if dim == 0 {
                // 0 means "copy from input shape"
                if idx < data_shape.len() {
                    output_shape.push(data_shape[idx]);
                    known_product *= data_shape[idx] as i64;
                } else {
                    return Err(ctx.shape_error(format!(
                        "Reshape: dimension 0 at index {} out of range (input rank is {})",
                        idx,
                        data_shape.len()
                    )));
                }
            } else if dim > 0 {
                output_shape.push(dim as usize);
                known_product *= dim;
            } else {
                return Err(ctx.shape_error(format!(
                    "Invalid reshape dimension: {} (must be > 0, or -1 to infer, or 0 to copy from input)",
                    dim
                )));
            }
        }

        // Compute inferred dimension
        if let Some(idx) = infer_dim {
            let inferred = total_elements as i64 / known_product;
            if inferred * known_product != total_elements as i64 {
                return Err(ctx.shape_error(format!(
                    "Cannot reshape {} elements into shape {:?} ({} elements)\n  \
                     Total element count must be preserved",
                    total_elements,
                    target_shape,
                    known_product * inferred
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
