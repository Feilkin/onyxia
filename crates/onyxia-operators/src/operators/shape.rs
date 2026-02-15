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
pub struct UnsqueezeOp;

impl Operator for UnsqueezeOp {
    fn name(&self) -> &str {
        "Unsqueeze"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for Unsqueeze - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for Unsqueeze - will be implemented in Tasks 024/025")
    }
}

/// Transpose operator - permutes tensor dimensions.
pub struct TransposeOp;

impl Operator for TransposeOp {
    fn name(&self) -> &str {
        "Transpose"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for Transpose - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for Transpose - will be implemented in Tasks 024/025")
    }
}

/// Concat operator - concatenates tensors along a dimension.
pub struct ConcatOp;

impl Operator for ConcatOp {
    fn name(&self) -> &str {
        "Concat"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for Concat - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for Concat - will be implemented in Tasks 024/025")
    }
}

/// Expand operator - broadcasts a tensor to a larger shape.
pub struct ExpandOp;

impl Operator for ExpandOp {
    fn name(&self) -> &str {
        "Expand"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for Expand - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for Expand - will be implemented in Tasks 024/025")
    }
}
