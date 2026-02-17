//! Shape manipulation operators.

use onyxia_core::{
    CompileCtx, DataType, DispatchCtx, Error, OpDispatch, Operator, Result, RuntimeTensor,
};
use std::sync::Arc;

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
