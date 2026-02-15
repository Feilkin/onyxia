//! Indexing and data access operators.

use onyxia_core::{
    BindingDesc, FoldCtx, InferenceCtx, Operator, PlanCtx, Result, Step, TensorShape, TensorValue,
};
use std::collections::HashMap;

/// Gather operator - selects elements from input along an axis using indices.
///
/// Performs indexed selection from a data tensor:
///   output[i][j][k] = data[indices[i][j]][k]  (for axis=0)
///
/// Commonly used for embedding lookup where:
/// - data: embedding table [vocab_size, hidden_dim]
/// - indices: token IDs [batch, seq]
/// - output: embeddings [batch, seq, hidden_dim]
pub struct GatherOp;

impl Operator for GatherOp {
    fn name(&self) -> &str {
        "Gather"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        if ctx.input_count() < 2 {
            return Err(onyxia_core::Error::ShapeInference(
                "Gather requires 2 inputs (data and indices)".to_string(),
            ));
        }

        // Get axis (defaults to 0 per ONNX spec)
        let axis: i64 = ctx.attr_i64_or("axis", 0);

        // Extract static dimensions
        let data_shape = match ctx.input_shape(0)? {
            TensorShape::Static(dims) => dims,
            TensorShape::Symbolic(_) => {
                return Err(onyxia_core::Error::ShapeInference(
                    "Symbolic shapes should be resolved before shape inference".to_string(),
                ));
            }
            TensorShape::Absent => {
                return Err(onyxia_core::Error::ShapeInference(
                    "Gather data input is absent".to_string(),
                ));
            }
        };

        let indices_shape = match ctx.input_shape(1)? {
            TensorShape::Static(dims) => dims,
            TensorShape::Symbolic(_) => {
                return Err(onyxia_core::Error::ShapeInference(
                    "Symbolic shapes should be resolved before shape inference".to_string(),
                ));
            }
            TensorShape::Absent => {
                return Err(onyxia_core::Error::ShapeInference(
                    "Gather indices input is absent".to_string(),
                ));
            }
        };

        // Normalize negative axis
        let normalized_axis = if axis < 0 {
            (data_shape.len() as i64 + axis) as usize
        } else {
            axis as usize
        };

        if normalized_axis >= data_shape.len() {
            return Err(onyxia_core::Error::ShapeInference(format!(
                "Gather axis {} is out of bounds for data shape {:?}",
                axis, data_shape
            )));
        }

        // Output shape: data[:axis] + indices + data[axis+1:]
        let mut output_shape = Vec::new();
        output_shape.extend_from_slice(&data_shape[..normalized_axis]);
        output_shape.extend_from_slice(indices_shape);
        output_shape.extend_from_slice(&data_shape[normalized_axis + 1..]);

        Ok(vec![TensorShape::Static(output_shape)])
    }

    fn try_fold(&self, ctx: &FoldCtx) -> Result<Vec<Option<TensorValue>>> {
        // If both data and indices are known, gather at compile time
        let Some(data_val) = ctx.input_value(0) else {
            return Ok(vec![None]);
        };
        let Some(indices_val) = ctx.input_value(1) else {
            return Ok(vec![None]);
        };

        let axis: i64 = ctx.attr_i64_or("axis", 0);

        // For axis=0 and 1D data, simple indexing
        if axis == 0 {
            let data_shape = ctx.input_shape(0)?;
            let TensorShape::Static(data_shape) = data_shape else {
                return Ok(vec![None]);
            };

            // Only support 1D data for now (e.g., selecting scalars from a list)
            if data_shape.len() == 1 {
                let result = match (data_val, indices_val) {
                    (TensorValue::I64(data), TensorValue::I64(indices)) => {
                        let mut result = Vec::new();
                        for &idx in indices {
                            let idx_usize = idx as usize;
                            if idx_usize >= data.len() {
                                return Err(onyxia_core::Error::ConstantFolding(format!(
                                    "Gather index {} out of bounds for data length {}",
                                    idx,
                                    data.len()
                                )));
                            }
                            result.push(data[idx_usize]);
                        }
                        TensorValue::I64(result)
                    }
                    (TensorValue::I32(data), TensorValue::I64(indices)) => {
                        let mut result = Vec::new();
                        for &idx in indices {
                            let idx_usize = idx as usize;
                            if idx_usize >= data.len() {
                                return Err(onyxia_core::Error::ConstantFolding(format!(
                                    "Gather index {} out of bounds for data length {}",
                                    idx,
                                    data.len()
                                )));
                            }
                            result.push(data[idx_usize]);
                        }
                        TensorValue::I32(result)
                    }
                    _ => return Ok(vec![None]),
                };
                return Ok(vec![Some(result)]);
            }
        }

        Ok(vec![None])
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // Get axis attribute (default to 0)
        let axis: i64 = ctx.attr_i64("axis").unwrap_or(0);

        // Get data shape
        let data_tensor = ctx.input_tensor(0)?;
        let data_shape = ctx.static_dims(&data_tensor.shape)?;

        // Normalize negative axis
        let normalized_axis = if axis < 0 {
            (data_shape.len() as i64 + axis) as usize
        } else {
            axis as usize
        };

        // Get output shape and calculate total elements
        let output_tensor = ctx.output_tensor(0)?;
        let output_shape = ctx.static_dims(&output_tensor.shape)?;
        let num_elements: usize = output_shape.iter().product();

        // Calculate inner_dim: product of data.shape[axis+1:]
        let inner_dim: usize = data_shape[normalized_axis + 1..].iter().product();
        let inner_dim = if inner_dim == 0 { 1 } else { inner_dim };

        // Configure workgroup size
        let workgroup_size: u32 = 256;
        let num_workgroups = (num_elements as u32 + workgroup_size - 1) / workgroup_size;

        // Prepare shader definitions
        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), workgroup_size.to_string());

        // Compile shader
        let shader_index = ctx.compile_shader(
            "gather",
            include_str!("../../shaders/indexing/gather.wgsl"),
            &shader_defs,
        )?;

        // Encode immediate data (must match ImmediateConstants struct in shader)
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(num_elements as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(inner_dim as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(normalized_axis as u32).to_le_bytes());

        // Create dispatch step with bindings and immediates
        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0)?, // data
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(1)?, // indices (I64 as u32 pairs)
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(0)?, // output
                    read_only: false,
                },
            ],
            workgroups: [num_workgroups, 1, 1],
            immediates: Some(immediates_data),
        }])
    }
}

/// Slice operator - extracts a slice from a tensor.
pub struct SliceOp;

impl Operator for SliceOp {
    fn name(&self) -> &str {
        "Slice"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for Slice - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for Slice - will be implemented in Tasks 024/025")
    }
}

/// ScatterND operator - writes updates to a tensor at specified indices.
pub struct ScatterNDOp;

impl Operator for ScatterNDOp {
    fn name(&self) -> &str {
        "ScatterND"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for ScatterND - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for ScatterND - will be implemented in Tasks 024/025")
    }
}

/// Range operator - generates a sequence of numbers.
pub struct RangeOp;

impl Operator for RangeOp {
    fn name(&self) -> &str {
        "Range"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for Range - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for Range - will be implemented in Tasks 024/025")
    }
}

/// Trilu operator - returns upper or lower triangular part of a matrix.
pub struct TriluOp;

impl Operator for TriluOp {
    fn name(&self) -> &str {
        "Trilu"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        let _ = ctx;
        todo!("Shape inference for Trilu - will be implemented in Tasks 024/025")
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let _ = ctx;
        todo!("Planning for Trilu - will be implemented in Tasks 024/025")
    }
}
