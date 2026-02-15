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
        if ctx.input_count() < 3 {
            return Err(onyxia_core::Error::ShapeInference(
                "Slice requires at least 3 inputs (data, starts, ends)".to_string(),
            ));
        }

        let data_shape = match ctx.input_shape(0)? {
            TensorShape::Static(dims) => dims,
            _ => {
                return Err(onyxia_core::Error::ShapeInference(
                    "Non-static shape".to_string(),
                ));
            }
        };

        let Some(_starts_val) = ctx.input_value(1) else {
            return Err(onyxia_core::Error::ShapeInference(
                "Slice starts must be constant".to_string(),
            ));
        };
        let Some(_ends_val) = ctx.input_value(2) else {
            return Err(onyxia_core::Error::ShapeInference(
                "Slice ends must be constant".to_string(),
            ));
        };

        // TODO: Implement full Slice shape inference logic
        // For now, return Unknown to avoid blocking compilation
        Ok(vec![TensorShape::Static(data_shape.to_vec())])
    }

    fn plan(&self, _ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // TODO: Implement Slice GPU planning
        Err(onyxia_core::Error::Planning(
            "Slice GPU planning not yet implemented".to_string(),
        ))
    }
}

/// ScatterND operator - writes updates to a tensor at specified indices.
pub struct ScatterNDOp;

impl Operator for ScatterNDOp {
    fn name(&self) -> &str {
        "ScatterND"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        if ctx.input_count() < 3 {
            return Err(onyxia_core::Error::ShapeInference(
                "ScatterND requires 3 inputs (data, indices, updates)".to_string(),
            ));
        }

        // Output has same shape as data input
        Ok(vec![ctx.input_shape(0)?.clone()])
    }

    fn plan(&self, _ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // TODO: Implement ScatterND GPU planning
        Err(onyxia_core::Error::Planning(
            "ScatterND GPU planning not yet implemented".to_string(),
        ))
    }
}

/// Range operator - generates a sequence of numbers.
pub struct RangeOp;

impl Operator for RangeOp {
    fn name(&self) -> &str {
        "Range"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        if ctx.input_count() != 3 {
            return Err(onyxia_core::Error::ShapeInference(format!(
                "Range requires 3 inputs (start, limit, delta), got {}",
                ctx.input_count()
            )));
        }

        // Try to compute output size from constant inputs
        let start_val = ctx.input_value(0);
        let limit_val = ctx.input_value(1);
        let delta_val = ctx.input_value(2);

        if let (Some(start), Some(limit), Some(delta)) = (start_val, limit_val, delta_val) {
            let size = match (start, limit, delta) {
                (TensorValue::I64(s), TensorValue::I64(l), TensorValue::I64(d)) => {
                    if s.len() == 1 && l.len() == 1 && d.len() == 1 {
                        let (s, l, d) = (s[0], l[0], d[0]);
                        if d == 0 {
                            return Err(onyxia_core::Error::ShapeInference(
                                "Range delta cannot be zero".to_string(),
                            ));
                        }
                        ((l - s) as f64 / d as f64).ceil() as usize
                    } else {
                        return Err(onyxia_core::Error::ShapeInference(
                            "Range inputs must be scalars".to_string(),
                        ));
                    }
                }
                (TensorValue::F32(s), TensorValue::F32(l), TensorValue::F32(d)) => {
                    if s.len() == 1 && l.len() == 1 && d.len() == 1 {
                        let (s, l, d) = (s[0], l[0], d[0]);
                        if d == 0.0 {
                            return Err(onyxia_core::Error::ShapeInference(
                                "Range delta cannot be zero".to_string(),
                            ));
                        }
                        ((l - s) / d).ceil() as usize
                    } else {
                        return Err(onyxia_core::Error::ShapeInference(
                            "Range inputs must be scalars".to_string(),
                        ));
                    }
                }
                _ => {
                    return Err(onyxia_core::Error::ShapeInference(
                        "Range: unsupported types".to_string(),
                    ));
                }
            };
            Ok(vec![TensorShape::Static(vec![size])])
        } else {
            // Inputs not constant - can't infer size
            Err(onyxia_core::Error::ShapeInference(
                "Range inputs must be constant".to_string(),
            ))
        }
    }

    fn try_fold(&self, ctx: &FoldCtx) -> Result<Vec<Option<TensorValue>>> {
        let Some(start) = ctx.input_value(0) else {
            return Ok(vec![None]);
        };
        let Some(limit) = ctx.input_value(1) else {
            return Ok(vec![None]);
        };
        let Some(delta) = ctx.input_value(2) else {
            return Ok(vec![None]);
        };

        let result = match (start, limit, delta) {
            (TensorValue::I64(s), TensorValue::I64(l), TensorValue::I64(d)) => {
                if s.len() != 1 || l.len() != 1 || d.len() != 1 {
                    return Err(onyxia_core::Error::ConstantFolding(
                        "Range inputs must be scalars".to_string(),
                    ));
                }
                let (start, limit, delta) = (s[0], l[0], d[0]);
                if delta == 0 {
                    return Err(onyxia_core::Error::ConstantFolding(
                        "Range delta cannot be zero".to_string(),
                    ));
                }
                let size = ((limit - start) as f64 / delta as f64).ceil() as usize;
                let values: Vec<i64> = (0..size).map(|i| start + i as i64 * delta).collect();
                TensorValue::I64(values)
            }
            (TensorValue::F32(s), TensorValue::F32(l), TensorValue::F32(d)) => {
                if s.len() != 1 || l.len() != 1 || d.len() != 1 {
                    return Err(onyxia_core::Error::ConstantFolding(
                        "Range inputs must be scalars".to_string(),
                    ));
                }
                let (start, limit, delta) = (s[0], l[0], d[0]);
                if delta == 0.0 {
                    return Err(onyxia_core::Error::ConstantFolding(
                        "Range delta cannot be zero".to_string(),
                    ));
                }
                let size = ((limit - start) / delta).ceil() as usize;
                let values: Vec<f32> = (0..size).map(|i| start + i as f32 * delta).collect();
                TensorValue::F32(values)
            }
            _ => return Ok(vec![None]),
        };

        Ok(vec![Some(result)])
    }

    fn plan(&self, _ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // TODO: Implement Range GPU planning
        Err(onyxia_core::Error::Planning(
            "Range GPU planning not yet implemented".to_string(),
        ))
    }
}

/// Trilu operator - returns upper or lower triangular part of a matrix.
pub struct TriluOp;

impl Operator for TriluOp {
    fn name(&self) -> &str {
        "Trilu"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        if ctx.input_count() == 0 {
            return Err(onyxia_core::Error::ShapeInference(
                "Trilu requires at least one input".to_string(),
            ));
        }
        // Trilu output has same shape as input
        Ok(vec![ctx.input_shape(0)?.clone()])
    }

    fn try_fold(&self, ctx: &FoldCtx) -> Result<Vec<Option<TensorValue>>> {
        let Some(input) = ctx.input_value(0) else {
            return Ok(vec![None]);
        };

        let input_shape = match ctx.input_shape(0)? {
            TensorShape::Static(dims) => dims,
            _ => return Ok(vec![None]),
        };

        if input_shape.len() < 2 {
            return Err(onyxia_core::Error::ConstantFolding(
                "Trilu requires input with rank >= 2".to_string(),
            ));
        }

        let rank = input_shape.len();
        let m = input_shape[rank - 2];
        let n = input_shape[rank - 1];

        let upper: i64 = ctx.attr_i64("upper").unwrap_or(1);

        let k: i64 = if ctx.input_count() > 1 {
            if let Some(TensorValue::I64(k_vals)) = ctx.input_value(1) {
                if k_vals.len() != 1 {
                    return Err(onyxia_core::Error::ConstantFolding(
                        "Trilu k must be scalar".to_string(),
                    ));
                }
                k_vals[0]
            } else {
                0
            }
        } else {
            0
        };

        match input {
            TensorValue::F32(vals) => {
                let batch_size: usize = input_shape[..rank - 2].iter().product();
                let matrix_size = m * n;
                let mut result = vals.clone();

                for batch_idx in 0..batch_size {
                    for row in 0..m {
                        for col in 0..n {
                            let idx = batch_idx * matrix_size + row * n + col;
                            let keep = if upper == 1 {
                                (row as i64) <= (col as i64) + k
                            } else {
                                (row as i64) >= (col as i64) + k
                            };
                            if !keep {
                                result[idx] = 0.0;
                            }
                        }
                    }
                }

                Ok(vec![Some(TensorValue::F32(result))])
            }
            _ => Ok(vec![None]),
        }
    }

    fn plan(&self, _ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // TODO: Implement Trilu GPU planning
        Err(onyxia_core::Error::Planning(
            "Trilu GPU planning not yet implemented".to_string(),
        ))
    }
}
