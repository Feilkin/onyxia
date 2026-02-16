//! Indexing and data access operators.

use onyxia_core::{
    BindingDesc, DataType, FoldCtx, InferenceCtx, Operator, PlanCtx, Result, Step, TensorData,
    TensorShape, TensorValue,
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
            return Err(ctx.shape_error("Gather requires 2 inputs (data and indices)"));
        }

        // Get axis (defaults to 0 per ONNX spec)
        let axis: i64 = ctx.attr_i64_or("axis", 0);

        // Extract static dimensions
        let data_shape = match ctx.input_shape(0)? {
            TensorShape::Static(dims) => dims,
            TensorShape::Symbolic(_) => {
                return Err(
                    ctx.shape_error("Symbolic shapes should be resolved before shape inference")
                );
            }
            TensorShape::Absent => {
                return Err(ctx.shape_error("Gather data input is absent"));
            }
            TensorShape::Unknown => {
                return Err(ctx.shape_error("Gather data input has unknown shape"));
            }
        };

        let indices_shape = match ctx.input_shape(1)? {
            TensorShape::Static(dims) => dims,
            TensorShape::Symbolic(_) => {
                return Err(
                    ctx.shape_error("Symbolic shapes should be resolved before shape inference")
                );
            }
            TensorShape::Absent => {
                return Err(ctx.shape_error("Gather indices input is absent"));
            }
            TensorShape::Unknown => {
                return Err(ctx.shape_error("Gather indices input has unknown shape"));
            }
        };

        // Normalize negative axis
        let normalized_axis = if axis < 0 {
            (data_shape.len() as i64 + axis) as usize
        } else {
            axis as usize
        };

        if normalized_axis >= data_shape.len() {
            let rank = data_shape.len();
            return Err(ctx.shape_error(format!(
                "Gather axis {} out of bounds for input rank {}\n  \
                 Valid axis range: [-{}, {})",
                axis, rank, rank, rank
            )));
        }

        // Output shape: data[:axis] + indices + data[axis+1:]
        let mut output_shape = Vec::new();
        output_shape.extend_from_slice(&data_shape[..normalized_axis]);
        output_shape.extend_from_slice(&indices_shape);
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
                let indices_shape = &indices_val.shape;

                let result = match (&data_val.data, &indices_val.data) {
                    (TensorData::I64(data), TensorData::I64(indices)) => {
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
                        TensorValue::new(
                            TensorData::I64(result),
                            indices_shape.clone(),
                            DataType::I64,
                        )
                    }
                    (TensorData::I32(data), TensorData::I64(indices)) => {
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
                        TensorValue::new(
                            TensorData::I32(result),
                            indices_shape.clone(),
                            DataType::I32,
                        )
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
        let num_workgroups = (num_elements as u32).div_ceil(workgroup_size);

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
                    "Slice requires static input shape".to_string(),
                ));
            }
        };

        let Some(starts_val) = ctx.input_value(1) else {
            return Err(onyxia_core::Error::ShapeInference(
                "Slice starts must be constant".to_string(),
            ));
        };
        let Some(ends_val) = ctx.input_value(2) else {
            return Err(onyxia_core::Error::ShapeInference(
                "Slice ends must be constant".to_string(),
            ));
        };

        // Parse starts and ends as i64 arrays
        let starts = match &starts_val.data {
            TensorData::I64(v) => v,
            TensorData::I32(v) => &v.iter().map(|&x| x as i64).collect::<Vec<_>>(),
            _ => {
                return Err(onyxia_core::Error::ShapeInference(
                    "Slice starts must be integer type".to_string(),
                ));
            }
        };

        let ends = match &ends_val.data {
            TensorData::I64(v) => v,
            TensorData::I32(v) => &v.iter().map(|&x| x as i64).collect::<Vec<_>>(),
            _ => {
                return Err(onyxia_core::Error::ShapeInference(
                    "Slice ends must be integer type".to_string(),
                ));
            }
        };

        // Get optional axes and steps inputs
        let axes: Vec<usize> = if ctx.input_count() > 3 {
            if let Some(axes_val) = ctx.input_value(3) {
                match &axes_val.data {
                    TensorData::I64(v) => v
                        .iter()
                        .map(|&a| {
                            if a < 0 {
                                (data_shape.len() as i64 + a) as usize
                            } else {
                                a as usize
                            }
                        })
                        .collect(),
                    TensorData::I32(v) => v
                        .iter()
                        .map(|&a| {
                            if a < 0 {
                                (data_shape.len() as i64 + a as i64) as usize
                            } else {
                                a as usize
                            }
                        })
                        .collect(),
                    _ => {
                        return Err(onyxia_core::Error::ShapeInference(
                            "Slice axes must be integer type".to_string(),
                        ));
                    }
                }
            } else {
                (0..starts.len()).collect()
            }
        } else {
            (0..starts.len()).collect()
        };

        let steps: Vec<i64> = if ctx.input_count() > 4 {
            if let Some(steps_val) = ctx.input_value(4) {
                match &steps_val.data {
                    TensorData::I64(v) => v.clone(),
                    TensorData::I32(v) => v.iter().map(|&x| x as i64).collect(),
                    _ => {
                        return Err(onyxia_core::Error::ShapeInference(
                            "Slice steps must be integer type".to_string(),
                        ));
                    }
                }
            } else {
                vec![1; starts.len()]
            }
        } else {
            vec![1; starts.len()]
        };

        // Compute output shape
        let mut output_shape = data_shape.to_vec();

        for (i, &axis) in axes.iter().enumerate() {
            if axis >= data_shape.len() {
                return Err(onyxia_core::Error::ShapeInference(format!(
                    "Slice axis {} out of range for rank {}",
                    axis,
                    data_shape.len()
                )));
            }

            let dim = data_shape[axis] as i64;
            let mut start = starts[i];
            let mut end = ends[i];
            let step = steps[i];

            if step == 0 {
                return Err(onyxia_core::Error::ShapeInference(
                    "Slice step cannot be 0".to_string(),
                ));
            }

            // Normalize negative indices
            if start < 0 {
                start += dim;
            }
            if end < 0 {
                end += dim;
            }

            // Clamp to valid range
            start = start.max(0).min(dim);
            end = end.max(0).min(dim);

            // Compute output dimension
            let output_dim = if step > 0 {
                if end > start {
                    ((end - start + step - 1) / step) as usize
                } else {
                    0
                }
            } else {
                if start > end {
                    ((start - end - step - 1) / (-step)) as usize
                } else {
                    0
                }
            };

            output_shape[axis] = output_dim;
        }

        Ok(vec![TensorShape::Static(output_shape)])
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let input_tensor = ctx.input_tensor(0)?;
        let input_shape = ctx.static_dims(&input_tensor.shape)?;
        let rank = input_shape.len();

        let output_tensor = ctx.output_tensor(0)?;
        let output_shape = ctx.static_dims(&output_tensor.shape)?;

        // Parse starts, ends, axes, steps from input values
        let starts_val = ctx
            .input_value(1)
            .ok_or_else(|| onyxia_core::Error::Planning("Slice starts not constant".into()))?;
        let ends_val = ctx
            .input_value(2)
            .ok_or_else(|| onyxia_core::Error::Planning("Slice ends not constant".into()))?;

        let starts: Vec<i64> = match &starts_val.data {
            TensorData::I64(v) => v.clone(),
            TensorData::I32(v) => v.iter().map(|&x| x as i64).collect(),
            _ => return Err(onyxia_core::Error::Planning("Invalid starts type".into())),
        };

        let ends: Vec<i64> = match &ends_val.data {
            TensorData::I64(v) => v.clone(),
            TensorData::I32(v) => v.iter().map(|&x| x as i64).collect(),
            _ => return Err(onyxia_core::Error::Planning("Invalid ends type".into())),
        };

        let axes: Vec<usize> = if ctx.input_count() > 3 {
            if let Some(axes_val) = ctx.input_value(3) {
                match &axes_val.data {
                    TensorData::I64(v) => v
                        .iter()
                        .map(|&a| {
                            if a < 0 {
                                (rank as i64 + a) as usize
                            } else {
                                a as usize
                            }
                        })
                        .collect(),
                    TensorData::I32(v) => v
                        .iter()
                        .map(|&a| {
                            if a < 0 {
                                (rank as i64 + a as i64) as usize
                            } else {
                                a as usize
                            }
                        })
                        .collect(),
                    _ => return Err(onyxia_core::Error::Planning("Invalid axes type".into())),
                }
            } else {
                (0..starts.len()).collect()
            }
        } else {
            (0..starts.len()).collect()
        };

        let steps: Vec<i64> = if ctx.input_count() > 4 {
            if let Some(steps_val) = ctx.input_value(4) {
                match &steps_val.data {
                    TensorData::I64(v) => v.clone(),
                    TensorData::I32(v) => v.iter().map(|&x| x as i64).collect(),
                    _ => return Err(onyxia_core::Error::Planning("Invalid steps type".into())),
                }
            } else {
                vec![1; starts.len()]
            }
        } else {
            vec![1; starts.len()]
        };

        // Build full starts/steps arrays for all dimensions
        let mut full_starts = vec![0i32; rank];
        let mut full_steps = vec![1i32; rank];

        for (i, &axis) in axes.iter().enumerate() {
            let dim = input_shape[axis] as i64;
            let mut start = starts[i];
            let mut end = ends[i];
            let step = steps[i];

            // Normalize negative indices
            if start < 0 {
                start += dim;
            }
            if end < 0 {
                end += dim;
            }

            // Clamp to valid range
            start = start.max(0).min(dim);

            full_starts[axis] = start as i32;
            full_steps[axis] = step as i32;
        }

        // Calculate strides
        let mut input_strides = vec![1usize; rank];
        let mut output_strides = vec![1usize; rank];

        for i in (0..rank.saturating_sub(1)).rev() {
            input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
        }

        let output_size: usize = output_shape.iter().product();
        let workgroup_size: u32 = 256;
        let num_workgroups = (output_size as u32).div_ceil(workgroup_size);

        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), workgroup_size.to_string());

        let shader_index = ctx.compile_shader(
            "slice",
            include_str!("../../shaders/indexing/slice.wgsl"),
            &shader_defs,
        )?;

        // Encode immediate constants
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(rank as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(output_size as u32).to_le_bytes());
        immediates_data.extend_from_slice(&0u32.to_le_bytes()); // _padding[0]
        immediates_data.extend_from_slice(&0u32.to_le_bytes()); // _padding[1]

        // Input strides (up to 7 dimensions)
        for i in 0..7 {
            let stride = if i < rank {
                input_strides[i] as u32
            } else {
                0u32
            };
            immediates_data.extend_from_slice(&stride.to_le_bytes());
        }

        // Output strides (up to 7 dimensions)
        for i in 0..7 {
            let stride = if i < rank {
                output_strides[i] as u32
            } else {
                0u32
            };
            immediates_data.extend_from_slice(&stride.to_le_bytes());
        }

        // Starts (up to 7 dimensions)
        for i in 0..7 {
            let start = if i < rank { full_starts[i] } else { 0i32 };
            immediates_data.extend_from_slice(&start.to_le_bytes());
        }

        // Steps (up to 7 dimensions)
        for i in 0..7 {
            let step = if i < rank { full_steps[i] } else { 1i32 };
            immediates_data.extend_from_slice(&step.to_le_bytes());
        }

        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0)?,
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(0)?,
                    read_only: false,
                },
            ],
            workgroups: [num_workgroups, 1, 1],
            immediates: Some(immediates_data),
        }])
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

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // ScatterND: scatter updates into data tensor at specified indices
        // Two-pass approach:
        //   1. Copy data to output
        //   2. Scatter updates at indices

        let data_tensor = ctx.input_tensor(0)?;
        let data_shape = ctx.static_dims(&data_tensor.shape)?;
        let data_size: usize = data_shape.iter().product();

        let indices_tensor = ctx.input_tensor(1)?;
        let indices_shape = ctx.static_dims(&indices_tensor.shape)?;

        let updates_tensor = ctx.input_tensor(2)?;
        let updates_shape = ctx.static_dims(&updates_tensor.shape)?;

        if indices_shape.is_empty() {
            return Err(onyxia_core::Error::Planning(
                "ScatterND indices cannot be scalar".to_string(),
            ));
        }

        let indices_last_dim = indices_shape[indices_shape.len() - 1];

        if indices_last_dim > data_shape.len() {
            return Err(onyxia_core::Error::Planning(format!(
                "ScatterND indices last dimension {} exceeds data rank {}",
                indices_last_dim,
                data_shape.len()
            )));
        }

        let num_updates: usize = updates_shape.iter().product();

        // Calculate output strides for index computation
        let mut output_strides = vec![1usize; data_shape.len()];
        for i in (0..data_shape.len().saturating_sub(1)).rev() {
            output_strides[i] = output_strides[i + 1] * data_shape[i + 1];
        }

        // Get reduction mode attribute (default: none/replace)
        let reduction_str = ctx
            .attr_string("reduction")
            .unwrap_or_else(|_| "none".to_string());
        let reduction: u32 = match reduction_str.as_str() {
            "none" => 0,
            "add" => 1,
            "mul" => 2,
            "max" => 3,
            "min" => 4,
            _ => {
                return Err(onyxia_core::Error::Planning(format!(
                    "ScatterND unknown reduction mode: {}",
                    reduction_str
                )));
            }
        };

        let workgroup_size: u32 = 256;

        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), workgroup_size.to_string());

        let shader_index = ctx.compile_shader(
            "scatternd",
            include_str!("../../shaders/indexing/scatternd.wgsl"),
            &shader_defs,
        )?;

        // Step 1: Copy data to output (indices_last_dim = 0 signals copy mode)
        let copy_workgroups = (data_size as u32).div_ceil(workgroup_size);

        let mut copy_immediates = Vec::new();
        copy_immediates.extend_from_slice(&(data_size as u32).to_le_bytes());
        copy_immediates.extend_from_slice(&0u32.to_le_bytes()); // indices_last_dim = 0 => copy mode
        copy_immediates.extend_from_slice(&0u32.to_le_bytes()); // reduction (unused in copy)
        copy_immediates.extend_from_slice(&0u32.to_le_bytes()); // pad0
        // output_strides (unused in copy mode, but must be present)
        for _ in 0..8 {
            copy_immediates.extend_from_slice(&0u32.to_le_bytes());
        }

        let copy_step = Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0)?, // data
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(1)?, // indices (unused in copy)
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(2)?, // updates (unused in copy)
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(0)?, // output
                    read_only: false,
                },
            ],
            workgroups: [copy_workgroups, 1, 1],
            immediates: Some(copy_immediates),
        };

        // Step 2: Scatter updates (indices_last_dim != 0 signals scatter mode)
        let scatter_workgroups = (num_updates as u32).div_ceil(workgroup_size);

        let mut scatter_immediates = Vec::new();
        scatter_immediates.extend_from_slice(&(num_updates as u32).to_le_bytes());
        scatter_immediates.extend_from_slice(&(indices_last_dim as u32).to_le_bytes());
        scatter_immediates.extend_from_slice(&reduction.to_le_bytes());
        scatter_immediates.extend_from_slice(&0u32.to_le_bytes()); // pad0

        // Encode output strides (up to 8 dimensions)
        for i in 0..8 {
            let stride = if i < output_strides.len() {
                output_strides[i] as u32
            } else {
                0u32
            };
            scatter_immediates.extend_from_slice(&stride.to_le_bytes());
        }

        let scatter_step = Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0)?, // data (unused in scatter)
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(1)?, // indices
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(2)?, // updates
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(0)?, // output
                    read_only: false,
                },
            ],
            workgroups: [scatter_workgroups, 1, 1],
            immediates: Some(scatter_immediates),
        };

        Ok(vec![copy_step, scatter_step])
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
            let size = match (&start.data, &limit.data, &delta.data) {
                (TensorData::I64(s), TensorData::I64(l), TensorData::I64(d)) => {
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
                (TensorData::F32(s), TensorData::F32(l), TensorData::F32(d)) => {
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

        let result = match (&start.data, &limit.data, &delta.data) {
            (TensorData::I64(s), TensorData::I64(l), TensorData::I64(d)) => {
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
                TensorValue::new(TensorData::I64(values), vec![size], DataType::I64)
            }
            (TensorData::F32(s), TensorData::F32(l), TensorData::F32(d)) => {
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
                TensorValue::new(TensorData::F32(values), vec![size], DataType::F32)
            }
            _ => return Ok(vec![None]),
        };

        Ok(vec![Some(result)])
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // Range should be folded away in most cases, but if we need GPU execution:

        // Get output shape and size
        let output_tensor = ctx.output_tensor(0)?;
        let output_dtype = output_tensor.dtype;
        let output_shape = ctx.static_dims(&output_tensor.shape)?;

        if output_shape.len() != 1 {
            return Err(onyxia_core::Error::Planning(format!(
                "Range output must be 1D, got {:?}",
                output_shape
            )));
        }

        let size = output_shape[0];

        // Try to get constant start, limit, delta values
        // If they're not constant at plan time, we have a problem since the shader
        // needs them as immediates
        let start_val = ctx
            .input_value(0)
            .ok_or_else(|| onyxia_core::Error::Planning("Range start not constant".into()))?;
        let _limit_val = ctx
            .input_value(1)
            .ok_or_else(|| onyxia_core::Error::Planning("Range limit not constant".into()))?;
        let delta_val = ctx
            .input_value(2)
            .ok_or_else(|| onyxia_core::Error::Planning("Range delta not constant".into()))?;

        // Handle I64 output by generating values on CPU and writing directly,
        // since the GPU shader only supports f32.
        if output_dtype == DataType::I64 {
            let (start, delta) = match (&start_val.data, &delta_val.data) {
                (TensorData::I64(s), TensorData::I64(d)) if s.len() == 1 && d.len() == 1 => {
                    (s[0], d[0])
                }
                _ => {
                    return Err(onyxia_core::Error::Planning(
                        "Range I64: start and delta must be I64 scalars".to_string(),
                    ));
                }
            };

            let mut data = Vec::with_capacity(size * 8);
            for i in 0..size {
                data.extend_from_slice(&(start + i as i64 * delta).to_le_bytes());
            }

            return Ok(vec![Step::WriteBuffer {
                dst: ctx.output(0)?,
                data,
            }]);
        }

        // Handle I32 output similarly
        if output_dtype == DataType::I32 {
            let (start, delta) = match (&start_val.data, &delta_val.data) {
                (TensorData::I32(s), TensorData::I32(d)) if s.len() == 1 && d.len() == 1 => {
                    (s[0], d[0])
                }
                (TensorData::I64(s), TensorData::I64(d)) if s.len() == 1 && d.len() == 1 => {
                    (s[0] as i32, d[0] as i32)
                }
                _ => {
                    return Err(onyxia_core::Error::Planning(
                        "Range I32: start and delta must be integer scalars".to_string(),
                    ));
                }
            };

            let mut data = Vec::with_capacity(size * 4);
            for i in 0..size {
                data.extend_from_slice(&(start + i as i32 * delta).to_le_bytes());
            }

            return Ok(vec![Step::WriteBuffer {
                dst: ctx.output(0)?,
                data,
            }]);
        }

        // F32 path: use GPU shader
        let (start, delta) = match (&start_val.data, &delta_val.data) {
            (TensorData::F32(s), TensorData::F32(d)) if s.len() == 1 && d.len() == 1 => {
                (s[0], d[0])
            }
            _ => {
                return Err(onyxia_core::Error::Planning(
                    "Range F32: start and delta must be F32 scalars".to_string(),
                ));
            }
        };

        let workgroup_size: u32 = 256;
        let num_workgroups = (size as u32).div_ceil(workgroup_size);

        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), workgroup_size.to_string());

        let shader_index = ctx.compile_shader(
            "range",
            include_str!("../../shaders/indexing/range.wgsl"),
            &shader_defs,
        )?;

        // Encode immediate constants
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&start.to_le_bytes());
        immediates_data.extend_from_slice(&delta.to_le_bytes());
        immediates_data.extend_from_slice(&(size as u32).to_le_bytes());
        immediates_data.extend_from_slice(&0u32.to_le_bytes()); // _pad

        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![BindingDesc {
                buffer: ctx.output(0)?,
                read_only: false,
            }],
            workgroups: [num_workgroups, 1, 1],
            immediates: Some(immediates_data),
        }])
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
            if let Some(k_val) = ctx.input_value(1) {
                if let TensorData::I64(k_vals) = &k_val.data {
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
            }
        } else {
            0
        };

        match &input.data {
            TensorData::F32(vals) => {
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

                Ok(vec![Some(TensorValue::new(
                    TensorData::F32(result),
                    input.shape.clone(),
                    DataType::F32,
                ))])
            }
            _ => Ok(vec![None]),
        }
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let input_tensor = ctx.input_tensor(0)?;
        let input_shape = ctx.static_dims(&input_tensor.shape)?;
        let rank = input_shape.len();

        if rank < 2 {
            return Err(onyxia_core::Error::Planning(format!(
                "Trilu requires input with rank >= 2, got rank {}",
                rank
            )));
        }

        let m = input_shape[rank - 2];
        let n = input_shape[rank - 1];
        let total_size: usize = input_shape.iter().product();

        // Get attributes
        let upper: i64 = ctx.attr_i64("upper").unwrap_or(1);

        // Get k from second input (if provided) or default to 0
        let k: i32 = if ctx.input_count() > 1 {
            if let Some(k_val) = ctx.input_value(1) {
                match &k_val.data {
                    TensorData::I64(v) => {
                        if v.len() != 1 {
                            return Err(onyxia_core::Error::Planning(
                                "Trilu k must be scalar".to_string(),
                            ));
                        }
                        v[0] as i32
                    }
                    TensorData::I32(v) => {
                        if v.len() != 1 {
                            return Err(onyxia_core::Error::Planning(
                                "Trilu k must be scalar".to_string(),
                            ));
                        }
                        v[0]
                    }
                    _ => {
                        return Err(onyxia_core::Error::Planning(
                            "Trilu k must be integer type".to_string(),
                        ));
                    }
                }
            } else {
                0
            }
        } else {
            0
        };

        let workgroup_size: u32 = 256;
        let num_workgroups = (total_size as u32).div_ceil(workgroup_size);

        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), workgroup_size.to_string());

        let shader_index = ctx.compile_shader(
            "trilu",
            include_str!("../../shaders/indexing/trilu.wgsl"),
            &shader_defs,
        )?;

        // Encode immediate constants
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(total_size as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(m as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(n as u32).to_le_bytes());
        immediates_data.extend_from_slice(&k.to_le_bytes());
        immediates_data.extend_from_slice(&(upper as u32).to_le_bytes());

        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0)?,
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(0)?,
                    read_only: false,
                },
            ],
            workgroups: [num_workgroups, 1, 1],
            immediates: Some(immediates_data),
        }])
    }
}
