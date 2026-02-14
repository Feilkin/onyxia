//! SliceOperator implementation for tensor slicing operations.

use crate::error::{CodegenError, Result};
use crate::inference::{InferenceContext, TensorValue};
use crate::operator::{OpOperator, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::TensorShape;
use std::collections::HashMap;

/// Operator for Slice operation (ONNX Slice operator).
///
/// Extracts a sub-tensor from the input tensor based on start/end indices,
/// axes, and steps. Supports:
/// - Multi-axis slicing
/// - Negative indices (counting from the end)
/// - Strided slicing (steps > 1)
/// - Reverse slicing (negative steps)
///
/// Inputs:
/// - data (input 0): Input tensor of any shape
/// - starts (input 1): 1D tensor of starting indices (int32/int64)
/// - ends (input 2): 1D tensor of ending indices (int32/int64)
/// - axes (input 3, optional): 1D tensor of axes to slice (int32/int64)
/// - steps (input 4, optional): 1D tensor of step sizes (int32/int64)
///
/// Outputs:
/// - output: Sliced tensor
pub struct SliceOperator;

impl OpOperator for SliceOperator {
    fn name(&self) -> &str {
        "Slice"
    }

    fn infer_output_shapes(&self, ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>> {
        if ctx.input_shapes.len() < 3 {
            return Err(CodegenError::InvalidShape(
                "Slice requires at least 3 inputs (data, starts, ends)".to_string(),
            ));
        }

        // Get data shape
        let data_shape = match &ctx.input_shapes[0] {
            TensorShape::Static(dims) => dims,
            TensorShape::Unknown | TensorShape::Absent => return Ok(vec![TensorShape::Unknown]),
            TensorShape::Dynamic(_) => {
                return Err(CodegenError::InvalidShape(
                    "Unexpected Dynamic shape after dimension resolution".to_string(),
                ));
            }
        };

        // Get starts, ends, axes, steps values
        let Some(starts_val) = ctx.input_value(1)? else {
            return Ok(vec![TensorShape::Unknown]);
        };
        let Some(ends_val) = ctx.input_value(2)? else {
            return Ok(vec![TensorShape::Unknown]);
        };

        let starts = match starts_val {
            TensorValue::I64(v) => v.as_slice(),
            TensorValue::I32(v) => {
                // Convert i32 to i64 for uniform handling
                let v64: Vec<i64> = v.iter().map(|&x| x as i64).collect();
                return self.compute_output_shape_from_params(
                    data_shape,
                    &v64,
                    &match ends_val {
                        TensorValue::I64(v) => v.clone(),
                        TensorValue::I32(v) => v.iter().map(|&x| x as i64).collect(),
                        _ => {
                            return Err(CodegenError::InvalidShape(
                                "Slice ends must be int32 or int64".to_string(),
                            ));
                        }
                    },
                    ctx.input_value(3)?,
                    ctx.input_value(4)?,
                );
            }
            _ => {
                return Err(CodegenError::InvalidShape(
                    "Slice starts must be int32 or int64".to_string(),
                ));
            }
        };

        let ends = match ends_val {
            TensorValue::I64(v) => v.as_slice(),
            TensorValue::I32(v) => {
                let v64: Vec<i64> = v.iter().map(|&x| x as i64).collect();
                return self.compute_output_shape_from_params(
                    data_shape,
                    starts,
                    &v64,
                    ctx.input_value(3)?,
                    ctx.input_value(4)?,
                );
            }
            _ => {
                return Err(CodegenError::InvalidShape(
                    "Slice ends must be int32 or int64".to_string(),
                ));
            }
        };

        self.compute_output_shape_from_params(
            data_shape,
            starts,
            ends,
            ctx.input_value(3)?,
            ctx.input_value(4)?,
        )
    }

    #[allow(clippy::needless_range_loop)]
    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        // Get data shape
        let data_info = ctx.input_info(0)?;
        let data_shape = ctx.static_shape(&data_info.shape)?;
        let rank = data_shape.len();

        // Parse starts, ends, axes, steps from constant inputs
        let starts = self.parse_int_input(ctx, 1, "starts")?;
        let ends = self.parse_int_input(ctx, 2, "ends")?;
        let axes = if ctx.input_ids.len() > 3 {
            self.parse_int_input(ctx, 3, "axes")?
        } else {
            // Default: all axes [0, 1, ..., rank-1]
            (0..rank as i64).collect()
        };
        let steps = if ctx.input_ids.len() > 4 {
            self.parse_int_input(ctx, 4, "steps")?
        } else {
            // Default: all 1s
            vec![1; starts.len()]
        };

        // Validate input lengths
        if starts.len() != ends.len() || starts.len() != axes.len() || starts.len() != steps.len() {
            return Err(CodegenError::InvalidShape(format!(
                "Slice parameter lengths mismatch: starts={}, ends={}, axes={}, steps={}",
                starts.len(),
                ends.len(),
                axes.len(),
                steps.len()
            )));
        }

        // Build per-dimension slice parameters (initially identity: full slice)
        let mut dim_starts = vec![0i32; rank];
        let mut dim_ends: Vec<i32> = data_shape.iter().map(|&d| d as i32).collect();
        let mut dim_steps = vec![1i32; rank];

        // Apply per-axis slicing parameters
        for i in 0..starts.len() {
            let axis = axes[i];
            let normalized_axis = if axis < 0 {
                (rank as i64 + axis) as usize
            } else {
                axis as usize
            };

            if normalized_axis >= rank {
                return Err(CodegenError::InvalidShape(format!(
                    "Slice axis {} is out of bounds for rank {}",
                    axis, rank
                )));
            }

            let dim_size = data_shape[normalized_axis] as i64;

            // Normalize negative indices
            let start = self.normalize_index(starts[i], dim_size);
            let end = self.normalize_index(ends[i], dim_size);
            let step = steps[i];

            if step == 0 {
                return Err(CodegenError::InvalidShape(
                    "Slice step cannot be 0".to_string(),
                ));
            }

            dim_starts[normalized_axis] = start as i32;
            dim_ends[normalized_axis] = end as i32;
            dim_steps[normalized_axis] = step as i32;
        }

        // Calculate output shape and strides
        let mut output_shape = Vec::with_capacity(rank);
        for i in 0..rank {
            let start = dim_starts[i];
            let end = dim_ends[i];
            let step = dim_steps[i];

            let slice_size = if step > 0 {
                ((end - start).max(0) + step - 1) / step
            } else {
                ((start - end).max(0) - step - 1) / (-step)
            };

            output_shape.push(slice_size.max(0) as usize);
        }

        let num_elements: usize = output_shape.iter().product();

        // Compute strides for indexing
        let mut input_strides = vec![1usize; rank];
        let mut output_strides = vec![1usize; rank];
        if rank > 0 {
            for i in (0..rank - 1).rev() {
                input_strides[i] = input_strides[i + 1] * data_shape[i + 1];
                output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
            }
        }

        // Configure workgroup size
        let workgroup_size: u32 = 256;
        let num_workgroups = (num_elements as u32).div_ceil(workgroup_size);

        // Prepare shader definitions
        let mut shader_defs = HashMap::new();
        shader_defs.insert(
            "WORKGROUP_SIZE".to_string(),
            ShaderDefValue::UInt(workgroup_size),
        );
        shader_defs.insert("MAX_RANK".to_string(), ShaderDefValue::UInt(rank as u32));

        // Validate rank (max 7 to fit in 128-byte immediate data limit)
        if rank > 7 {
            return Err(CodegenError::InvalidShape(format!(
                "Slice does not support tensors with rank > 7 (got rank {})",
                rank
            )));
        }

        // Compile shader
        let shader_index = ctx.compile_shader(
            "slice",
            include_str!("../../shaders/indexing/slice.wgsl"),
            shader_defs,
        )?;

        // Encode immediate data (must match ImmediateConstants struct in shader)
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(rank as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(num_elements as u32).to_le_bytes());

        // Pad to align to 16 bytes (vec4 alignment in WGSL)
        immediates_data.extend_from_slice(&[0u8; 8]);

        // Add per-dimension parameters (up to 7 dimensions to fit in 128 bytes)
        for i in 0..7 {
            if i < rank {
                immediates_data.extend_from_slice(&(input_strides[i] as u32).to_le_bytes());
            } else {
                immediates_data.extend_from_slice(&0u32.to_le_bytes());
            }
        }

        for i in 0..7 {
            if i < rank {
                immediates_data.extend_from_slice(&(output_strides[i] as u32).to_le_bytes());
            } else {
                immediates_data.extend_from_slice(&0u32.to_le_bytes());
            }
        }

        for i in 0..7 {
            if i < rank {
                immediates_data.extend_from_slice(&dim_starts[i].to_le_bytes());
            } else {
                immediates_data.extend_from_slice(&0i32.to_le_bytes());
            }
        }

        for i in 0..7 {
            if i < rank {
                immediates_data.extend_from_slice(&dim_steps[i].to_le_bytes());
            } else {
                immediates_data.extend_from_slice(&0i32.to_le_bytes());
            }
        }

        // Create dispatch step with bindings and immediates
        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0), // data
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(0), // output
                    read_only: false,
                },
            ],
            workgroups: [num_workgroups, 1, 1],
            immediates: Some(immediates_data),
        }])
    }
}

impl SliceOperator {
    /// Normalize an index (handle negative indices).
    fn normalize_index(&self, idx: i64, dim_size: i64) -> i64 {
        if idx < 0 {
            (dim_size + idx).max(0).min(dim_size)
        } else {
            idx.min(dim_size)
        }
    }

    /// Parse an integer input (i32 or i64) from a constant tensor.
    fn parse_int_input(
        &self,
        ctx: &PlanContext<'_>,
        input_idx: usize,
        name: &str,
    ) -> Result<Vec<i64>> {
        let tensor_id = ctx.input_ids[input_idx];
        let tensor_info = ctx.graph.tensor_info.get(tensor_id).ok_or_else(|| {
            CodegenError::InvalidShape(format!("Slice {} tensor not found", name))
        })?;

        let Some(ref bytes) = tensor_info.initializer else {
            return Err(CodegenError::InvalidShape(format!(
                "Slice {} must be a constant (initializer)",
                name
            )));
        };

        let TensorShape::Static(ref dims) = tensor_info.shape else {
            return Err(CodegenError::InvalidShape(format!(
                "Slice {} shape must be static",
                name
            )));
        };

        let element_count: usize = dims.iter().product();

        match tensor_info.dtype {
            onyxia_onnx::DataType::I64 => {
                if bytes.len() != element_count * 8 {
                    return Err(CodegenError::InvalidShape(format!(
                        "Slice {} I64 size mismatch",
                        name
                    )));
                }
                Ok(bytes
                    .chunks_exact(8)
                    .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                    .collect())
            }
            onyxia_onnx::DataType::I32 => {
                if bytes.len() != element_count * 4 {
                    return Err(CodegenError::InvalidShape(format!(
                        "Slice {} I32 size mismatch",
                        name
                    )));
                }
                Ok(bytes
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()) as i64)
                    .collect())
            }
            _ => Err(CodegenError::InvalidShape(format!(
                "Slice {} must be int32 or int64",
                name
            ))),
        }
    }

    /// Compute output shape from slice parameters.
    fn compute_output_shape_from_params(
        &self,
        data_shape: &[usize],
        starts: &[i64],
        ends: &[i64],
        axes_val: Option<&TensorValue>,
        steps_val: Option<&TensorValue>,
    ) -> Result<Vec<TensorShape>> {
        let rank = data_shape.len();

        // Parse axes (default: all axes)
        let axes = if let Some(val) = axes_val {
            match val {
                TensorValue::I64(v) => v.clone(),
                TensorValue::I32(v) => v.iter().map(|&x| x as i64).collect(),
                _ => {
                    return Err(CodegenError::InvalidShape(
                        "Slice axes must be int32 or int64".to_string(),
                    ));
                }
            }
        } else {
            (0..rank as i64).collect()
        };

        // Parse steps (default: all 1s)
        let steps = if let Some(val) = steps_val {
            match val {
                TensorValue::I64(v) => v.clone(),
                TensorValue::I32(v) => v.iter().map(|&x| x as i64).collect(),
                _ => {
                    return Err(CodegenError::InvalidShape(
                        "Slice steps must be int32 or int64".to_string(),
                    ));
                }
            }
        } else {
            vec![1; starts.len()]
        };

        // Validate lengths
        if starts.len() != ends.len() || starts.len() != axes.len() || starts.len() != steps.len() {
            return Err(CodegenError::InvalidShape(format!(
                "Slice parameter lengths mismatch: starts={}, ends={}, axes={}, steps={}",
                starts.len(),
                ends.len(),
                axes.len(),
                steps.len()
            )));
        }

        // Start with input shape
        let mut output_dims = data_shape.to_vec();

        // Apply slicing to specified axes
        for i in 0..starts.len() {
            let axis = axes[i];
            let normalized_axis = if axis < 0 {
                (rank as i64 + axis) as usize
            } else {
                axis as usize
            };

            if normalized_axis >= rank {
                return Err(CodegenError::InvalidShape(format!(
                    "Slice axis {} is out of bounds for rank {}",
                    axis, rank
                )));
            }

            let dim_size = data_shape[normalized_axis] as i64;
            let start = self.normalize_index(starts[i], dim_size);
            let end = self.normalize_index(ends[i], dim_size);
            let step = steps[i];

            if step == 0 {
                return Err(CodegenError::InvalidShape(
                    "Slice step cannot be 0".to_string(),
                ));
            }

            // Calculate slice size
            let slice_size = if step > 0 {
                ((end - start).max(0) + step - 1) / step
            } else {
                ((start - end).max(0) - step - 1) / (-step)
            };

            output_dims[normalized_axis] = slice_size.max(0) as usize;
        }

        Ok(vec![TensorShape::Static(output_dims)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::BufferRef;
    use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
    use std::collections::HashMap;

    fn create_slice_test_graph() -> Graph {
        let mut graph = Graph::new();

        // Add data input tensor
        graph.add_tensor(TensorInfo {
            name: "data".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![10, 20, 30]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add starts tensor (constant)
        let starts_bytes = vec![1i64, 5i64, 10i64]
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        graph.add_tensor(TensorInfo {
            name: "starts".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![3]),
            kind: TensorKind::Weight,
            initializer: Some(starts_bytes),
        });

        // Add ends tensor (constant)
        let ends_bytes = vec![5i64, 15i64, 25i64]
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        graph.add_tensor(TensorInfo {
            name: "ends".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![3]),
            kind: TensorKind::Weight,
            initializer: Some(ends_bytes),
        });

        // Add output tensor
        graph.add_tensor(TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4, 10, 15]),
            kind: TensorKind::Output,
            initializer: None,
        });

        graph.inputs = vec!["data".to_string()];
        graph.outputs = vec!["output".to_string()];

        graph
    }

    #[test]
    fn test_slice_kernel_plan() {
        let graph = create_slice_test_graph();
        let mut node = Node::new("Slice");
        node.inputs = vec!["data".to_string(), "starts".to_string(), "ends".to_string()];
        node.outputs = vec!["output".to_string()];

        let input_ids = vec![0, 1, 2];
        let output_ids = vec![3];
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let steps = SliceOperator.plan(&mut ctx).expect("Planning should succeed");

        // Verify we got exactly one dispatch step
        assert_eq!(steps.len(), 1);

        match &steps[0] {
            Step::Dispatch {
                shader_index,
                bindings,
                workgroups,
                immediates,
            } => {
                // Verify shader was compiled
                assert_eq!(*shader_index, 0);

                // Verify bindings: 1 read-only input + 1 read-write output
                assert_eq!(bindings.len(), 2);

                assert_eq!(bindings[0].buffer, BufferRef::Tensor(0)); // data
                assert!(bindings[0].read_only);

                assert_eq!(bindings[1].buffer, BufferRef::Tensor(3)); // output
                assert!(!bindings[1].read_only);

                // Verify workgroups: output shape is [4, 10, 15] = 600 elements
                // With workgroup size 256, we need 3 workgroups
                let expected_workgroups = (600 + 255) / 256;
                assert_eq!(workgroups[0], expected_workgroups);

                // Verify immediates are present
                assert!(immediates.is_some());
            }
            _ => panic!("Expected Dispatch step"),
        }
    }

    #[test]
    fn test_normalize_index() {
        let operator = SliceOperator;

        // Positive indices
        assert_eq!(operator.normalize_index(0, 10), 0);
        assert_eq!(operator.normalize_index(5, 10), 5);
        assert_eq!(operator.normalize_index(10, 10), 10);
        assert_eq!(operator.normalize_index(15, 10), 10); // Clamped

        // Negative indices
        assert_eq!(operator.normalize_index(-1, 10), 9);
        assert_eq!(operator.normalize_index(-5, 10), 5);
        assert_eq!(operator.normalize_index(-10, 10), 0);
        assert_eq!(operator.normalize_index(-15, 10), 0); // Clamped
    }
}
