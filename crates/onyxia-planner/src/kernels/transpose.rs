//! TransposeKernel implementation for tensor dimension permutation.

use crate::error::Result;
use crate::inference::{InferenceContext, TensorValue};
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::TensorShape;
use std::collections::HashMap;

/// Kernel for transpose operation (ONNX Transpose operator).
///
/// Permutes tensor dimensions according to a permutation vector.
/// For a tensor with shape [d0, d1, ..., dn] and permutation perm,
/// output has shape [d_perm[0], d_perm[1], ..., d_perm[n]].
///
/// If perm is not specified, defaults to reversing all dimensions.
pub struct TransposeKernel;

impl OpKernel for TransposeKernel {
    fn name(&self) -> &str {
        "Transpose"
    }

    fn infer_output_shapes(&self, ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>> {
        if ctx.input_shapes.is_empty() {
            return Err(crate::error::CodegenError::InvalidShape(
                "Transpose requires at least one input".to_string(),
            ));
        }

        // Extract static dimensions (Phase 1 already resolved Dynamic dims)
        let input_dims = match &ctx.input_shapes[0] {
            TensorShape::Static(dims) => dims,
            TensorShape::Unknown | TensorShape::Absent => {
                return Ok(vec![TensorShape::Unknown]);
            }
            TensorShape::Dynamic(_) => {
                return Err(crate::error::CodegenError::InvalidShape(
                    "Unexpected Dynamic shape after dimension resolution".to_string(),
                ));
            }
        };

        let rank = input_dims.len();

        // Read perm attribute (optional)
        let perm: Option<Vec<i64>> = ctx.node.attr("perm").ok();
        let perm = perm.unwrap_or_else(|| (0..rank as i64).rev().collect());

        // Validate perm
        if perm.len() != rank {
            return Err(crate::error::CodegenError::InvalidShape(format!(
                "Transpose perm length {} does not match input rank {}",
                perm.len(),
                rank
            )));
        }

        // Compute output shape by permuting input dimensions
        let output_dims: Vec<usize> = perm.iter().map(|&p| input_dims[p as usize]).collect();

        Ok(vec![TensorShape::Static(output_dims)])
    }

    fn try_fold(&self, ctx: &InferenceContext<'_>) -> Result<Vec<Option<TensorValue>>> {
        // Try to constant-fold if input is known
        let Some(input) = ctx.input_value(0)? else {
            return Ok(vec![None]);
        };

        // Get input shape
        let input_shape = match &ctx.input_shapes[0] {
            TensorShape::Static(dims) => dims,
            _ => return Ok(vec![None]),
        };

        let rank = input_shape.len();

        // Read perm attribute (optional)
        let perm: Option<Vec<i64>> = ctx.node.attr("perm").ok();
        let perm = perm.unwrap_or_else(|| (0..rank as i64).rev().collect());

        // Helper function to transpose a flat array based on shape and permutation
        fn transpose_values<T: Clone>(
            values: &[T],
            input_shape: &[usize],
            perm: &[i64],
        ) -> Vec<T> {
            let rank = input_shape.len();
            let num_elements: usize = input_shape.iter().product();

            // Compute output shape
            let output_shape: Vec<usize> = perm.iter().map(|&p| input_shape[p as usize]).collect();

            // Compute strides for input and output
            let mut input_strides = vec![1; rank];
            let mut output_strides = vec![1; rank];
            for i in (0..rank - 1).rev() {
                input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
                output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
            }

            let mut result = vec![values[0].clone(); num_elements];

            // For each element in the output
            for out_idx in 0..num_elements {
                // Compute multi-dimensional index in output
                let mut out_coords = vec![0; rank];
                let mut temp = out_idx;
                for i in 0..rank {
                    out_coords[i] = temp / output_strides[i];
                    temp %= output_strides[i];
                }

                // Map to input coordinates using inverse permutation
                let mut in_coords = vec![0; rank];
                for i in 0..rank {
                    in_coords[perm[i] as usize] = out_coords[i];
                }

                // Compute flat input index
                let in_idx: usize = in_coords
                    .iter()
                    .zip(input_strides.iter())
                    .map(|(c, s)| c * s)
                    .sum();

                result[out_idx] = values[in_idx].clone();
            }

            result
        }

        // Apply transpose based on value type
        let result = match input {
            TensorValue::F32(vals) => {
                TensorValue::F32(transpose_values(vals, input_shape, &perm))
            }
            TensorValue::I64(vals) => {
                TensorValue::I64(transpose_values(vals, input_shape, &perm))
            }
            TensorValue::I32(vals) => {
                TensorValue::I32(transpose_values(vals, input_shape, &perm))
            }
            TensorValue::Bool(vals) => {
                TensorValue::Bool(transpose_values(vals, input_shape, &perm))
            }
            TensorValue::U8(vals) => {
                TensorValue::U8(transpose_values(vals, input_shape, &perm))
            }
        };

        Ok(vec![Some(result)])
    }

    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        // Get input and output shapes
        let input_info = ctx.input_info(0)?;
        let input_shape = ctx.static_shape(&input_info.shape)?;

        let output_info = ctx.output_info(0)?;
        let output_shape = ctx.static_shape(&output_info.shape)?;

        let rank = input_shape.len();
        let num_elements: usize = output_shape.iter().product();

        // Read perm attribute (optional)
        let perm: Option<Vec<i64>> = ctx.node.attr("perm").ok();
        let perm = perm.unwrap_or_else(|| (0..rank as i64).rev().collect());

        // Validate rank (max 6 dimensions)
        if rank > 6 {
            return Err(crate::error::CodegenError::InvalidShape(format!(
                "Transpose supports maximum rank 6, got {}",
                rank
            )));
        }

        // Compute strides for input and output
        let input_strides = compute_strides(&input_shape);
        let output_strides = compute_strides(&output_shape);

        // Configure workgroup size
        let workgroup_size: u32 = 256;
        let total_workgroups = (num_elements as u32 + workgroup_size - 1) / workgroup_size;

        // WebGPU limit: each dispatch dimension must be <= 65535
        // Distribute workgroups across X, Y (and Z if needed) dimensions
        const MAX_DISPATCH_DIM: u32 = 65535;
        let (workgroups_x, workgroups_y, workgroups_z) = if total_workgroups <= MAX_DISPATCH_DIM {
            (total_workgroups, 1, 1)
        } else {
            let workgroups_y = (total_workgroups + MAX_DISPATCH_DIM - 1) / MAX_DISPATCH_DIM;
            if workgroups_y <= MAX_DISPATCH_DIM {
                (MAX_DISPATCH_DIM, workgroups_y, 1)
            } else {
                // Very large tensors: use all 3 dimensions
                let workgroups_z = (workgroups_y + MAX_DISPATCH_DIM - 1) / MAX_DISPATCH_DIM;
                (MAX_DISPATCH_DIM, MAX_DISPATCH_DIM, workgroups_z)
            }
        };

        // Prepare shader definitions
        let mut shader_defs = HashMap::new();
        shader_defs.insert(
            "WORKGROUP_SIZE".to_string(),
            ShaderDefValue::UInt(workgroup_size),
        );

        // Compile shader
        let shader_index = ctx.compile_shader(
            "transpose",
            include_str!("../../shaders/indexing/transpose.wgsl"),
            shader_defs,
        )?;

        // Encode immediate data (must match ImmediateConstants struct in shader)
        let mut immediates_data = Vec::new();

        // rank (u32)
        immediates_data.extend_from_slice(&(rank as u32).to_le_bytes());

        // num_elements (u32)
        immediates_data.extend_from_slice(&(num_elements as u32).to_le_bytes());

        // dispatch_size_x (u32) - needed for multi-dimensional dispatch thread ID calculation
        immediates_data.extend_from_slice(&workgroups_x.to_le_bytes());

        // input_strides (6 x u32, pad with 0)
        for i in 0..6 {
            let stride = if i < rank { input_strides[i] as u32 } else { 0 };
            immediates_data.extend_from_slice(&stride.to_le_bytes());
        }

        // output_strides (6 x u32, pad with 0)
        for i in 0..6 {
            let stride = if i < rank {
                output_strides[i] as u32
            } else {
                0
            };
            immediates_data.extend_from_slice(&stride.to_le_bytes());
        }

        // perm (6 x u32, pad with 0)
        for i in 0..6 {
            let p = if i < rank { perm[i] as u32 } else { 0 };
            immediates_data.extend_from_slice(&p.to_le_bytes());
        }

        // Create dispatch step with bindings and immediates
        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0),
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(0),
                    read_only: false,
                },
            ],
            workgroups: [workgroups_x, workgroups_y, workgroups_z],
            immediates: Some(immediates_data),
        }])
    }
}

/// Compute row-major strides from shape.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::InferenceContext;
    use crate::plan::BufferRef;
    use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
    use std::collections::HashMap;

    #[test]
    fn test_transpose_kernel_shape_inference_2d() {
        let kernel = TransposeKernel;
        let mut node = Node::new("Transpose");
        node.attributes.insert(
            "perm".to_string(),
            onyxia_onnx::AttributeValue::Ints(vec![1i64, 0i64]),
        );

        let input_shapes = vec![TensorShape::Static(vec![4, 8])];

        let graph = onyxia_onnx::Graph::new();
        let output_shapes = kernel
            .infer_output_shapes(&{
                let input_values = vec![None; input_shapes.len()];
                InferenceContext::new(&node, &graph, input_shapes.clone(), input_values)
            })
            .expect("Shape inference should succeed");

        assert_eq!(output_shapes.len(), 1);
        assert_eq!(output_shapes[0], TensorShape::Static(vec![8, 4]));
    }

    #[test]
    fn test_transpose_kernel_shape_inference_3d() {
        let kernel = TransposeKernel;
        let mut node = Node::new("Transpose");
        node.attributes.insert(
            "perm".to_string(),
            onyxia_onnx::AttributeValue::Ints(vec![2i64, 0i64, 1i64]),
        );

        let input_shapes = vec![TensorShape::Static(vec![2, 3, 4])];

        let graph = onyxia_onnx::Graph::new();
        let output_shapes = kernel
            .infer_output_shapes(&{
                let input_values = vec![None; input_shapes.len()];
                InferenceContext::new(&node, &graph, input_shapes.clone(), input_values)
            })
            .expect("Shape inference should succeed");

        assert_eq!(output_shapes.len(), 1);
        assert_eq!(output_shapes[0], TensorShape::Static(vec![4, 2, 3]));
    }

    #[test]
    fn test_transpose_kernel_default_perm() {
        let kernel = TransposeKernel;
        let node = Node::new("Transpose"); // No perm attribute

        let input_shapes = vec![TensorShape::Static(vec![2, 3, 4])];

        let graph = onyxia_onnx::Graph::new();
        let output_shapes = kernel
            .infer_output_shapes(&{
                let input_values = vec![None; input_shapes.len()];
                InferenceContext::new(&node, &graph, input_shapes.clone(), input_values)
            })
            .expect("Shape inference should succeed");

        assert_eq!(output_shapes.len(), 1);
        // Default perm reverses dimensions: [2,3,4] -> [4,3,2]
        assert_eq!(output_shapes[0], TensorShape::Static(vec![4, 3, 2]));
    }

    fn create_transpose_test_graph_2d() -> Graph {
        let mut graph = Graph::new();

        // Input: [2, 3]
        graph.add_tensor(TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![2, 3]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Output: [3, 2]
        graph.add_tensor(TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![3, 2]),
            kind: TensorKind::Output,
            initializer: None,
        });

        graph.inputs = vec!["input".to_string()];
        graph.outputs = vec!["output".to_string()];

        graph
    }

    #[test]
    fn test_transpose_kernel_plan_2d() {
        let graph = create_transpose_test_graph_2d();
        let mut node = Node::new("Transpose");
        node.inputs = vec!["input".to_string()];
        node.outputs = vec!["output".to_string()];
        node.attributes.insert(
            "perm".to_string(),
            onyxia_onnx::AttributeValue::Ints(vec![1i64, 0i64]),
        );

        let input_ids = vec![0];
        let output_ids = vec![1];
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

        let steps = TransposeKernel
            .plan(&mut ctx)
            .expect("Planning should succeed");

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

                assert_eq!(bindings[0].buffer, BufferRef::Tensor(0));
                assert!(bindings[0].read_only);

                assert_eq!(bindings[1].buffer, BufferRef::Tensor(1));
                assert!(!bindings[1].read_only);

                // Verify workgroup count: ceil(6 / 256) = 1
                assert_eq!(*workgroups, [1, 1, 1]);

                // Verify immediates were encoded
                assert!(immediates.is_some());
                let imm = immediates.as_ref().unwrap();

                // Check structure: rank + num_elements + dispatch_size_x + 6 strides + 6 strides + 6 perm = 21 u32 = 84 bytes
                assert_eq!(imm.len(), 84);

                // Decode first two values
                let rank = u32::from_le_bytes([imm[0], imm[1], imm[2], imm[3]]);
                let num_elements = u32::from_le_bytes([imm[4], imm[5], imm[6], imm[7]]);

                assert_eq!(rank, 2);
                assert_eq!(num_elements, 6);
            }
            _ => panic!("Expected Dispatch step"),
        }

        // Verify shader was compiled
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].label, "transpose");
        assert_eq!(shaders[0].entry_point, "main");
    }

    fn create_transpose_test_graph_3d() -> Graph {
        let mut graph = Graph::new();

        // Input: [2, 3, 4]
        graph.add_tensor(TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![2, 3, 4]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Output: [4, 2, 3] (perm = [2, 0, 1])
        graph.add_tensor(TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4, 2, 3]),
            kind: TensorKind::Output,
            initializer: None,
        });

        graph.inputs = vec!["input".to_string()];
        graph.outputs = vec!["output".to_string()];

        graph
    }

    #[test]
    fn test_transpose_kernel_plan_3d() {
        let graph = create_transpose_test_graph_3d();
        let mut node = Node::new("Transpose");
        node.inputs = vec!["input".to_string()];
        node.outputs = vec!["output".to_string()];
        node.attributes.insert(
            "perm".to_string(),
            onyxia_onnx::AttributeValue::Ints(vec![2i64, 0i64, 1i64]),
        );

        let input_ids = vec![0];
        let output_ids = vec![1];
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

        let steps = TransposeKernel
            .plan(&mut ctx)
            .expect("Planning should succeed");

        assert_eq!(steps.len(), 1);

        match &steps[0] {
            Step::Dispatch {
                shader_index,
                bindings,
                workgroups,
                immediates,
            } => {
                assert_eq!(*shader_index, 0);
                assert_eq!(bindings.len(), 2);

                // 2*3*4 = 24 elements, ceil(24/256) = 1
                assert_eq!(*workgroups, [1, 1, 1]);

                // Verify immediates structure
                assert!(immediates.is_some());
                let imm = immediates.as_ref().unwrap();
                assert_eq!(imm.len(), 84);

                let rank = u32::from_le_bytes([imm[0], imm[1], imm[2], imm[3]]);
                let num_elements = u32::from_le_bytes([imm[4], imm[5], imm[6], imm[7]]);

                assert_eq!(rank, 3);
                assert_eq!(num_elements, 24);
            }
            _ => panic!("Expected Dispatch step"),
        }
    }

    #[test]
    fn test_compute_strides() {
        // Test 2D: [2, 3] -> strides [3, 1]
        let strides = compute_strides(&[2, 3]);
        assert_eq!(strides, vec![3, 1]);

        // Test 3D: [2, 3, 4] -> strides [12, 4, 1]
        let strides = compute_strides(&[2, 3, 4]);
        assert_eq!(strides, vec![12, 4, 1]);

        // Test 1D: [10] -> strides [1]
        let strides = compute_strides(&[10]);
        assert_eq!(strides, vec![1]);

        // Test 4D: [2, 3, 4, 5] -> strides [60, 20, 5, 1]
        let strides = compute_strides(&[2, 3, 4, 5]);
        assert_eq!(strides, vec![60, 20, 5, 1]);
    }
}
