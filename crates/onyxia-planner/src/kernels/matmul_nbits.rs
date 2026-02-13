//! MatMulNBitsKernel implementation for N-bit quantized matrix multiplication.
//!
//! This is a Microsoft ONNX Runtime contrib operator that performs matrix
//! multiplication with quantized weights. The most common use case is Q4
//! quantization (4-bit weights) for efficient LLM inference.

use crate::error::Result;
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::TensorShape;
use std::collections::HashMap;

/// Kernel for N-bit quantized matrix multiplication (ONNX MatMulNBits operator).
///
/// Computes C = A × B where:
/// - A: [M, K] — activations (f32)
/// - B: quantized weights in packed format
/// - scales: per-block dequantization scales
/// - zero_points: optional per-block zero points
/// - C: [M, N] — output (f32)
///
/// The weights are stored as N-bit integers, packed multiple values per byte.
/// Each block of weights shares a scale and zero_point for dequantization.
pub struct MatMulNBitsKernel;

impl OpKernel for MatMulNBitsKernel {
    fn name(&self) -> &str {
        "MatMulNBits"
    }

    fn infer_output_shapes(
        &self,        _graph: &onyxia_onnx::Graph,        node: &onyxia_onnx::Node,
        input_shapes: &[TensorShape],
    ) -> Result<Vec<TensorShape>> {
        // MatMulNBits: [M, K] × quantized[N, ...] -> [M, N]
        // N and K are provided as attributes
        if input_shapes.is_empty() {
            return Err(crate::error::CodegenError::InvalidShape(
                "MatMulNBits requires at least one input (activations)".to_string(),
            ));
        }

        // Extract static dimensions (Phase 1 already resolved Dynamic dims)
        let a_dims = match &input_shapes[0] {
            TensorShape::Static(dims) => dims,
            TensorShape::Unknown | TensorShape::Absent => return Ok(vec![TensorShape::Unknown]),
            TensorShape::Dynamic(_) => {
                return Err(crate::error::CodegenError::InvalidShape(
                    "Unexpected Dynamic shape after dimension resolution".to_string(),
                ));
            }
        };

        if a_dims.len() < 2 {
            return Err(crate::error::CodegenError::InvalidShape(format!(
                "MatMulNBits requires at least 2D activation tensor, got: {:?}",
                a_dims
            )));
        }

        // Read N from attributes (output dimension)
        let n = node
            .attr::<i64>("N")
            .map_err(|e| crate::error::CodegenError::OnnxError(e))? as usize;

        // Output shape: [M, N]
        let m = a_dims[a_dims.len() - 2];
        Ok(vec![TensorShape::Static(vec![m, n])])
    }

    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        // Get input shapes
        let a_info = ctx.input_info(0)?;
        let a_shape = ctx.static_shape(&a_info.shape)?;

        // Validate shape
        if a_shape.len() < 2 {
            return Err(crate::error::CodegenError::InvalidShape(format!(
                "MatMulNBits expects at least 2D activation input, got: {:?}",
                a_shape
            )));
        }

        // Extract matrix dimensions
        let m = a_shape[a_shape.len() - 2];
        let k_a = a_shape[a_shape.len() - 1];

        // Read attributes
        let k_attr = ctx
            .node
            .attr::<i64>("K")
            .map_err(|e| crate::error::CodegenError::OnnxError(e))?;
        let n = ctx
            .node
            .attr::<i64>("N")
            .map_err(|e| crate::error::CodegenError::OnnxError(e))?;
        let bits = ctx
            .node
            .attr::<i64>("bits")
            .map_err(|e| crate::error::CodegenError::OnnxError(e))?;
        let block_size = ctx
            .node
            .attr::<i64>("block_size")
            .map_err(|e| crate::error::CodegenError::OnnxError(e))?;

        // Validate K dimension
        if k_a != k_attr as usize {
            return Err(crate::error::CodegenError::InvalidShape(format!(
                "MatMulNBits K dimension mismatch: activation has K={}, attribute says K={}",
                k_a, k_attr
            )));
        }
        let k = k_a;

        // Validate bits (only Q4 supported for now)
        if bits != 4 {
            return Err(crate::error::CodegenError::UnsupportedOp(format!(
                "MatMulNBits currently only supports 4-bit quantization, got {} bits",
                bits
            )));
        }

        // Check if zero_points are provided
        let has_zero_points = ctx.node.inputs.len() >= 4 && !ctx.node.inputs[3].is_empty();

        // Calculate quantization parameters
        let n_blocks_per_col = ((k as u32 + block_size as u32 - 1) / block_size as u32) as u32;

        // Workgroup sizes
        let workgroup_x: u32 = 16;
        let workgroup_y: u32 = 16;

        // Prepare shader definitions
        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_X".to_string(), ShaderDefValue::UInt(workgroup_x));
        shader_defs.insert("WORKGROUP_Y".to_string(), ShaderDefValue::UInt(workgroup_y));
        if has_zero_points {
            shader_defs.insert("HAS_ZERO_POINTS".to_string(), ShaderDefValue::Bool(true));
        }

        // Compile shader
        let shader_index = ctx.compile_shader(
            "matmul_q4",
            include_str!("../../shaders/matmul/matmul_q4.wgsl"),
            shader_defs,
        )?;

        // Calculate workgroup dimensions
        let workgroups_x = (n as u32 + workgroup_x - 1) / workgroup_x;
        let workgroups_y = (m as u32 + workgroup_y - 1) / workgroup_y;

        // Encode immediate data (must match ImmediateConstants struct in shader)
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(m as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(n as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(k as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(block_size as u32).to_le_bytes());
        immediates_data.extend_from_slice(&n_blocks_per_col.to_le_bytes());

        // Create bindings based on whether zero_points are provided
        let mut bindings = vec![
            BindingDesc {
                buffer: ctx.input(0), // matrix A (activations)
                read_only: true,
            },
            BindingDesc {
                buffer: ctx.input(1), // packed_weights
                read_only: true,
            },
            BindingDesc {
                buffer: ctx.input(2), // scales
                read_only: true,
            },
        ];

        if has_zero_points {
            bindings.push(BindingDesc {
                buffer: ctx.input(3), // zero_points
                read_only: true,
            });
        }

        bindings.push(BindingDesc {
            buffer: ctx.output(0), // matrix C (output)
            read_only: false,
        });

        // Create dispatch step
        Ok(vec![Step::Dispatch {
            shader_index,
            bindings,
            workgroups: [workgroups_x, workgroups_y, 1],
            immediates: Some(immediates_data),
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::BufferRef;
    use onyxia_onnx::{AttributeValue, DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
    use std::collections::HashMap;

    fn create_matmul_nbits_test_graph(m: usize, k: usize, n: usize) -> Graph {
        let mut graph = Graph::new();

        // Add matrix A [M, K] - activations
        graph.add_tensor(TensorInfo {
            name: "A".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![m, k]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add packed weights (shape calculation based on Q4 format)
        let block_size = 32;
        let n_blocks_per_col = (k + block_size - 1) / block_size;
        let blob_size = block_size / 2; // 4 bits per weight, 2 weights per byte
        graph.add_tensor(TensorInfo {
            name: "B".to_string(),
            dtype: DataType::U8,
            shape: TensorShape::Static(vec![n, n_blocks_per_col, blob_size]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add scales [N, n_blocks_per_col]
        graph.add_tensor(TensorInfo {
            name: "scales".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![n, n_blocks_per_col]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add zero_points [N, n_blocks_per_col_padded] - optional
        graph.add_tensor(TensorInfo {
            name: "zero_points".to_string(),
            dtype: DataType::U8,
            shape: TensorShape::Static(vec![n, n_blocks_per_col]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add output matrix C [M, N]
        graph.add_tensor(TensorInfo {
            name: "C".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![m, n]),
            kind: TensorKind::Output,
            initializer: None,
        });

        graph.inputs = vec!["A".to_string(), "B".to_string(), "scales".to_string()];
        graph.outputs = vec!["C".to_string()];

        graph
    }

    #[test]
    fn test_matmul_nbits_kernel_attributes() {
        // Test that we can correctly read MatMulNBits attributes
        let mut node = Node::new("MatMulNBits");
        node.attributes
            .insert("K".to_string(), AttributeValue::Int(8));
        node.attributes
            .insert("N".to_string(), AttributeValue::Int(16));
        node.attributes
            .insert("bits".to_string(), AttributeValue::Int(4));
        node.attributes
            .insert("block_size".to_string(), AttributeValue::Int(32));

        let k: i64 = node.attr("K").unwrap();
        let n: i64 = node.attr("N").unwrap();
        let bits: i64 = node.attr("bits").unwrap();
        let block_size: i64 = node.attr("block_size").unwrap();

        assert_eq!(k, 8);
        assert_eq!(n, 16);
        assert_eq!(bits, 4);
        assert_eq!(block_size, 32);
    }

    #[test]
    fn test_matmul_nbits_kernel_plan() {
        let graph = create_matmul_nbits_test_graph(4, 8, 16);
        let mut node = Node::new("MatMulNBits");
        node.inputs = vec![
            "A".to_string(),
            "B".to_string(),
            "scales".to_string(),
            "zero_points".to_string(),
        ];
        node.outputs = vec!["C".to_string()];

        // Add attributes
        node.attributes
            .insert("K".to_string(), AttributeValue::Int(8));
        node.attributes
            .insert("N".to_string(), AttributeValue::Int(16));
        node.attributes
            .insert("bits".to_string(), AttributeValue::Int(4));
        node.attributes
            .insert("block_size".to_string(), AttributeValue::Int(32));

        let input_ids = vec![0, 1, 2, 3];
        let output_ids = vec![4];
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

        let steps = MatMulNBitsKernel
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

                // Verify bindings: 4 read-only inputs + 1 read-write output
                assert_eq!(bindings.len(), 5);

                assert_eq!(bindings[0].buffer, BufferRef::Tensor(0)); // A
                assert!(bindings[0].read_only);

                assert_eq!(bindings[1].buffer, BufferRef::Tensor(1)); // B
                assert!(bindings[1].read_only);

                assert_eq!(bindings[2].buffer, BufferRef::Tensor(2)); // scales
                assert!(bindings[2].read_only);

                assert_eq!(bindings[3].buffer, BufferRef::Tensor(3)); // zero_points
                assert!(bindings[3].read_only);

                assert_eq!(bindings[4].buffer, BufferRef::Tensor(4)); // C
                assert!(!bindings[4].read_only);

                // Verify workgroup count
                // Output is [4, 16], workgroup size is [16, 16]
                // workgroups_x = (16 + 15) / 16 = 1
                // workgroups_y = (4 + 15) / 16 = 1
                assert_eq!(workgroups[0], 1);
                assert_eq!(workgroups[1], 1);
                assert_eq!(workgroups[2], 1);

                // Verify immediates are present
                assert!(immediates.is_some());
                let imm = immediates.as_ref().unwrap();
                // Should have 5 u32 values: M, N, K, block_size, n_blocks_per_col
                assert_eq!(imm.len(), 5 * 4);
            }
            _ => panic!("Expected a Dispatch step"),
        }
    }

    #[test]
    fn test_matmul_nbits_kernel_no_zero_points() {
        // Test the 3-input variant (without zero_points)
        let graph = create_matmul_nbits_test_graph(4, 8, 16);
        let mut node = Node::new("MatMulNBits");
        node.inputs = vec!["A".to_string(), "B".to_string(), "scales".to_string()];
        node.outputs = vec!["C".to_string()];

        // Add attributes
        node.attributes
            .insert("K".to_string(), AttributeValue::Int(8));
        node.attributes
            .insert("N".to_string(), AttributeValue::Int(16));
        node.attributes
            .insert("bits".to_string(), AttributeValue::Int(4));
        node.attributes
            .insert("block_size".to_string(), AttributeValue::Int(32));

        let input_ids = vec![0, 1, 2];
        let output_ids = vec![4];
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

        let steps = MatMulNBitsKernel
            .plan(&mut ctx)
            .expect("Planning should succeed");

        // Verify we got exactly one dispatch step
        assert_eq!(steps.len(), 1);

        match &steps[0] {
            Step::Dispatch {
                shader_index: _,
                bindings,
                workgroups: _,
                immediates: _,
            } => {
                // Verify bindings: 3 read-only inputs + 1 read-write output (no zero_points)
                assert_eq!(bindings.len(), 4);

                assert_eq!(bindings[0].buffer, BufferRef::Tensor(0)); // A
                assert_eq!(bindings[1].buffer, BufferRef::Tensor(1)); // B
                assert_eq!(bindings[2].buffer, BufferRef::Tensor(2)); // scales
                assert_eq!(bindings[3].buffer, BufferRef::Tensor(4)); // C (output)
            }
            _ => panic!("Expected a Dispatch step"),
        }
    }

    #[test]
    fn test_matmul_nbits_infer_output_shapes() {
        let mut node = Node::new("MatMulNBits");
        node.attributes
            .insert("K".to_string(), AttributeValue::Int(8));
        node.attributes
            .insert("N".to_string(), AttributeValue::Int(16));
        node.attributes
            .insert("bits".to_string(), AttributeValue::Int(4));
        node.attributes
            .insert("block_size".to_string(), AttributeValue::Int(32));

        let input_shapes = vec![TensorShape::Static(vec![4, 8])];

        let kernel = MatMulNBitsKernel;
        let graph = onyxia_onnx::Graph::new();
        let output_shapes = kernel
            .infer_output_shapes(&graph, &node, &input_shapes)
            .expect("Shape inference should succeed");

        assert_eq!(output_shapes.len(), 1);
        assert_eq!(output_shapes[0], TensorShape::Static(vec![4, 16]));
    }
}
