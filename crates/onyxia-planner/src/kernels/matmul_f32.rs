//! MatMulF32Kernel implementation for f32 matrix multiplication.

use crate::error::Result;
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::TensorShape;
use std::collections::HashMap;

/// Kernel for f32 matrix multiplication (ONNX MatMul operator).
///
/// Computes C = A × B where:
/// - A: [M, K]
/// - B: [K, N]
/// - C: [M, N]
///
/// Uses tiled algorithm with shared memory for efficiency.
pub struct MatMulF32Kernel;

impl OpKernel for MatMulF32Kernel {
    fn name(&self) -> &str {
        "MatMulF32"
    }

    fn infer_output_shapes(
        &self,
        _graph: &onyxia_onnx::Graph,
        _node: &onyxia_onnx::Node,
        input_shapes: &[TensorShape],
    ) -> Result<Vec<TensorShape>> {
        // MatMul: [M, K] × [K, N] -> [M, N]
        if input_shapes.len() < 2 {
            return Err(crate::error::CodegenError::InvalidShape(
                "MatMul requires two inputs".to_string(),
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

        let b_dims = match &input_shapes[1] {
            TensorShape::Static(dims) => dims,
            TensorShape::Unknown | TensorShape::Absent => return Ok(vec![TensorShape::Unknown]),
            TensorShape::Dynamic(_) => {
                return Err(crate::error::CodegenError::InvalidShape(
                    "Unexpected Dynamic shape after dimension resolution".to_string(),
                ));
            }
        };

        if a_dims.len() < 2 || b_dims.len() < 2 {
            return Err(crate::error::CodegenError::InvalidShape(format!(
                "MatMul requires at least 2D tensors, got A: {:?}, B: {:?}",
                a_dims, b_dims
            )));
        }

        // Output shape: [M, N]
        let m = a_dims[a_dims.len() - 2];
        let n = b_dims[b_dims.len() - 1];
        Ok(vec![TensorShape::Static(vec![m, n])])
    }

    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        // Get input shapes
        let a_info = ctx.input_info(0)?;
        let a_shape = ctx.static_shape(&a_info.shape)?;
        let b_info = ctx.input_info(1)?;
        let b_shape = ctx.static_shape(&b_info.shape)?;

        // Validate shapes
        if a_shape.len() < 2 || b_shape.len() < 2 {
            return Err(crate::error::CodegenError::InvalidShape(format!(
                "MatMul expects 2D or higher dimensional inputs, got A: {:?}, B: {:?}",
                a_shape, b_shape
            )));
        }

        // Extract matrix dimensions (handle batch dimensions later if needed)
        // For now, assume 2D matrices
        let m = a_shape[a_shape.len() - 2];
        let k_a = a_shape[a_shape.len() - 1];
        let k_b = b_shape[b_shape.len() - 2];
        let n = b_shape[b_shape.len() - 1];

        // Validate K dimensions match
        if k_a != k_b {
            return Err(crate::error::CodegenError::InvalidShape(format!(
                "MatMul K dimensions must match, got A: [.., {}, {}], B: [.., {}, {}]",
                m, k_a, k_b, n
            )));
        }
        let k = k_a;

        // Tile sizes for workgroup computation
        let tile_m: u32 = 16;
        let tile_n: u32 = 16;
        let tile_k: u32 = 16;

        // Prepare shader definitions
        let mut shader_defs = HashMap::new();
        shader_defs.insert("TILE_M".to_string(), ShaderDefValue::UInt(tile_m));
        shader_defs.insert("TILE_N".to_string(), ShaderDefValue::UInt(tile_n));
        shader_defs.insert("TILE_K".to_string(), ShaderDefValue::UInt(tile_k));

        // Compile shader
        let shader_index = ctx.compile_shader(
            "matmul_f32",
            include_str!("../../shaders/matmul/matmul_f32.wgsl"),
            shader_defs,
        )?;

        // Calculate workgroup dimensions
        // Each workgroup computes a TILE_M × TILE_N tile of the output
        let workgroups_x = (n as u32 + tile_n - 1) / tile_n;
        let workgroups_y = (m as u32 + tile_m - 1) / tile_m;

        // Encode immediate data (must match ImmediateConstants struct in shader)
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(m as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(n as u32).to_le_bytes());
        immediates_data.extend_from_slice(&(k as u32).to_le_bytes());

        // Create dispatch step with bindings and immediates
        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0), // matrix A
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(1), // matrix B
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(0), // matrix C
                    read_only: false,
                },
            ],
            workgroups: [workgroups_x, workgroups_y, 1],
            immediates: Some(immediates_data),
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::BufferRef;
    use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
    use std::collections::HashMap;

    fn create_matmul_test_graph(m: usize, k: usize, n: usize) -> Graph {
        let mut graph = Graph::new();

        // Add matrix A [M, K]
        graph.add_tensor(TensorInfo {
            name: "A".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![m, k]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add matrix B [K, N]
        graph.add_tensor(TensorInfo {
            name: "B".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![k, n]),
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

        graph.inputs = vec!["A".to_string(), "B".to_string()];
        graph.outputs = vec!["C".to_string()];

        graph
    }

    #[test]
    fn test_matmul_kernel_plan() {
        let graph = create_matmul_test_graph(4, 8, 4);
        let mut node = Node::new("MatMul");
        node.inputs = vec!["A".to_string(), "B".to_string()];
        node.outputs = vec!["C".to_string()];

        let input_ids = vec![0, 1];
        let output_ids = vec![2];
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

        let steps = MatMulF32Kernel
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

                // Verify bindings: 2 read-only inputs + 1 read-write output
                assert_eq!(bindings.len(), 3);

                assert_eq!(bindings[0].buffer, BufferRef::Tensor(0));
                assert!(bindings[0].read_only);

                assert_eq!(bindings[1].buffer, BufferRef::Tensor(1));
                assert!(bindings[1].read_only);

                assert_eq!(bindings[2].buffer, BufferRef::Tensor(2));
                assert!(!bindings[2].read_only);

                // Verify workgroup count
                // TILE_M = TILE_N = 16
                // M=4, N=4: workgroups_x = ceil(4/16) = 1, workgroups_y = ceil(4/16) = 1
                assert_eq!(*workgroups, [1, 1, 1]);

                // Verify immediates contain M, N, K
                let imm = immediates.as_ref().unwrap();
                assert_eq!(imm.len(), 12); // 3 u32s = 12 bytes

                let m = u32::from_le_bytes([imm[0], imm[1], imm[2], imm[3]]);
                let n = u32::from_le_bytes([imm[4], imm[5], imm[6], imm[7]]);
                let k = u32::from_le_bytes([imm[8], imm[9], imm[10], imm[11]]);

                assert_eq!(m, 4);
                assert_eq!(n, 4);
                assert_eq!(k, 8);
            }
            _ => panic!("Expected Dispatch step"),
        }

        // Verify shader was compiled
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].label, "matmul_f32");
        assert_eq!(shaders[0].entry_point, "main");
    }

    #[test]
    fn test_matmul_kernel_large_matrices() {
        // Test with larger matrices that require multiple workgroups
        let graph = create_matmul_test_graph(64, 128, 96);
        let mut node = Node::new("MatMul");
        node.inputs = vec!["A".to_string(), "B".to_string()];
        node.outputs = vec!["C".to_string()];

        let input_ids = vec![0, 1];
        let output_ids = vec![2];
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

        let steps = MatMulF32Kernel
            .plan(&mut ctx)
            .expect("Planning should succeed");

        match &steps[0] {
            Step::Dispatch {
                workgroups,
                immediates,
                ..
            } => {
                // M=64, N=96, TILE_M=16, TILE_N=16
                // workgroups_x = ceil(96/16) = 6
                // workgroups_y = ceil(64/16) = 4
                assert_eq!(*workgroups, [6, 4, 1]);

                // Verify dimensions
                let imm = immediates.as_ref().unwrap();
                let m = u32::from_le_bytes([imm[0], imm[1], imm[2], imm[3]]);
                let n = u32::from_le_bytes([imm[4], imm[5], imm[6], imm[7]]);
                let k = u32::from_le_bytes([imm[8], imm[9], imm[10], imm[11]]);

                assert_eq!(m, 64);
                assert_eq!(n, 96);
                assert_eq!(k, 128);
            }
            _ => panic!("Expected Dispatch step"),
        }
    }

    #[test]
    fn test_matmul_kernel_invalid_shapes() {
        let mut graph = Graph::new();

        // Create matrices with mismatched K dimensions
        graph.add_tensor(TensorInfo {
            name: "A".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4, 8]),
            kind: TensorKind::Input,
            initializer: None,
        });

        graph.add_tensor(TensorInfo {
            name: "B".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![16, 4]), // K=16 doesn't match A's K=8
            kind: TensorKind::Input,
            initializer: None,
        });

        graph.add_tensor(TensorInfo {
            name: "C".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4, 4]),
            kind: TensorKind::Output,
            initializer: None,
        });

        graph.inputs = vec!["A".to_string(), "B".to_string()];
        graph.outputs = vec!["C".to_string()];

        let mut node = Node::new("MatMul");
        node.inputs = vec!["A".to_string(), "B".to_string()];
        node.outputs = vec!["C".to_string()];

        let input_ids = vec![0, 1];
        let output_ids = vec![2];
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

        // Should fail due to mismatched K dimensions
        let result = MatMulF32Kernel.plan(&mut ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_kernel_square_matrices() {
        // Test with square matrices
        let graph = create_matmul_test_graph(32, 32, 32);
        let mut node = Node::new("MatMul");
        node.inputs = vec!["A".to_string(), "B".to_string()];
        node.outputs = vec!["C".to_string()];

        let input_ids = vec![0, 1];
        let output_ids = vec![2];
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

        let steps = MatMulF32Kernel
            .plan(&mut ctx)
            .expect("Planning should succeed");

        match &steps[0] {
            Step::Dispatch { workgroups, .. } => {
                // 32x32 matrix with TILE_M=TILE_N=16
                // workgroups_x = ceil(32/16) = 2
                // workgroups_y = ceil(32/16) = 2
                assert_eq!(*workgroups, [2, 2, 1]);
            }
            _ => panic!("Expected Dispatch step"),
        }
    }
}
