//! RotaryEmbeddingKernel implementation for Rotary Position Embedding (RoPE).

use crate::error::Result;
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::TensorShape;
use std::collections::HashMap;

/// Kernel for Rotary Position Embedding (Microsoft contrib op "RotaryEmbedding").
///
/// Applies rotation to pairs of embedding dimensions based on position.
/// Used in transformer attention to inject positional information.
///
/// Inputs:
/// - input: tensor to rotate, typically [batch, seq_len, num_heads, head_dim]
/// - position_ids: position indices [batch, seq_len] (I64)
/// - cos_cache: precomputed cosine values [max_seq, head_dim/2]
/// - sin_cache: precomputed sine values [max_seq, head_dim/2]
///
/// Attributes:
/// - num_heads: number of attention heads
/// - interleaved: 0 = split-half layout (default), 1 = interleaved pairs
pub struct RotaryEmbeddingKernel;

impl OpKernel for RotaryEmbeddingKernel {
    fn name(&self) -> &str {
        "RotaryEmbedding"
    }

    fn infer_output_shapes(
        &self,
        _node: &onyxia_onnx::Node,
        input_shapes: &[TensorShape],
        _dynamic_dimensions: &HashMap<String, usize>,
    ) -> Result<Vec<TensorShape>> {
        // Output shape = input shape (same tensor with rotated values)
        if input_shapes.is_empty() {
            return Err(crate::error::CodegenError::InvalidShape(
                "RotaryEmbedding requires at least one input".to_string(),
            ));
        }
        Ok(vec![input_shapes[0].clone()])
    }

    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        // Read attributes
        let num_heads: i64 = ctx.node.attr("num_heads")?;
        let interleaved: i64 = ctx.node.attr("interleaved").unwrap_or(0); // Default: split-half

        // Get input shape: [batch, seq_len, num_heads, head_dim]
        let input_info = ctx.input_info(0)?;
        let input_shape = ctx.resolve_shape(&input_info.shape)?;

        // Validate input shape
        if input_shape.len() != 4 {
            return Err(crate::error::CodegenError::InvalidShape(format!(
                "RotaryEmbedding expects 4D input [batch, seq_len, num_heads, head_dim], got shape {:?}",
                input_shape
            )));
        }

        let batch_size = input_shape[0] as u32;
        let seq_len = input_shape[1] as u32;
        let heads_in_shape = input_shape[2] as u32;
        let head_dim = input_shape[3] as u32;

        // Validate num_heads attribute matches input shape
        if heads_in_shape != num_heads as u32 {
            return Err(crate::error::CodegenError::InvalidShape(format!(
                "RotaryEmbedding: num_heads attribute ({}) doesn't match input shape num_heads ({})",
                num_heads, heads_in_shape
            )));
        }

        // Calculate total number of threads needed (one per pair)
        let total_pairs = batch_size * seq_len * (num_heads as u32) * (head_dim / 2);

        // Configure workgroup size
        let workgroup_size: u32 = 256;
        let num_workgroups = total_pairs.div_ceil(workgroup_size);

        // Prepare shader definitions
        let mut shader_defs = HashMap::new();
        shader_defs.insert(
            "WORKGROUP_SIZE".to_string(),
            ShaderDefValue::UInt(workgroup_size),
        );

        // Compile shader
        let shader_index = ctx.compile_shader(
            "rotary_embedding",
            include_str!("../../shaders/attention/rotary_embedding.wgsl"),
            shader_defs,
        )?;

        // Encode immediate data (must match ImmediateConstants struct in shader)
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&batch_size.to_le_bytes());
        immediates_data.extend_from_slice(&seq_len.to_le_bytes());
        immediates_data.extend_from_slice(&(num_heads as u32).to_le_bytes());
        immediates_data.extend_from_slice(&head_dim.to_le_bytes());
        immediates_data.extend_from_slice(&(interleaved as u32).to_le_bytes());

        // Create dispatch step with bindings and immediates
        // Bindings order must match shader: input, position_ids, cos_cache, sin_cache, output
        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0), // input tensor
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(1), // position_ids (I64)
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(2), // cos_cache
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(3), // sin_cache
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(0), // output tensor
                    read_only: false,
                },
            ],
            workgroups: [num_workgroups, 1, 1],
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

    fn create_rotary_embedding_test_graph() -> (Graph, Node) {
        let mut graph = Graph::new();

        // Input tensor: [batch=1, seq_len=2, num_heads=1, head_dim=4]
        graph.add_tensor(TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 2, 1, 4]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Position IDs: [batch=1, seq_len=2] (I64)
        graph.add_tensor(TensorInfo {
            name: "position_ids".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![1, 2]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Cos cache: [max_seq=4, head_dim/2=2]
        graph.add_tensor(TensorInfo {
            name: "cos_cache".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4, 2]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Sin cache: [max_seq=4, head_dim/2=2]
        graph.add_tensor(TensorInfo {
            name: "sin_cache".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4, 2]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Output tensor: same shape as input
        graph.add_tensor(TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 2, 1, 4]),
            kind: TensorKind::Output,
            initializer: None,
        });

        let mut node = Node::new("RotaryEmbedding");
        node.inputs = vec![
            "input".to_string(),
            "position_ids".to_string(),
            "cos_cache".to_string(),
            "sin_cache".to_string(),
        ];
        node.outputs = vec!["output".to_string()];
        node.attributes
            .insert("num_heads".to_string(), AttributeValue::Int(1));
        node.attributes
            .insert("interleaved".to_string(), AttributeValue::Int(0));

        (graph, node)
    }

    #[test]
    fn test_rotary_embedding_kernel_shape_inference() {
        let kernel = RotaryEmbeddingKernel;
        let node = Node::new("RotaryEmbedding");
        let input_shapes = vec![
            TensorShape::Static(vec![1, 8, 4, 256]), // input
            TensorShape::Static(vec![1, 8]),         // position_ids
            TensorShape::Static(vec![8192, 128]),    // cos_cache
            TensorShape::Static(vec![8192, 128]),    // sin_cache
        ];
        let dynamic_dimensions = HashMap::new();

        let output_shapes = kernel
            .infer_output_shapes(&node, &input_shapes, &dynamic_dimensions)
            .expect("Shape inference should succeed");

        assert_eq!(output_shapes.len(), 1);
        assert_eq!(output_shapes[0], TensorShape::Static(vec![1, 8, 4, 256]));
    }

    #[test]
    fn test_rotary_embedding_kernel_plan() {
        let (graph, node) = create_rotary_embedding_test_graph();

        let input_ids = vec![0, 1, 2, 3]; // input, position_ids, cos_cache, sin_cache
        let output_ids = vec![4]; // output
        let dynamic_dimensions = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let steps = RotaryEmbeddingKernel
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

                assert_eq!(bindings[0].buffer, BufferRef::Tensor(0)); // input
                assert!(bindings[0].read_only);

                assert_eq!(bindings[1].buffer, BufferRef::Tensor(1)); // position_ids
                assert!(bindings[1].read_only);

                assert_eq!(bindings[2].buffer, BufferRef::Tensor(2)); // cos_cache
                assert!(bindings[2].read_only);

                assert_eq!(bindings[3].buffer, BufferRef::Tensor(3)); // sin_cache
                assert!(bindings[3].read_only);

                assert_eq!(bindings[4].buffer, BufferRef::Tensor(4)); // output
                assert!(!bindings[4].read_only);

                // Verify workgroup count
                // Input shape: [1, 2, 1, 4] -> total_pairs = 1 * 2 * 1 * (4/2) = 4
                // Workgroups = ceil(4 / 256) = 1
                assert_eq!(*workgroups, [1, 1, 1]);

                // Verify immediates are present
                assert!(immediates.is_some());
                let imm_data = immediates.as_ref().unwrap();
                // 5 u32 values: batch_size, seq_len, num_heads, head_dim, interleaved
                assert_eq!(imm_data.len(), 5 * 4); // 5 fields * 4 bytes each
            }
            _ => panic!("Expected Dispatch step"),
        }

        // Verify shader was compiled
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].label, "rotary_embedding");
        assert_eq!(shaders[0].entry_point, "main");
    }

    #[test]
    fn test_rotary_embedding_kernel_attributes() {
        let (graph, mut node) = create_rotary_embedding_test_graph();

        // Test with different num_heads
        node.attributes
            .insert("num_heads".to_string(), AttributeValue::Int(1));
        node.attributes
            .insert("interleaved".to_string(), AttributeValue::Int(1));

        let input_ids = vec![0, 1, 2, 3];
        let output_ids = vec![4];
        let dynamic_dimensions = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let result = RotaryEmbeddingKernel.plan(&mut ctx);
        assert!(
            result.is_ok(),
            "Planning with valid attributes should succeed"
        );
    }

    #[test]
    fn test_rotary_embedding_default_interleaved() {
        let (graph, mut node) = create_rotary_embedding_test_graph();

        // Remove interleaved attribute to test default value
        node.attributes.remove("interleaved");

        let input_ids = vec![0, 1, 2, 3];
        let output_ids = vec![4];
        let dynamic_dimensions = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let result = RotaryEmbeddingKernel.plan(&mut ctx);
        assert!(
            result.is_ok(),
            "Planning with missing interleaved attribute should use default (0)"
        );
    }

    #[test]
    fn test_rotary_embedding_invalid_shape() {
        let mut graph = Graph::new();

        // Invalid input shape: 3D instead of 4D
        graph.add_tensor(TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 2, 4]),
            kind: TensorKind::Input,
            initializer: None,
        });

        graph.add_tensor(TensorInfo {
            name: "position_ids".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![1, 2]),
            kind: TensorKind::Input,
            initializer: None,
        });

        graph.add_tensor(TensorInfo {
            name: "cos_cache".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4, 2]),
            kind: TensorKind::Input,
            initializer: None,
        });

        graph.add_tensor(TensorInfo {
            name: "sin_cache".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4, 2]),
            kind: TensorKind::Input,
            initializer: None,
        });

        graph.add_tensor(TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 2, 4]),
            kind: TensorKind::Output,
            initializer: None,
        });

        let mut node = Node::new("RotaryEmbedding");
        node.inputs = vec![
            "input".to_string(),
            "position_ids".to_string(),
            "cos_cache".to_string(),
            "sin_cache".to_string(),
        ];
        node.outputs = vec!["output".to_string()];
        node.attributes
            .insert("num_heads".to_string(), AttributeValue::Int(1));

        let input_ids = vec![0, 1, 2, 3];
        let output_ids = vec![4];
        let dynamic_dimensions = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let result = RotaryEmbeddingKernel.plan(&mut ctx);
        assert!(
            result.is_err(),
            "Planning with invalid input shape should fail"
        );
    }
}
