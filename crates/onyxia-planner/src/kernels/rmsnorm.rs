//! RmsNormKernel implementation for RMS normalization.

use crate::error::Result;
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::TensorShape;
use std::collections::HashMap;

/// Kernel for RMS Normalization (ONNX SimplifiedLayerNormalization operator).
///
/// RMSNorm(x) = x / RMS(x) * weight where RMS(x) = sqrt(mean(x²) + epsilon).
/// Used in modern LLMs like Llama and Gemma.
pub struct RmsNormKernel;

impl OpKernel for RmsNormKernel {
    fn name(&self) -> &str {
        "RMSNorm"
    }

    fn infer_output_shapes(
        &self,
        _node: &onyxia_onnx::Node,
        input_shapes: &[TensorShape],
    ) -> Result<Vec<TensorShape>> {
        // RMSNorm normalizes over the last dimension: output shape equals input shape
        if input_shapes.is_empty() {
            return Err(crate::error::CodegenError::InvalidShape(
                "RMSNorm requires at least one input".to_string(),
            ));
        }
        Ok(vec![input_shapes[0].clone()])
    }

    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        // Extract epsilon from node attributes (default to 1e-5 if not present)
        let epsilon: f32 = ctx.node.attr("epsilon").unwrap_or(1e-5);

        // Get input shape: [batch, seq_len, hidden_dim]
        let input_info = ctx.input_info(0)?;
        let input_shape = ctx.static_shape(&input_info.shape)?;

        // Validate input shape
        if input_shape.len() < 2 {
            return Err(crate::error::CodegenError::InvalidShape(format!(
                "RMSNorm expects input with at least 2 dimensions, got {:?}",
                input_shape
            )));
        }

        // Calculate dimensions
        // For 3D: [batch, seq_len, hidden_dim]
        // For 2D: [batch_seq, hidden_dim]
        let hidden_dim = *input_shape.last().unwrap();
        let batch_seq: usize = input_shape[..input_shape.len() - 1].iter().product();

        // Split batch_seq into batch and seq_len for the shader
        // For simplicity, we'll use batch=1 and seq_len=batch_seq
        let batch_size = 1u32;
        let seq_len = batch_seq as u32;

        // Configure workgroup size (256 threads per workgroup)
        let workgroup_size: u32 = 256;
        let mut shader_defs = HashMap::new();
        shader_defs.insert(
            "WORKGROUP_SIZE".to_string(),
            ShaderDefValue::UInt(workgroup_size),
        );

        // Compile shader
        let shader_index = ctx.compile_shader(
            "rmsnorm",
            include_str!("../../shaders/normalization/rmsnorm.wgsl"),
            shader_defs,
        )?;

        // Each workgroup handles one [batch, seq] position
        let num_workgroups = batch_seq as u32;

        // Encode immediate data (must match ImmediateConstants struct in shader)
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&batch_size.to_le_bytes());
        immediates_data.extend_from_slice(&seq_len.to_le_bytes());
        immediates_data.extend_from_slice(&(hidden_dim as u32).to_le_bytes());
        immediates_data.extend_from_slice(&epsilon.to_le_bytes());

        // Create dispatch step with bindings and immediates
        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0), // input data
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.input(1), // weight/scale
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::BufferRef;
    use onyxia_onnx::{AttributeValue, DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
    use std::collections::HashMap;

    fn create_rmsnorm_test_graph() -> Graph {
        let mut graph = Graph::new();

        // Add input tensor [2, 768]
        graph.add_tensor(TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![2, 768]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add weight tensor [768]
        graph.add_tensor(TensorInfo {
            name: "weight".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![768]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add output tensor [2, 768]
        graph.add_tensor(TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![2, 768]),
            kind: TensorKind::Output,
            initializer: None,
        });

        graph.inputs = vec!["input".to_string(), "weight".to_string()];
        graph.outputs = vec!["output".to_string()];

        graph
    }

    #[test]
    fn test_rmsnorm_kernel_plan() {
        let graph = create_rmsnorm_test_graph();
        let mut node = Node::new("SimplifiedLayerNormalization");
        node.inputs = vec!["input".to_string(), "weight".to_string()];
        node.outputs = vec!["output".to_string()];
        node.attributes
            .insert("epsilon".to_string(), AttributeValue::Float(1e-6));

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

        let steps = RmsNormKernel
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

                // Verify workgroup count: 2 (one per batch_seq position)
                assert_eq!(*workgroups, [2, 1, 1]);

                // Verify immediates contain correct values
                let imm = immediates.as_ref().unwrap();
                assert_eq!(imm.len(), 16); // 3 u32s + 1 f32 = 16 bytes

                let batch_size = u32::from_le_bytes([imm[0], imm[1], imm[2], imm[3]]);
                let seq_len = u32::from_le_bytes([imm[4], imm[5], imm[6], imm[7]]);
                let hidden_dim = u32::from_le_bytes([imm[8], imm[9], imm[10], imm[11]]);
                let epsilon = f32::from_le_bytes([imm[12], imm[13], imm[14], imm[15]]);

                assert_eq!(batch_size, 1);
                assert_eq!(seq_len, 2);
                assert_eq!(hidden_dim, 768);
                assert_eq!(epsilon, 1e-6);
            }
            _ => panic!("Expected Dispatch step"),
        }

        // Verify shader was compiled
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].label, "rmsnorm");
        assert_eq!(shaders[0].entry_point, "main");
    }

    #[test]
    fn test_rmsnorm_kernel_default_epsilon() {
        let graph = create_rmsnorm_test_graph();
        let mut node = Node::new("SimplifiedLayerNormalization");
        node.inputs = vec!["input".to_string(), "weight".to_string()];
        node.outputs = vec!["output".to_string()];
        // No epsilon attribute - should use default 1e-5

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

        let steps = RmsNormKernel
            .plan(&mut ctx)
            .expect("Planning should succeed");

        match &steps[0] {
            Step::Dispatch { immediates, .. } => {
                let imm = immediates.as_ref().unwrap();
                let epsilon = f32::from_le_bytes([imm[12], imm[13], imm[14], imm[15]]);
                assert_eq!(epsilon, 1e-5); // Default value
            }
            _ => panic!("Expected Dispatch step"),
        }
    }

    #[test]
    fn test_rmsnorm_kernel_3d_input() {
        let mut graph = Graph::new();

        // Add 3D input tensor [1, 4, 512]
        graph.add_tensor(TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 4, 512]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add weight tensor [512]
        graph.add_tensor(TensorInfo {
            name: "weight".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![512]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add output tensor [1, 4, 512]
        graph.add_tensor(TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 4, 512]),
            kind: TensorKind::Output,
            initializer: None,
        });

        graph.inputs = vec!["input".to_string(), "weight".to_string()];
        graph.outputs = vec!["output".to_string()];

        let mut node = Node::new("SimplifiedLayerNormalization");
        node.inputs = vec!["input".to_string(), "weight".to_string()];
        node.outputs = vec!["output".to_string()];

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

        let steps = RmsNormKernel
            .plan(&mut ctx)
            .expect("Planning should succeed");

        match &steps[0] {
            Step::Dispatch {
                workgroups,
                immediates,
                ..
            } => {
                // Verify workgroup count: 4 (batch=1 * seq_len=4)
                assert_eq!(*workgroups, [4, 1, 1]);

                // Verify dimensions
                let imm = immediates.as_ref().unwrap();
                let seq_len = u32::from_le_bytes([imm[4], imm[5], imm[6], imm[7]]);
                let hidden_dim = u32::from_le_bytes([imm[8], imm[9], imm[10], imm[11]]);

                assert_eq!(seq_len, 4); // batch * seq_len = 1 * 4
                assert_eq!(hidden_dim, 512);
            }
            _ => panic!("Expected Dispatch step"),
        }
    }
}
